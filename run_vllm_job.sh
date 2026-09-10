#!/usr/bin/env bash
# Start the vLLM generation Jobs: `VLLM_REPLICAS` identical Jobs (default 2), each one GPU, LoRA serving on, the
# shared bucket mounted read-only, port 8000 exposed through the jobs proxy. Prints one Job ID per line on stdout
# and nothing else, so `run_all.sh` can capture them. The trainer-side `lora_proxy.py` fans requests out across
# the replicas and broadcasts every adapter load to all of them, so from trl's point of view this is one server.
#
#   ./run_vllm_job.sh            # uses the defaults below
#   VLLM_REPLICAS=1 ./run_vllm_job.sh
#
# This side installs nothing. It needs no trl and no worker extension: on the adapter-only sync path the
# trainer hands vLLM a *path* over `/v1/load_lora_adapter`, and vLLM's own filesystem LoRA loader does the
# rest. That is the whole reason the bucket works here -- the transport is a directory, not a NCCL group.
set -euo pipefail

cd "$(dirname "$0")"

BUCKET=${BUCKET:-aminediroHF/asyncgrpo-lora-buckets}
MODEL=${MODEL:-Qwen/Qwen2.5-Math-1.5B}
# Pinned, not `latest`: vLLM 0.28.0 dropped `NCCLTrainerSendWeightsArgs`, which trl's weight transfer imports
# at module load, so the trainer side cannot even import `trl.experimental.async_grpo` against it. 0.27.1 is
# also the version this branch's LoRA path was validated on.
VLLM_TAG=${VLLM_TAG:-v0.27.1}
FLAVOR=${VLLM_FLAVOR:-h200}
TIMEOUT=${VLLM_TIMEOUT:-4h}
# Prompt + completion. The sanity set's MATH prompts are short; 4096 leaves the rest of the H200 for KV cache.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
# A capacity bound, not the rank served, and it must be one of 1, 8, 16, 32, 64, 128, 256, 320, 512.
MAX_LORA_RANK=${MAX_LORA_RANK:-1}
# At least `max_staleness + 2`: the trainer keeps `max_staleness + 1` adapter versions registered so a rollout that
# started under an older policy can finish under it, and each sync loads the next version before unloading the
# oldest. At `+ 1` vLLM silently LRU-evicts a still-servable policy on every sync (PR #7017 bugbot finding).
MAX_LORAS=${MAX_LORAS:-6}
VLLM_REPLICAS=${VLLM_REPLICAS:-2}

for replica in $(seq 1 "$VLLM_REPLICAS"); do
uvx hf jobs run \
    --name "asyncgrpo-lora-buckets-vllm-${replica}" --flavor "$FLAVOR" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    --expose 8000 \
    ` # Read-only: the server only ever reads adapters. The trainer Job mounts the same bucket read-write, at ` \
    ` # the same path -- the paths must match, because the path the trainer sends is resolved over there. ` \
    -v "hf://buckets/${BUCKET}:/lora:ro" \
    -e "MODEL=${MODEL}" \
    -e "MAX_MODEL_LEN=${MAX_MODEL_LEN}" \
    -e "MAX_LORA_RANK=${MAX_LORA_RANK}" \
    -e "MAX_LORAS=${MAX_LORAS}" \
    -e HF_HOME=/tmp/hf \
    -e PYTHONUNBUFFERED=1 \
    ` # /pause, /resume and /server_info are all gated on dev mode; the trainer needs all three. ` \
    -e VLLM_SERVER_DEV_MODE=1 \
    ` # Exposes /v1/load_lora_adapter, which is how each new adapter version reaches the engine. ` \
    -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
    -- "vllm/vllm-openai:${VLLM_TAG}" bash -c '
set -euo pipefail
echo "=== mounts ==="; grep /lora /proc/mounts || echo "no /lora mount!"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --host 0.0.0.0 so the jobs proxy can reach it; on localhost the exposed port answers nothing.
# --logprobs-mode processed_logprobs: the PPO denominator comes from these logprobs.
# --generation-config vllm: ignore the model card sampling defaults, which a server-side top_p would apply to
#   every rollout and which collapsed every async run that inherited them.
# --weight-transfer-config: kept even though the adapter path never builds a NCCL group. The trainer only
#   picks a sync mode after the server is already up, and merged sync is the fallback if the probe rejects the
#   adapter -- without this flag that fallback would have nowhere to go.
exec vllm serve "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.85 \
    --logprobs-mode processed_logprobs \
    --generation-config vllm \
    --weight-transfer-config "{\"backend\":\"nccl\"}" \
    --enable-lora --max-lora-rank "$MAX_LORA_RANK" --max-loras "$MAX_LORAS" --max-cpu-loras 8
' 2>&1 | grep -oE '[0-9a-f]{24}' | head -1
done
