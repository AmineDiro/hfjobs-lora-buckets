#!/usr/bin/env bash
# Start the trainer Job: FSDP2 across the flavor's GPUs, the shared bucket mounted read-write, and generation
# served by the vLLM Jobs whose IDs are the arguments. Prints the Job ID on stdout.
#
#   ./run_trainer_job.sh <vllm_job_id> [<vllm_job_id> ...]
#
# The trainer never talks to `https://<id>--8000.hf.jobs` directly. That URL needs an `Authorization: Bearer`
# header on every request (an unauthenticated call gets a 401), and nothing in the async GRPO stack can add
# one: `VLLMClient` calls `requests` bare, and the rollout worker builds its own `aiohttp` session inside a
# spawned child process. So `lora_proxy.py` runs alongside the trainer: it adds the header, routes each rollout
# to the replica most likely to hold its KV prefix, broadcasts adapter loads to every replica, and answers
# `/health` only when all of them do -- and `vllm_server_base_url` stays `http://localhost:8000` exactly as it
# would on a single node.
set -euo pipefail

cd "$(dirname "$0")"

[ $# -ge 1 ] || { echo "usage: run_trainer_job.sh <vllm_job_id> [<vllm_job_id> ...]" >&2; exit 1; }
UPSTREAM_URLS=$(for id in "$@"; do printf 'https://%s--8000.hf.jobs,' "$id"; done)
UPSTREAM_URLS=${UPSTREAM_URLS%,}
BUCKET=${BUCKET:-aminediroHF/asyncgrpo-lora-buckets}
MODEL=${MODEL:-Qwen/Qwen2.5-Math-1.5B}
# huggingface/trl PR #7017 (`asyncgrpo-lora`) at its 2026-09-08 head.
TRL_SHA=${TRL_SHA:-9d2f38056aa0c79be500525536b2830fc47dd35a}
VLLM_TAG=${VLLM_TAG:-v0.27.1}
# h200x2 is the floor for "distributed": two FSDP2 ranks, which is what exercises the collective adapter save
# (`save_lora_adapter` all-gathers each LoRA parameter with `DTensor.full_tensor()`).
FLAVOR=${TRAIN_FLAVOR:-h200x2}
TIMEOUT=${TRAIN_TIMEOUT:-4h}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-20}
LORA_RANK=${LORA_RANK:-1}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-3000}
WEIGHT_SYNC_STEPS=${WEIGHT_SYNC_STEPS:-4}
MAX_STALENESS=${MAX_STALENESS:-4}
# Concurrent rollout requests through the public jobs proxy; see the note in async_grpo_lora_buckets.py.
MAX_INFLIGHT=${MAX_INFLIGHT:-128}
# Token-budgeted packing (0 = off, the reference's fixed 1 sample per micro-batch). See async_grpo_lora_buckets.py.
TOKEN_BUDGET=${TOKEN_BUDGET:-0}
GRAD_ACCUM=${GRAD_ACCUM:-6}
# 1 = trl's default (recompute the forward in the backward, low memory); 0 = keep activations, ~30% faster fwd+bwd.
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
# Buffered rollout samples. Sized to the staleness window by default; see async_grpo_lora_buckets.py.
QUEUE_MAXSIZE=${QUEUE_MAXSIZE:-$((MAX_STALENESS * 128))}
PROJECT=${PROJECT:-async-grpo-lora-buckets}
RUN_TAG=${RUN_TAG:-r${LORA_RANK}}
# One directory per run inside the bucket: adapters under `.vllm_lora/`, checkpoints beside them, and the
# final adapter in `final-adapter/`. Both Jobs mount the bucket at /lora, so this path resolves in both.
OUTPUT_DIR=${OUTPUT_DIR:-/lora/sanity-lora-${RUN_TAG}}

uvx hf jobs run \
    --name asyncgrpo-lora-buckets-train --flavor "$FLAVOR" --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    -v "hf://buckets/${BUCKET}:/lora" \
    -v "$PWD/src:/work" \
    -e "UPSTREAM_URLS=${UPSTREAM_URLS}" \
    ` # A replica may hold this many more in-flight rollouts than the least-loaded one before prefix affinity yields. ` \
    -e "PROXY_IMBALANCE=${PROXY_IMBALANCE:-8}" \
    ` # How often a replica whose mount has not seen the new adapter yet is re-offered it. ` \
    -e "PROXY_LORA_RETRY_S=${PROXY_LORA_RETRY_S:-2}" \
    -e "MODEL_ID=${MODEL}" \
    -e "TRL_SHA=${TRL_SHA}" \
    -e "OUTPUT_DIR=${OUTPUT_DIR}" \
    -e "MAX_STEPS=${MAX_STEPS}" \
    -e "SAVE_STEPS=${SAVE_STEPS}" \
    -e "LORA_RANK=${LORA_RANK}" \
    -e "MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH}" \
    -e "WEIGHT_SYNC_STEPS=${WEIGHT_SYNC_STEPS}" \
    -e "MAX_STALENESS=${MAX_STALENESS}" \
    -e "MAX_INFLIGHT=${MAX_INFLIGHT}" \
    -e "TOKEN_BUDGET=${TOKEN_BUDGET}" \
    -e "GRAD_ACCUM=${GRAD_ACCUM}" \
    -e "GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}" \
    -e "QUEUE_MAXSIZE=${QUEUE_MAXSIZE}" \
    -e "PROJECT=${PROJECT}" \
    ` # No timestamp: the same RUN_TAG has to name the same trackio run in every Job of a chain, so a resume ` \
    ` # appends to the existing curve instead of starting a second one beside it. ` \
    -e "RUN_NAME=${RUN_NAME:-$RUN_TAG}" \
    -e HF_HOME=/tmp/hf \
    -e PYTHONUNBUFFERED=1 \
    -e TRL_EXPERIMENTAL_SILENCE=1 \
    -e ACCELERATE_LOG_LEVEL=info \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -- "vllm/vllm-openai:${VLLM_TAG}" bash -c '
set -euo pipefail
echo "=== mounts ==="; grep /lora /proc/mounts || { echo "no /lora mount!"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# The trl revision under test, as a codeload tarball rather than git+https because the vLLM image ships no git.
# GitHub rate-limits datacenter IPs, so the install is retried rather than losing the Job to a 429.
for attempt in 1 2 3 4 5; do
    pip install -q "https://codeload.github.com/huggingface/trl/tar.gz/${TRL_SHA}" \
        peft kernels trackio math-verify latex2sympy2_extended && break
    echo "pip install failed (attempt $attempt), retrying in 30s"
    sleep 30
done
python3 -c "import trl, peft, vllm; print(f\"trl {trl.__version__} peft {peft.__version__} vllm {vllm.__version__}\")"

# Localhost stand-in for the remote vLLM Jobs: bearer token, prefix-aware routing, adapter broadcast.
# Its stdout goes to the Job log (prefixed) so routing stats and adapter broadcasts are visible from outside.
python3 /work/lora_proxy.py > >(sed -u "s/^/[lora_proxy] /") 2>&1 &
PROXY_PID=$!
sleep 3
kill -0 $PROXY_PID || { echo "proxy died"; exit 1; }

# Fail fast with a clear message rather than 240s into the trainer'"'"'s own readiness wait.
echo "=== waiting for every vLLM replica through the proxy (up to 900s) ==="
for i in $(seq 1 180); do
    curl -sf http://localhost:8000/health > /dev/null 2>&1 && { echo "vLLM reachable after $((i * 5))s"; break; }
    sleep 5
done
curl -sf http://localhost:8000/health > /dev/null 2>&1 || { echo "!!! vLLM unreachable through the proxy"; curl -s http://localhost:8000/health; exit 1; }

# Replicas may be reused across runs, but trl always numbers adapters `trl-policy-v1, v2, ...` starting at v1 in
# every run, so any adapter left registered by a previous run is a name collision waiting to happen (vLLM
# answers 400 "has already been loaded", which trl treats as fatal). Unload them through the proxy, which broadcasts.
echo "=== sweeping stale trl-policy-* adapters off every replica ==="
for u in ${UPSTREAM_URLS//,/ }; do
    curl -s -H "Authorization: Bearer $HF_TOKEN" "$u/v1/models" \
        | python3 -c "import sys, json; [print(m[\"id\"]) for m in json.load(sys.stdin).get(\"data\", []) if m[\"id\"].startswith(\"trl-policy-\")]" || true
done | sort -u | while read -r name; do
    [ -n "$name" ] || continue
    echo "unloading stale adapter $name"
    curl -s -X POST -H "Content-Type: application/json" -d "{\"lora_name\": \"$name\"}" http://localhost:8000/v1/unload_lora_adapter > /dev/null || true
done

echo "=== server lora_config / parallel_config ==="
curl -s "http://localhost:8000/server_info?config_format=json" | python3 -c \
    "import json,sys; c=json.load(sys.stdin)[\"vllm_config\"]; print(\"lora:\", c[\"lora_config\"]); \
     print(\"tp:\", c[\"parallel_config\"][\"tensor_parallel_size\"], \"dp:\", c[\"parallel_config\"][\"data_parallel_size\"])"

# `--num_processes` from the Job'"'"'s actual GPU count, so one accelerate config covers every flavor.
NPROC=$(nvidia-smi -L | wc -l)
echo "=== trainer: FSDP2 on $NPROC rank(s) ==="
export SERVE_URL=http://localhost:8000
accelerate launch --config_file /work/fsdp2.yaml --num_processes "$NPROC" /work/async_grpo_lora_buckets.py
' 2>&1 | grep -oE '[0-9a-f]{24}' | head -1
