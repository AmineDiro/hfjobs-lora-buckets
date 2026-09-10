# hfjobs-lora-buckets

Async GRPO with LoRA across [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs): one trainer Job, N vLLM Jobs, a [Storage Bucket](https://huggingface.co/docs/hub/storage-buckets) as the adapter transport, and a small proxy in front of the replicas. No shared node, no NCCL.

Companion code for the blog post [Async GRPO with LoRA across Hugging Face Jobs](https://huggingface.co/blog/asyncgrpo-lora-hfjobs). The training metrics of every run in the post are on the [trackio dashboard](https://huggingface.co/spaces/aminediroHF/async-grpo-lora-buckets).

## How it works

- The trainer Job runs TRL's `AsyncGRPOTrainer` with a LoRA adapter under FSDP2 ([TRL PR #7017](https://github.com/huggingface/trl/pull/7017)). Every `WEIGHT_SYNC_STEPS` optimizer steps it saves the adapter to `/lora/<run>/.vllm_lora/trl-policy-vN` and asks vLLM to load that path.
- Every vLLM Job mounts the same bucket read-only at `/lora` and loads the adapter with `/v1/load_lora_adapter`. Nothing in TRL or vLLM is patched. The bucket is the shared filesystem.
- `src/lora_proxy.py` runs on the trainer Job at `127.0.0.1:8000` and makes the replicas look like one server. It adds the bearer token that exposed Job ports require, routes each completion to the replica most likely to hold its KV prefix, and broadcasts adapter loads, pause and resume to every replica.

## Layout

| file | what it is |
|---|---|
| `run_all.sh` | end to end: bucket, `VLLM_REPLICAS` vLLM Jobs, wait for every `/health`, trainer Job. `--wait` cancels the servers when the trainer finishes |
| `run_vllm_job.sh` | the vLLM Jobs on their own, one Job ID per line |
| `run_trainer_job.sh` | the trainer Job on its own, given the vLLM Job IDs |
| `stop.sh` | cancel every Job of the last `run_all.sh` |
| `src/async_grpo_lora_buckets.py` | the training script: the sanity recipe, output on the bucket |
| `src/lora_proxy.py` | the proxy: bearer token, KV-prefix routing, adapter broadcast |
| `src/fsdp2.yaml` | accelerate FSDP2 config with `SHARDED_STATE_DICT` |
| `tests/test_lora_proxy.py` | the proxy against two fake vLLM servers, no GPU needed |
| `run_preflight.sh`, `src/preflight_*.py` | two `cpu-basic` Jobs that check the bucket mount and the Jobs proxy before spending GPU money |
| `run_latency.sh`, `src/latency_*.py` | the two-Job harness that measured bucket write-to-read latency |

`src/` is mounted into the Jobs at `/work`.

## Run it

You need the `hf` CLI logged in with a token that can create Jobs and Buckets. `run_all.sh` creates the bucket if it does not exist.

```sh
hf auth login
MAX_STEPS=20 RUN_TAG=smoke ./run_all.sh --wait        # ~15 min, three Jobs, cancels the servers when done
MAX_STEPS=500 ./run_all.sh --wait                     # the reference batch shape, ~3.5 h
TOKEN_BUDGET=16384 GRAD_ACCUM=6 GRADIENT_CHECKPOINTING=0 PROXY_LORA_RETRY_S=0.5 \
  MAX_INFLIGHT=384 QUEUE_MAXSIZE=768 MAX_STEPS=500 ./run_all.sh --wait   # same recipe, ~55 min
./stop.sh                                             # cancel everything now
```

Everything is an environment variable with a default. The ones you are most likely to change:

| variable | default | meaning |
|---|---|---|
| `BUCKET` | `aminediroHF/asyncgrpo-lora-buckets` | the bucket mounted at `/lora` in every Job |
| `MODEL` | `Qwen/Qwen2.5-Math-1.5B` | base model, served and trained |
| `VLLM_REPLICAS` | `2` | number of vLLM Jobs |
| `VLLM_FLAVOR`, `TRAIN_FLAVOR` | `h200`, `h200x2` | Job hardware |
| `LORA_RANK` | `1` | adapter rank; `MAX_LORA_RANK` on the servers follows |
| `MAX_STEPS`, `SAVE_STEPS` | `100`, `20` | optimizer steps and checkpoint interval |
| `WEIGHT_SYNC_STEPS`, `MAX_STALENESS` | `4`, `4` | adapter publish cadence and staleness window; `MAX_LORAS` is `MAX_STALENESS + 2` |
| `MAX_INFLIGHT`, `QUEUE_MAXSIZE` | `128`, `MAX_STALENESS * 128` | rollout concurrency and buffer size |
| `TOKEN_BUDGET`, `GRAD_ACCUM`, `GRADIENT_CHECKPOINTING` | `0`, `6`, `1` | token-budget packing, accumulation, checkpointing |
| `PROXY_IMBALANCE`, `PROXY_LORA_RETRY_S` | `8`, `2` | routing spill threshold and adapter-load retry interval |
| `RUN_TAG` | `r${LORA_RANK}` | bucket directory and trackio run name; reuse it to resume from the checkpoint in the bucket |

The trainer Job installs TRL from a pinned commit (`TRL_SHA`) and both Jobs use `vllm/vllm-openai:${VLLM_TAG}`, `v0.27.1` by default.

The two GPU flavors bill about $20 per hour while all three Jobs run. The server Jobs have no natural end, so use `--wait` or `./stop.sh`.

## Tests

```sh
pip install pytest pytest-asyncio aiohttp
pytest tests/
```

## Related

- [TRL PR #7017](https://github.com/huggingface/trl/pull/7017): LoRA support and adapter-only vLLM sync for `AsyncGRPOTrainer`.
- [hf-mount-repro](https://github.com/AmineDiro/hf-mount-repro): the two-script reproduction of the bucket mount negative-cache stall, since fixed in `hf-mount`.
