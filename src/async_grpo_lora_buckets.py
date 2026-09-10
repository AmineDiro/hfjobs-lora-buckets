# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Async GRPO LoRA on the `sail/Sanity-Test-R1D-1.5B` sanity set, with the adapter shipped through an HF Storage
Bucket to a vLLM server running in a *different* HF Job.

The recipe is `asyncgrpo-lora-run/sanity_lora.py` unchanged: the LoRA and batch hyperparameters of oat's
`scripts/lora/bf16_grpo_tis_lora.sh` on `Qwen/Qwen2.5-Math-1.5B` -- `--lora_rank 1 --lora_alpha 2`,
`--learning_rate 0.00004 --lr_scheduler constant`, `--num_samples 8`, `--temperature 1 --top_p 1`,
`--generate_max_length 3000`, `--max_model_len 4096`, `--train_batch_size 128`,
`--train_batch_size_per_device 1`, `--beta 0`, `--num_ppo_epochs 1`, `--prompt_template qwen_math`. What is new
is only *where the adapter goes*.

Three things in that script this run does NOT reproduce, and none of them can be expressed here:

* `--critic_type drgrpo`. Dr. GRPO means advantages are not divided by the group's reward std and the token
  loss is normalized by the constant `generate_max_length`. `AsyncGRPOTrainer` hard-codes the opposite on both
  counts -- `_score_group` computes `(scored - scored.mean()) / (scored.std() + 1e-8)` and the loss divides by
  the batch's token count. So this is **vanilla** GRPO at the reference's batch shape, not Dr. GRPO.
* `--tis_c 2`. Truncated importance sampling with a clip constant is the mechanism the Precision-RL paper is
  about. `AsyncGRPOTrainer` has no equivalent option; it takes vLLM's own logprobs as the PPO denominator,
  which corrects the trainer/generator mismatch differently and is what the `ratio` metric measures.
* `--prompt_data ./data/train/math_12k`. The reference LoRA run trains on MATH-12k. This trains on
  `sail/Sanity-Test-R1D-1.5B`, the 1460-problem sanity set, because a filtered set where the base model scores
  between 20% and 80% makes a broken *pipeline* visible in tens of steps -- which is what this run is for.

`--beta 0` matches by construction rather than by configuration: `AsyncGRPOConfig` has no KL coefficient and
the trainer builds no reference model, so there is no KL penalty to switch off. On Slurm both sides shared the node's `/scratch`, so `<output_dir>/.vllm_lora/`
was an ordinary local directory. Here the trainer and the server are separate Jobs on separate machines, and the
one filesystem they both see is a Storage Bucket mounted read-write at the same path in each (`/lora`). The
trainer writes `trl-policy-vN` into it, hands the path to `lora_proxy.py`, which broadcasts it to every vLLM
replica, and each replica reads it off its own mount.

Two things about the bucket shape the run, both measured (`BUCKET_LATENCY_RESULTS.md`) rather than assumed:

* Publishing is fast, and reading no longer lags. The trainer's `close()` returns in under a millisecond and the
  object is on the Hub ~2.5s later. Until 2026-09-08 the server's mount then cached a "not found" for ~30s if it had
  been asked about the path before the upload landed, which is exactly what the trainer's first
  `load_lora_adapter` did; hf-mount's negative-cache TTL was shortened after we reported it, and the same probe now
  sees a new file ~3s after the write. The remaining per-replica retry lives in `lora_proxy.py`: vLLM preloads the
  adapter when `/v1/load_lora_adapter` is called and answers `No adapter found for <path>` while its mount has not
  caught up, so the proxy re-offers the path to that replica alone until it takes it, and rolls the load back
  everywhere if any replica fails for a real reason.
* Every filesystem call the publish path makes works on it -- `os.makedirs`, write, `os.rename` of a directory,
  `shutil.rmtree`, and the `mmap` safetensors does when vLLM loads the adapter. `save_lora_adapter`'s atomic
  rename therefore keeps its meaning, and no part of trl needs patching for the bucket.

Checkpoints go to the same bucket, so `save_strategy="steps"` costs one write and nothing else: the run's
adapter survives the Job, and a requeue resumes from it. That needs `fsdp_state_dict_type: SHARDED_STATE_DICT`
-- FSDP2 + PEFT under `FULL_STATE_DICT` silently checkpoints only rank 0's shard of each adapter tensor
(https://github.com/huggingface/accelerate/pull/4206), which loses ~38% of the learned reward on resume.

Launched by `run_trainer_job.sh`; see `README.md` for the whole setup.
"""

import json
import logging
import os
import re
import threading
import time

import torch
from datasets import load_dataset
from math_verify import parse, verify
from peft import LoraConfig
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

import trl.experimental.async_grpo.async_grpo_trainer as async_grpo_trainer
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer



MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Math-1.5B")  # the reference LoRA recipe's base model
# The bucket, mounted read-write at the same absolute path in both Jobs. `<output_dir>/.vllm_lora/` is what the
# server reads, so this path has to resolve identically in the server's container -- which is the entire reason
# both Jobs mount at `/lora` rather than at a per-Job location.
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/lora/sanity-lora")
RUN_NAME = os.environ.get("RUN_NAME", f"buckets-{os.environ.get('JOB_ID', 'local')}")
PROJECT = os.environ.get("PROJECT", "async-grpo-lora-buckets")
# The reference LoRA recipe's `--generate_max_length 3000`.
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "3000"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "20"))
LORA_RANK = int(os.environ.get("LORA_RANK", "1"))
# 128 completions per optimizer step -- the reference's `--train_batch_size 128` -- derived from the rank count
# that `accelerate launch` actually gave us rather than assumed. `MICROBATCH=1` is the reference's
# `--train_batch_size_per_device 1` and is what bounds peak memory; `FixedCountBatcher` (`token_budget=0`) does not.
MICROBATCH = int(os.environ.get("MICROBATCH", "1"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
# Token-budgeted packing, off by default. With `TOKEN_BUDGET > 0` every micro-batch packs as many samples as fit in
# `TOKEN_BUDGET` tokens per rank (padding-free, one row per rank), and an optimizer step is `GRAD_ACCUM` such
# micro-batches -- so samples per step becomes *variable*, roughly `GRAD_ACCUM x WORLD_SIZE x TOKEN_BUDGET / mean
# sample length`. At ~1250 tokens per sample, 16384 x 6 x 2 ranks is ~130-140 samples per step in 6 micro-batches
# instead of 64. The reference recipe's `--train_batch_size_per_device 1` ran this 1.5B model at 4% MFU; this is the
# lever that fixes it, at the cost of no longer holding 128 samples per step exactly.
TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "0"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "6"))
# `AsyncGRPOConfig` defaults `gradient_checkpointing` to True (unlike `TrainingArguments`), which is why a 16k-token
# row fits in ~25 GB of a 141 GB H200 -- and why the backward re-runs the forward. With this much headroom, off is
# the faster setting; the token budget is the knob that then fills the memory.
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "1") == "1"
MAX_STALENESS = int(os.environ.get("MAX_STALENESS", "4"))
# Every 4 optimizer steps, which is both the reference's broadcast cadence and what makes the bucket's ~20s
# publish latency a few percent of iteration time instead of a fixed tax on every step.
WEIGHT_SYNC_STEPS = int(os.environ.get("WEIGHT_SYNC_STEPS", "4"))
# Concurrent rollout requests. Left at auto this is `max_staleness x pdtb x accum x num_processes` = 512, which
# is the reference's rollout batch and is right when the server is a socket away. Here every one of them is an
# HTTPS connection through the public jobs proxy, a hop shared with everyone else's Jobs and one the preflight
# did not measure under load, so it is capped instead. Transport errors are retried by the rollout worker with
# backoff, so an over-large value degrades rather than fails -- but there is no reason to find out.
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "128"))
# One optimizer step's worth of samples per unit of staleness. The default of 1024 is eight steps of buffer
# against a four-step staleness window, so half of what the queue holds is already too old to be trained on by
# the time it is dequeued: the first smoke run of this setup dropped 158 samples as stale in 12 steps, with
# samples sitting 213s in the queue. Sizing the queue to the staleness window makes the rollout worker block
# instead of generating rollouts that will be thrown away.
QUEUE_MAXSIZE = int(os.environ.get("QUEUE_MAXSIZE", str(MAX_STALENESS * 128)))


# Response template for the rollout worker. `add_response_schema` raises "Unrecognized chat template" for
# Qwen2.5-Math, so it is set by hand: this model renders plain ChatML, the assistant turn opening with
# `<|im_start|>assistant\n` and closing with `<|im_end|>`. New-style `response_template` rather than the legacy
# `response_schema`, which transformers >= 5.13 no longer reads.
QWEN_MATH_RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {"content": {"close_pattern": r"<\|im_end\|>\s*|$", "content": "text"}},
}


# --- Untie lm_head from embed_tokens ----------------------------------------------------------------------
#
# FSDP2 cannot place one parameter in two `fully_shard` groups, and accelerate's FSDP2 path always carves
# `embed_tokens` into its own unit while leaving `lm_head` to the root unit. Qwen2.5-Math-1.5B sets
# `tie_word_embeddings=True`, so those two modules hold the *same* tensor and `accelerator.prepare` dies with
# "Parameter 'base_model.model.model.embed_tokens.weight' is shared with a parameter already managed by another
# FSDP group". NO_WRAP, SIZE_BASED_WRAP and `fsdp_ignored_modules` on either module were all measured and all
# still fail; giving `lm_head` its own copy is the only thing that works.
#
# Numerically inert here: LoRA's `all-linear` targets neither `lm_head` nor `embed_tokens`, so both stay frozen
# at the checkpoint's values for the whole run. vLLM keeps serving the original tied checkpoint plus the
# adapter, which is what the `ratio` metric checks.
_create_model_from_path = async_grpo_trainer.create_model_from_path


def _create_model_untied(model_id, **kwargs):
    model = _create_model_from_path(model_id, **kwargs)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    model.config.tie_word_embeddings = False  # stops accelerate's `tie_weights()` re-tying them
    return model


async_grpo_trainer.create_model_from_path = _create_model_untied


# --- Reward: oat's `boxed_reward_fn` under the `qwen_math` prompt template ----------------------------------
#
# `trl.rewards.accuracy_reward` is not equivalent and was measurably worse on this dataset: it returns `None`
# when the *gold* fails to parse (160 of these 1460 golds do not parse under `math_verify`, and trl drops those
# samples), and it extracts the *first* `\boxed{}` where oat takes the last, which rewards early guessing.
_SUBS = [
    (r"\left", ""),
    (r"\right", ""),
    (r"\!", ""),
    (r"\,", ""),
    (r"\;", ""),
    (r"\ ", ""),
    (r"\$", ""),
    ("$", ""),
    (r"\%", ""),
    ("%", ""),
    (r"\dfrac", r"\frac"),
    (r"\tfrac", r"\frac"),
    (r"^\circ", ""),
    (r"\circ", ""),
    (r"\text{", "{"),
    (r"\mbox{", "{"),
]


def _normalize(s: str) -> str:
    """Normalize LaTeX the way oat's `grade_answer_mathd` does before comparing."""
    s = str(s).strip()
    for old, new in _SUBS:
        s = s.replace(old, new)
    s = re.sub(r"\\frac\s+", r"\\frac", s)  # `\frac {40}7` -> `\frac{40}7`
    s = re.sub(r"\\frac\{([^{}]+)\}\s*(\d)", r"\\frac{\1}{\2}", s)  # `\frac{40}7` -> `\frac{40}{7}`
    s = re.sub(r"\\sqrt\s+", r"\\sqrt", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separator: `1,657` -> `1657`
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s.removesuffix(".0")


def _last_boxed(text: str) -> str | None:
    """Contents of the LAST `\\boxed{...}`, matching oat's `extract_boxed_answer` (which uses rfind)."""
    start = text.rfind(r"\boxed")
    if start == -1:
        return None
    open_brace = text.find("{", start)
    if open_brace == -1:
        return None
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : i]
    return None


def qwen_math_reward(completions, solution, completion_ids, **kwargs):
    """1.0 if the completion terminates and boxes the right answer, else 0.0."""
    # math_verify implements its timeouts with `signal.alarm`, which only works on the main thread.
    # AsyncGRPOTrainer scores rollouts via `asyncio.to_thread`, so timeouts must be disabled there or every
    # symbolic comparison raises. `trl.rewards.accuracy_reward` does the same dance.
    on_main_thread = threading.current_thread() is threading.main_thread()
    parsing_timeout = 10 if on_main_thread else None
    verify_timeout = 5 if on_main_thread else None
    if not on_main_thread:
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    rewards = []
    for completion, gold, ids in zip(completions, solution, completion_ids, strict=True):
        # oat overrides the grade to 0 for any rollout that hit the length budget without emitting EOS
        # (`if no_eos[i][j]: reward = 0`), regardless of what it had written by then. Without this a rollout
        # that boxes an answer and then rambles past the budget still collects the reward.
        if len(ids) >= MAX_COMPLETION_LENGTH:
            rewards.append(0.0)
            continue
        # No `</think>` requirement: Qwen2.5-Math is not a reasoning model and never emits one, so requiring it
        # would score every rollout 0.0 and leave the run with no gradient at all. oat's `qwen_math` template --
        # the one the reference LoRA recipe selects -- grades the last `\boxed{}` directly.
        answer = _last_boxed(completion[0]["content"])
        if answer is None:
            rewards.append(0.0)
        elif _normalize(answer) == _normalize(gold):
            rewards.append(1.0)
        else:
            # Symbolic fallback for answers that are equivalent but written differently.
            parsed_gold = parse(gold, parsing_timeout=parsing_timeout)
            parsed_answer = parse(answer, parsing_timeout=parsing_timeout)
            rewards.append(
                float(
                    bool(parsed_gold)
                    and bool(parsed_answer)
                    and bool(verify(parsed_gold, parsed_answer, timeout_seconds=verify_timeout))
                )
            )
    return rewards


def format_sample(sample):
    # `prompt` is already conversational and already ends with the "put your final answer within \boxed{}"
    # instruction, so it is passed through untouched. Only the rule-based ground truth has to be lifted out of
    # the `reward_model` struct into the `solution` column the reward function reads.
    return {"prompt": sample["prompt"], "solution": sample["reward_model"]["ground_truth"]}


def main() -> None:
    # trl logs through `accelerate.logging`, i.e. the stdlib root logger, which has no handler in a bare script,
    # so INFO records fall to `logging.lastResort` (stderr, pinned at WARNING) and vanish -- including the
    # weight-sync timings and stale-sample drops this run is read by. Scoped to `trl` rather than
    # `basicConfig`, which would also unleash urllib3 and aiohttp.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    trl_logger = logging.getLogger("trl")
    trl_logger.addHandler(handler)
    trl_logger.setLevel(logging.INFO)

    dataset = load_dataset("sail/Sanity-Test-R1D-1.5B", split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.response_template = QWEN_MATH_RESPONSE_TEMPLATE

    config = AsyncGRPOConfig(
        output_dir=OUTPUT_DIR,
        bf16=True,
        num_generations=8,
        temperature=1.0,
        top_p=1.0,
        max_completion_length=MAX_COMPLETION_LENGTH,
        learning_rate=4e-5,  # the reference LoRA recipe's `--learning_rate 0.00004`, constant schedule
        adam_beta2=0.95,  # oat's `PPOArgs` default; HF `TrainingArguments` defaults to 0.999
        per_device_train_batch_size=MICROBATCH,
        gradient_accumulation_steps=GRAD_ACCUM if TOKEN_BUDGET > 0 else 128 // (MICROBATCH * WORLD_SIZE),
        # `token_budget=0` selects `FixedCountBatcher` over `TokenBudgetBatcher` and keeps the reference's exact
        # 128 completions per step. Any `token_budget > 0` (including the `None` default, which resolves to the
        # server's max_model_len) packs rows to a *token* target and never reads `per_device_train_batch_size`,
        # so samples per step become variable -- see TOKEN_BUDGET above.
        token_budget=TOKEN_BUDGET,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        max_staleness=MAX_STALENESS,
        max_inflight_tasks=MAX_INFLIGHT,
        queue_maxsize=QUEUE_MAXSIZE,
        weight_sync_steps=WEIGHT_SYNC_STEPS,
        max_steps=MAX_STEPS,
        # Checkpoints land in the same bucket the adapters are published through, so keeping them costs one
        # write. `.vllm_lora/` is a serving cache -- each version is deleted a sync after it leaves the
        # staleness window -- so without these the run would finish and leave no adapter behind.
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        vllm_server_base_url=os.environ.get("SERVE_URL", "http://localhost:8000"),
        report_to="trackio",
        run_name=RUN_NAME,
        project=PROJECT,
        trackio_space_id=PROJECT,
    )
    trainer = AsyncGRPOTrainer(
        model=MODEL_ID,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=qwen_math_reward,
        # Plain LoRA: no `modules_to_save`, no DoRA, no trained bias, nothing on `lm_head` or `embed_tokens`.
        # Each of those would make the adapter unservable by vLLM and silently drop the run back to merged
        # sync -- which across a bucket would mean shipping the whole 3 GB checkpoint every sync.
        peft_config=LoraConfig(r=LORA_RANK, lora_alpha=2 * LORA_RANK, target_modules="all-linear"),
    )

    # Resume if the bucket already holds a checkpoint, so a requeued Job continues instead of restarting.
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    if last_checkpoint:
        print(f"[resume] {last_checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"[done] global_step={trainer.state.global_step}", flush=True)
    # The final adapter, kept outside `.vllm_lora/` so the serving cache's deletion policy cannot reach it.
    # Every rank calls this: materializing a sharded adapter parameter all-gathers, and only rank 0 then writes.
    final = os.path.join(OUTPUT_DIR, "final-adapter")
    async_grpo_trainer.save_lora_adapter(
        trainer.accelerator.unwrap_model(trainer.model), trainer.accelerator, trainer._adapter_name, final
    )
    if trainer.accelerator.is_main_process:
        with open(os.path.join(OUTPUT_DIR, "run.json"), "w") as f:
            json.dump({"run_name": RUN_NAME, "project": PROJECT, "steps": trainer.state.global_step}, f)
        print(f"[final] adapter -> {final}", flush=True)


if __name__ == "__main__":
    main()
