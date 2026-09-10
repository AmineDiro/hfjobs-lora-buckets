#!/usr/bin/env bash
# End to end: create the bucket if needed, start the vLLM Jobs, wait for all of them to serve, start the trainer Job.
#
#   ./run_all.sh                 # launch both and return
#   ./run_all.sh --wait          # ...then block until the trainer finishes and cancel the server Job
#
# Order matters: the trainer needs the vLLM Job's ID to build its exposed URL, so the server has to exist
# first. Waiting for /health before launching the trainer is not cosmetic either -- the trainer's own
# readiness wait is 240s by default, and a cold model load plus image pull can exceed that.
#
# All Job IDs land in `.last_run`, which `stop.sh` reads.
set -euo pipefail

cd "$(dirname "$0")"

BUCKET=${BUCKET:-aminediroHF/asyncgrpo-lora-buckets}
LORA_RANK=${LORA_RANK:-1}
export BUCKET LORA_RANK
# vLLM's `--max-lora-rank` accepts only 1, 8, 16, 32, 64, 128, 256, 320, 512, and it is a capacity bound, so
# the smallest allowed value at or above the adapter's rank is the right one.
export MAX_LORA_RANK=${MAX_LORA_RANK:-$(python3 -c "
r = $LORA_RANK
print(next(v for v in (1, 8, 16, 32, 64, 128, 256, 320, 512) if v >= r))")}
export MAX_STALENESS=${MAX_STALENESS:-4}
export MAX_LORAS=${MAX_LORAS:-$((MAX_STALENESS + 2))}

echo "=== bucket ==="
uvx hf buckets create "${BUCKET#*/}" --private 2>/dev/null || echo "bucket $BUCKET already exists"

export VLLM_REPLICAS=${VLLM_REPLICAS:-2}
echo "=== $VLLM_REPLICAS vLLM Job(s) (rank<=$MAX_LORA_RANK, $MAX_LORAS adapter slots) ==="
VLLM_IDS=$(./run_vllm_job.sh | tr '\n' ' ')
VLLM_IDS=${VLLM_IDS% }
[ "$(echo "$VLLM_IDS" | wc -w)" -eq "$VLLM_REPLICAS" ] || {
    echo "!!! started $(echo "$VLLM_IDS" | wc -w) of $VLLM_REPLICAS vLLM Jobs: '$VLLM_IDS'"
    for id in $VLLM_IDS; do uvx hf jobs cancel "$id" 2>/dev/null || true; done
    exit 1
}
for id in $VLLM_IDS; do echo "vllm job: $id   url: https://${id}--8000.hf.jobs"; done

echo "=== waiting for every replica's /health (up to 900s) ==="
TOKEN=$(uvx hf auth token 2>/dev/null | tail -1)
for id in $VLLM_IDS; do
    url="https://${id}--8000.hf.jobs"
    for i in $(seq 1 180); do
        code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" "$url/health" || true)
        [ "$code" = "200" ] && { echo "$id serving after $((i * 5))s"; break; }
        stage=$(uvx hf jobs inspect "$id" 2>/dev/null | grep -oE "'stage': '[A-Z]+'" | head -1)
        case "$stage" in
            *ERROR*|*COMPLETED*|*CANCELED*)
                echo "!!! vLLM Job $id reached $stage before serving; last log lines:"
                uvx hf jobs logs "$id" 2>/dev/null | tail -30
                for other in $VLLM_IDS; do uvx hf jobs cancel "$other" 2>/dev/null || true; done
                exit 1
                ;;
        esac
        sleep 5
    done
    curl -sf -H "Authorization: Bearer $TOKEN" "$url/health" > /dev/null || {
        echo "!!! $id never became ready; last log lines:"; uvx hf jobs logs "$id" 2>/dev/null | tail -40
        for other in $VLLM_IDS; do uvx hf jobs cancel "$other" 2>/dev/null || true; done
        exit 1
    }
done

echo "=== trainer Job ==="
# shellcheck disable=SC2086
TRAIN_ID=$(./run_trainer_job.sh $VLLM_IDS)
[ -n "$TRAIN_ID" ] || { echo "!!! could not start the trainer Job"; for id in $VLLM_IDS; do uvx hf jobs cancel "$id"; done; exit 1; }
echo "train job: $TRAIN_ID"

cat > .last_run <<IDS
VLLM_IDS="$VLLM_IDS"
TRAIN_ID=$TRAIN_ID
PROJECT=${PROJECT:-async-grpo-lora-buckets}
IDS

cat <<TXT

=== running ===
  trackio     https://huggingface.co/spaces/$(uvx hf auth whoami 2>/dev/null | grep -oE 'user=[^ ]+' | cut -d= -f2)/${PROJECT:-async-grpo-lora-buckets}
  bucket      https://huggingface.co/buckets/$BUCKET
  trainer log uvx hf jobs logs -f $TRAIN_ID
  server logs $(for id in $VLLM_IDS; do printf 'uvx hf jobs logs -f %s   ' "$id"; done)
  stop all    ./stop.sh
TXT

if [ "${1:-}" = "--wait" ]; then
    echo "=== waiting for the trainer Job ==="
    uvx hf jobs wait "$TRAIN_ID" || true
    echo "=== trainer finished; cancelling the server Jobs so they stop billing ==="
    for id in $VLLM_IDS; do uvx hf jobs cancel "$id" 2>/dev/null || true; done
fi
