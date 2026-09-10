#!/usr/bin/env bash
# Bucket write/read latency decomposition -- see BUCKET_LATENCY_EXPERIMENT.md.
#
#   ./run_latency.sh
#
# Two `cpu-basic` Jobs share one Storage Bucket mounted at /lora. Side A (writer) serves an HTTP API on an exposed
# port; side B (reader) drives ~100 trials and times, on its own clock, how long after a write the object is
# readable through the bucket HTTP API (upload flush) versus visible on its own mount (metadata cache).
set -euo pipefail
cd "$(dirname "$0")"

BUCKET=${BUCKET:-aminediroHF/bucket-latency-probe}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
TRIALS_PER_CELL=${TRIALS_PER_CELL:-10}
TIMEOUT=${TIMEOUT:-2h}
READER_SCRIPT=${READER_SCRIPT:-latency_reader.py}   # or latency_reader_gated.py for the follow-up
export HF_TOKEN=${HF_TOKEN:-$(uvx hf auth token 2>/dev/null | tail -1)}

uvx hf buckets create "${BUCKET#*/}" --private 2>/dev/null || true

echo "=== writer (A): bucket $BUCKET at /lora, port 8000 exposed ==="
WRITER=$(uvx hf jobs run \
    --name bucket-latency-writer --flavor cpu-basic --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    --expose 8000 \
    -v "hf://buckets/${BUCKET}:/lora" \
    -v "$PWD/src:/work" \
    -e "BUCKET=${BUCKET}" \
    -- python:3.12 python /work/latency_writer.py 2>&1 | grep -oE '[0-9a-f]{24}' | head -1)
WRITER_URL="https://${WRITER}--8000.hf.jobs"
echo "writer job: $WRITER   url: $WRITER_URL"

echo "=== reader (B) ==="
READER=$(uvx hf jobs run \
    --name bucket-latency-reader --flavor cpu-basic --timeout "$TIMEOUT" --detach --secrets HF_TOKEN \
    -v "hf://buckets/${BUCKET}:/lora" \
    -v "$PWD/src:/work" \
    -e "WRITER_URL=${WRITER_URL}" \
    -e "BUCKET=${BUCKET}" \
    -e "RUN_ID=${RUN_ID}" \
    -e "TRIALS_PER_CELL=${TRIALS_PER_CELL}" ${EXTRA_ENV:-} \
    -- python:3.12 python /work/${READER_SCRIPT} 2>&1 | grep -oE '[0-9a-f]{24}' | head -1)
echo "reader job: $READER"

cat > .latency_ids <<IDS
WRITER=$WRITER
READER=$READER
WRITER_URL=$WRITER_URL
RUN_ID=$RUN_ID
BUCKET=$BUCKET
IDS
echo
echo "follow:  uvx hf jobs logs -f $READER"
echo "stop:    uvx hf jobs cancel $WRITER; uvx hf jobs cancel $READER"
