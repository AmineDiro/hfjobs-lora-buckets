#!/usr/bin/env bash
# Preflight: measure the two platform properties the disaggregated LoRA run depends on, before spending GPU money.
#
#   ./run_preflight.sh
#
# Starts two `cpu-basic` Jobs (about $0.01/hour each) that share one Storage Bucket:
#
#   server  -- probes what the mount supports (`os.rename` of a directory above all, since that is how
#              `save_lora_adapter` publishes an adapter), then publishes adapter-sized beacons on a cadence and
#              serves an HTTP endpoint on an exposed port.
#   client  -- measures how long after each beacon is written its own mount can see it and hash-verify it, and
#              probes the exposed port: whether it needs a bearer token, and how long a response the jobs proxy
#              will hold open.
#
# Read the client's `[vis] SUMMARY` line and its `[proxy] hold` ladder. Those two numbers set
# `LORA_LOAD_RETRY_S` and the completion-length ceiling in `run_trainer_job.sh`.
set -euo pipefail

cd "$(dirname "$0")"
BUCKET=${BUCKET:-aminediroHF/asyncgrpo-lora-buckets}
BEACONS=${BEACONS:-12}

echo "=== starting preflight server (bucket $BUCKET at /lora, port 8000 exposed) ==="
SERVER_ID=$(uvx hf jobs run \
    --name preflight-bucket-server --flavor cpu-basic --timeout 40m --detach \
    -v "hf://buckets/${BUCKET}:/lora" \
    -v "$PWD/src:/work" \
    -e "BEACONS=${BEACONS}" \
    --expose 8000 \
    -- python:3.12 python /work/preflight_server.py 2>&1 | grep -oE '[0-9a-f]{24}' | head -1)
echo "server job: $SERVER_ID"

SERVER_URL="https://${SERVER_ID}--8000.hf.jobs"
echo "server url:  $SERVER_URL"

# The server probes the filesystem before it binds the port, so give it a moment to get there.
sleep 45

echo "=== starting preflight client ==="
CLIENT_ID=$(uvx hf jobs run \
    --name preflight-bucket-client --flavor cpu-basic --timeout 40m --detach --secrets HF_TOKEN \
    -v "hf://buckets/${BUCKET}:/lora" \
    -v "$PWD/src:/work" \
    -e "SERVER_URL=${SERVER_URL}" \
    -e "BEACONS=${BEACONS}" \
    -- python:3.12 bash -c 'pip install -q requests && python /work/preflight_client.py' 2>&1 \
    | grep -oE '[0-9a-f]{24}' | head -1)
echo "client job: $CLIENT_ID"

cat > .preflight_ids <<IDS
SERVER_ID=$SERVER_ID
CLIENT_ID=$CLIENT_ID
SERVER_URL=$SERVER_URL
IDS

echo
echo "follow:  uvx hf jobs logs -f $CLIENT_ID"
echo "         uvx hf jobs logs -f $SERVER_ID"
echo "stop:    uvx hf jobs cancel $SERVER_ID; uvx hf jobs cancel $CLIENT_ID"
