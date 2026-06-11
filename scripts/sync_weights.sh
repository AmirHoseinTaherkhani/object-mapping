#!/bin/bash
# Run this on your Mac in a separate terminal while training runs on Vast.ai.
# Usage: bash scripts/sync_weights.sh <port> <host>
# Example: bash scripts/sync_weights.sh 25297 ssh8.vast.ai

PORT=${1:?usage: sync_weights.sh PORT HOST}
HOST=${2:?usage: sync_weights.sh PORT HOST}
LOCAL_DIR="$(dirname "$0")/../cloud_weights"
mkdir -p "$LOCAL_DIR"

echo "Syncing weights from $HOST:$PORT every 10 minutes. Ctrl+C to stop."
while true; do
    echo "[$(date '+%H:%M:%S')] Syncing..."
    rsync -avz -e "ssh -p $PORT" \
        root@$HOST:/workspace/runs/train/yolov8s_v3/weights/ \
        "$LOCAL_DIR/" 2>&1 | grep -E "\.pt|error|done"
    echo "[$(date '+%H:%M:%S')] Next sync in 10 minutes."
    sleep 600
done
