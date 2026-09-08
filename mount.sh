#!/usr/bin/env bash
# Mount the RunPod S3 volume locally as a read-write folder.
# Uses our own WebDAV server (boto3-backed) + macOS's BUILT-IN WebDAV client.
# No macFUSE, no kernel extension, no rclone — Apple's own signed driver only.
#
# Usage: bash mount.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$DIR/.serve.pid"
PORT=8765
MOUNT="$DIR/mnt/runpod"

mkdir -p "$MOUNT"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Already mounted at $MOUNT"
  exit 0
fi

echo "Starting WebDAV server on localhost:$PORT ..."
python "$DIR/serve.py" --port "$PORT" &
PID=$!
echo "$PID" > "$PID_FILE"

for i in $(seq 1 15); do
  if curl -sf "http://localhost:$PORT/" &>/dev/null; then
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Error: server failed to start — check credentials in .env" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
  sleep 1
done

mount_webdav -S "http://localhost:$PORT" "$MOUNT"

echo ""
echo "Volume mounted at: $MOUNT  (read-write)"
echo ""
echo "Open in VS Code:   code \"$MOUNT\""
echo "Open in Finder:    open \"$MOUNT\""
echo "To unmount:        bash unmount.sh"
