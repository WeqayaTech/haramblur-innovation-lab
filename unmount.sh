#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$DIR/.serve.pid"
MOUNT="$DIR/mnt/runpod"

if mount | grep -q " $MOUNT "; then
  diskutil unmount force "$MOUNT" 2>/dev/null || umount "$MOUNT" 2>/dev/null || true
  echo "Unmounted $MOUNT"
else
  echo "Nothing mounted at $MOUNT"
fi

if [ -f "$PID_FILE" ]; then
  kill -TERM "$(cat "$PID_FILE")" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "WebDAV server stopped."
fi
pkill -f "serve.py" 2>/dev/null || true
