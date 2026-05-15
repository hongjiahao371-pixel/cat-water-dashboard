#!/bin/sh
set -eu
while true; do
  python3 /app/hd_live_worker.py || true
  sleep 1
done
