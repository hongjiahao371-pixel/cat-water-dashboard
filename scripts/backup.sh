#!/bin/sh
# 备份运行数据(JSON 配置/事件/标签/状态) 与 ROI 配置，保留最近 14 份
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO/backups"
LOG_DIR="$REPO/logs"
mkdir -p "$DEST" "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
tar -czf "$DEST/backup-$STAMP.tar.gz" \
  -C "$REPO" \
  --exclude=\x27data/recent_review_clips\x27 \
  data/*.json settings push models/roi_drinking.json 2>/dev/null
ls -1t "$DEST"/backup-*.tar.gz | tail -n +15 | xargs -r rm -f
echo "$STAMP backup done: $(du -h "$DEST/backup-$STAMP.tar.gz" | cut -f1)"
