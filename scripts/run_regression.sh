#!/bin/bash
# Daily classifier accuracy regression for cat-water-dashboard.
# Run: docker exec cat-water python3 - < tests/regression_accuracy.py
# Alerts to Feishu webhook on FAIL or crash. Retains 30 days of logs.
set -u
REPO="$(cd "$(dirname "$0")/.." \&\& pwd)"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/regression_$(date +%Y%m%d).log"

docker exec -i cat-water-detector python3 - < "$REPO/tests/regression_accuracy.py" > "$LOG" 2>&1
EXIT=$?
RESULT=$(grep -oE 'RESULT: (PASS|FAIL)' "$LOG" | head -1 | awk '{print $2}')
RESULT=${RESULT:-FAILED}

# prune logs older than 30 days
find "$LOG_DIR" -name 'regression_*.log' -mtime +30 -delete 2>/dev/null

if [ "$EXIT" -ne 0 ] || [ "$RESULT" != "PASS" ]; then
  WEBHOOK=$(python3 -c "import json;print(json.load(open('$REPO/data/push.json')).get('openclawWebhook',''))" 2>/dev/null)
  if [ -n "$WEBHOOK" ]; then
    TAIL=$(tail -5 "$LOG")
    python3 - "$WEBHOOK" "$RESULT" "$TAIL" <<'PYEOF'
import sys, json, urllib.request
hook, result, tail = sys.argv[1], sys.argv[2], sys.argv[3]
payload = json.dumps({"msg_type": "text", "content": {"text": f"cat-water 分类器回归 {result}\n{tail}"}})
urllib.request.urlopen(urllib.request.Request(hook, data=payload.encode(), headers={"Content-Type": "application/json"}), timeout=10)
PYEOF
  fi
  echo "$(date +%F\ %T) FAILED (exit=$EXIT)" >> "$LOG_DIR/cron.log"
  exit 1
fi

echo "$(date +%F\ %T) OK -> $LOG" >> "$LOG_DIR/cron.log"
