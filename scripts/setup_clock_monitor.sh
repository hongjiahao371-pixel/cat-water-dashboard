#!/bin/sh
# 相机时钟漂移监控 + ONVIF 自动对时：一键安装/自检（宿主机 crontab，每日 04:15）
# 前提: .env 已配置 CAMERA_IP（必填）；CAMERA_USER/CAMERA_PASS（自动对时需要，只测不纠可不填）
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY=/usr/bin/python3
PYPATH="$REPO/pylibs"

[ -f "$REPO/.env" ] || { echo "缺少 $REPO/.env（可从 .env.example 复制）"; exit 1; }
set -a; . "$REPO/.env"; set +a
[ -n "${CAMERA_IP:-}" ] || { echo "请先在 .env 配置 CAMERA_IP（python3 scripts/discover_camera.py 可发现）"; exit 1; }

[ -d "$PYPATH/onvif" ] || {
  echo "首次安装: 下载 ONVIF 依赖到 $PYPATH ..."
  $PY -m pip install -q --target "$PYPATH" onvif-zeep-async || { echo "安装失败"; exit 1; }
}
PYTHONPATH="$PYPATH" $PY -c "import onvif" || { echo "onvif 库仍不可用"; exit 1; }

mkdir -p "$REPO/logs" "$REPO/data"
CAMERA_OFFSET_FILE="$REPO/data/camera_offset.json" PYTHONPATH="$PYPATH" $PY "$REPO/scripts/sync_camera_time.py" || exit 1

MARKER="# cat-water-clock-monitor"
CRON_LINE="15 4 * * * cd $REPO && set -a && . $REPO/.env && set +a && CAMERA_OFFSET_FILE=$REPO/data/camera_offset.json PYTHONPATH=$PYPATH $PY $REPO/scripts/sync_camera_time.py >> $REPO/logs/camera_drift.log 2>&1 $MARKER"
if [ "$(id -u)" = "0" ]; then
  (crontab -l 2>/dev/null | grep -v "cat-water-clock-monitor"; echo "$CRON_LINE") | crontab -
  echo "已安装 root crontab（每日 04:15 漂移检查+超阈值自动对时）"
else
  (crontab -l 2>/dev/null | grep -v "cat-water-clock-monitor"; echo "$CRON_LINE") | crontab - 2>/dev/null && echo "已安装当前用户 crontab（每日 04:15）" || {
    echo "当前用户无 crontab 权限，请用 root 执行:"; echo "  sudo $0"; exit 1; }
fi
