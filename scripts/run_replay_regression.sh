#!/bin/sh
# Nightly accuracy regression for the ACTIVE drinking decision path (ROI).
#
# Replays the blind-labeled 2026-09-09 window (93 segments, predictions-blind
# human labels; 1 true episode + ~55 min labeled non-drinking including five
# cat-present-no-drink risk stretches) through the production detector code
# and roi_drinking.json in an isolated no-network container, then asserts the
# accepted baseline (episode TP>=1, episode FP=0, recall>=0.95, prec>=0.85).
#
# Baseline accepted 2026-09-21 (RECHECK-20260909-RESULT.txt):
#   episode 1/1, recall 1.000, precision 0.913, duration error +4.7%.
# Runtime ~25 min at 2 CPUs. Results: logs/replay_regression.log (+ dated copy).
set -u
REPO=/volume1/docker/cat-water-dashboard
A=$REPO/accuracy-20260908
OUT=$A/recovery-20260918
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d)
IMAGE=ghcr.io/hongjiahao371-pixel/cat-water-dashboard:latest

docker run --rm --name cat-replay-regression --network none --cpus 2 -m 1536m \
  -v /volume1/docker/easynvr_rec:/recordings:ro \
  -v "$REPO":/source:ro \
  -v "$OUT":/out \
  $IMAGE python3 -u /source/accuracy-20260908/full-day/replay_realtime.py \
  --source-dir /source \
  --manifest /out/recheck-20260909-manifest.json \
  --mode roi --output /out/replay-regression-roi.json \
  --fps 5 --width 960 --height 720 \
  --model-dir /source/models \
  --settings /source/accuracy-20260908/full-day/frozen-runtime-20260909/settings.json \
  --calibration /source/accuracy-20260908/full-day/frozen-runtime-20260909/classifier_calibration.json \
  --runtime-env /source/accuracy-20260908/full-day/frozen-runtime-20260909/detector-env.json \
  --roi-config /source/models/roi_drinking.json
REPLAY_RC=$?

docker run --rm --network none \
  -v "$REPO":/source:ro \
  -v "$OUT":/out \
  $IMAGE python3 /source/tests/replay_regression_baseline.py \
  /out/replay-regression-roi.json
ASSERT_RC=$?

{
  echo "=== $STAMP replay_rc=$REPLAY_RC assert_rc=$ASSERT_RC"
} >> "$LOG_DIR/replay_regression.log"
cp "$OUT/replay-regression-roi.json" "$LOG_DIR/replay_regression_$STAMP.json" 2>/dev/null
exit $((REPLAY_RC != 0 || ASSERT_RC != 0))
