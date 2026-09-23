#!/usr/bin/env python3
"""
Classifier accuracy regression for cat-water-drinking detection.

Runs inside the detector container (has cv2 + onnxruntime + shared data/media).

Evaluates the deployed drinking classifier against human/archive-labeled review
candidates and asserts that discrimination has not regressed — both overall and
on night (IR) samples specifically.

Assertions (baselines set 2026-08-04 from the retrained model):
  - confirmed crops scoring >= 0.75:  >= 0.85   (real drinking must score high)
  - rejected crops scoring >= 0.60:  <= 0.15   (false positives must stay low)
  - night rejected crops >= 0.60:    <= 0.20   (night IR false positives)
Usage:
  docker exec cat-water-detector python3 - < regression_accuracy.py
"""
import json
import sys
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DATA = "/app/data"
MEDIA = "/app/media"
MODEL = "/app/models/cat_drinking_classifier.onnx"

# ── thresholds under test ──────────────────────────────────────────
RECALL_THRESHOLD = 0.75     # confirmed (real drinking) must score at least this
FP_THRESHOLD = 0.60         # rejected must score below this
NIGHT_FP_THRESHOLD = 0.60

# ── assertion baselines ────────────────────────────────────────────
MIN_CONFIRMED_GE_075 = 0.85
MAX_REJECTED_GE_060 = 0.15
MAX_NIGHT_REJECTED_GE_060 = 0.20


def score(session, path):
    img = cv2.imread(path)
    if img is None or img.size == 0:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    tensor = np.transpose((resized - MEAN) / STD, (2, 0, 1))[np.newaxis]
    logit = float(np.asarray(
        session.run(None, {session.get_inputs()[0].name: tensor})[0]).reshape(-1)[0])
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def main():
    items = json.load(open(f"{DATA}/review_candidates.json"))["items"]
    labels = json.load(open(f"{DATA}/review_candidate_labels.json"))

    if not labels or not items:
        print("FAIL: no labeled review candidates available to test against")
        sys.exit(1)

    session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    print(f"model: {MODEL} providers={session.get_providers()}")

    buckets = defaultdict(list)   # (status, profile) -> scores
    unreadable = 0
    for item in items:
        rid = item["id"]
        status = labels.get(rid, {}).get("recovery_status")
        if status not in ("confirmed", "rejected"):
            continue
        profile = item.get("profile", "?")
        path = f"{MEDIA}/review_candidates/{rid}-crop.jpg"
        s = score(session, path)
        if s is None:
            unreadable += 1
            continue
        buckets[(status, profile)].append(s)

    def rate(seq, thr):
        return sum(1 for x in seq if x >= thr) / len(seq) if seq else 0.0

    def report(status):
        confirmed = buckets.get((status, "day"), []) + buckets.get((status, "?"), [])
        night = buckets.get((status, "night"), [])
        return confirmed, night

    conf_day, conf_night = report("confirmed")
    rej_day, rej_night = report("rejected")

    c_ge = rate(conf_day + conf_night, RECALL_THRESHOLD)
    r_ge = rate(rej_day + rej_night, FP_THRESHOLD)
    n_ge = rate(rej_night, NIGHT_FP_THRESHOLD)

    print(f"\nconfirmed crops: n={len(conf_day)+len(conf_night)} "
          f"(day={len(conf_day)}, night={len(conf_night)})")
    print(f"  >= {RECALL_THRESHOLD}: {c_ge:.1%}  (require >= {MIN_CONFIRMED_GE_075:.0%})")
    if conf_night:
        print(f"  night >= {RECALL_THRESHOLD}: {rate(conf_night, RECALL_THRESHOLD):.1%}")
    print(f"rejected crops: n={len(rej_day)+len(rej_night)} "
          f"(day={len(rej_day)}, night={len(rej_night)})")
    print(f"  >= {FP_THRESHOLD}: {r_ge:.1%}  (require <= {MAX_REJECTED_GE_060:.0%})")
    print(f"  night >= {NIGHT_FP_THRESHOLD}: {n_ge:.1%}  (require <= {MAX_NIGHT_REJECTED_GE_060:.0%})")
    if unreadable:
        print(f"unreadable crops skipped: {unreadable}")

    # night positive coverage (few samples expected; informational)
    night_pos = rate(conf_night, RECALL_THRESHOLD) if conf_night else None

    ok = (c_ge >= MIN_CONFIRMED_GE_075
          and r_ge <= MAX_REJECTED_GE_060
          and n_ge <= MAX_NIGHT_REJECTED_GE_060)
    # Since 2026-09-21 the production drinking decision is the ROI classifier
    # (roi_drinking.json); this crop model is only the automatic fallback, so
    # its gates are telemetry, not the release gate. The active-path gate is
    # scripts/run_replay_regression.sh (blind-label replay baseline).
    print(f"\nRESULT (legacy-fallback telemetry, gating regression is "
          f"replay_regression): {'PASS' if ok else 'BELOW-HISTORIC-BASELINE'}")
    print("  confirmed>=%.2f %s" % (RECALL_THRESHOLD, "OK" if c_ge >= MIN_CONFIRMED_GE_075 else "LOW"))
    print("  rejected>=%.2f %s" % (FP_THRESHOLD, "OK" if r_ge <= MAX_REJECTED_GE_060 else "HIGH"))
    print("  night-rejected>=%.2f %s" % (NIGHT_FP_THRESHOLD, "OK" if n_ge <= MAX_NIGHT_REJECTED_GE_060 else "HIGH"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
