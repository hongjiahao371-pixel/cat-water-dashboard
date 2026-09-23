#!/usr/bin/env python3
"""Assert the ROI replay of the blind-labeled 2026-09-09 window against the
accepted baseline. Runs inside the replay container after replay_realtime.py.

Accepted baseline (2026-09-21 blind-label recheck, production roi_drinking.json
enabled/0.80, replay of 93 labeled segments):
  - the single true drinking episode detected (>= 1 overlapping episode)
  - zero episodes in labeled non-drinking coverage (episode-level FP = 0)
  - second-level recall >= 0.95, precision >= 0.85 (0.5 s grid)
Fails loudly with the offending metrics; exits 0 only on PASS.
"""
import json
import sys
from datetime import datetime, timedelta, timezone

CST = timezone(timedelta(hours=8))

OUT_ROI = sys.argv[1] if len(sys.argv) > 1 else "/out/replay-regression-roi.json"
LABELS = "/source/accuracy-20260908/full-day/heldout-reference-20260909/labels.json"
DAY_COVERAGE_START = "2026-09-09T00:00:18+08:00"
DAY_COVERAGE_END = "2026-09-09T01:34:00+08:00"

MIN_RECALL = 0.95
MIN_PRECISION = 0.85


def epoch(name):
    # Explicit CST: containers run UTC; manifest/labels were built on the NAS
    # in Asia/Shanghai. Naive .timestamp() would shift truth by 8 hours.
    return datetime.strptime(name[:14], "%Y%m%d%H%M%S").replace(tzinfo=CST).timestamp()


def load_truth():
    d = json.load(open(LABELS))
    coverage, drink = [], []
    for r in d["rows"]:
        pts0, pts1 = r.get("reviewed_pts") or [0, 0]
        s0, s1 = epoch(r["source"]) + float(pts0), epoch(r["source"]) + float(pts1)
        if s1 <= epoch("20260909000000") or s0 >= epoch("20260909013400"):
            continue  # partial previous-day row / beyond regression window
        coverage.append((s0, s1))
        for v in r.get("drinking_visits", []):
            lap = v.get("supported_lapping_pts")
            if lap:
                drink.append((epoch(r["source"]) + lap[0], epoch(r["source"]) + lap[1]))
    drink.sort()
    merged = []
    for a, b in drink:
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    cov = []
    for a, b in sorted(coverage):
        if cov and a <= cov[-1][1]:
            cov[-1][1] = max(cov[-1][1], b)
        else:
            cov.append([a, b])
    return cov, merged


def load_episodes(path):
    ev = json.load(open(path)).get("events", [])
    eps, cur = [], None
    for e in ev:
        t = e.get("type", "")
        ts = float(e["observed_at"])
        if t == "drink_start":
            cur = {"start": ts}
        elif t == "drink_end" and cur is not None:
            cur["end"] = ts
            eps.append(cur)
            cur = None
    return eps


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    cov, drink = load_truth()
    t0, t1 = drink[0]
    eps = load_episodes(OUT_ROI)
    tp = [e for e in eps if overlap(e["start"], e["end"], t0, t1) > 0]
    fp = [e for e in eps if overlap(e["start"], e["end"], t0, t1) <= 0
          and sum(overlap(e["start"], e["end"], a, b) for a, b in cov) > 0.5]
    step = 0.5
    tp_s = fp_s = fn_s = 0.0
    for a, b in cov:
        t = a
        while t < b:
            in_truth = any(x0 <= t <= x1 for x0, x1 in drink)
            in_pred = any(e["start"] <= t <= e["end"] for e in eps)
            if in_truth and in_pred:
                tp_s += step
            elif in_truth:
                fn_s += step
            elif in_pred:
                fp_s += step
            t += step
    recall = tp_s / (tp_s + fn_s) if tp_s + fn_s else 0.0
    precision = tp_s / (tp_s + fp_s) if tp_s + fp_s else 0.0
    print("episodes: tp=%d fp=%d | second-level: recall=%.3f precision=%.3f"
          % (len(tp), len(fp), recall, precision))
    failures = []
    if not tp:
        failures.append("true episode missed")
    if fp:
        failures.append("episode-level false positives in labeled coverage")
    if recall < MIN_RECALL:
        failures.append("recall %.3f < %.2f" % (recall, MIN_RECALL))
    if precision < MIN_PRECISION:
        failures.append("precision %.3f < %.2f" % (precision, MIN_PRECISION))
    if failures:
        print("RESULT: FAIL — " + "; ".join(failures))
        sys.exit(1)
    print("RESULT: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
