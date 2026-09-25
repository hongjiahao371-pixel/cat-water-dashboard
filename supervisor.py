#!/usr/bin/env python3
"""Single-container process supervisor for all cat-water workers.

Spawns every worker as a child process, restarts crashed children with
exponential backoff, forwards SIGTERM/SIGINT for a clean full stop, and
publishes an aggregate status file consumed by the container healthcheck
and /api/health. Workers keep writing their own status JSONs into /app/data,
so the dashboard and notification stack need no changes.

Configuration: the WORKERS list below. Add a name+cmd pair to run another
worker inside the same container.
"""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

APP = Path(__file__).resolve().parent
DATA = APP / "data"
LOG_DIR = DATA / "worker_logs"
STATUS_FILE = DATA / "supervisor_status.json"
CST = timezone(timedelta(hours=8))

# (name, cmd, optional): optional workers (e.g. the 5.5GB LocateAnything
# semantic verifier) are skipped with a clear status after repeated startup
# failures instead of crash-looping forever - fresh installs without the
# optional model stay healthy.
WORKERS = [
    ("dashboard", ["python3", "-u", "server.py"], False),
    ("detector", ["python3", "-u", "detection_worker.py"], False),
    ("camera", ["python3", "-u", "camera_worker.py"], False),
    ("transcode", ["python3", "-u", "hd_live_worker.py"], False),
    ("archive-verifier", ["python3", "-u", "archive_verifier_worker.py"], False),
    ("daily-audit", ["python3", "-u", "daily_audit_worker.py"], False),
    ("notifier", ["python3", "-u", "notification_worker.py"], False),
    ("model-trainer", ["python3", "-u", "classifier_calibration_worker.py"], False),
    ("locate-verifier", ["python3", "-u", "locate_verifier_worker.py"], True),
]
GIVE_UP_AFTER = 3  # consecutive failures for optional workers

BACKOFF_BASE = 5.0
BACKOFF_MAX = 60.0
HEALTHY_UPTIME = 300.0  # a child running this long resets its failure counter

procs = {}
meta = {}
stopping = False


def now():
    return time.time()


def stamp():
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def spawn(name, cmd):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    # rotate worker logs at ~20 MB so they cannot grow unbounded
    try:
        if log_path.exists() and log_path.stat().st_size > 20 * 1024 * 1024:
            rotated = log_path.with_suffix(".log.1")
            log_path.replace(rotated)
    except OSError:
        pass
    handle = open(log_path, "ab")
    proc = subprocess.Popen(cmd, cwd=str(APP), stdout=handle, stderr=subprocess.STDOUT)
    handle.close()
    procs[name] = proc
    m = meta.setdefault(name, {"restarts": 0, "last_exit_code": None})
    m["pid"] = proc.pid
    m["started_at"] = now()
    print(f"[{stamp()}] started {name} pid={proc.pid}", flush=True)


def write_status():
    payload = {
        "updated_unix": now(),
        "updated_at": datetime.now(CST).isoformat(),
        "stopping": stopping,
        "workers": {
            name: {
                "pid": meta.get(name, {}).get("pid"),
                "running": name in procs and procs[name].poll() is None,
                "restarts": meta.get(name, {}).get("restarts", 0),
                "given_up": meta.get(name, {}).get("given_up", False),
                "last_exit_code": meta.get(name, {}).get("last_exit_code"),
                "started_at": meta.get(name, {}).get("started_at"),
            } for name, _cmd, _opt in WORKERS
        },
    }
    tmp = STATUS_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, STATUS_FILE)
    except OSError:
        pass


def handle_signal(signum, frame):
    global stopping
    stopping = True
    print(f"[{stamp()}] received signal {signum}, stopping children…", flush=True)


def main():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    for name, cmd, optional in WORKERS:
        spawn(name, cmd)
    write_status()
    while not stopping:
        time.sleep(5)
        respawn = []
        for name, cmd, optional in WORKERS:
            if meta.get(name, {}).get("given_up"):
                continue
            proc = procs.get(name)
            if proc is None or proc.poll() is None:
                continue
            code = proc.poll()
            m = meta[name]
            uptime = now() - m.get("started_at", 0)
            m["last_exit_code"] = code
            m["pid"] = None
            if uptime >= HEALTHY_UPTIME:
                m["failures"] = 0
            failures = m.get("failures", 0) + 1
            m["failures"] = failures
            if optional and failures >= GIVE_UP_AFTER:
                m["given_up"] = True
                print(f"[{stamp()}] optional worker {name} failed {failures}x - "
                      f"giving up (optional component; fix the cause and restart "
                      f"the container to re-enable)", flush=True)
                continue
            delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** min(failures, 5)))
            respawn.append((name, cmd, m.get("exit_at", 0) or now(), delay))
            m["exit_at"] = now()
            print(f"[{stamp()}] {name} exited code={code}, "
                  f"restart in {delay:.0f}s (failures={failures})", flush=True)
        for name, cmd, exit_at, delay in respawn:
            if now() - meta[name]["exit_at"] >= delay:
                meta[name]["restarts"] = meta[name].get("restarts", 0) + 1
                spawn(name, cmd)
        write_status()
    # graceful stop: TERM, wait up to 10s, then KILL survivors
    for name, proc in procs.items():
        if proc.poll() is None:
            proc.terminate()
    deadline = now() + 10
    for name, proc in procs.items():
        while proc.poll() is None and now() < deadline:
            time.sleep(0.2)
        if proc.poll() is None:
            proc.kill()
    write_status()
    print(f"[{stamp()}] supervisor stopped", flush=True)


if __name__ == "__main__":
    main()
