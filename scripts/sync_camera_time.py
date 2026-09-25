#!/usr/bin/env python3
"""Camera clock drift check + ONVIF auto-sync (async onvif-zeep-async).

Env: CAMERA_IP, CAMERA_ONVIF_PORT (default 2020), CAMERA_USER, CAMERA_PASS,
DRIFT_SYNC_THRESHOLD (seconds, default 10; auto-sync when exceeded),
CAMERA_OFFSET_FILE (drift history JSON).
"""
import asyncio, json, os, sys, tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from onvif import ONVIFCamera

CAMERA_IP = os.environ.get("CAMERA_IP", "")
CAMERA_ONVIF_PORT = int(os.environ.get("CAMERA_ONVIF_PORT", "2020"))
CAMERA_USER = os.environ.get("CAMERA_USER", "")
CAMERA_PASS = os.environ.get("CAMERA_PASS", "")
DRIFT_SYNC_THRESHOLD = float(os.environ.get("DRIFT_SYNC_THRESHOLD", "10"))
CAMERA_OFFSET_FILE = os.environ.get("CAMERA_OFFSET_FILE", str(Path(__file__).resolve().parent.parent / "data" / "camera_offset.json"))
CST = timezone(timedelta(hours=8))


async def get_utc(devicemgmt):
    dt = await devicemgmt.GetSystemDateAndTime()
    u = dt.UTCDateTime
    return datetime(u.Date.Year, u.Date.Month, u.Date.Day,
                    u.Time.Hour, u.Time.Minute, u.Time.Second, tzinfo=timezone.utc)


async def set_utc(devicemgmt, dt_utc):
    params = devicemgmt.create_type("SetSystemDateAndTime")
    params.DateTimeType = "Manual"
    params.DaylightSavings = False
    params.TimeZone = "CST-8"
    params.UTCDateTime = {
        "Date": {"Year": dt_utc.year, "Month": dt_utc.month, "Day": dt_utc.day},
        "Time": {"Hour": dt_utc.hour, "Minute": dt_utc.minute, "Second": dt_utc.second},
    }
    await devicemgmt.SetSystemDateAndTime(params)


def record(nas_time, drift, synced, tag):
    f = Path(CAMERA_OFFSET_FILE)
    f.parent.mkdir(parents=True, exist_ok=True)
    hist = {}
    if f.exists():
        try: hist = json.loads(f.read_text())
        except Exception: pass
    hist.setdefault("history", []).append({
        "time": nas_time.isoformat(), "offset_seconds": round(drift, 1),
        "synced": synced, "tag": tag})
    hist["history"] = hist["history"][-100:]
    tmp = Path(str(f) + ".tmp")
    tmp.write_text(json.dumps(hist, indent=2))
    os.replace(tmp, f)


async def main():
    if not CAMERA_IP:
        print("CAMERA_IP not configured"); sys.exit(2)
    cam = ONVIFCamera(CAMERA_IP, CAMERA_ONVIF_PORT, CAMERA_USER, CAMERA_PASS)
    await cam.update_xaddrs()
    devicemgmt = await cam.create_devicemgmt_service()
    cam_utc = await get_utc(devicemgmt)
    nas_utc = datetime.now(timezone.utc)
    drift = (nas_utc - cam_utc).total_seconds()
    synced = False
    tag = "OK" if abs(drift) < 5 else "WARN"
    if abs(drift) > DRIFT_SYNC_THRESHOLD:
        if not CAMERA_USER or not CAMERA_PASS:
            tag = "NEEDS_CREDS"
            print(f"Camera drift: {drift:.1f}s exceeds threshold but no credentials configured")
        else:
            print(f"Camera drift: {drift:.1f}s - auto-syncing...")
            now = datetime.now(timezone.utc)
            await set_utc(devicemgmt, now)
            await asyncio.sleep(1)
            cam_utc2 = await get_utc(devicemgmt)
            drift = (datetime.now(timezone.utc) - cam_utc2).total_seconds()
            synced = True; tag = "SYNCED"
    print(f"Camera drift: {drift:.1f}s [{tag}] cam_utc={cam_utc.isoformat()}")
    record(datetime.now(CST), drift, synced, tag)
    if tag == "SYNC_FAIL":
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
