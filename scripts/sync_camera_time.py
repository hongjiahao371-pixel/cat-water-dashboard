#!/usr/bin/env python3
import sys, os
from datetime import datetime
from onvif import ONVIFCamera

CAMERA_IP = os.environ.get('CAMERA_IP', '192.168.5.41')
CAMERA_PORT = int(os.environ.get('CAMERA_ONVIF_PORT', '2020'))
CAMERA_USER = os.environ.get('CAMERA_USER', '')
CAMERA_PASS = os.environ.get('CAMERA_PASS', '')
DRIFT_WARN = 30
DRIFT_SYNC = 120

def get_time(cam):
    dt = cam.devicemgmt.GetSystemDateAndTime()
    u = dt.UTCDateTime
    return datetime(u.Date.Year, u.Date.Month, u.Date.Day,
                    u.Time.Hour, u.Time.Minute, u.Time.Second)

def main():
    print(f"[{datetime.now().isoformat()}] Camera time sync check")
    if not CAMERA_USER or not CAMERA_PASS:
        print('  ERROR: CAMERA_USER and CAMERA_PASS must be configured')
        sys.exit(2)
    try:
        cam = ONVIFCamera(CAMERA_IP, CAMERA_PORT, CAMERA_USER, CAMERA_PASS)
        cam_utc = get_time(cam)
        now_utc = datetime.utcnow()
        drift = abs((now_utc - cam_utc).total_seconds())
        print(f"  Camera UTC: {cam_utc.isoformat()}")
        print(f"  NAS    UTC: {now_utc.isoformat()}")
        print(f"  Drift: {drift:.0f}s")
        if drift <= DRIFT_WARN:
            print("  Status: OK")
        elif drift <= DRIFT_SYNC:
            print(f"  Status: WARNING drift={drift:.0f}s > {DRIFT_WARN}s")
        else:
            print(f"  Status: SYNCING drift={drift:.0f}s > {DRIFT_SYNC}s")
            now = datetime.utcnow()
            cam.devicemgmt.SetSystemDateAndTime({
                'DateTimeType': 'Manual', 'DaylightSavings': False,
                'TimeZone': {'TZ': 'CST-8'},
                'UTCDateTime': {
                    'Date': {'Year': now.year, 'Month': now.month, 'Day': now.day},
                    'Time': {'Hour': now.hour, 'Minute': now.minute, 'Second': now.second}
                }
            })
            new_drift = abs((datetime.utcnow() - get_time(cam)).total_seconds())
            print(f"  Sync OK, new drift={new_drift:.0f}s")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
