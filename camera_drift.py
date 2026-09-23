#!/usr/bin/env python3
"""Camera drift monitor with auto-sync - called from detection_worker.py."""
import json, os, base64, hashlib, re as _re, tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

CAMERA_IP = os.environ.get('CAMERA_IP', '')
CAMERA_ONVIF_PORT = int(os.environ.get('CAMERA_ONVIF_PORT', '2020'))
CAMERA_USER = os.environ.get('CAMERA_USER', '')
CAMERA_PASS = os.environ.get('CAMERA_PASS', '')
DRIFT_SYNC_THRESHOLD = float(os.environ.get('DRIFT_SYNC_THRESHOLD', '10'))
CAMERA_OFFSET_FILE = os.environ.get('CAMERA_OFFSET_FILE', '/app/data/camera_offset.json')
CST = timezone(timedelta(hours=8))

def _onvif_request(soap_body):
    """Send ONVIF request. WS-Security header only when credentials are set;
    GetSystemDateAndTime is readable unauthenticated on this camera."""
    import urllib.request
    nonce = os.urandom(16)
    created = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    auth_header = ''
    if CAMERA_USER:
        digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + CAMERA_PASS.encode()).digest()).decode()
        nonce_b64 = base64.b64encode(nonce).decode()
        auth_header = (
            '<Security s:mustUnderstand="1" xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">'
            '<UsernameToken>'
            '<Username>' + CAMERA_USER + '</Username>'
            '<Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">' + digest + '</Password>'
            '<Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd#Base64Binary">' + nonce_b64 + '</Nonce>'
            '<Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">' + created + '</Created>'
            '</UsernameToken></Security>')
    envelope = (
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"'
        ' xmlns:tds="http://www.onvif.org/ver10/device/wsdl"'
        ' xmlns:tt="http://www.onvif.org/ver10/schema">'
        '<s:Header>' + auth_header +
        '</s:Header>'
        '<s:Body>' + soap_body + '</s:Body>'
        '</s:Envelope>'
    )
    req = urllib.request.Request(
        'http://' + CAMERA_IP + ':' + str(CAMERA_ONVIF_PORT) + '/onvif/device_service',
        data=envelope.encode(),
        headers={'Content-Type': 'application/soap+xml; charset=utf-8'},
        method='POST'
    )
    resp = urllib.request.urlopen(req, timeout=10)
    body = resp.read().decode()
    # Check for SOAP fault
    if '<s:Fault>' in body or '<SOAP-ENV:Fault>' in body:
        m = _re.search(r'<SOAP-ENV:Text[^>]*>(.*?)</SOAP-ENV:Text>', body)
        err = m.group(1) if m else 'unknown SOAP fault'
        raise RuntimeError('ONVIF SOAP fault: ' + err)
    return body

def _get_camera_time():
    """Get camera local datetime (CST). Returns None on parse failure."""
    body = _onvif_request('<tds:GetSystemDateAndTime xmlns:tds="http://www.onvif.org/ver10/device/wsdl"/>')
    m = _re.search(r'<tt:LocalDateTime>(.*?)</tt:LocalDateTime>', body, _re.DOTALL)
    if not m:
        return None
    try:
        def p(tag):
            return int(_re.search(r'<tt:' + tag + r'>(\d+)</tt:' + tag + '>', m.group(1)).group(1))
        return datetime(p('Year'), p('Month'), p('Day'), p('Hour'), p('Minute'), p('Second'), tzinfo=CST)
    except (AttributeError, ValueError):
        return None

def _sync_camera_time():
    """Set camera time to match NAS via ONVIF SetSystemDateAndTime."""
    now_utc = datetime.now(timezone.utc)
    soap = (
        '<tds:SetSystemDateAndTime xmlns:tds="http://www.onvif.org/ver10/device/wsdl">'
        '<tds:DateTimeType>Manual</tds:DateTimeType>'
        '<tds:DaylightSavings>false</tds:DaylightSavings>'
        '<tds:TimeZone><tt:TZ xmlns:tt="http://www.onvif.org/ver10/schema">CST-8</tt:TZ></tds:TimeZone>'
        '<tds:UTCDateTime>'
        '<tt:Time xmlns:tt="http://www.onvif.org/ver10/schema">'
        '<tt:Hour>%d</tt:Hour><tt:Minute>%d</tt:Minute><tt:Second>%d</tt:Second>'
        '</tt:Time>'
        '<tt:Date xmlns:tt="http://www.onvif.org/ver10/schema">'
        '<tt:Year>%d</tt:Year><tt:Month>%d</tt:Month><tt:Day>%d</tt:Day>'
        '</tt:Date>'
        '</tds:UTCDateTime>'
        '</tds:SetSystemDateAndTime>' % (now_utc.hour, now_utc.minute, now_utc.second, now_utc.year, now_utc.month, now_utc.day)
    )
    _onvif_request(soap)

def check_camera_drift():
    """Check camera drift, auto-sync if above threshold.

    Time reading needs no credentials on this camera; auto-sync does, so a
    large drift without credentials is reported as NEEDS_CREDS instead of
    skipping the whole check.
    """
    try:
        cam_time = _get_camera_time()
        if cam_time is None:
            print("Camera drift: failed to parse camera time")
            return
        nas_time = datetime.now(CST)
        drift = (nas_time - cam_time).total_seconds()

        # Auto-sync if drift exceeds threshold
        if abs(drift) > DRIFT_SYNC_THRESHOLD:
            if not CAMERA_USER or not CAMERA_PASS:
                print("Camera drift: %.1fs exceeds threshold but ONVIF "
                      "credentials are not configured" % drift)
                tag = "NEEDS_CREDS"
                _record_offset(cam_time, nas_time, drift, False, tag)
                return
            print("Camera drift: %.1fs - auto-syncing..." % drift)
            try:
                _sync_camera_time()
                import time
                time.sleep(1)
                cam_time2 = _get_camera_time()
                if cam_time2:
                    drift = (datetime.now(CST) - cam_time2).total_seconds()
                    tag = "SYNCED"
                else:
                    tag = "SYNCED(?)"
            except Exception as e:
                print("Camera time sync failed: %s" % e)
                tag = "SYNC_FAIL"
        else:
            tag = "OK" if abs(drift) < 5 else "WARN"

        print("Camera drift: %.1fs [%s] cam=%s nas=%s" % (
            drift, tag,
            cam_time.strftime('%H:%M:%S'),
            nas_time.strftime('%H:%M:%S')
        ))
        _record_offset(cam_time, nas_time, drift, tag == 'SYNCED', tag)
    except Exception as e:
        print("Camera drift check failed: %s" % e)


def _record_offset(cam_time, nas_time, drift, synced, tag):
    offset_file = Path(CAMERA_OFFSET_FILE)
    offset_file.parent.mkdir(parents=True, exist_ok=True)
    history = {}
    if offset_file.exists():
        try:
            history = json.loads(offset_file.read_text())
        except Exception:
            pass
    history.setdefault('history', []).append({
        'time': nas_time.isoformat(),
        'offset_seconds': round(drift, 1),
        'synced': synced,
        'tag': tag,
    })
    history['history'] = history['history'][-100:]
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=offset_file.parent,
        prefix=f'.{offset_file.name}.', suffix='.tmp', delete=False,
    ) as handle:
        json.dump(history, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    os.replace(tmp_path, offset_file)

if __name__ == '__main__':
    check_camera_drift()
