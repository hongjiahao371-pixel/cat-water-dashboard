#!/usr/bin/env python3
"""Discover ONVIF cameras on the LAN and verify their clock via ONVIF.

Zero dependencies (stdlib only), no credentials required for discovery or
time reading on most cameras. Helps you fill CAMERA_SOURCE_URL / CAMERA_IP
in .env:

    python3 scripts/discover_camera.py            # discover + verify time
    python3 scripts/discover_camera.py 192.168.1.100

Then point a test player at rtsp://user:pass@<ip>:554/<profile> to confirm
the stream, and put the URL into .env as CAMERA_SOURCE_URL.
"""
import base64
import hashlib
import os
import re
import socket
import sys
from datetime import datetime, timedelta, timezone

CST = timezone(timedelta(hours=8))
DISCOVERY_ADDR = ("239.255.255.250", 3702)

PROBE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"'
    ' xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"'
    ' xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"'
    ' xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
    '<e:Header><w:MessageID>urn:uuid:ce6d7a4c-0a1f-4a86-a5e0-1a7f1d0e3c01</w:MessageID>'
    '<w:To e:mustUnderstand="1">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
    '<w:Action e:mustUnderstand="1">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>'
    '</e:Header><e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>'
    '</e:Envelope>'
)

TIME_BODY = '<tds:GetSystemDateAndTime xmlns="http://www.onvif.org/ver10/device/wsdl"/>'


def discover(timeout=4.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.settimeout(timeout)
    sock.bind(("0.0.0.0", 0))
    sock.sendto(PROBE.encode(), DISCOVERY_ADDR)
    found = {}
    try:
        while True:
            data, addr = sock.recvfrom(4096)
            found.setdefault(addr[0], data.decode("utf-8", "replace"))
    except socket.timeout:
        pass
    finally:
        sock.close()
    return found


def read_time(ip, port=80):
    """Read camera clock (ONVIF). Returns (iso_or_None, drift_seconds_or_None)."""
    envelope = ('<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"'
                ' xmlns:tds="http://www.onvif.org/ver10/device/wsdl">'
                "<s:Body>" + TIME_BODY + "</s:Body></s:Envelope>")
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://%s:%d/onvif/device_service" % (ip, port),
            data=envelope.encode(),
            headers={"Content-Type": "application/soap+xml; charset=utf-8"},
            method="POST")
        body = urllib.request.urlopen(req, timeout=6).read().decode("utf-8", "replace")
    except Exception as exc:
        return None, None, str(exc)
    try:
        vals = {tag: int(re.search(r"<tt:%s>(\d+)</tt:%s>" % (tag, tag), body).group(1))
                for tag in ("Year", "Month", "Day", "Hour", "Minute", "Second")}
    except AttributeError:
        return None, None, "no datetime in response"
    if "UTCDateTime" in body:
        cam = datetime(vals["Year"], vals["Month"], vals["Day"], vals["Hour"],
                       vals["Minute"], vals["Second"], tzinfo=timezone.utc)
    else:
        cam = datetime(vals["Year"], vals["Month"], vals["Day"], vals["Hour"],
                       vals["Minute"], vals["Second"], tzinfo=CST)
    drift = (cam - datetime.now(timezone.utc)).total_seconds()
    return cam.isoformat(), drift, ""


def main():
    direct = sys.argv[1] if len(sys.argv) > 1 else ""
    if direct:
        targets = {direct: ""}
    else:
        targets = discover()
        if not targets:
            print("no ONVIF cameras answered the LAN discovery probe "
                  "(some cameras need ONVIF enabled in their web UI)")
            return
    print("found %d ONVIF device(s):\n" % len(targets))
    for ip, body in sorted(targets.items()):
        model = ""
        m = re.search(r"<dn:hardware>(.*?)</dn:hardware>", body)
        if m:
            model = m.group(1)[:60]
        iso, drift, err = read_time(ip)
        line = "  %s  %s" % (ip, model)
        if iso:
            line += "  | clock drift %+.1fs (%s)" % (drift, iso)
        elif err:
            line += "  | time read: %s" % err
        print(line)
    print("\nnext steps:")
    print("  1. confirm the RTSP stream, e.g. ffplay 'rtsp://user:pass@<ip>:554/stream1'")
    print("  2. put that URL into .env as CAMERA_SOURCE_URL")
    print("  3. docker compose up -d")
    print("  optional (clock auto-sync / HD playback): set CAMERA_IP / "
          "EASYNVR_* in .env")


if __name__ == "__main__":
    main()
