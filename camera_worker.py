#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

STREAM_URL = os.environ.get('CAMERA_STREAM_URL', '').strip()
SNAPSHOT_SOURCE = os.environ.get('CAMERA_SNAPSHOT_SOURCE', '')
SNAPSHOT_DIR = Path(os.environ.get('CAMERA_SNAPSHOT_DIR', '/app/media/live'))
SNAPSHOT_NAME = os.environ.get('CAMERA_SNAPSHOT_NAME', 'latest.jpg')
STATUS_FILE = Path(os.environ.get('CAMERA_STATUS_FILE', '/app/data/camera_status.json'))
CAPTURE_INTERVAL_SEC = int(os.environ.get('CAMERA_CAPTURE_INTERVAL_SEC', '5'))
SNAPSHOT_STALE_SEC = int(os.environ.get('CAMERA_SNAPSHOT_STALE_SEC', '45'))
CAPTURE_TIMEOUT_SEC = int(os.environ.get('CAMERA_CAPTURE_TIMEOUT_SEC', '20'))
FFMPEG_BIN = os.environ.get('FFMPEG_BIN', 'ffmpeg')

def safe_stream_url(value):
    return re.sub(r'(://)[^/@]+@', r'\1***@', value)


def write_status(ok, detail='', snapshot=''):
    payload = {
        'ok': ok,
        'stream_url': safe_stream_url(STREAM_URL),
        'snapshot': snapshot,
        'detail': detail,
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'updated_unix': time.time(),
    }
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=STATUS_FILE.parent,
        prefix=f'.{STATUS_FILE.name}.', suffix='.tmp', delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    os.replace(tmp_path, STATUS_FILE)


def capture_once():
    if not STREAM_URL and not SNAPSHOT_SOURCE:
        write_status(False, detail='CAMERA_STREAM_URL is not configured')
        return False
    final_file = SNAPSHOT_DIR / SNAPSHOT_NAME
    final_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = final_file.with_name(f'.{final_file.stem}.{os.getpid()}.tmp{final_file.suffix}')
    if SNAPSHOT_SOURCE:
        source_file = Path(SNAPSHOT_SOURCE)
        if not source_file.exists():
            write_status(False, detail=f'snapshot source missing: {source_file}')
            return False
        shutil.copyfile(source_file, tmp_file)
        os.replace(tmp_file, final_file)
        write_status(True, detail='snapshot sync ok', snapshot=f'/media/live/{final_file.name}')
        return True
    cmd = [
        FFMPEG_BIN,
        '-y',
        '-loglevel', 'error',
        '-nostdin',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-analyzeduration', '0',
        '-probesize', '32768',
    ]
    if STREAM_URL.startswith('rtsp://'):
        cmd += ['-rtsp_transport', 'tcp', '-rw_timeout', '15000000']
    elif STREAM_URL.startswith(('http://', 'https://')):
        cmd += [
            '-rw_timeout', '15000000',
            '-reconnect', '1',
            '-reconnect_at_eof', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '5',
        ]
    cmd += [
        '-i', STREAM_URL,
        '-map', '0:v:0',
        '-frames:v', '1',
        '-q:v', '3',
        str(tmp_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(5, CAPTURE_TIMEOUT_SEC))
    if result.returncode != 0 or not tmp_file.exists():
        detail = (result.stderr or result.stdout or 'capture failed').strip()
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)
        if final_file.exists():
            age = time.time() - final_file.stat().st_mtime
            if age <= SNAPSHOT_STALE_SEC:
                write_status(True, detail=f'using cached snapshot after capture error: {detail}', snapshot=f'/media/live/{final_file.name}')
                return True
        write_status(False, detail=detail)
        return False
    os.replace(tmp_file, final_file)
    write_status(True, detail='live frame captured', snapshot=f'/media/live/{final_file.name}')
    return True


if __name__ == '__main__':
    write_status(False, detail='camera worker starting')
    while True:
        try:
            capture_once()
        except subprocess.TimeoutExpired:
            write_status(False, detail='capture timeout')
        except Exception as exc:
            write_status(False, detail=f'unexpected error: {exc}')
        time.sleep(CAPTURE_INTERVAL_SEC)
