#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

RTSP_URL = os.environ.get('RTSP_URL', '').strip()
LISTEN_URL = os.environ.get('LISTEN_URL', 'http://0.0.0.0:39100/live.flv')
FFMPEG_BIN = os.environ.get('FFMPEG_BIN', '/usr/bin/ffmpeg')
SCALE_WIDTH = os.environ.get('SCALE_WIDTH', '1280')
CRF = os.environ.get('CRF', '23')
PRESET = os.environ.get('PRESET', 'ultrafast')
GOP = os.environ.get('GOP', '30')
MAXRATE = os.environ.get('MAXRATE', '4000k')
BUFSIZE = os.environ.get('BUFSIZE', '8000k')
OUTPUT_FPS = os.environ.get('OUTPUT_FPS', '8')
HWACCEL_MODE = os.environ.get('HWACCEL_MODE', 'qsv').lower()
STATUS_FILE = Path(os.environ.get('TRANSCODE_STATUS_FILE', '/app/data/transcode_status.json'))
INPUT_TIMEOUT_US = int(os.environ.get('INPUT_TIMEOUT_US', '15000000'))
MAX_RESTART_DELAY_SEC = float(os.environ.get('MAX_RESTART_DELAY_SEC', '20'))
STABLE_RUN_SEC = float(os.environ.get('STABLE_RUN_SEC', '20'))
STATUS_INTERVAL_SEC = float(os.environ.get('STATUS_INTERVAL_SEC', '5'))

running = True
child = None


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def write_status(**values):
    payload = {
        'ok': bool(values.pop('ok', False)),
        'updated_at': now_iso(),
        'updated_unix': time.time(),
        'output_fps': int(OUTPUT_FPS),
        'scale_width': int(SCALE_WIDTH),
        **values,
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


def terminate_child():
    global child
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=8)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


def stop(*_):
    global running
    running = False
    terminate_child()


def base_input_cmd():
    if not RTSP_URL:
        raise RuntimeError('RTSP_URL is not configured')
    cmd = [
        FFMPEG_BIN, '-nostdin', '-loglevel', 'warning',
        '-analyzeduration', '5M', '-probesize', '5M',
    ]
    source = RTSP_URL.lower()
    if source.startswith('rtsp://'):
        cmd += ['-rtsp_transport', 'tcp', '-rw_timeout', str(INPUT_TIMEOUT_US)]
    elif source.startswith(('http://', 'https://')):
        # EasyNVR serves an endless FLV stream. Reconnecting that stream within
        # the same FFmpeg process can retain stale FLV packet state, so allow the
        # supervisor to restart FFmpeg from a clean stream header instead.
        cmd += ['-rw_timeout', str(INPUT_TIMEOUT_US)]
    return cmd + ['-i', RTSP_URL, '-an']


def qsv_cmd():
    return base_input_cmd() + [
        '-vf', f'scale={SCALE_WIDTH}:-2,format=nv12',
        '-r', OUTPUT_FPS,
        '-c:v', 'h264_qsv',
        '-preset', PRESET,
        '-look_ahead', '0', '-bf', '0',
        '-g', GOP, '-keyint_min', GOP,
        '-maxrate', MAXRATE, '-bufsize', BUFSIZE,
        '-f', 'flv', '-listen', '1', LISTEN_URL,
    ]


def cpu_cmd():
    return base_input_cmd() + [
        '-c:v', 'libx264',
        '-preset', PRESET,
        '-tune', 'zerolatency',
        '-pix_fmt', 'yuv420p',
        '-g', GOP, '-keyint_min', GOP,
        '-crf', CRF,
        '-maxrate', MAXRATE, '-bufsize', BUFSIZE,
        '-r', OUTPUT_FPS,
        '-vf', f'scale={SCALE_WIDTH}:-2',
        '-f', 'flv', '-listen', '1', LISTEN_URL,
    ]


def command_chain():
    if HWACCEL_MODE == 'qsv':
        return [('qsv', qsv_cmd()), ('cpu-fallback', cpu_cmd())]
    return [('cpu', cpu_cmd())]


def main():
    global child
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    restart_count = 0
    restart_delay = 1.0
    write_status(ok=False, phase='starting', restart_count=restart_count, active_mode='')

    while running:
        try:
            commands = command_chain()
        except RuntimeError as exc:
            write_status(ok=False, phase='configuration_error', restart_count=restart_count,
                         active_mode='', detail=str(exc))
            time.sleep(min(MAX_RESTART_DELAY_SEC, restart_delay))
            restart_delay = min(MAX_RESTART_DELAY_SEC, restart_delay * 2)
            continue

        launched = False
        active_mode = ''
        for mode, command in commands:
            print(f'[hd_live_worker] starting mode={mode} hwaccel={HWACCEL_MODE} fps={OUTPUT_FPS}', flush=True)
            child = subprocess.Popen(command)
            time.sleep(2)
            code = child.poll()
            if code is None:
                launched = True
                active_mode = mode
                break
            print(f'[hd_live_worker] mode={mode} exited early code={code}, trying next', flush=True)

        if not launched:
            restart_count += 1
            write_status(ok=False, phase='restarting', restart_count=restart_count,
                         active_mode='', detail='ffmpeg exited during startup')
            time.sleep(restart_delay)
            restart_delay = min(MAX_RESTART_DELAY_SEC, restart_delay * 2)
            continue

        started_at = time.monotonic()
        last_status_at = 0.0
        while running:
            code = child.poll()
            uptime_sec = round(time.monotonic() - started_at, 1)
            if code is not None:
                print(f'[hd_live_worker] child exited code={code}, restarting', flush=True)
                restart_count += 1
                write_status(ok=False, phase='restarting', restart_count=restart_count,
                             active_mode=active_mode, uptime_sec=uptime_sec,
                             detail=f'ffmpeg exited code={code}')
                if uptime_sec >= STABLE_RUN_SEC:
                    restart_delay = 1.0
                else:
                    restart_delay = min(MAX_RESTART_DELAY_SEC, restart_delay * 2)
                break
            if time.monotonic() - last_status_at >= STATUS_INTERVAL_SEC:
                write_status(ok=True, phase='running', restart_count=restart_count,
                             active_mode=active_mode, uptime_sec=uptime_sec, detail='stream active')
                last_status_at = time.monotonic()
            time.sleep(1)

        if running and child and child.poll() is not None:
            time.sleep(restart_delay)

    terminate_child()
    write_status(ok=False, phase='stopped', restart_count=restart_count, active_mode='', detail='worker stopped')


if __name__ == '__main__':
    main()
