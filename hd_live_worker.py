#!/usr/bin/env python3
import os, signal, subprocess, time

RTSP_URL = os.environ.get('RTSP_URL', 'rtsp://admin:Hjh1981940361@192.168.5.41:554/stream1')
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
QSV_DEVICE = os.environ.get('QSV_DEVICE', '/dev/dri/renderD128')

running = True
child = None

def stop(*_):
    global running, child
    running = False
    if child and child.poll() is None:
        child.terminate()

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

def base_input_cmd():
    return [
        FFMPEG_BIN, '-nostdin', '-loglevel', 'warning',
        '-rtsp_transport', 'tcp',
        '-analyzeduration', '5M', '-probesize', '5M',
        '-i', RTSP_URL, '-an',
    ]

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

while running:
    launched = False
    for mode, cmd in command_chain():
        print(f'[hd_live_worker] starting mode={mode} hwaccel={HWACCEL_MODE} fps={OUTPUT_FPS} url={LISTEN_URL}', flush=True)
        child = subprocess.Popen(cmd)
        time.sleep(2)
        code = child.poll()
        if code is None:
            launched = True
            break
        print(f'[hd_live_worker] mode={mode} exited early code={code}, trying next', flush=True)
    if not launched:
        time.sleep(2)
        continue
    while running:
        code = child.poll()
        if code is not None:
            print(f'[hd_live_worker] child exited code={code}, restarting', flush=True)
            break
        time.sleep(1)
    if running and child and child.poll() is not None:
        time.sleep(1)
