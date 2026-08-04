#!/usr/bin/env python3
"""cat-water-dashboard health watchdog.

Runs from cron every 5 min. Checks container health + heartbeat freshness of
detector/transcode/camera. Alerts to the configured Feishu webhook on failure,
rate-limited by a cooldown to avoid spamming.
"""
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

REPO = pathlib.Path('/volume1/docker/cat-water-dashboard')
DATA = REPO / 'data'
LOG_DIR = REPO / 'logs'
STATE_FILE = DATA / 'health_alert_state.json'
COOLDOWN_SEC = 6 * 3600
FRESHNESS_SEC = 600  # heartbeat older than 10 min => stale

EXPECTED = {
    'cat-water-archive-verifier', 'cat-water-camera', 'cat-water-dashboard',
    'cat-water-daily-audit', 'cat-water-detector', 'cat-water-model-trainer',
    'cat-water-notifier', 'cat-water-qsv-transcode',
}
STATUS_FILES = [
    ('detector', 'detector_status.json'),
    ('transcode', 'transcode_status.json'),
    ('camera', 'camera_status.json'),
]

problems = []


def fresh(payload):
    return isinstance(payload, dict) and time.time() - payload.get('updated_unix', 0) < FRESHNESS_SEC


# 1. containers
try:
    out = subprocess.run(
        ['docker', 'ps', '--filter', 'name=cat-water-', '--format', '{{.Names}}|{{.Status}}'],
        capture_output=True, text=True, timeout=30,
    ).stdout
    running, unhealthy = set(), []
    for line in out.strip().splitlines():
        if not line:
            continue
        name, status = line.split('|', 1)
        running.add(name)
        if 'unhealthy' in status.lower():
            unhealthy.append(f'{name}:{status}')
    missing = EXPECTED - running
    if missing:
        problems.append('容器未运行: ' + ', '.join(sorted(missing)))
    if unhealthy:
        problems.append('容器不健康: ' + ', '.join(unhealthy))
except Exception as exc:
    problems.append(f'docker ps 失败: {exc}')

# 2. heartbeat freshness
for name, filename in STATUS_FILES:
    path = DATA / filename
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        problems.append(f'{name}: 状态文件不可读({exc})')
        continue
    if not payload.get('ok'):
        problems.append(f'{name}: ok=false phase={payload.get("phase", payload.get("detail", "?"))}')
    elif not fresh(payload):
        problems.append(f'{name}: 心跳过期(>{FRESHNESS_SEC}s)')

LOG_DIR.mkdir(parents=True, exist_ok=True)
stamp = time.strftime('%Y-%m-%d %H:%M:%S')


def webhook_url():
    try:
        payload = json.loads((DATA / 'push.json').read_text(encoding='utf-8'))
        return str(payload.get('openclawWebhook', '') or '')
    except Exception:
        return ''


if not problems:
    print(f'{stamp} OK')
    sys.exit(0)

print(f'{stamp} PROBLEMS: {problems}')
state = {}
if STATE_FILE.exists():
    try:
        state = json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except Exception:
        state = {}
if time.time() - state.get('last_alert_unix', 0) >= COOLDOWN_SEC:
    hook = webhook_url()
    if hook:
        message = 'cat-water 健康告警\n' + '\n'.join(problems)
        payload = json.dumps({'msg_type': 'text', 'content': {'text': message}})
        try:
            req = urllib.request.Request(hook, data=payload.encode(),
                                         headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=10)
            sent = True
        except Exception as exc:
            sent = False
            problems.append(f'webhook 发送失败: {exc}')
        if sent:
            (LOG_DIR / 'cron.log').open('a').write(f'{stamp} [HEALTH ALERT SENT]\n')
        STATE_FILE.write_text(json.dumps({'last_alert_unix': time.time(), 'problems': problems}))
    else:
        (LOG_DIR / 'cron.log').open('a').write(f'{stamp} [HEALTH PROBLEM, 无 webhook]\n')
sys.exit(2)
