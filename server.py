#!/usr/bin/env python3
import json
import os
import shutil
import socket
import subprocess
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Lock
import re
from datetime import datetime, timezone, timedelta
from time import time, sleep
from urllib.error import URLError
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DATA_FILE = DATA_DIR / 'mock.json'
SETTINGS_FILE = DATA_DIR / 'settings.json'
PUSH_FILE = DATA_DIR / 'push.json'
PUSH_LOG_FILE = DATA_DIR / 'openclaw-push-log.json'
CAMERA_STATUS_FILE = DATA_DIR / 'camera_status.json'
DETECTOR_STATUS_FILE = DATA_DIR / 'detector_status.json'
BACKFILL_STATUS_FILE = DATA_DIR / 'backfill_status.json'
LABEL_STATS_FILE = DATA_DIR / 'label_stats.json'
POSITIVE_DIR = DATA_DIR / 'positive'
NEGATIVE_DIR = DATA_DIR / 'negative'
LEGACY_CAMERA_LIVE_STREAM_URL = os.environ.get('CAMERA_LIVE_STREAM_URL', '')
CAMERA_LIVE_STREAM_URL_HD = os.environ.get('CAMERA_LIVE_STREAM_URL_HD', 'http://192.168.5.17:39100/live.flv')
CAMERA_LIVE_STREAM_URL_SMOOTH = os.environ.get('CAMERA_LIVE_STREAM_URL_SMOOTH', '')
EASYNVR_BASE_URL = os.environ.get('EASYNVR_BASE_URL', 'http://192.168.5.17:10000')
EASYNVR_USERNAME = os.environ.get('EASYNVR_USERNAME', 'admin')
EASYNVR_PASSWORD = os.environ.get('EASYNVR_PASSWORD', 'admin')
EASYNVR_CHANNEL_ID = os.environ.get('EASYNVR_CHANNEL_ID', 'OpD6HNpPU8j0X_profile_1')
EASYNVR_CHANNEL_ID_HD = os.environ.get('EASYNVR_CHANNEL_ID_HD', EASYNVR_CHANNEL_ID)
EASYNVR_CHANNEL_ID_SMOOTH = os.environ.get('EASYNVR_CHANNEL_ID_SMOOTH', 'OpD6HNpPU8j0X_profile_2')
MEDIA_DIR = BASE_DIR / 'media'
MEDIA_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
FILE_LOCK = Lock()
STREAM_LOCK = Lock()
EASYNVR_SESSION = {'token': '', 'expired_at': 0}

def live_flv_url(mode='smooth'):
    if mode == 'hd':
        explicit_hd = CAMERA_LIVE_STREAM_URL_HD.strip()
        if explicit_hd:
            return explicit_hd
        channel_id = EASYNVR_CHANNEL_ID_HD
        return f'{EASYNVR_BASE_URL}/live/{channel_id}.flv'

    explicit = CAMERA_LIVE_STREAM_URL_SMOOTH.strip()
    if explicit:
        return explicit
    legacy = LEGACY_CAMERA_LIVE_STREAM_URL.strip()
    if legacy and 'OpML1KfFPOkPF_profile_1' not in legacy:
        return legacy
    channel_id = EASYNVR_CHANNEL_ID_SMOOTH
    return f'{EASYNVR_BASE_URL}/live/{channel_id}.flv'

DEFAULT_SETTINGS = {
    'cameraName': 'cat-water-cam-01',
    'rtspUrl': 'rtsp://admin:***@192.168.5.30:554/stream1',
    'roi': {'x1': 820, 'y1': 540, 'x2': 1320, 'y2': 980},
    'fpsSample': 2,
    'minPresenceSec': 3,
    'mergeGapSec': 5,
    'mlPerSec': 0.15,
    'enableNoDrinkAlert': True,
    'enableDailyReport': True,
    'enableManualReview': True,
}

DEFAULT_PUSH = {
    'openclawWebhook': 'http://openclaw.local/api/integrations/openclaw/push',
    'pushMode': '日报 + 异常提醒',
    'summaryTemplate': '今日饮水次数：{{drink_count}}\n预估饮水量：{{estimated_total_ml}} ml\n最近一次饮水：{{last_drink_time}}',
    'feishuTarget': 'chat:cat-health-report',
    'noDrinkHours': 8,
    'lowCountPercent': 30,
    'enableFeishuDaily': True,
    'enableFeishuAlert': True,
    'enableLowIntakeAlert': True,
}

DEFAULT_LABEL_STATS = {
    'confirmed': 0,
    'false_positive': 0,
    'last_labeled_at': '',
    'threshold_confirmed': 50,
    'threshold_false_positive': 20,
}


def read_json(path, default, skip_lock=False):
    def _read():
        if path.exists():
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding='utf-8')
                return json.loads(json.dumps(default))
        return json.loads(json.dumps(default))
    if skip_lock:
        return _read()
    with FILE_LOCK:
        return _read()


def write_json(path, payload, skip_lock=False):
    def _write():
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    if skip_lock:
        _write()
    else:
        with FILE_LOCK:
            _write()


def load_dashboard_data(skip_lock=False):
    return read_json(DATA_FILE, {'overview': {}, 'trends': {}, 'events': []}, skip_lock=skip_lock)


def save_dashboard_data(data, skip_lock=False):
    write_json(DATA_FILE, data, skip_lock=skip_lock)
    _rotate_events(data)

EVENT_ARCHIVE_DIR = DATA_DIR / 'events_archive'
EVENT_RETENTION_DAYS = int(os.environ.get('EVENT_RETENTION_DAYS', '90'))
EVENT_ACTIVE_LIMIT = int(os.environ.get('EVENT_ACTIVE_LIMIT', '500'))

def _rotate_events(data):
    """Archive events older than retention period, keep active file lean."""
    events = data.get('events', [])
    if len(events) <= EVENT_ACTIVE_LIMIT:
        return
    EVENT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    # Group events by date
    from collections import defaultdict
    by_date = defaultdict(list)
    for e in events:
        ts = e.get('start_time', '')
        date_key = ts[:10] if 'T' in ts else 'unknown'
        by_date[date_key].append(e)
    # Find dates older than retention
    cutoff = (datetime.now(CST) - timedelta(days=EVENT_RETENTION_DAYS)).strftime('%Y-%m-%d')
    to_archive = {d: evts for d, evts in by_date.items() if d < cutoff and d != 'unknown'}
    if not to_archive:
        return
    # Write archive files
    for date_key, evts in to_archive.items():
        archive_file = EVENT_ARCHIVE_DIR / f'events-{date_key}.json'
        existing = []
        if archive_file.exists():
            try:
                existing = json.loads(archive_file.read_text(encoding='utf-8'))
            except Exception:
                pass
        existing_ids = {e.get('id') for e in existing}
        for e in evts:
            if e.get('id') not in existing_ids:
                existing.append(e)
        archive_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding='utf-8')
    # Remove archived events from active data
    archived_ids = {e.get('id') for evts in to_archive.values() for e in evts}
    data['events'] = [e for e in events if e.get('id') not in archived_ids]
    print(f"Archived {len(archived_ids)} events to {EVENT_ARCHIVE_DIR}")


def load_settings():
    data = read_json(SETTINGS_FILE, DEFAULT_SETTINGS)
    return {**DEFAULT_SETTINGS, **data}


def save_settings(data):
    write_json(SETTINGS_FILE, data)


def load_push():
    data = read_json(PUSH_FILE, DEFAULT_PUSH)
    return {**DEFAULT_PUSH, **data}


def save_push(data):
    write_json(PUSH_FILE, data)


def load_label_stats():
    return read_json(LABEL_STATS_FILE, DEFAULT_LABEL_STATS)


def save_label_stats(data):
    write_json(LABEL_STATS_FILE, data)


def load_camera_status():
    return read_json(CAMERA_STATUS_FILE, {'ok': False, 'stream_url': '', 'snapshot': '', 'detail': 'camera worker not started', 'updated_at': ''})


def load_detector_status():
    return read_json(DETECTOR_STATUS_FILE, {'ok': False, 'phase': 'detector not started', 'stream_url': '', 'updated_at': '', 'cat_last_seen_at': None})


def load_backfill_status():
    return read_json(BACKFILL_STATUS_FILE, {
        'ok': False,
        'phase': 'backfill worker not started',
        'updated_at': '',
        'last_scan_at': '',
        'last_clip': '',
        'last_event_at': '',
        'processed_count': 0,
        'pending_count': 0,
        'detail': '',
    })


def _read_meminfo():
    values = {}
    try:
        for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
            if ':' not in line:
                continue
            key, raw = line.split(':', 1)
            values[key.strip()] = int(raw.strip().split()[0]) * 1024
    except Exception:
        return {}
    return values


def _read_cgroup_bytes(paths):
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            try:
                raw = path.read_text(encoding='utf-8').strip()
                if raw.isdigit():
                    return int(raw)
            except Exception:
                pass
    return None


def _container_metrics():
    memory_usage = _read_cgroup_bytes([
        '/sys/fs/cgroup/memory.current',
        '/sys/fs/cgroup/memory/memory.usage_in_bytes',
    ])
    memory_limit = _read_cgroup_bytes([
        '/sys/fs/cgroup/memory.max',
        '/sys/fs/cgroup/memory/memory.limit_in_bytes',
    ])
    memory_percent = None
    if memory_usage is not None and memory_limit and memory_limit > 0 and memory_limit < 1 << 60:
        memory_percent = round(memory_usage / memory_limit * 100, 1)
    return {
        'memory_used_mb': round(memory_usage / 1024 / 1024, 1) if memory_usage is not None else None,
        'memory_limit_mb': round(memory_limit / 1024 / 1024, 1) if memory_limit and memory_limit < 1 << 60 else None,
        'memory_percent': memory_percent,
    }


def load_system_metrics():
    load1, load5, load15 = os.getloadavg()
    meminfo = _read_meminfo()
    mem_total = meminfo.get('MemTotal', 0)
    mem_available = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
    mem_used = max(0, mem_total - mem_available)
    mem_percent = round(mem_used / mem_total * 100, 1) if mem_total else None
    root_disk = shutil.disk_usage('/')
    data_disk = shutil.disk_usage(str(DATA_DIR))
    detector = load_detector_status()
    return {
        'ok': True,
        'host': {
            'loadavg': {'1m': round(load1, 2), '5m': round(load5, 2), '15m': round(load15, 2)},
            'memory': {
                'used_mb': round(mem_used / 1024 / 1024, 1) if mem_total else None,
                'total_mb': round(mem_total / 1024 / 1024, 1) if mem_total else None,
                'percent': mem_percent,
            },
            'disk_root': {
                'used_gb': round((root_disk.used) / 1024 / 1024 / 1024, 2),
                'total_gb': round((root_disk.total) / 1024 / 1024 / 1024, 2),
                'percent': round(root_disk.used / root_disk.total * 100, 1) if root_disk.total else None,
            },
            'disk_data': {
                'used_gb': round((data_disk.used) / 1024 / 1024 / 1024, 2),
                'total_gb': round((data_disk.total) / 1024 / 1024 / 1024, 2),
                'percent': round(data_disk.used / data_disk.total * 100, 1) if data_disk.total else None,
            },
        },
        'container': _container_metrics(),
        'detector': {
            'frame_poll_fps': detector.get('frame_poll_fps'),
            'full_frame_fps': detector.get('full_frame_fps'),
            'phase': detector.get('phase'),
        },
    }


def clean_time(ts):
    """Strip timezone from ISO timestamp: '2026-05-07T17:07:36+08:00' -> '2026-05-07T17:07:36'"""
    if not ts or not isinstance(ts, str):
        return ts or ''
    # Remove timezone suffix like +08:00, +09:00, Z, etc.
    return re.sub(r'[+-]\d{2}:\d{2}$', '', ts).rstrip('Z')

def fmt_time(ts):
    """Format timestamp for display: '2026-05-07T17:07:36' -> '17:07:36'"""
    clean = clean_time(ts)
    if 'T' in clean:
        date_part, time_part = clean.split('T', 1)
        return time_part[:8] if len(time_part) >= 8 else time_part
    return clean


def ensure_event_fields(evt):
    """Add formatted display fields to an event dict if missing."""
    if not evt.get('title'):
        start = fmt_time(evt.get('start_time', ''))
        evt['title'] = f"{start} · 猫咪喝水" if start else '饮水事件'
    if not evt.get('desc'):
        evt['desc'] = '猫咪靠近饮水机并停留'
    if not evt.get('metric'):
        dur = float(evt.get('duration_sec', 0) or 0)
        evt['metric'] = f"{round(dur, 1)}s" if dur % 1 else f"{int(dur)}s"
    if not evt.get('tag'):
        evt['tag'] = '喝水事件'
    if not evt.get('img'):
        evt['img'] = evt.get('snapshot_url', '')
    if not evt.get('timeRange'):
        s = fmt_time(evt.get('start_time', ''))
        e = fmt_time(evt.get('end_time', ''))
        evt['timeRange'] = f"{s} ~ {e}" if s and e else s or '--'
    if not evt.get('estimatedMl'):
        evt['estimatedMl'] = evt.get('estimated_ml', 0)
    return evt

CST = timezone(timedelta(hours=8))

def parse_duration(metric_str):
    """Parse duration string like '21.2s' or '54.9s' to float seconds."""
    try:
        return float(str(metric_str).rstrip('s') or 0)
    except (ValueError, TypeError):
        return 0.0


def compute_last_drink_ago(end_time_str):
    """Calculate relative time from now for a cleaned ISO timestamp"""
    if not end_time_str or not isinstance(end_time_str, str):
        return ''
    try:
        clean = clean_time(end_time_str)
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=CST)
        diff = datetime.now(CST) - dt
        secs = diff.total_seconds()
        if secs < 0:
            return '刚刚'
        if secs < 60:
            return '刚刚'
        if secs < 3600:
            return f'{int(secs / 60)} 分钟前'
        if secs < 86400:
            h = int(secs / 3600)
            m = int((secs % 3600) / 60)
            return f'{h} 小时 {m} 分钟前'
        return f'{int(secs / 86400)} 天前'
    except Exception:
        return ''

def find_event(events, event_id):
    return next((event for event in events if event.get('id') == event_id), None)

def update_trends(data):
    """Rebuild daily trends from events."""
    from collections import defaultdict
    events = data.get('events', [])
    daily = defaultdict(lambda: {'count': 0, 'ml': 0.0})
    for evt in events:
        if evt.get("label") == "false_positive":
            continue
        ts = evt.get('start_time', '')
        if ts and 'T' in ts:
            date = ts.split('T')[0]  # '2026-05-07'
            dur_str = str(evt.get('metric', '0')).rstrip('s')
            try:
                dur = float(dur_str)
            except ValueError:
                dur = 0
            daily[date]['count'] += 1
            daily[date]['ml'] += round(float(evt.get('estimatedMl', 0) or 0), 1)
    labels = sorted(daily.keys())
    data['trends'] = {
        'labels': labels,
        'counts': [daily[d]['count'] for d in labels],
        'ml': [daily[d]['ml'] for d in labels],
    }

def recalc_overview(data):
    """Recalculate overview stats from events, excluding false_positives."""
    from datetime import datetime, timedelta
    overview = data.setdefault('overview', {})
    events = data.get('events', [])
    today_str = datetime.now(CST).strftime('%Y-%m-%d')
    week_start = (datetime.now(CST) - timedelta(days=6)).strftime('%Y-%m-%d')
    today_events = [e for e in events if e.get('start_time','').startswith(today_str) and e.get('label') != 'false_positive']
    week_events = [e for e in events if e.get('start_time','')[:10] >= week_start and e.get('label') != 'false_positive']
    today_count = len(today_events)
    today_ml = round(sum(float(e.get('estimatedMl', 0) or 0) for e in today_events), 1)
    today_dur = int(sum(parse_duration(e.get('metric','0')) for e in today_events))
    overview['drinkCount'] = today_count
    overview['waterMl'] = today_ml
    overview['durationSec'] = today_dur
    overview['avgDuration'] = round(today_dur / max(today_count, 1), 1)
    overview['weekCount'] = len(week_events)
    overview['weekMl'] = round(sum(float(e.get('estimatedMl', 0) or 0) for e in week_events), 1)
    overview['weekDuration'] = int(sum(parse_duration(e.get('metric','0')) for e in week_events))
    valid_events = [e for e in events if e.get("label") != "false_positive"]
    if valid_events and valid_events[0].get("end_time"):
        overview['lastDrinkAgo'] = compute_last_drink_ago(valid_events[0]["end_time"])
    elif not overview.get('lastDrinkAgo'):
        overview['lastDrinkAgo'] = '...'
    update_health(overview)
    _compute_peak_hours(overview, today_events)
    update_trends(data)

def update_health(overview):
    """Set health status based on real data."""
    count = int(overview.get('drinkCount', 0) or 0)
    if count == 0:
        overview['healthMain'] = '监控中'
        overview['healthDesc'] = '实时检测已运行，等待猫咪前来饮水。'
        overview['todayAssessment'] = '今日未饮水'
    elif count < 5:
        overview['healthMain'] = '少量活动'
        overview['healthDesc'] = f'今日已记录 {count} 次饮水事件，猫咪饮水正常。'
        overview['todayAssessment'] = '饮水偏少'
    elif count < 15:
        overview['healthMain'] = '饮水正常'
        overview['healthDesc'] = f'今日已记录 {count} 次饮水，水量 {overview.get("waterMl", 0)}ml。'
        overview['todayAssessment'] = '饮水正常'
    else:
        overview['healthMain'] = '频繁饮水'
        overview['healthDesc'] = f'今日已记录 {count} 次饮水，注意观察是否异常。'
        overview['todayAssessment'] = '饮水充足'

def _compute_peak_hours(overview, today_events):
    hour_buckets = {'凌晨 (0-6点)': 0, '上午 (6-12点)': 0, '下午 (12-18点)': 0, '晚上 (18-24点)': 0}
    for e in today_events:
        ts = e.get('start_time', '')
        if 'T' in ts:
            try:
                h = int(ts.split('T')[1].split(':')[0])
            except Exception:
                continue
            if h < 6:
                hour_buckets['凌晨 (0-6点)'] += 1
            elif h < 12:
                hour_buckets['上午 (6-12点)'] += 1
            elif h < 18:
                hour_buckets['下午 (12-18点)'] += 1
            else:
                hour_buckets['晚上 (18-24点)'] += 1
    best = max(hour_buckets, key=hour_buckets.get)
    best_count = hour_buckets[best]
    overview['peakHours'] = f'{best} ({best_count}次)' if best_count > 0 else '暂无数据'
    overview['peakHoursCount'] = best_count

def mjpeg_snapshot_path(mode='smooth'):
    name = 'latest-hd.jpg' if mode == 'hd' else 'latest.jpg'
    return MEDIA_DIR / 'live' / name


def stream_mjpeg(handler, mode='smooth'):
    snapshot = mjpeg_snapshot_path(mode)
    if not snapshot.exists():
        return handler._send_json({'ok': False, 'error': 'snapshot not ready'}, 503)
    boundary = 'frame'
    handler.send_response(200)
    handler.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={boundary}')
    handler.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
    handler.send_header('Pragma', 'no-cache')
    handler.send_header('Connection', 'close')
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.end_headers()
    last_sent = 0.0
    # Serve MJPEG with a heartbeat so the browser doesn't think it's dead
    last_heartbeat = time()
    try:
        while True:
            if not snapshot.exists():
                sleep(0.3)
                continue
            stat = snapshot.stat()
            if stat.st_mtime <= last_sent:
                # Send a keepalive comment every 5s to prevent connection timeout
                now = time()
                if now - last_heartbeat > 5:
                    handler.wfile.write(f'--{boundary}\r\n'.encode('utf-8'))
                    handler.wfile.write(b'Content-Type: text/plain\r\n\r\n')
                    handler.wfile.write(b'keepalive\r\n')
                    handler.wfile.flush()
                    last_heartbeat = now
                sleep(0.2)
                continue
            frame = snapshot.read_bytes()
            last_sent = stat.st_mtime
            handler.wfile.write(f'--{boundary}\r\n'.encode('utf-8'))
            handler.wfile.write(b'Content-Type: image/jpeg\r\n')
            handler.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode('utf-8'))
            handler.wfile.write(frame)
            handler.wfile.write(b'\r\n')
            handler.wfile.flush()
            last_heartbeat = time()
            sleep(0.05)
    except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
        return



def easynvr_login():
    with STREAM_LOCK:
        if EASYNVR_SESSION['token'] and EASYNVR_SESSION['expired_at'] > time() + 60:
            return EASYNVR_SESSION['token']
        payload = json.dumps({'username': EASYNVR_USERNAME, 'password': EASYNVR_PASSWORD}).encode('utf-8')
        request = Request(f'{EASYNVR_BASE_URL}/login', data=payload, headers={'Content-Type': 'application/json'})
        response = urlopen(request, timeout=10)
        data = json.loads(response.read().decode('utf-8'))
        token = data.get('token')
        if not token:
            EASYNVR_SESSION['token'] = None
            return None
        EASYNVR_SESSION['token'] = token
        EASYNVR_SESSION['expired_at'] = int(data.get('expired_at', int(time()) + 3600))
        return token


def easynvr_play_address():
    token = easynvr_login()
    request = Request(
        f'{EASYNVR_BASE_URL}/channels/{EASYNVR_CHANNEL_ID_HD}/play',
        data=b'',
        headers={'Authorization': f'Bearer {token}'},
        method='POST',
    )
    response = urlopen(request, timeout=10)
    data = json.loads(response.read().decode('utf-8'))
    return data.get('address', {}).get('hls', '')


def fetch_hls_playlist():
    playlist_path = easynvr_play_address()
    playlist_url = urljoin(EASYNVR_BASE_URL, playlist_path)
    response = urlopen(Request(playlist_url, headers={'User-Agent': 'cat-water-dashboard/1.0'}), timeout=15)
    final_url = response.geturl()
    body = response.read().decode('utf-8', 'replace')
    base_url = final_url.rsplit('/', 1)[0] + '/'
    rewritten_lines = []
    for line in body.splitlines():
        if not line or line.startswith('#'):
            rewritten_lines.append(line)
            continue
        absolute = urljoin(base_url, line)
        rewritten_lines.append(f'/api/live-hls-segment?url={quote(absolute, safe="")}')
    return '\n'.join(rewritten_lines) + '\n'


def proxy_binary(url, handler):
    upstream = urlopen(Request(url, headers={'User-Agent': 'cat-water-dashboard/1.0'}), timeout=15)
    try:
        upstream.fp._sock.settimeout(30)
    except Exception:
        pass
    handler.send_response(200)
    handler.send_header('Content-Type', upstream.headers.get('Content-Type', 'application/octet-stream'))
    handler.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
    handler.send_header('Pragma', 'no-cache')
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.end_headers()
    try:
        while True:
            chunk = upstream.read(128 * 1024)
            if not chunk:
                break
            handler.wfile.write(chunk)
    except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
        pass
    finally:
        try:
            upstream.close()
        except Exception:
            pass


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        super().end_headers()

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get('Content-Length', '0') or '0')
        raw = self.rfile.read(length) if length else b'{}'
        try:
            return json.loads(raw.decode('utf-8') or '{}')
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError('请求体不是合法 JSON')

    def _read_request_json(self):
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({'ok': False, 'error': str(exc)}, 400)
            return None
        if not isinstance(payload, dict):
            self._send_json({'ok': False, 'error': '请求体必须是 JSON 对象'}, 400)
            return None
        return payload

    def _proxy_live_stream(self, mode='smooth'):
        url = live_flv_url(mode)
        request = Request(url, headers={'User-Agent': 'cat-water-dashboard/1.0'})
        try:
            upstream = urlopen(request, timeout=15)
        except (URLError, OSError) as exc:
            return self._send_json({'ok': False, 'error': f'live stream unavailable: {exc}'}, 502)

        # Set socket timeout so read() doesn't block forever if upstream stalls
        try:
            upstream.fp._sock.settimeout(30)
        except Exception:
            pass

        self.send_response(200)
        content_type = upstream.headers.get('Content-Type', 'video/x-flv')
        if content_type == 'application/octet-stream':
            content_type = 'video/x-flv'
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Connection', 'close')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Range')
        self.end_headers()

        try:
            while True:
                chunk = upstream.read(128 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            pass
        finally:
            try:
                upstream.close()
            except Exception:
                pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == '/api/dashboard/overview':
            data = load_dashboard_data()
            events = [ensure_event_fields(e) for e in data.get('events', [])]
            data["events"] = events
            # Recalc dynamic fields (lastDrinkAgo changes every minute)
            overview = data.setdefault('overview', {})
            valid_events = [e for e in events if e.get("label") != "false_positive"]
            if valid_events and valid_events[0].get("end_time"):
                overview['lastDrinkAgo'] = compute_last_drink_ago(valid_events[0]["end_time"])
            elif not overview.get('lastDrinkAgo'):
                overview['lastDrinkAgo'] = '等待猫咪饮水...'
            return self._send_json(data)
        if path == '/api/events':
            data = load_dashboard_data()
            event_id = parse_qs(parsed.query).get('id', [None])[0]
            if event_id:
                event = find_event(data.get('events', []), event_id)
                if event is None:
                    return self._send_json({'ok': False, 'error': 'Event Not Found'}, 404)
                return self._send_json({'event': ensure_event_fields(event)})
            events = [ensure_event_fields(e) for e in data.get('events', [])]
            return self._send_json({'events': events})
        if path.startswith('/api/events/'):
            data = load_dashboard_data()
            event_id = path.rsplit('/', 1)[-1]
            event = find_event(data.get('events', []), event_id)
            if event is None:
                return self._send_json({'ok': False, 'error': 'Event Not Found'}, 404)
            return self._send_json({'event': ensure_event_fields(event)})
        if path == '/api/settings':
            return self._send_json(load_settings())
        if path == '/api/push-settings':
            return self._send_json(load_push())
        if path == '/api/push-log':
            return self._send_json({'items': read_json(PUSH_LOG_FILE, [])})
        if path == '/api/camera-status':
            return self._send_json(load_camera_status())
        if path == '/api/detector-status':
            return self._send_json(load_detector_status())
        if path == '/api/backfill-status':
            return self._send_json(load_backfill_status())
        if path == '/api/system-metrics':
            return self._send_json(load_system_metrics())
        if path == '/api/live-hls.m3u8':
            try:
                playlist = fetch_hls_playlist()
            except Exception as exc:
                return self._send_json({'ok': False, 'error': f'hls unavailable: {exc}'}, 502)
            body = playlist.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
            return
        if path == '/api/live-hls-segment':
            segment_url = parse_qs(parsed.query).get('url', [''])[0]
            if not segment_url:
                return self._send_json({'ok': False, 'error': 'missing url'}, 400)
            # SSRF protection: only allow EasyNVR URLs
            from urllib.parse import urlparse as _urlparse
            parsed_url = _urlparse(unquote(segment_url))
            allowed_hosts = (EASYNVR_BASE_URL.replace('http://','').replace('https://','').split(':')[0],)
            if parsed_url.hostname not in allowed_hosts and not parsed_url.hostname.endswith('.' + allowed_hosts[0]):
                return self._send_json({'ok': False, 'error': 'url not allowed'}, 403)
            try:
                return proxy_binary(unquote(segment_url), self)
            except Exception as exc:
                return self._send_json({'ok': False, 'error': f'hls segment unavailable: {exc}'}, 502)
        if path == '/api/live-preview.flv':
            mode = parse_qs(parsed.query).get('mode', ['smooth'])[0]
            return self._proxy_live_stream('hd' if mode == 'hd' else 'smooth')
        if path == '/api/live.mjpeg':
            mode = parse_qs(parsed.query).get('mode', ['smooth'])[0]
            return stream_mjpeg(self, 'hd' if mode == 'hd' else 'smooth')
        
        if path == '/api/event-video':
            event_id = parse_qs(parsed.query).get('id', [None])[0]
            if not event_id:
                return self._send_json({'ok': False, 'error': 'missing event id'}, 400)
            data = load_dashboard_data()
            event = find_event(data.get('events', []), event_id)
            if not event:
                return self._send_json({'ok': False, 'error': 'event not found'}, 404)
            start_str = event.get('start_time', '')
            end_str = event.get('end_time', '')
            if not start_str:
                return self._send_json({'ok': False, 'error': 'event has no start_time'}, 400)
            try:
                start_dt = datetime.fromisoformat(start_str)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=CST)
            except Exception:
                return self._send_json({'ok': False, 'error': 'invalid start_time format'}, 400)
            if end_str:
                try:
                    end_dt = datetime.fromisoformat(end_str)
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=CST)
                except Exception:
                    end_dt = start_dt + timedelta(seconds=10)
            else:
                end_dt = start_dt + timedelta(seconds=10)
            PRE_BUFFER = 15
            POST_BUFFER = 10
            clip_start = start_dt - timedelta(seconds=PRE_BUFFER)
            clip_end = end_dt + timedelta(seconds=POST_BUFFER)
            clip_duration = max(4, (clip_end - clip_start).total_seconds())
            # Collect recording files across date boundaries
            dates = set()
            d = clip_start
            while d <= clip_end:
                dates.add(d.strftime('%Y%m%d'))
                d += timedelta(days=1)
            all_files = []
            for ds in sorted(dates):
                rec_dir = Path(f'/easynvr_rec/easynvr_rec/OpD6HNpPU8j0X/profile_1/{ds}')
                if rec_dir.exists():
                    for f in sorted(rec_dir.glob('*.mp4')):
                        if len(f.name) >= 14 and f.name[:14].isdigit():
                            all_files.append(f)
            if not all_files:
                return self._send_json({'ok': False, 'error': 'no recording files found'}, 404)
            clip_start_str = clip_start.strftime('%Y%m%d%H%M%S')
            start_file = None
            start_file_idx = 0
            for i, f in enumerate(all_files):
                if f.name[:14] <= clip_start_str:
                    start_file = f
                    start_file_idx = i
                elif start_file:
                    break
            if start_file is None:
                start_file = all_files[0]
                start_file_idx = 0
            file_start_dt = datetime.strptime(start_file.name[:14], '%Y%m%d%H%M%S').replace(tzinfo=CST)
            offset_sec = max(0, (clip_start - file_start_dt).total_seconds())
            # Gather all files needed from start_file onwards
            needed_files = [start_file]
            remaining = clip_duration + offset_sec
            idx = start_file_idx + 1
            while remaining > 62 and idx < len(all_files):
                needed_files.append(all_files[idx])
                remaining -= 62
                idx += 1
            tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                if len(needed_files) == 1:
                    cmd = ['ffmpeg', '-y', '-ss', f'{offset_sec:.3f}', '-i', str(needed_files[0]),
                           '-t', f'{clip_duration:.3f}',
                           '-vf', 'scale=960:-2', '-c:v', 'libx264',
                           '-preset', 'ultrafast', '-crf', '23', '-an',
                           '-movflags', '+faststart', tmp_path]
                    proc = subprocess.run(cmd, capture_output=True, timeout=30)
                else:
                    list_path = tmp_path + '.list.txt'
                    with open(list_path, 'w') as f:
                        for nf in needed_files:
                            f.write(f"file '{nf}'\n")
                    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path,
                           '-ss', f'{offset_sec:.3f}', '-t', f'{clip_duration:.3f}',
                           '-vf', 'scale=960:-2', '-c:v', 'libx264',
                           '-preset', 'ultrafast', '-crf', '23', '-an',
                           '-movflags', '+faststart', tmp_path]
                    proc = subprocess.run(cmd, capture_output=True, timeout=120)
                    try: os.unlink(list_path)
                    except: pass
                if proc.returncode != 0:
                    err_msg = proc.stderr.decode('utf-8', errors='replace')[-200:]
                    Path(tmp_path).unlink(missing_ok=True)
                    return self._send_json({'ok': False, 'error': f'ffmpeg failed: {err_msg}'}, 500)
                file_size = Path(tmp_path).stat().st_size
                if file_size == 0:
                    Path(tmp_path).unlink(missing_ok=True)
                    return self._send_json({'ok': False, 'error': 'extracted clip is empty'}, 500)
                self.send_response(200)
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Content-Length', str(file_size))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                with open(tmp_path, 'rb') as vf:
                    while True:
                        chunk = vf.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except subprocess.TimeoutExpired:
                Path(tmp_path).unlink(missing_ok=True)
                return self._send_json({'ok': False, 'error': 'ffmpeg timed out'}, 500)
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
            return

        if path == '/api/labels/stats':
            stats = load_label_stats()
            stats['total_labeled'] = stats['confirmed'] + stats['false_positive']
            stats['ready_to_train'] = (
                stats['confirmed'] >= stats['threshold_confirmed'] and
                stats['false_positive'] >= stats['threshold_false_positive']
            )
            return self._send_json(stats)
        if path == '/api/health':
            return self._send_json({'ok': True, 'service': 'cat-water-dashboard-api'})
        if path == "/developer" or path == "/developer/" or path == "/dev.html":
            dev_path = Path("/app/data/dev.html")
            if not dev_path.exists():
                dev_path = Path("/app/data/developer.html")
            if dev_path.exists():
                content = dev_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            return self._send_json({"error": "Developer page not found"}, 404)
        if path == '/api/models':
            models_dir = BASE_DIR / 'models'
            models = []
            if models_dir.exists():
                for f in sorted(models_dir.iterdir()):
                    if f.suffix == '.onnx':
                        models.append({
                            'name': f.name,
                            'size_mb': round(f.stat().st_size / 1024 / 1024, 1),
                        })
            settings = load_settings()
            active = settings.get('activeModel', 'yolov8n_640_opset21.onnx')
            # Mark active model
            for m in models:
                m['active'] = m['name'] == active
            return self._send_json({'models': models, 'active': active})
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/api/events':
            with FILE_LOCK:
                data = load_dashboard_data(skip_lock=True)
            payload = self._read_request_json()
            if payload is None:
                return
            event_id = str(payload.get('event_id') or payload.get('id') or f"event-{len(data.get('events', [])) + 1}")
            duration_sec = float(payload.get('duration_sec', 0) or 0)
            estimated_ml = float(payload.get('estimated_ml', 0) or 0)
            events = data.setdefault('events', [])
            existing = find_event(events, event_id)
            # Merge with existing event data (drink_end updates should preserve drink_start fields)
            prev = existing if existing else {}
            snapshot = (payload.get('snapshot_url') or payload.get('snapshot')
                        or prev.get('img')
                        or 'https://images.unsplash.com/photo-1519052537078-e6302a4968d4?auto=format&fit=crop&w=640&q=80')
            # Never lower confidence on event update (drink_end may carry stale zero score)
            existing_conf = float(prev.get('confidence', 0) or 0)
            if 'confidence' in payload:
                new_conf = float(payload.get('confidence', 0) or 0)
                confidence = new_conf if new_conf > existing_conf else existing_conf
            else:
                confidence = existing_conf
            raw_start = payload.get('start_time') or prev.get('start_time') or ''
            raw_end = payload.get('end_time') or prev.get('end_time') or ''
            start_time = clean_time(raw_start)
            end_time = clean_time(raw_end)
            dur = duration_sec if duration_sec else float(str(prev.get('metric', '0')).rstrip('s') or 0)
            ml = estimated_ml if estimated_ml else float(prev.get('estimatedMl', 0) or 0)
            display_start = fmt_time(start_time)
            event = {
                'id': event_id,
                'start_time': start_time,
                'end_time': end_time,
                'title': payload.get('title') or f"{display_start} · 猫咪喝水",
                'desc': payload.get('summary') or payload.get('desc') or '猫咪靠近饮水机并停留超过3秒',
                'metric': f"{round(dur, 1)}s" if dur % 1 else f"{int(dur)}s",
                'tag': payload.get('tag', '喝水事件'),
                'img': snapshot,
                'timeRange': f"{display_start} ~ {fmt_time(end_time)}",
                'confidence': confidence,
                'estimatedMl': round(ml, 1),
                'summary': payload.get('summary', '由本地分析程序自动生成'),
                'clip': payload.get('clip_url', ''),
                'debugCurveUrl': payload.get('debug_curve_url', ''),
            }
            if existing is not None:
                events.remove(existing)
            events.insert(0, event)
            overview = data.setdefault('overview', {})
            if existing is None:
                overview['drinkCount'] = int(overview.get('drinkCount', 0)) + 1
                overview['waterMl'] = round(float(overview.get('waterMl', 0) or 0) + estimated_ml, 1)
                overview['durationSec'] = int(float(overview.get('durationSec', 0) or 0) + duration_sec)
            else:
                previous_estimated_ml = float(existing.get('estimatedMl', 0) or 0)
                previous_duration_sec = float(str(existing.get('metric', '0')).rstrip('s') or 0)
                overview['waterMl'] = round(float(overview.get('waterMl', 0) or 0) - previous_estimated_ml + estimated_ml, 1)
                overview['durationSec'] = int(float(overview.get('durationSec', 0) or 0) - previous_duration_sec + duration_sec)
            count = max(overview['drinkCount'], 1)
            overview['avgDuration'] = round(overview['durationSec'] / count, 1)
            overview['lastDrink'] = fmt_time(payload.get('end_time', payload.get('start_time', overview.get('lastDrink', '--:--'))))
            overview['lastDrinkAgo'] = '刚刚'
            update_health(overview)
            update_trends(data)
            save_dashboard_data(data, skip_lock=True)
            return self._send_json({'ok': True, 'event': event, 'overview': overview}, 201)

        if path == '/api/fountain-zone':
            payload = self._read_request_json()
            if payload is None:
                return
            settings = load_settings()
            zone = payload.get('fountainZone')
            if zone and isinstance(zone, dict):
                settings['fountainZone'] = {
                    'x1': int(zone.get('x1', 0)),
                    'y1': int(zone.get('y1', 0)),
                    'x2': int(zone.get('x2', 1280)),
                    'y2': int(zone.get('y2', 720)),
                }
            if 'frameWidth' in payload:
                settings['frameWidth'] = int(payload['frameWidth'])
            if 'frameHeight' in payload:
                settings['frameHeight'] = int(payload['frameHeight'])
            save_settings(settings)
            return self._send_json({'ok': True, 'fountainZone': settings.get('fountainZone')})

        if path == '/api/daily-summary':
            payload = self._read_request_json()
            if payload is None:
                return
            with FILE_LOCK:
                data = load_dashboard_data(skip_lock=True)
            overview = data.setdefault('overview', {})
            overview['drinkCount'] = payload.get('drink_count', overview.get('drinkCount', 0))
            overview['waterMl'] = payload.get('estimated_total_ml', overview.get('waterMl', 0))
            overview['durationSec'] = payload.get('total_duration_sec', overview.get('durationSec', 0))
            overview['avgDuration'] = payload.get('avg_duration_sec', overview.get('avgDuration', 0))
            overview['lastDrink'] = payload.get('last_drink_time', overview.get('lastDrink', '--:--'))
            overview['lastDrinkAgo'] = '刚刚'
            save_dashboard_data(data, skip_lock=True)
            return self._send_json({'ok': True, 'overview': overview})

        if path == '/api/settings':
            payload = self._read_request_json()
            if payload is None:
                return
            current = load_settings()
            current.update(payload)
            save_settings(current)
            return self._send_json({'ok': True, 'settings': current})

        if path == '/api/push-settings':
            payload = self._read_request_json()
            if payload is None:
                return
            current = load_push()
            current.update(payload)
            save_push(current)
            return self._send_json({'ok': True, 'push': current})

        if path == '/api/integrations/openclaw/push':
            payload = self._read_request_json()
            if payload is None:
                return
            history = read_json(PUSH_LOG_FILE, [])
            history.insert(0, payload)
            write_json(PUSH_LOG_FILE, history[:50])
            return self._send_json({'ok': True, 'queued': True, 'message': '已记录 OpenClaw 推送请求（当前为占位实现）'})

        if path == '/api/events/label':
            payload = self._read_request_json()
            if payload is None:
                return
            event_id = str(payload.get('event_id', ''))
            label = str(payload.get('label', ''))
            if not event_id or label not in ('confirmed', 'false_positive'):
                return self._send_json({'ok': False, 'error': '需要 event_id 和 label (confirmed/false_positive)'}, 400)
            with FILE_LOCK:
                data = load_dashboard_data(skip_lock=True)
            events = data.get('events', [])
            event = find_event(events, event_id)
            if event is None:
                return self._send_json({'ok': False, 'error': 'Event Not Found'}, 404)
            # Update event - handle re-labeling
            old_label = event.get('label', '')
            if old_label and old_label != label:
                # Remove old snapshot from previous directory
                old_dir = POSITIVE_DIR if old_label == 'confirmed' else NEGATIVE_DIR
                old_target = old_dir / f"{event_id}.jpg"
                if old_target.exists():
                    old_target.unlink()
            event['label'] = label
            event['labeled_at'] = datetime.now(CST).isoformat()
            # Copy snapshot to positive/ or negative/ directory
            img_path = event.get('img', '')
            snapshot_copied = False
            if img_path and img_path.startswith('/media/'):
                # /media/events/xxx.jpg -> BASE_DIR/media/events/xxx.jpg
                src_rel = img_path.lstrip('/')
                src_path = BASE_DIR / src_rel
                target_dir = POSITIVE_DIR if label == 'confirmed' else NEGATIVE_DIR
                target_dir.mkdir(parents=True, exist_ok=True)
                target_name = f"{event_id}.jpg"
                target_path = target_dir / target_name
                if src_path.exists():
                    shutil.copy2(str(src_path), str(target_path))
                    snapshot_copied = True
            # Update label stats
            stats = load_label_stats()
            if old_label and old_label in stats:
                stats[old_label] = max(0, stats[old_label] - 1)
            stats[label] = stats.get(label, 0) + 1
            stats['last_labeled_at'] = datetime.now(CST).isoformat()
            save_label_stats(stats)
            recalc_overview(data)
            save_dashboard_data(data, skip_lock=True)
            stats['total_labeled'] = stats['confirmed'] + stats['false_positive']
            stats['ready_to_train'] = (
                stats['confirmed'] >= stats['threshold_confirmed'] and
                stats['false_positive'] >= stats['threshold_false_positive']
            )
            return self._send_json({'ok': True, 'event': event, 'stats': stats, 'snapshot_copied': snapshot_copied})

        if path == '/data/label_results.json':
            payload = self._read_request_json()
            if payload is None:
                return
            write_json(BASE_DIR / 'data' / 'label_results.json', payload)
            return self._send_json({'ok': True, 'saved': len(payload)})

        if path == '/api/models/activate':
            payload = self._read_request_json()
            if payload is None:
                return
            model_name = str(payload.get('model', ''))
            if not model_name:
                return self._send_json({'ok': False, 'error': '缺少 model 参数'}, 400)
            model_path = BASE_DIR / 'models' / model_name
            if not model_path.exists():
                return self._send_json({'ok': False, 'error': f'模型文件不存在: {model_name}'}, 404)
            settings = load_settings()
            old_model = settings.get('activeModel', '')
            settings['activeModel'] = model_name
            save_settings(settings)
            return self._send_json({'ok': True, 'active': model_name, 'previous': old_model,
                                    'message': f'模型已切换为 {model_name}，检测器将在下一个周期自动热加载'})

        return self._send_json({'ok': False, 'error': 'Not Found'}, 404)


if __name__ == '__main__':
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    server = ThreadingHTTPServer(('0.0.0.0', 39095), Handler)
    print('cat-water-dashboard api listening on 0.0.0.0:39095')
    print('Local access: http://127.0.0.1:39095/')
    print(f'Network access: http://{local_ip}:39095/')
    server.serve_forever()