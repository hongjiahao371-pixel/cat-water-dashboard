#!/usr/bin/env python3
import json
import hmac
import os
import shutil
import socket
import subprocess
import tempfile
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Lock, RLock
import re
from datetime import datetime, timezone, timedelta
from time import time, sleep
from urllib.error import URLError
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from notification_delivery import deliver as deliver_notification
from notification_delivery import public_webhook_url, validate_webhook_url, webhook_configured

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DATA_FILE = DATA_DIR / 'mock.json'
SETTINGS_FILE = DATA_DIR / 'settings.json'
PUSH_FILE = DATA_DIR / 'push.json'
PUSH_LOG_FILE = DATA_DIR / 'openclaw-push-log.json'
CAMERA_STATUS_FILE = DATA_DIR / 'camera_status.json'
DETECTOR_STATUS_FILE = DATA_DIR / 'detector_status.json'
TRANSCODE_STATUS_FILE = DATA_DIR / 'transcode_status.json'
BACKFILL_STATUS_FILE = DATA_DIR / 'backfill_status.json'
ARCHIVE_VERIFIER_STATUS_FILE = DATA_DIR / 'archive_verifier_status.json'
NOTIFICATION_STATUS_FILE = DATA_DIR / 'notification_status.json'
LABEL_STATS_FILE = DATA_DIR / 'label_stats.json'
TRAINING_MANIFEST_FILE = DATA_DIR / 'training_manifest.json'
REVIEW_CANDIDATES_FILE = DATA_DIR / 'review_candidates.json'
REVIEW_LABELS_FILE = DATA_DIR / 'review_candidate_labels.json'
DAILY_AUDIT_CANDIDATES_FILE = DATA_DIR / 'daily_audit_candidates.json'
DAILY_AUDIT_STATUS_FILE = DATA_DIR / 'daily_audit_status.json'
VOLUME_CALIBRATIONS_FILE = DATA_DIR / 'volume_calibrations.json'
MODEL_TRAINING_STATUS_FILE = DATA_DIR / 'model_training_status.json'
POSITIVE_DIR = DATA_DIR / 'positive'
NEGATIVE_DIR = DATA_DIR / 'negative'
LABEL_ORPHAN_DIR = DATA_DIR / 'label_orphans'
LEGACY_CAMERA_LIVE_STREAM_URL = os.environ.get('CAMERA_LIVE_STREAM_URL', '')
CAMERA_LIVE_STREAM_URL_HD = os.environ.get('CAMERA_LIVE_STREAM_URL_HD', 'http://192.168.5.17:39100/live.flv')
CAMERA_LIVE_STREAM_URL_SMOOTH = os.environ.get('CAMERA_LIVE_STREAM_URL_SMOOTH', '')
EASYNVR_BASE_URL = os.environ.get('EASYNVR_BASE_URL', 'http://192.168.5.17:10000')
EASYNVR_USERNAME = os.environ.get('EASYNVR_USERNAME', 'admin')
EASYNVR_PASSWORD = os.environ.get('EASYNVR_PASSWORD', 'admin')
API_TOKEN = os.environ.get('CAT_WATER_API_TOKEN', '').strip()
EASYNVR_CHANNEL_ID = os.environ.get('EASYNVR_CHANNEL_ID', 'OpD6HNpPU8j0X_profile_1')
EASYNVR_CHANNEL_ID_HD = os.environ.get('EASYNVR_CHANNEL_ID_HD', EASYNVR_CHANNEL_ID)
EASYNVR_CHANNEL_ID_SMOOTH = os.environ.get('EASYNVR_CHANNEL_ID_SMOOTH', 'OpD6HNpPU8j0X_profile_2')
MEDIA_DIR = BASE_DIR / 'media'
MEDIA_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
FILE_LOCK = RLock()
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
    'mergeGapSec': 45,
    'mlPerSec': 0.15,
    'mlPerActiveSec': 0.45,
    'mlPerActiveSecLow': 0.30,
    'mlPerActiveSecHigh': 0.70,
    'volumeCalibration': {},
    'enableNoDrinkAlert': True,
    'enableDailyReport': True,
    'enableManualReview': True,
}

DEFAULT_PUSH = {
    'openclawWebhook': '',
    'pushMode': '日报 + 异常提醒',
    'summaryTemplate': '今日饮水次数：{{drink_count}}\n预估饮水量：{{estimated_total_ml}} ml\n最近一次饮水：{{last_drink_time}}',
    'feishuTarget': '',
    'noDrinkHours': 8,
    'lowCountPercent': 30,
    'dailyReportTime': '21:30',
    'alertCooldownHours': 6,
    'enableFeishuDaily': True,
    'enableFeishuAlert': True,
    'enableLowIntakeAlert': True,
    'enableReviewCandidateAlert': True,
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
            except (json.JSONDecodeError, OSError) as exc:
                # Never overwrite an unreadable data file with defaults: doing so
                # silently deletes historical events after an interrupted write.
                backup = path.with_name(f'{path.name}.corrupt-{datetime.now(CST).strftime("%Y%m%d%H%M%S")}')
                try:
                    shutil.copy2(path, backup)
                except OSError:
                    pass
                print(f'Warning: failed to read {path}: {exc}; preserved original at {backup}')
                return json.loads(json.dumps(default))
        return json.loads(json.dumps(default))
    if skip_lock:
        return _read()
    with FILE_LOCK:
        return _read()


def write_json(path, payload, skip_lock=False):
    def _write():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = Path(handle.name)
        os.replace(tmp_path, path)
    if skip_lock:
        _write()
    else:
        with FILE_LOCK:
            _write()


def review_candidate_payloads(skip_lock=False):
    return [
        read_json(REVIEW_CANDIDATES_FILE, {'version': 1, 'items': []}, skip_lock=skip_lock),
        read_json(DAILY_AUDIT_CANDIDATES_FILE, {'version': 1, 'items': []}, skip_lock=skip_lock),
    ]


def find_review_candidate(sample_id, skip_lock=False):
    for payload in review_candidate_payloads(skip_lock=skip_lock):
        candidate = next((item for item in payload.get('items', [])
                          if isinstance(item, dict) and item.get('id') == sample_id), None)
        if candidate is not None:
            return candidate
    return None


def load_review_candidates():
    payloads = review_candidate_payloads()
    labels = read_json(REVIEW_LABELS_FILE, {})
    event_by_id = {}
    for event in load_dashboard_data().get('events', []):
        for event_id in _unique_event_values(event.get('id', ''), event.get('sourceEventIds', [])):
            event_by_id[event_id] = event
    items = []
    counts = {'unlabeled': 0, 'drinking': 0, 'not_drinking': 0, 'no_cat': 0}
    for payload in payloads:
        for source in payload.get('items', []):
            if not isinstance(source, dict) or not source.get('id'):
                continue
            item = dict(source)
            annotation = labels.get(item['id'], {})
            item['label'] = annotation.get('label', '')
            item['labeled_at'] = annotation.get('labeled_at', '')
            item['recovery_event_id'] = annotation.get('recovery_event_id', '')
            item['recovery_status'] = annotation.get('recovery_status', '')
            item['recovery_trigger'] = annotation.get('recovery_trigger', '')
            linked_event = event_by_id.get(item['recovery_event_id'])
            if linked_event is not None:
                verification = linked_event.get('verificationStatus', '') or 'unverified'
                if item['recovery_status'].startswith('covered_'):
                    item['recovery_status'] = f'covered_{verification}'
                elif verification in ('confirmed', 'rejected', 'uncertain', 'error'):
                    item['recovery_status'] = verification
            counts[item['label'] or 'unlabeled'] += 1
            items.append(item)
    return {
        'version': 2,
        'updated_at': max((payload.get('updated_at', '') for payload in payloads), default=''),
        'items': items,
        'counts': counts,
    }


def label_review_candidate(sample_id, label):
    if label not in ('drinking', 'not_drinking', 'no_cat'):
        raise ValueError('label must be drinking/not_drinking/no_cat')
    if find_review_candidate(sample_id) is None:
        raise KeyError(sample_id)
    with FILE_LOCK:
        labels = read_json(REVIEW_LABELS_FILE, {}, skip_lock=True)
        annotation = dict(labels.get(sample_id, {}))
        annotation.update({
            'label': label,
            'labeled_at': datetime.now(CST).isoformat(),
        })
        labels[sample_id] = annotation
        write_json(REVIEW_LABELS_FILE, labels, skip_lock=True)
    return labels[sample_id]


def load_dashboard_data(skip_lock=False):
    return read_json(DATA_FILE, {'overview': {}, 'trends': {}, 'events': []}, skip_lock=skip_lock)


def save_dashboard_data(data, skip_lock=False):
    _rotate_events(data)
    write_json(DATA_FILE, data, skip_lock=skip_lock)

EVENT_ARCHIVE_DIR = DATA_DIR / 'events_archive'
EVENT_RETENTION_DAYS = int(os.environ.get('EVENT_RETENTION_DAYS', '90'))
EVENT_ACTIVE_LIMIT = int(os.environ.get('EVENT_ACTIVE_LIMIT', '500'))
MERGE_ADJACENT_EVENT_GAP_SEC = float(os.environ.get('MERGE_ADJACENT_EVENT_GAP_SEC', '45'))
CAMERA_STATUS_STALE_SEC = int(os.environ.get('CAMERA_STATUS_STALE_SEC', '90'))
DETECTOR_STATUS_STALE_SEC = int(os.environ.get('DETECTOR_STATUS_STALE_SEC', '30'))
TRANSCODE_STATUS_STALE_SEC = int(os.environ.get('TRANSCODE_STATUS_STALE_SEC', '30'))
BACKFILL_STATUS_STALE_SEC = int(os.environ.get('BACKFILL_STATUS_STALE_SEC', '900'))
ARCHIVE_VERIFIER_STATUS_STALE_SEC = int(os.environ.get('ARCHIVE_VERIFIER_STATUS_STALE_SEC', '600'))
NOTIFICATION_STATUS_STALE_SEC = int(os.environ.get('NOTIFICATION_STATUS_STALE_SEC', '180'))
DAILY_AUDIT_STATUS_STALE_SEC = int(os.environ.get('DAILY_AUDIT_STATUS_STALE_SEC', '900'))
MODEL_TRAINING_STATUS_STALE_SEC = int(os.environ.get('MODEL_TRAINING_STATUS_STALE_SEC', '28800'))
MAX_REQUEST_BYTES = int(os.environ.get('MAX_REQUEST_BYTES', str(256 * 1024)))

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
        write_json(archive_file, existing, skip_lock=True)
    # Remove archived events from active data
    archived_ids = {e.get('id') for evts in to_archive.values() for e in evts}
    data['events'] = [e for e in events if e.get('id') not in archived_ids]
    print(f"Archived {len(archived_ids)} events to {EVENT_ARCHIVE_DIR}")


def load_archived_events(start_date='', end_date=''):
    events = []
    if not EVENT_ARCHIVE_DIR.is_dir():
        return events
    for path in sorted(EVENT_ARCHIVE_DIR.glob('events-*.json')):
        match = re.fullmatch(r'events-(\d{4}-\d{2}-\d{2})\.json', path.name)
        if not match:
            continue
        day = match.group(1)
        if start_date and day < start_date:
            continue
        if end_date and day > end_date:
            continue
        payload = read_json(path, [])
        if isinstance(payload, list):
            for event in payload:
                if isinstance(event, dict):
                    item = dict(event)
                    item['archived'] = True
                    events.append(item)
    return events


def find_event_with_archive(event_id, active_data=None):
    data = active_data if isinstance(active_data, dict) else load_dashboard_data()
    event = find_event(data.get('events', []), event_id)
    if event is not None:
        return event
    return find_event(load_archived_events(), event_id)


def archive_status():
    files = []
    total = 0
    earliest = ''
    latest = ''
    if EVENT_ARCHIVE_DIR.is_dir():
        for path in sorted(EVENT_ARCHIVE_DIR.glob('events-*.json')):
            match = re.fullmatch(r'events-(\d{4}-\d{2}-\d{2})\.json', path.name)
            if not match:
                continue
            payload = read_json(path, [])
            count = len(payload) if isinstance(payload, list) else 0
            day = match.group(1)
            files.append({'date': day, 'events': count, 'file': path.name})
            total += count
            earliest = earliest or day
            latest = day
    return {
        'retention_days': EVENT_RETENTION_DAYS,
        'active_limit': EVENT_ACTIVE_LIMIT,
        'archive_files': len(files),
        'archived_events': total,
        'earliest_date': earliest,
        'latest_date': latest,
        'files': files,
    }


def load_settings():
    data = read_json(SETTINGS_FILE, DEFAULT_SETTINGS)
    return {**DEFAULT_SETTINGS, **data}


def save_settings(data):
    write_json(SETTINGS_FILE, data)


def public_settings(data):
    result = dict(data)
    rtsp_url = result.get('rtspUrl')
    if isinstance(rtsp_url, str):
        result['rtspUrl'] = re.sub(r'(://)[^/@]+@', r'\1***@', rtsp_url)
    return result


def load_push():
    data = read_json(PUSH_FILE, DEFAULT_PUSH)
    return {**DEFAULT_PUSH, **data}


def save_push(data):
    write_json(PUSH_FILE, data)


def public_push(data):
    result = dict(data)
    webhook = str(result.get('openclawWebhook', '') or '')
    result['openclawWebhook'] = public_webhook_url(webhook)
    result['webhookConfigured'] = webhook_configured(webhook)
    return result


def normalize_push_update(current, payload):
    allowed = {
        'openclawWebhook', 'pushMode', 'summaryTemplate', 'feishuTarget',
        'noDrinkHours', 'lowCountPercent', 'dailyReportTime', 'alertCooldownHours',
        'enableFeishuDaily', 'enableFeishuAlert', 'enableLowIntakeAlert',
        'enableReviewCandidateAlert',
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f'unsupported push settings: {", ".join(unknown)}')
    result = dict(current)
    if 'openclawWebhook' in payload:
        webhook = str(payload['openclawWebhook'] or '').strip()
        if '***' not in webhook:
            result['openclawWebhook'] = validate_webhook_url(webhook)
    if 'pushMode' in payload:
        mode = str(payload['pushMode'])
        if mode not in ('日报 + 异常提醒', '仅异常提醒', '仅日报'):
            raise ValueError('unsupported push mode')
        result['pushMode'] = mode
    if 'summaryTemplate' in payload:
        template = str(payload['summaryTemplate'])
        if len(template) > 2000:
            raise ValueError('summary template is too long')
        result['summaryTemplate'] = template
    if 'feishuTarget' in payload:
        target = str(payload['feishuTarget']).strip()
        if len(target) > 200:
            raise ValueError('Feishu target is too long')
        result['feishuTarget'] = target
    for field, minimum, maximum in (
        ('noDrinkHours', 1, 168),
        ('lowCountPercent', 1, 99),
        ('alertCooldownHours', 1, 48),
    ):
        if field in payload:
            result[field] = _bounded_int(payload[field], field, minimum, maximum)
    if 'dailyReportTime' in payload:
        clock = str(payload['dailyReportTime'])
        if not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d', clock):
            raise ValueError('dailyReportTime must use HH:MM')
        result['dailyReportTime'] = clock
    for field in (
        'enableFeishuDaily', 'enableFeishuAlert', 'enableLowIntakeAlert',
        'enableReviewCandidateAlert',
    ):
        if field in payload:
            if not isinstance(payload[field], bool):
                raise ValueError(f'{field} must be a boolean')
            result[field] = payload[field]
    return result


def load_label_stats():
    return read_json(LABEL_STATS_FILE, DEFAULT_LABEL_STATS)


def save_label_stats(data):
    write_json(LABEL_STATS_FILE, data)


def _bounded_int(value, field, minimum, maximum):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f'{field} must be an integer')
    if result < minimum or result > maximum:
        raise ValueError(f'{field} must be between {minimum} and {maximum}')
    return result


def normalize_zone(zone, frame_width, frame_height, field='fountainZone'):
    if not isinstance(zone, dict):
        raise ValueError(f'{field} must be an object')
    x1 = _bounded_int(zone.get('x1'), f'{field}.x1', 0, frame_width - 1)
    y1 = _bounded_int(zone.get('y1'), f'{field}.y1', 0, frame_height - 1)
    x2 = _bounded_int(zone.get('x2'), f'{field}.x2', 1, frame_width)
    y2 = _bounded_int(zone.get('y2'), f'{field}.y2', 1, frame_height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'{field} must have a positive area')
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}


def _label_directory(label):
    return POSITIVE_DIR if label == 'confirmed' else NEGATIVE_DIR


def _safe_event_filename(event_id):
    event_id = str(event_id or '')
    return event_id if re.fullmatch(r'[A-Za-z0-9_.-]{1,180}', event_id) else None


def _local_media_path(url):
    if not isinstance(url, str) or not url.startswith('/media/'):
        return None
    candidate = (BASE_DIR / url.lstrip('/')).resolve()
    media_root = MEDIA_DIR.resolve()
    return candidate if candidate.is_relative_to(media_root) else None


def _event_training_sample(event):
    label = event.get('label')
    event_id = _safe_event_filename(event.get('id'))
    if label not in ('confirmed', 'false_positive') or not event_id:
        return None
    return _label_directory(label) / f'{event_id}.jpg'


def _event_crop_path(event):
    event_id = _safe_event_filename(event.get('id'))
    return MEDIA_DIR / 'events' / f'crop-{event_id}.jpg' if event_id else None


def _event_crop_url(event):
    explicit = event.get('crop', '')
    explicit_path = _local_media_path(explicit)
    if explicit_path is not None and explicit_path.is_file():
        return explicit
    crop_path = _event_crop_path(event)
    if crop_path is not None and crop_path.is_file():
        return f'/media/events/{crop_path.name}'
    return ''


def _event_training_source(event):
    crop_url = _event_crop_url(event)
    for url, kind in ((crop_url, 'crop'), (event.get('img', ''), 'snapshot')):
        source = _local_media_path(url)
        if source is not None and source.is_file():
            return source, kind
    return None, ''


def training_sample_inventory(data):
    inventory = {
        'confirmed': 0,
        'false_positive': 0,
        'missing_event_ids': [],
    }
    for event in data.get('events', []):
        label = event.get('label')
        if label not in ('confirmed', 'false_positive'):
            continue
        sample = _event_training_sample(event)
        if sample and sample.is_file():
            inventory[label] += 1
        else:
            inventory['missing_event_ids'].append(event.get('id'))
    return inventory


def sync_training_samples(data, archive_stale=True):
    """Make the on-disk training folders match the current event labels."""
    expected = {POSITIVE_DIR: set(), NEGATIVE_DIR: set()}
    copied = []
    missing = []
    source_counts = {'crop': 0, 'snapshot': 0}
    for event in data.get('events', []):
        label = event.get('label')
        if label not in ('confirmed', 'false_positive'):
            continue
        target = _event_training_sample(event)
        if target is None:
            missing.append({'event_id': event.get('id'), 'reason': 'unsafe event id'})
            continue
        expected[target.parent].add(target.name)
        source, source_kind = _event_training_source(event)
        if source is None:
            missing.append({'event_id': event.get('id'), 'reason': 'crop and snapshot missing'})
            continue
        source_counts[source_kind] += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        if (source_kind == 'crop' or not target.exists() or
                source.stat().st_mtime_ns > target.stat().st_mtime_ns):
            shutil.copy2(source, target)
            copied.append(str(target))

    orphaned = []
    if archive_stale:
        archive_root = LABEL_ORPHAN_DIR / datetime.now(CST).strftime('%Y%m%d_%H%M%S')
        for directory in (POSITIVE_DIR, NEGATIVE_DIR):
            if not directory.exists():
                continue
            for sample in directory.glob('*.jpg'):
                if sample.name in expected[directory]:
                    continue
                destination = archive_root / directory.name / sample.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(sample), str(destination))
                orphaned.append(str(destination))
    inventory = training_sample_inventory(data)
    return {
        'copied': len(copied),
        'orphaned': len(orphaned),
        'missing': missing,
        'source_counts': source_counts,
        **inventory,
    }


def build_training_manifest(data):
    """Create a stable, auditable hand-off for future model training."""
    records = []
    for event in data.get('events', []):
        label = event.get('label')
        if label not in ('confirmed', 'false_positive'):
            continue
        sample = _event_training_sample(event)
        records.append({
            'event_id': event.get('id'),
            'label': 'drinking' if label == 'confirmed' else 'not_drinking',
            'snapshot': event.get('img', ''),
            'classifier_crop': _event_crop_url(event),
            'classifier_crops': event.get('crops', []),
            'clip': event.get('clip', ''),
            'clips': event.get('clips', []),
            'evidence': event.get('evidenceUrl', ''),
            'evidence_urls': event.get('evidenceUrls', []),
            'training_sample': str(sample) if sample and sample.is_file() else '',
            'training_source': _event_training_source(event)[1],
            'usable': bool(sample and sample.is_file()),
            'trigger_source': event.get('triggerSource', ''),
            'profile': event.get('profile', ''),
            'cat_confidence': event.get('catConfidence', event.get('confidence')),
            'drinking_confidence': event.get('drinkingConfidence', event.get('clfConfidence')),
            'labeled_at': event.get('labeled_at', ''),
        })
    manifest = {
        'generated_at': datetime.now(CST).isoformat(timespec='seconds'),
        'schema_version': 2,
        'inventory': training_sample_inventory(data),
        'records': records,
    }
    write_json(TRAINING_MANIFEST_FILE, manifest)
    return manifest


def label_quality_summary(data):
    by_source = {}
    for event in data.get('events', []):
        label = event.get('label')
        if label not in ('confirmed', 'false_positive'):
            continue
        source = event.get('triggerSource') or 'unknown'
        bucket = by_source.setdefault(source, {'confirmed': 0, 'false_positive': 0})
        bucket[label] += 1
    for bucket in by_source.values():
        total = bucket['confirmed'] + bucket['false_positive']
        bucket['total'] = total
        bucket['precision'] = round(bucket['confirmed'] / total, 3) if total else None
    return by_source


def rebuild_label_stats(data, previous=None):
    """Use event labels as the source of truth instead of stale counters."""
    previous = previous or DEFAULT_LABEL_STATS
    stats = {
        'confirmed': 0,
        'false_positive': 0,
        'last_labeled_at': previous.get('last_labeled_at', ''),
        'threshold_confirmed': previous.get('threshold_confirmed', DEFAULT_LABEL_STATS['threshold_confirmed']),
        'threshold_false_positive': previous.get('threshold_false_positive', DEFAULT_LABEL_STATS['threshold_false_positive']),
    }
    for event in data.get('events', []):
        label = event.get('label')
        if label in ('confirmed', 'false_positive'):
            stats[label] += 1
            labeled_at = event.get('labeled_at', '')
            if labeled_at and labeled_at > stats['last_labeled_at']:
                stats['last_labeled_at'] = labeled_at
    return stats


def _bounded_float(value, field, minimum, maximum):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{field} must be a number')
    if not minimum <= result <= maximum:
        raise ValueError(f'{field} must be between {minimum} and {maximum}')
    return result


def normalize_settings_update(current, payload):
    allowed = {
        'cameraName', 'rtspUrl', 'roi', 'fountainZone', 'waterZone', 'waterSurfaceY',
        'frameWidth', 'frameHeight', 'fpsSample', 'minPresenceSec', 'mergeGapSec',
        'mlPerSec', 'mlPerActiveSec', 'mlPerActiveSecLow', 'mlPerActiveSecHigh',
        'enableNoDrinkAlert', 'enableDailyReport', 'enableManualReview',
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f'unsupported settings: {", ".join(unknown)}')

    next_settings = dict(current)
    frame_width = _bounded_int(
        payload.get('frameWidth', current.get('frameWidth', 1280)), 'frameWidth', 2, 8192)
    frame_height = _bounded_int(
        payload.get('frameHeight', current.get('frameHeight', 720)), 'frameHeight', 2, 8192)
    next_settings['frameWidth'] = frame_width
    next_settings['frameHeight'] = frame_height

    for field in ('roi', 'fountainZone', 'waterZone'):
        if field in payload:
            next_settings[field] = normalize_zone(payload[field], frame_width, frame_height, field)

    if 'waterSurfaceY' in payload:
        next_settings['waterSurfaceY'] = _bounded_int(
            payload['waterSurfaceY'], 'waterSurfaceY', 0, frame_height)
    if 'cameraName' in payload:
        camera_name = str(payload['cameraName']).strip()
        if not camera_name or len(camera_name) > 100:
            raise ValueError('cameraName must be 1-100 characters')
        next_settings['cameraName'] = camera_name
    if 'rtspUrl' in payload:
        rtsp_url = str(payload['rtspUrl']).strip()
        if len(rtsp_url) > 2048:
            raise ValueError('rtspUrl is too long')
        # The settings page masks an existing password with asterisks. Do not
        # replace a usable saved value with that display-only placeholder.
        if rtsp_url and '***' not in rtsp_url:
            next_settings['rtspUrl'] = rtsp_url
    if 'fpsSample' in payload:
        next_settings['fpsSample'] = _bounded_int(payload['fpsSample'], 'fpsSample', 1, 30)
    if 'minPresenceSec' in payload:
        next_settings['minPresenceSec'] = _bounded_float(
            payload['minPresenceSec'], 'minPresenceSec', 0, 3600)
    if 'mergeGapSec' in payload:
        next_settings['mergeGapSec'] = _bounded_float(payload['mergeGapSec'], 'mergeGapSec', 0, 3600)
    if 'mlPerSec' in payload:
        next_settings['mlPerSec'] = _bounded_float(payload['mlPerSec'], 'mlPerSec', 0, 10)
    for field in ('mlPerActiveSec', 'mlPerActiveSecLow', 'mlPerActiveSecHigh'):
        if field in payload:
            next_settings[field] = _bounded_float(payload[field], field, 0, 2)
    if next_settings.get('mlPerActiveSecLow', 0) > next_settings.get('mlPerActiveSecHigh', 0):
        raise ValueError('mlPerActiveSecLow must not exceed mlPerActiveSecHigh')
    for field in ('enableNoDrinkAlert', 'enableDailyReport', 'enableManualReview'):
        if field in payload:
            if not isinstance(payload[field], bool):
                raise ValueError(f'{field} must be a boolean')
            next_settings[field] = payload[field]
    return next_settings


def resolve_model_path(model_name):
    name = str(model_name or '')
    if not name or Path(name).name != name or Path(name).suffix.lower() != '.onnx':
        raise ValueError('model must be an .onnx filename without a path')
    models_root = (BASE_DIR / 'models').resolve()
    path = (models_root / name).resolve()
    if path.parent != models_root or not path.is_file():
        raise FileNotFoundError(name)
    return path


def _status_with_freshness(payload, stale_after_sec, missing_phase):
    status = dict(payload)
    updated_unix = status.get('updated_unix')
    if not isinstance(updated_unix, (int, float)):
        raw_updated = str(status.get('updated_at') or '')
        try:
            parsed = datetime.fromisoformat(raw_updated.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=CST)
            updated_unix = parsed.timestamp()
        except (TypeError, ValueError):
            updated_unix = None
    age_sec = round(max(0, time() - updated_unix), 1) if isinstance(updated_unix, (int, float)) else None
    stale = age_sec is None or age_sec > stale_after_sec
    status['age_sec'] = age_sec
    status['stale'] = stale
    if stale:
        status['ok'] = False
        status['phase'] = 'stale'
        previous_detail = str(status.get('detail') or '')
        status['detail'] = missing_phase if not previous_detail else f'{missing_phase}; previous: {previous_detail}'
    return status


def load_camera_status():
    return _status_with_freshness(read_json(CAMERA_STATUS_FILE, {
        'ok': False, 'stream_url': '', 'snapshot': '', 'detail': 'camera worker not started', 'updated_at': '',
    }), CAMERA_STATUS_STALE_SEC, 'camera status is stale')


def load_detector_status():
    return _status_with_freshness(read_json(DETECTOR_STATUS_FILE, {
        'ok': False, 'phase': 'detector not started', 'stream_url': '', 'updated_at': '', 'cat_last_seen_at': None,
    }), DETECTOR_STATUS_STALE_SEC, 'detector status is stale')


def load_transcode_status():
    return _status_with_freshness(read_json(TRANSCODE_STATUS_FILE, {
        'ok': False, 'phase': 'transcode worker not started', 'updated_at': '', 'detail': '',
    }), TRANSCODE_STATUS_STALE_SEC, 'transcode status is stale')


def load_backfill_status():
    return _status_with_freshness(read_json(BACKFILL_STATUS_FILE, {
        'ok': False,
        'phase': 'backfill worker not started',
        'updated_at': '',
        'last_scan_at': '',
        'last_clip': '',
        'last_event_at': '',
        'processed_count': 0,
        'pending_count': 0,
        'detail': '',
    }), BACKFILL_STATUS_STALE_SEC, 'backfill worker is not running')


def load_archive_verifier_status():
    return _status_with_freshness(read_json(ARCHIVE_VERIFIER_STATUS_FILE, {
        'ok': False, 'phase': 'archive verifier not started', 'updated_at': '',
    }), ARCHIVE_VERIFIER_STATUS_STALE_SEC, 'archive verifier status is stale')


def load_notification_status():
    return _status_with_freshness(read_json(NOTIFICATION_STATUS_FILE, {
        'ok': False, 'phase': 'notification worker not started', 'updated_at': '',
        'configured': False, 'pending': 0,
    }), NOTIFICATION_STATUS_STALE_SEC, 'notification status is stale')


def load_daily_audit_status():
    return _status_with_freshness(read_json(DAILY_AUDIT_STATUS_FILE, {
        'ok': False, 'phase': 'daily audit worker not started', 'updated_at': '',
    }), DAILY_AUDIT_STATUS_STALE_SEC, 'daily audit status is stale')


def load_model_training_status():
    return _status_with_freshness(read_json(MODEL_TRAINING_STATUS_FILE, {
        'ok': False, 'phase': 'model calibration worker not started', 'updated_at': '',
        'confirmed': 0, 'false_positive': 0, 'ready': False,
    }), MODEL_TRAINING_STATUS_STALE_SEC, 'model calibration status is stale')


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
    camera = load_camera_status()
    transcode = load_transcode_status()
    backfill = load_backfill_status()
    archive_verifier = load_archive_verifier_status()
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
        'services': {
            'camera': {'ok': camera.get('ok', False), 'phase': camera.get('phase', ''), 'age_sec': camera.get('age_sec')},
            'detector': {'ok': detector.get('ok', False), 'phase': detector.get('phase', ''), 'age_sec': detector.get('age_sec')},
            'transcode': {'ok': transcode.get('ok', False), 'phase': transcode.get('phase', ''), 'age_sec': transcode.get('age_sec')},
            'backfill': {'ok': backfill.get('ok', False), 'phase': backfill.get('phase', ''), 'age_sec': backfill.get('age_sec')},
            'archive_verifier': {'ok': archive_verifier.get('ok', False), 'phase': archive_verifier.get('phase', ''), 'age_sec': archive_verifier.get('age_sec')},
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
    if evt.get('label') == 'confirmed':
        evt['tag'] = '人工确认'
    elif evt.get('label') == 'false_positive':
        evt['tag'] = '人工排除'
    elif evt.get('verificationStatus') == 'confirmed':
        evt['tag'] = '高清复核通过'
    elif evt.get('verificationStatus') == 'rejected':
        evt['tag'] = '高清复核未通过'
    elif evt.get('verificationStatus') == 'uncertain':
        evt['tag'] = '待人工复核'
    elif evt.get('verificationStatus') == 'pending':
        evt['tag'] = '待高清复核'
    elif event_stats_status(evt) == 'excluded_unverified_legacy':
        evt['tag'] = '历史未复核'
    elif not evt.get('tag'):
        evt['tag'] = '喝水事件'
    if not evt.get('img'):
        evt['img'] = evt.get('snapshot_url', '')
    if not evt.get('timeRange'):
        s = fmt_time(evt.get('start_time', ''))
        e = fmt_time(evt.get('end_time', ''))
        evt['timeRange'] = f"{s} ~ {e}" if s and e else s or '--'
    if not evt.get('estimatedMl'):
        evt['estimatedMl'] = evt.get('estimated_ml', 0)
    if not evt.get('volumeMethod'):
        evt['volumeMethod'] = ('legacy_event_duration_v1'
                               if float(evt.get('estimatedMl', 0) or 0) > 0 else 'unavailable')
    evt['statisticsStatus'] = event_stats_status(evt)
    return evt

CST = timezone(timedelta(hours=8))

def parse_duration(metric_str):
    """Parse duration string like '21.2s' or '54.9s' to float seconds."""
    try:
        return float(str(metric_str).rstrip('s') or 0)
    except (ValueError, TypeError):
        return 0.0

def fmt_duration(seconds):
    seconds = float(seconds or 0)
    return f"{round(seconds, 1)}s" if seconds % 1 else f"{int(seconds)}s"

def parse_event_dt(ts):
    if not ts or not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(clean_time(ts))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=CST)
        return dt
    except Exception:
        return None

def event_duration_sec(evt):
    dur = parse_duration(evt.get('metric', '0'))
    if dur:
        return dur
    start = parse_event_dt(evt.get('start_time', ''))
    end = parse_event_dt(evt.get('end_time', ''))
    if start and end and end >= start:
        return (end - start).total_seconds()
    return 0.0


def event_stats_status(evt):
    """Return the trusted statistics state for an event."""
    label = evt.get('label', '')
    if label == 'confirmed':
        return 'included'
    if label == 'false_positive':
        return 'excluded_rejected'
    verification = evt.get('verificationStatus', '')
    if verification == 'confirmed':
        return 'included'
    if verification == 'rejected':
        return 'excluded_rejected'
    if verification in ('pending', 'uncertain', 'error') or evt.get('recoveryCandidate'):
        return 'excluded_pending'
    method = str(evt.get('volumeMethod', ''))
    if method in ('legacy_event_duration_v1', 'unavailable', ''):
        return 'excluded_unverified_legacy'
    return 'excluded_pending'


def event_counts_for_stats(evt):
    """Only human-confirmed or archive-confirmed events affect totals."""
    return event_stats_status(evt) == 'included'


def calculate_volume_calibration(payload, data=None):
    if not isinstance(payload, dict):
        raise ValueError('calibration payload must be an object')
    start = parse_event_dt(str(payload.get('start_time') or ''))
    end = parse_event_dt(str(payload.get('end_time') or ''))
    if not start or not end:
        raise ValueError('start_time and end_time must be valid dates')
    duration_sec = (end - start).total_seconds()
    if duration_sec < 60 or duration_sec > 72 * 3600:
        raise ValueError('calibration period must be between 1 minute and 72 hours')
    start_weight = _bounded_float(payload.get('start_weight_g'), 'start_weight_g', 0, 100000)
    end_weight = _bounded_float(payload.get('end_weight_g'), 'end_weight_g', 0, 100000)
    water_loss = start_weight - end_weight
    if water_loss <= 0:
        raise ValueError('end_weight_g must be lower than start_weight_g')

    data = data or load_dashboard_data()
    matched = []
    active_sec = 0.0
    for event in data.get('events', []):
        event_start = parse_event_dt(event.get('start_time', ''))
        event_end = parse_event_dt(event.get('end_time', '')) or event_start
        method = str(event.get('volumeMethod', ''))
        event_active_sec = float(event.get('activeDrinkSec', 0) or 0)
        if not event_counts_for_stats(event) or not event_start or not event_end:
            continue
        if event_start < start or event_end > end:
            continue
        if not method.endswith('_v2') or event_active_sec <= 0:
            continue
        active_sec += event_active_sec
        matched.append({
            'id': event.get('id', ''),
            'start_time': event.get('start_time', ''),
            'active_drink_sec': round(event_active_sec, 3),
        })
    if active_sec < 3:
        raise ValueError('calibration period has less than 3 seconds of trusted active drinking')

    rate = water_loss / active_sec
    warnings = []
    if len(matched) < 2:
        warnings.append('建议至少覆盖 2 次已确认饮水，单次结果容易受称重误差影响')
    if duration_sec > 24 * 3600:
        warnings.append('校准区间超过 24 小时，蒸发和设备残水可能放大误差')
    if rate < 0.05 or rate > 2.0:
        warnings.append('计算结果超出 0.05–2.00 ml/秒安全范围，请检查称重和时间区间')
    return {
        'start_time': start.isoformat(timespec='seconds'),
        'end_time': end.isoformat(timespec='seconds'),
        'period_hours': round(duration_sec / 3600, 2),
        'start_weight_g': round(start_weight, 2),
        'end_weight_g': round(end_weight, 2),
        'water_loss_ml': round(water_loss, 2),
        'event_count': len(matched),
        'active_drink_sec': round(active_sec, 3),
        'calculated_ml_per_active_sec': round(rate, 4),
        'matched_events': matched,
        'warnings': warnings,
        'can_apply': 0.05 <= rate <= 2.0,
    }


def apply_volume_calibration(payload):
    preview = calculate_volume_calibration(payload)
    if not preview['can_apply']:
        raise ValueError('calculated rate is outside the safe range and cannot be applied')
    apply_history = payload.get('recalculate_history', True)
    if not isinstance(apply_history, bool):
        raise ValueError('recalculate_history must be a boolean')

    rate = preview['calculated_ml_per_active_sec']
    low_rate = round(max(0.01, rate * 0.75), 4)
    high_rate = round(min(2.0, rate * 1.25), 4)
    applied_at = datetime.now(CST).isoformat(timespec='seconds')
    calibration_id = f"calibration-{datetime.now(CST).strftime('%Y%m%d-%H%M%S')}"
    record = {
        'id': calibration_id,
        **preview,
        'ml_per_active_sec_low': low_rate,
        'ml_per_active_sec_high': high_rate,
        'recalculate_history': apply_history,
        'applied_at': applied_at,
    }
    recalculated = 0
    with FILE_LOCK:
        settings = load_settings()
        settings['mlPerActiveSec'] = rate
        settings['mlPerActiveSecLow'] = low_rate
        settings['mlPerActiveSecHigh'] = high_rate
        settings['volumeCalibration'] = {
            'id': calibration_id,
            'appliedAt': applied_at,
            'eventCount': preview['event_count'],
            'activeDrinkSec': preview['active_drink_sec'],
            'waterLossMl': preview['water_loss_ml'],
        }
        save_settings(settings)

        if apply_history:
            data = load_dashboard_data(skip_lock=True)
            for event in data.get('events', []):
                active_sec = float(event.get('activeDrinkSec', 0) or 0)
                if active_sec <= 0 or not str(event.get('volumeMethod', '')).endswith('_v2'):
                    continue
                event['mlPerActiveSec'] = rate
                event['estimatedMl'] = round(active_sec * rate, 1)
                event['estimatedMlLow'] = round(active_sec * low_rate, 1)
                event['estimatedMlHigh'] = round(active_sec * high_rate, 1)
                event['volumeCalibrationId'] = calibration_id
                recalculated += 1
            recalc_overview(data)
            save_dashboard_data(data, skip_lock=True)

        history = read_json(VOLUME_CALIBRATIONS_FILE, {'version': 1, 'items': []}, skip_lock=True)
        history.setdefault('items', []).insert(0, record)
        history['items'] = history['items'][:50]
        history['updated_at'] = applied_at
        write_json(VOLUME_CALIBRATIONS_FILE, history, skip_lock=True)
    return {
        'ok': True,
        'calibration': record,
        'settings': public_settings(settings),
        'recalculated_event_count': recalculated,
    }


def event_clip_window(evt, archive_verification=False):
    start_dt = parse_event_dt(evt.get('start_time', ''))
    if not start_dt:
        raise ValueError('event has no valid start_time')
    end_dt = parse_event_dt(evt.get('end_time', '')) or start_dt + timedelta(seconds=10)
    if end_dt < start_dt:
        end_dt = start_dt + timedelta(seconds=10)
    if archive_verification:
        # EasyNVR profile_1 recordings lag profile_2 realtime by a variable
        # amount (observed 103-116 seconds), so use a bounded search window.
        end_offset = 20 if evt.get('recoveryCandidate') else 60
        return start_dt - timedelta(seconds=150), end_dt - timedelta(seconds=end_offset)
    return start_dt - timedelta(seconds=15), end_dt + timedelta(seconds=10)

def refresh_event_display(evt):
    start = fmt_time(evt.get('start_time', ''))
    end = fmt_time(evt.get('end_time', ''))
    if start:
        evt['title'] = f"{start} · 猫咪喝水"
    evt['timeRange'] = f"{start} ~ {end}" if start and end else start or '--'
    evt['metric'] = fmt_duration(event_duration_sec(evt))
    return evt


def _unique_event_values(*values):
    unique = []
    for value in values:
        if isinstance(value, list):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            if candidate and candidate not in unique:
                unique.append(candidate)
    return unique


def merge_event_pair(base, incoming):
    base_duration = event_duration_sec(base)
    incoming_duration = event_duration_sec(incoming)
    base_start = parse_event_dt(base.get('start_time', ''))
    incoming_start = parse_event_dt(incoming.get('start_time', ''))
    base_end = parse_event_dt(base.get('end_time', '')) or base_start
    incoming_end = parse_event_dt(incoming.get('end_time', '')) or incoming_start
    starts = [dt for dt in (base_start, incoming_start) if dt]
    ends = [dt for dt in (base_end, incoming_end) if dt]
    if starts:
        base['start_time'] = min(starts).isoformat(timespec='seconds')
    if ends:
        base['end_time'] = max(ends).isoformat(timespec='seconds')
    base['confidence'] = max(float(base.get('confidence', 0) or 0), float(incoming.get('confidence', 0) or 0))
    base['estimatedMl'] = round(float(base.get('estimatedMl', 0) or 0) + float(incoming.get('estimatedMl', 0) or 0), 1)
    base['estimatedMlLow'] = round(float(base.get('estimatedMlLow', 0) or 0) + float(incoming.get('estimatedMlLow', 0) or 0), 1)
    base['estimatedMlHigh'] = round(float(base.get('estimatedMlHigh', 0) or 0) + float(incoming.get('estimatedMlHigh', 0) or 0), 1)
    base['activeDrinkSec'] = round(float(base.get('activeDrinkSec', 0) or 0) + float(incoming.get('activeDrinkSec', 0) or 0), 1)
    methods = _unique_event_values(base.get('volumeMethod', ''), incoming.get('volumeMethod', ''))
    if methods:
        base['volumeMethod'] = methods[0] if len(methods) == 1 else 'mixed'
    base['metric'] = fmt_duration(base_duration + incoming_duration)
    if not base.get('img') and incoming.get('img'):
        base['img'] = incoming.get('img')
    clips = _unique_event_values(base.get('clips', []), base.get('clip', ''),
                                 incoming.get('clips', []), incoming.get('clip', ''))
    crops = _unique_event_values(base.get('crops', []), base.get('crop', ''),
                                incoming.get('crops', []), incoming.get('crop', ''))
    evidences = _unique_event_values(base.get('evidenceUrls', []), base.get('evidenceUrl', ''),
                                     incoming.get('evidenceUrls', []), incoming.get('evidenceUrl', ''))
    snapshots = _unique_event_values(base.get('snapshots', []), base.get('img', ''), incoming.get('img', ''))
    source_event_ids = _unique_event_values(base.get('sourceEventIds', []), base.get('id', ''),
                                             incoming.get('sourceEventIds', []), incoming.get('id', ''))
    if clips:
        base['clips'] = clips
        base['clip'] = base.get('clip') or incoming.get('clip') or clips[0]
    if crops:
        base['crops'] = crops
        base['crop'] = base.get('crop') or incoming.get('crop') or crops[0]
    if evidences:
        base['evidenceUrls'] = evidences
        base['evidenceUrl'] = base.get('evidenceUrl') or incoming.get('evidenceUrl') or evidences[0]
    if snapshots:
        base['snapshots'] = snapshots
    if source_event_ids:
        base['sourceEventIds'] = source_event_ids
    if incoming.get('label') == 'confirmed' or base.get('label') == 'confirmed':
        base['label'] = 'confirmed'
    incoming_verification = incoming.get('verificationStatus', '')
    base_verification = base.get('verificationStatus', '')
    if incoming_verification and not base_verification:
        base['verificationStatus'] = incoming_verification
        base['archiveVerification'] = incoming.get('archiveVerification', {})
    elif incoming_verification and base_verification != incoming_verification:
        base['verificationStatus'] = 'uncertain'
        base['archiveVerification'] = {
            'status': 'uncertain',
            'reason': 'merged_events_have_conflicting_verification',
            'verified_at': datetime.now(CST).isoformat(),
        }
    trigger_parts = []
    for evt in (base, incoming):
        trigger = evt.get('triggerSource') or evt.get('trigger_source')
        if trigger and trigger not in trigger_parts:
            trigger_parts.append(trigger)
    if trigger_parts:
        base['triggerSource'] = ','.join(trigger_parts)
    base['summary'] = base.get('summary') or incoming.get('summary', '')
    base['mergedCount'] = int(base.get('mergedCount', 1) or 1) + int(incoming.get('mergedCount', 1) or 1)
    return refresh_event_display(base)

def effective_event_merge_gap_sec():
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding='utf-8'))
        return float(data.get('mergeGapSec', MERGE_ADJACENT_EVENT_GAP_SEC) or MERGE_ADJACENT_EVENT_GAP_SEC)
    except Exception:
        return MERGE_ADJACENT_EVENT_GAP_SEC

def consolidate_adjacent_events(data, gap_sec=None):
    """Merge short adjacent automatic events from the same visit."""
    if gap_sec is None:
        gap_sec = effective_event_merge_gap_sec()
    events = data.get('events', [])
    if not events:
        return 0
    false_positive = [e for e in events if e.get('label') == 'false_positive']
    incomplete = [
        e for e in events
        if e.get('label') != 'false_positive' and (not e.get('end_time') or event_duration_sec(e) <= 0)
    ]
    candidates = [
        e for e in events
        if e.get('label') != 'false_positive' and e not in incomplete
    ]
    candidates.sort(key=lambda e: parse_event_dt(e.get('start_time', '')) or datetime.min.replace(tzinfo=CST))
    merged = []
    merge_count = 0
    for evt in candidates:
        evt = refresh_event_display(dict(evt))
        if not merged:
            merged.append(evt)
            continue
        prev = merged[-1]
        prev_end = parse_event_dt(prev.get('end_time', '')) or parse_event_dt(prev.get('start_time', ''))
        curr_start = parse_event_dt(evt.get('start_time', ''))
        same_day = prev.get('start_time', '')[:10] == evt.get('start_time', '')[:10]
        gap = (curr_start - prev_end).total_seconds() if prev_end and curr_start else None
        if same_day and gap is not None and 0 <= gap <= gap_sec:
            merge_event_pair(prev, evt)
            merge_count += 1
        else:
            merged.append(evt)
    merged.extend(refresh_event_display(dict(e)) for e in incomplete)
    merged.extend(refresh_event_display(dict(e)) for e in false_positive)
    merged.sort(key=lambda e: parse_event_dt(e.get('start_time', '')) or datetime.min.replace(tzinfo=CST), reverse=True)
    data['events'] = merged
    return merge_count


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
        if not event_counts_for_stats(evt):
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
        'ml': [round(daily[d]['ml'], 1) for d in labels],
    }

def recalc_overview(data):
    """Recalculate overview stats from events accepted by human or verifier."""
    from datetime import datetime, timedelta
    overview = data.setdefault('overview', {})
    events = data.get('events', [])
    today_str = datetime.now(CST).strftime('%Y-%m-%d')
    week_start = (datetime.now(CST) - timedelta(days=6)).strftime('%Y-%m-%d')
    today_events = [e for e in events if e.get('start_time','').startswith(today_str) and event_counts_for_stats(e)]
    week_events = [e for e in events if e.get('start_time','')[:10] >= week_start and event_counts_for_stats(e)]
    today_pending = [e for e in events if e.get('start_time','').startswith(today_str)
                     and event_stats_status(e) == 'excluded_pending']
    week_pending = [e for e in events if e.get('start_time','')[:10] >= week_start
                    and event_stats_status(e) == 'excluded_pending']
    legacy_excluded = [e for e in events if event_stats_status(e) == 'excluded_unverified_legacy']
    today_count = len(today_events)
    today_ml = round(sum(float(e.get('estimatedMl', 0) or 0) for e in today_events), 1)
    today_ml_low = round(sum(float(e.get('estimatedMlLow', e.get('estimatedMl', 0)) or 0) for e in today_events), 1)
    today_ml_high = round(sum(float(e.get('estimatedMlHigh', e.get('estimatedMl', 0)) or 0) for e in today_events), 1)
    today_v2_count = sum(str(e.get('volumeMethod', '')).endswith('_v2') for e in today_events)
    today_dur = int(sum(parse_duration(e.get('metric','0')) for e in today_events))
    overview['drinkCount'] = today_count
    overview['waterMl'] = today_ml
    overview['waterMlLow'] = today_ml_low
    overview['waterMlHigh'] = today_ml_high
    overview['volumeV2Count'] = today_v2_count
    overview['volumeEstimateMethod'] = (
        'active_lapping_v2' if today_count > 0 and today_v2_count == today_count else 'mixed')
    overview['durationSec'] = today_dur
    overview['avgDuration'] = round(today_dur / max(today_count, 1), 1)
    overview['weekCount'] = len(week_events)
    overview['weekMl'] = round(sum(float(e.get('estimatedMl', 0) or 0) for e in week_events), 1)
    overview['weekDuration'] = int(sum(parse_duration(e.get('metric','0')) for e in week_events))
    overview['todayPendingCount'] = len(today_pending)
    overview['weekPendingCount'] = len(week_pending)
    overview['legacyExcludedCount'] = len(legacy_excluded)
    overview['statisticsPolicy'] = 'human_or_archive_confirmed_v2'
    overview['todayCount'] = today_count
    valid_events = sorted(
        [e for e in events if event_counts_for_stats(e)],
        key=lambda e: parse_event_dt(e.get('end_time') or e.get('start_time') or '') or datetime.min.replace(tzinfo=CST),
        reverse=True,
    )
    if valid_events and (valid_events[0].get("end_time") or valid_events[0].get("start_time")):
        last_ts = valid_events[0].get("end_time") or valid_events[0].get("start_time")
        overview['lastDrink'] = fmt_time(last_ts)
        overview['lastDrinkAt'] = last_ts
        overview['lastDrinkAgo'] = compute_last_drink_ago(last_ts)
    elif not overview.get('lastDrinkAgo'):
        overview['lastDrinkAt'] = ''
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
        overview['healthDesc'] = f'今日已记录 {count} 次饮水事件，饮水偏少，建议继续观察。'
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


def upsert_event(payload):
    """Atomically apply one detector event update to the dashboard data."""
    with FILE_LOCK:
        data = load_dashboard_data(skip_lock=True)
        event_id = str(payload.get('event_id') or payload.get('id') or f"event-{len(data.get('events', [])) + 1}")
        duration_sec = float(payload.get('duration_sec', 0) or 0)
        estimated_ml = float(payload.get('estimated_ml', 0) or 0)
        active_drink_sec = float(payload.get('active_drink_sec', 0) or 0)
        events = data.setdefault('events', [])
        existing = find_event(events, event_id)
        prev = existing if existing else {}
        snapshot = (payload.get('snapshot_url') or payload.get('snapshot') or prev.get('img')
                    or 'https://images.unsplash.com/photo-1519052537078-e6302a4968d4?auto=format&fit=crop&w=640&q=80')
        existing_conf = float(prev.get('confidence', 0) or 0)
        new_conf = float(payload.get('confidence', 0) or 0)
        confidence = max(existing_conf, new_conf)
        start_time = clean_time(payload.get('start_time') or prev.get('start_time') or '')
        end_time = clean_time(payload.get('end_time') or prev.get('end_time') or '')
        dur = duration_sec if duration_sec else float(str(prev.get('metric', '0')).rstrip('s') or 0)
        ml = estimated_ml if estimated_ml else float(prev.get('estimatedMl', 0) or 0)
        active_sec = (active_drink_sec if 'active_drink_sec' in payload
                      else float(prev.get('activeDrinkSec', 0) or 0))
        ml_per_active_sec = float(
            payload.get('ml_per_active_sec', prev.get('mlPerActiveSec', 0.45)) or 0.45)
        settings = load_settings()
        ml_low_rate = float(settings.get('mlPerActiveSecLow', 0.30) or 0.30)
        ml_high_rate = float(settings.get('mlPerActiveSecHigh', 0.70) or 0.70)
        volume_method = payload.get('volume_method') or prev.get('volumeMethod', 'legacy_event_duration_v1')
        verification_status = prev.get('verificationStatus', '')
        if not verification_status and str(volume_method).endswith('_v2'):
            verification_status = 'pending'
        display_start = fmt_time(start_time)
        event = {
            'id': event_id,
            'start_time': start_time,
            'end_time': end_time,
            'title': payload.get('title') or prev.get('title') or f"{display_start} · 猫咪喝水",
            'desc': payload.get('summary') or payload.get('desc') or prev.get('desc') or '自动饮水判定事件',
            'metric': f"{round(dur, 1)}s" if dur % 1 else f"{int(dur)}s",
            'tag': payload.get('tag') or prev.get('tag') or '喝水事件',
            'img': snapshot,
            'crop': payload.get('crop_url') or prev.get('crop', ''),
            'crops': _unique_event_values(prev.get('crops', []), prev.get('crop', ''), payload.get('crop_url', '')),
            'timeRange': f"{display_start} ~ {fmt_time(end_time)}",
            'confidence': confidence,
            'estimatedMl': round(ml, 1),
            'estimatedMlLow': round(active_sec * ml_low_rate, 1) if active_sec else prev.get('estimatedMlLow', 0),
            'estimatedMlHigh': round(active_sec * ml_high_rate, 1) if active_sec else prev.get('estimatedMlHigh', 0),
            'activeDrinkSec': round(active_sec, 1),
            'mlPerActiveSec': round(ml_per_active_sec, 3),
            'volumeMethod': volume_method,
            'summary': payload.get('summary') or prev.get('summary') or '由本地分析程序自动生成',
            'triggerSource': payload.get('trigger_source') or payload.get('triggerSource') or prev.get('triggerSource', ''),
            'profile': payload.get('profile') or prev.get('profile', ''),
            'endReason': payload.get('end_reason') or prev.get('endReason', ''),
            'clfConfidence': payload.get('clf_confidence', prev.get('clfConfidence', None)),
            'drinkingConfidence': payload.get('drinking_confidence', prev.get('drinkingConfidence', None)),
            'catConfidence': payload.get('cat_confidence', prev.get('catConfidence', None)),
            'clip': payload.get('clip_url') or prev.get('clip', ''),
            'clips': _unique_event_values(prev.get('clips', []), prev.get('clip', ''), payload.get('clip_url', '')),
            'evidenceUrl': payload.get('evidence_url') or prev.get('evidenceUrl', ''),
            'evidenceUrls': _unique_event_values(prev.get('evidenceUrls', []), prev.get('evidenceUrl', ''), payload.get('evidence_url', '')),
            'debugCurveUrl': payload.get('debug_curve_url') or prev.get('debugCurveUrl', ''),
            'sourceEventIds': _unique_event_values(prev.get('sourceEventIds', []), event_id),
            'label': prev.get('label', ''),
            'labeled_at': prev.get('labeled_at', ''),
            'verificationStatus': verification_status,
            'archiveVerification': prev.get('archiveVerification', {}),
        }
        if existing is not None:
            events.remove(existing)
        events.insert(0, event)
        consolidate_adjacent_events(data)
        recalc_overview(data)
        overview = data.setdefault('overview', {})
        save_dashboard_data(data, skip_lock=True)
        return event, overview


def update_event_verification(event_id, payload):
    allowed_statuses = {'confirmed', 'rejected', 'uncertain', 'error'}
    status = str(payload.get('status', ''))
    if status not in allowed_statuses:
        raise ValueError('status must be confirmed/rejected/uncertain/error')
    detail = {
        'status': status,
        'verified_at': datetime.now(CST).isoformat(),
    }
    for key in ('samples', 'candidates', 'anchors'):
        if key in payload:
            detail[key] = max(0, int(payload[key]))
    for key in ('max_classifier', 'max_cat'):
        if key in payload:
            detail[key] = round(max(0.0, min(1.0, float(payload[key]))), 4)
    for key in ('first_anchor_sec', 'last_anchor_sec', 'anchor_span_sec', 'active_drink_sec'):
        if key in payload:
            detail[key] = round(max(0.0, min(600.0, float(payload[key]))), 3)
    for key in ('activated', 'night_mode_seen'):
        if key in payload:
            detail[key] = bool(payload[key])
    if payload.get('reason'):
        detail['reason'] = str(payload['reason'])[:240]
    with FILE_LOCK:
        data = load_dashboard_data(skip_lock=True)
        event = find_event(data.get('events', []), event_id)
        if event is None:
            raise KeyError(event_id)
        event['verificationStatus'] = status
        event['archiveVerification'] = detail
        current_method = event.get('volumeMethod', 'legacy_event_duration_v1')
        archive_active_duration = float(detail.get('active_drink_sec', 0) or 0)
        current_active_duration = float(event.get('activeDrinkSec', 0) or 0)
        should_refresh_volume = (
            event.get('recoveryCandidate') or
            current_method in ('', 'unavailable', 'legacy_event_duration_v1', 'archive_anchor_span_v2') or
            (current_method == 'active_lapping_v2'
             and archive_active_duration > current_active_duration + 1.0)
        )
        if status == 'confirmed' and should_refresh_volume:
            active_duration = archive_active_duration
            if event.get('recoveryCandidate') and active_duration <= 0:
                active_duration = detail.get('anchor_span_sec', 0.0) + 0.5
            active_duration = max(0.0, min(120.0, active_duration))
            if event.get('recoveryCandidate'):
                active_duration = max(3.0, active_duration)
                start_dt = parse_event_dt(event.get('start_time', ''))
                if start_dt:
                    event['end_time'] = (start_dt + timedelta(seconds=active_duration)).isoformat(timespec='seconds')
                event['metric'] = fmt_duration(active_duration)
                event['summary'] = '人工候选经 EasyNVR 高清录像复核恢复'
            elif active_duration > event_duration_sec(event):
                start_dt = parse_event_dt(event.get('start_time', ''))
                if start_dt:
                    event['end_time'] = (start_dt + timedelta(seconds=active_duration)).isoformat(timespec='seconds')
                event['metric'] = fmt_duration(active_duration)
            settings = load_settings()
            ml_per_sec = float(settings.get('mlPerActiveSec', 0.45) or 0.45)
            ml_low_rate = float(settings.get('mlPerActiveSecLow', 0.30) or 0.30)
            ml_high_rate = float(settings.get('mlPerActiveSecHigh', 0.70) or 0.70)
            event['activeDrinkSec'] = round(active_duration, 1)
            event['mlPerActiveSec'] = round(ml_per_sec, 3)
            event['estimatedMl'] = round(active_duration * ml_per_sec, 1)
            event['estimatedMlLow'] = round(active_duration * ml_low_rate, 1)
            event['estimatedMlHigh'] = round(active_duration * ml_high_rate, 1)
            event['volumeMethod'] = 'archive_anchor_clusters_v2'
        if (status == 'confirmed' and not event.get('recoveryCandidate')
                and archive_active_duration > 0
                and (archive_active_duration > current_active_duration + 1.0
                     or current_method == 'archive_anchor_clusters_v2')):
            summary = f'EasyNVR 高清复核确认，有效舔水时长 {archive_active_duration:.1f} 秒'
            event['summary'] = summary
            event['desc'] = summary
        if event.get('recoveryCandidate') and event.get('reviewSampleId'):
            labels = read_json(REVIEW_LABELS_FILE, {}, skip_lock=True)
            annotation = dict(labels.get(event['reviewSampleId'], {}))
            annotation['recovery_event_id'] = event_id
            annotation['recovery_status'] = status
            labels[event['reviewSampleId']] = annotation
            write_json(REVIEW_LABELS_FILE, labels, skip_lock=True)
        ensure_event_fields(event)
        refresh_event_display(event)
        recalc_overview(data)
        save_dashboard_data(data, skip_lock=True)
    return event


def sync_archive_audit_review(candidate, label, annotation, labels, data, linked_event):
    sample_id = candidate['id']
    events = data.setdefault('events', [])
    now = datetime.now(CST).isoformat(timespec='seconds')
    if label != 'drinking':
        if linked_event is not None and linked_event.get('auditRecovery'):
            linked_event['label'] = 'false_positive'
            linked_event['labeled_at'] = now
            linked_event['verificationStatus'] = 'rejected'
            linked_event['archiveVerification'] = {
                **linked_event.get('archiveVerification', {}),
                'status': 'rejected', 'verified_at': now,
                'reason': f'audit_candidate_relabelled_{label}',
            }
        annotation.update({
            'recovery_status': 'rejected',
            'recovery_trigger': 'daily_archive_audit',
        })
        labels[sample_id] = annotation
        write_json(REVIEW_LABELS_FILE, labels, skip_lock=True)
        recalc_overview(data)
        save_dashboard_data(data, skip_lock=True)
        sync_training_samples(data, archive_stale=False)
        return {'action': 'suppressed', 'event': linked_event}

    if linked_event is not None:
        if linked_event.get('auditRecovery'):
            linked_event['label'] = 'confirmed'
            linked_event['labeled_at'] = now
            linked_event['verificationStatus'] = 'confirmed'
            linked_event.setdefault('archiveVerification', {})['status'] = 'confirmed'
        annotation.update({
            'recovery_status': 'confirmed',
            'recovery_trigger': 'daily_archive_audit',
        })
        labels[sample_id] = annotation
        write_json(REVIEW_LABELS_FILE, labels, skip_lock=True)
        recalc_overview(data)
        save_dashboard_data(data, skip_lock=True)
        sync_training_samples(data, archive_stale=False)
        return {'action': 'existing', 'event': linked_event}

    candidate_dt = parse_event_dt(candidate.get('created_at', ''))
    if candidate_dt is None:
        raise ValueError('audit candidate has no valid created_at')
    active_duration = max(3.0, min(120.0, float(candidate.get('active_drink_sec', 0) or 0)))
    end_dt = candidate_dt + timedelta(seconds=active_duration)
    settings = load_settings()
    rate = float(settings.get('mlPerActiveSec', 0.45) or 0.45)
    low_rate = float(settings.get('mlPerActiveSecLow', 0.30) or 0.30)
    high_rate = float(settings.get('mlPerActiveSecHigh', 0.70) or 0.70)
    event_id = f'recovered-{sample_id}'
    archive_detail = dict(candidate.get('archive_verification') or {})
    archive_detail.update({
        'status': 'confirmed', 'verified_at': now,
        'reason': 'human_confirmed_daily_archive_audit',
    })
    event = {
        'id': event_id,
        'start_time': candidate_dt.isoformat(timespec='seconds'),
        'end_time': end_dt.isoformat(timespec='seconds'),
        'title': f'{candidate_dt.strftime("%H:%M:%S")} · 录像巡检恢复饮水',
        'desc': '每日 EasyNVR 高清巡检发现并经人工确认',
        'metric': fmt_duration(active_duration),
        'tag': '人工确认',
        'img': candidate.get('snapshot_url', ''),
        'crop': candidate.get('crop_url', ''),
        'crops': _unique_event_values(candidate.get('crop_url', '')),
        'timeRange': f'{candidate_dt.strftime("%H:%M:%S")} ~ {end_dt.strftime("%H:%M:%S")}',
        'confidence': float(candidate.get('classifier_confidence', 0) or 0),
        'estimatedMl': round(active_duration * rate, 1),
        'estimatedMlLow': round(active_duration * low_rate, 1),
        'estimatedMlHigh': round(active_duration * high_rate, 1),
        'activeDrinkSec': round(active_duration, 1),
        'mlPerActiveSec': round(rate, 3),
        'volumeMethod': 'archive_anchor_clusters_v2',
        'summary': '每日高清录像巡检恢复，人工确认后计入统计',
        'triggerSource': 'nightly_archive_audit',
        'profile': candidate.get('profile', ''),
        'sourceEventIds': [event_id],
        'label': 'confirmed',
        'labeled_at': now,
        'verificationStatus': 'confirmed',
        'archiveVerification': archive_detail,
        'recoveryCandidate': True,
        'auditRecovery': True,
        'reviewSampleId': sample_id,
    }
    events.insert(0, event)
    annotation.update({
        'recovery_event_id': event_id,
        'recovery_status': 'confirmed',
        'recovery_trigger': 'daily_archive_audit',
    })
    labels[sample_id] = annotation
    write_json(REVIEW_LABELS_FILE, labels, skip_lock=True)
    recalc_overview(data)
    save_dashboard_data(data, skip_lock=True)
    sync_training_samples(data, archive_stale=False)
    return {'action': 'created_confirmed', 'event': event}


def sync_review_candidate_recovery(sample_id, label):
    """Create or suppress a pending event derived from a human-reviewed sample."""
    with FILE_LOCK:
        candidate = find_review_candidate(sample_id, skip_lock=True)
        if candidate is None:
            raise KeyError(sample_id)
        labels = read_json(REVIEW_LABELS_FILE, {}, skip_lock=True)
        annotation = dict(labels.get(sample_id, {}))
        recovery_event_id = annotation.get('recovery_event_id', '')
        data = load_dashboard_data(skip_lock=True)
        events = data.setdefault('events', [])
        linked_event = find_event(events, recovery_event_id) if recovery_event_id else None
        if candidate.get('source') == 'nightly_archive_audit':
            return sync_archive_audit_review(
                candidate, label, annotation, labels, data, linked_event)

        if label != 'drinking':
            if linked_event is not None and linked_event.get('recoveryCandidate'):
                linked_event['label'] = 'false_positive'
                linked_event['labeled_at'] = datetime.now(CST).isoformat()
                linked_event['verificationStatus'] = 'rejected'
                linked_event['archiveVerification'] = {
                    'status': 'rejected',
                    'reason': f'review_candidate_relabelled_{label}',
                    'verified_at': datetime.now(CST).isoformat(),
                }
                annotation['recovery_status'] = 'rejected'
                recalc_overview(data)
                save_dashboard_data(data, skip_lock=True)
            labels[sample_id] = annotation
            write_json(REVIEW_LABELS_FILE, labels, skip_lock=True)
            return {'action': 'suppressed' if linked_event else 'none', 'event': linked_event}

        if linked_event is not None:
            if linked_event.get('recoveryCandidate') and linked_event.get('verificationStatus') == 'rejected':
                linked_event['label'] = ''
                linked_event['labeled_at'] = ''
                linked_event['verificationStatus'] = 'pending'
                linked_event['archiveVerification'] = {'status': 'pending'}
                annotation['recovery_status'] = 'pending'
                recalc_overview(data)
                save_dashboard_data(data, skip_lock=True)
            elif not linked_event.get('recoveryCandidate') and not event_counts_for_stats(linked_event):
                linked_event = None
                annotation.pop('recovery_event_id', None)
                annotation.pop('recovery_status', None)
            if linked_event is None:
                pass
            else:
                annotation['recovery_trigger'] = 'human_review'
                labels[sample_id] = annotation
                write_json(REVIEW_LABELS_FILE, labels, skip_lock=True)
                return {'action': 'existing', 'event': linked_event}

        candidate_dt = parse_event_dt(candidate.get('created_at', ''))
        if candidate_dt is None:
            raise ValueError('review candidate has no valid created_at')
        nearby = next((event for event in events
                       if event.get('label') != 'false_positive'
                       and event.get('verificationStatus') != 'rejected'
                       and parse_event_dt(event.get('start_time', ''))
                       and abs((parse_event_dt(event.get('start_time')) - candidate_dt).total_seconds()) <= 60), None)
        if nearby is not None:
            annotation['recovery_event_id'] = nearby.get('id', '')
            annotation['recovery_status'] = 'linked_existing'
            labels[sample_id] = annotation
            write_json(REVIEW_LABELS_FILE, labels, skip_lock=True)
            return {'action': 'linked_existing', 'event': nearby}

        event_id = f'recovered-{sample_id}'
        event = {
            'id': event_id,
            'start_time': candidate_dt.isoformat(timespec='seconds'),
            'end_time': candidate_dt.isoformat(timespec='seconds'),
            'title': f'{candidate_dt.strftime("%H:%M:%S")} · 待复核饮水候选',
            'desc': '人工标记为喝水，等待 EasyNVR 高清录像二次复核',
            'metric': '0s',
            'tag': '待高清复核',
            'img': candidate.get('snapshot_url', ''),
            'crop': candidate.get('crop_url', ''),
            'crops': _unique_event_values(candidate.get('crop_url', '')),
            'timeRange': candidate_dt.strftime('%H:%M:%S'),
            'confidence': float(candidate.get('classifier_confidence', 0) or 0),
            'estimatedMl': 0,
            'estimatedMlLow': 0,
            'estimatedMlHigh': 0,
            'activeDrinkSec': 0,
            'mlPerActiveSec': 0.45,
            'volumeMethod': 'pending_archive_verification',
            'summary': '来自人工审核候选，尚未计入饮水统计',
            'triggerSource': 'human_review_candidate',
            'profile': candidate.get('profile', ''),
            'sourceEventIds': [event_id],
            'label': '',
            'labeled_at': '',
            'verificationStatus': 'pending',
            'archiveVerification': {'status': 'pending'},
            'recoveryCandidate': True,
            'reviewSampleId': sample_id,
        }
        events.insert(0, event)
        annotation['recovery_event_id'] = event_id
        annotation['recovery_status'] = 'pending'
        annotation['recovery_trigger'] = 'human_review'
        labels[sample_id] = annotation
        write_json(REVIEW_LABELS_FILE, labels, skip_lock=True)
        recalc_overview(data)
        save_dashboard_data(data, skip_lock=True)
        return {'action': 'created', 'event': event}


def sync_review_candidate_probe(sample_id):
    """Create a statistics-excluded archive probe for an unlabeled candidate."""
    with FILE_LOCK:
        candidate = find_review_candidate(sample_id, skip_lock=True)
        if candidate is None:
            raise KeyError(sample_id)
        labels = read_json(REVIEW_LABELS_FILE, {}, skip_lock=True)
        annotation = dict(labels.get(sample_id, {}))
        if annotation.get('label'):
            return {'action': 'human_labeled', 'event': None}
        data = load_dashboard_data(skip_lock=True)
        events = data.setdefault('events', [])
        linked = find_event(events, annotation.get('recovery_event_id', ''))
        if linked is not None:
            return {'action': 'existing', 'event': linked}
        candidate_dt = parse_event_dt(candidate.get('created_at', ''))
        if candidate_dt is None:
            raise ValueError('review candidate has no valid created_at')
        nearby = min(
            (event for event in events if parse_event_dt(event.get('start_time', ''))
             and abs((parse_event_dt(event.get('start_time')) - candidate_dt).total_seconds()) <= 60),
            key=lambda event: abs((parse_event_dt(event.get('start_time')) - candidate_dt).total_seconds()),
            default=None,
        )
        if nearby is not None:
            verification = nearby.get('verificationStatus', '') or 'unverified'
            annotation.update({
                'recovery_event_id': nearby.get('id', ''),
                'recovery_status': f'covered_{verification}',
                'recovery_trigger': 'automatic_archive_probe',
            })
            labels[sample_id] = annotation
            write_json(REVIEW_LABELS_FILE, labels, skip_lock=True)
            return {'action': 'covered_existing', 'event': nearby}

        event_id = f'auto-recovered-{sample_id}'
        event = {
            'id': event_id,
            'start_time': candidate_dt.isoformat(timespec='seconds'),
            'end_time': candidate_dt.isoformat(timespec='seconds'),
            'title': f'{candidate_dt.strftime("%H:%M:%S")} · 自动高清探测候选',
            'desc': '实时边界样本，等待 EasyNVR 高清录像自动复核',
            'metric': '0s',
            'tag': '待高清复核',
            'img': candidate.get('snapshot_url', ''),
            'crop': candidate.get('crop_url', ''),
            'crops': _unique_event_values(candidate.get('crop_url', '')),
            'timeRange': candidate_dt.strftime('%H:%M:%S'),
            'confidence': float(candidate.get('classifier_confidence', 0) or 0),
            'estimatedMl': 0,
            'estimatedMlLow': 0,
            'estimatedMlHigh': 0,
            'activeDrinkSec': 0,
            'mlPerActiveSec': 0.45,
            'volumeMethod': 'pending_archive_verification',
            'summary': '自动高清探测尚未确认，不计入饮水统计',
            'triggerSource': 'automatic_archive_probe',
            'profile': candidate.get('profile', ''),
            'sourceEventIds': [event_id],
            'label': '',
            'labeled_at': '',
            'verificationStatus': 'pending',
            'archiveVerification': {'status': 'pending'},
            'recoveryCandidate': True,
            'automatedRecovery': True,
            'reviewSampleId': sample_id,
        }
        events.insert(0, event)
        annotation.update({
            'recovery_event_id': event_id,
            'recovery_status': 'pending',
            'recovery_trigger': 'automatic_archive_probe',
        })
        labels[sample_id] = annotation
        write_json(REVIEW_LABELS_FILE, labels, skip_lock=True)
        recalc_overview(data)
        save_dashboard_data(data, skip_lock=True)
        return {'action': 'created', 'event': event}


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
        self.end_headers()
        self.wfile.write(body)

    def _authenticated(self):
        if not API_TOKEN:
            return True
        supplied = self.headers.get('X-API-Token', '')
        if not supplied:
            authorization = self.headers.get('Authorization', '')
            if authorization.lower().startswith('bearer '):
                supplied = authorization[7:].strip()
        return bool(supplied) and hmac.compare_digest(supplied, API_TOKEN)

    def _require_auth(self):
        if self._authenticated():
            return True
        self._send_json({'ok': False, 'error': '需要 API 授权令牌'}, 401)
        return False

    def _read_json(self):
        try:
            length = int(self.headers.get('Content-Length', '0') or '0')
        except ValueError:
            raise ValueError('Content-Length is invalid')
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError(f'request body must be at most {MAX_REQUEST_BYTES} bytes')
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
        self.send_response(405)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == '/api/auth-status':
            return self._send_json({
                'enabled': bool(API_TOKEN),
                'authenticated': self._authenticated(),
            })
        if path == '/api/dashboard/overview':
            with FILE_LOCK:
                data = load_dashboard_data(skip_lock=True)
                events = [ensure_event_fields(e) for e in data.get('events', [])]
                data["events"] = events
                consolidate_adjacent_events(data)
                recalc_overview(data)
                overview = data.setdefault('overview', {})
                if not overview.get('lastDrinkAgo'):
                    overview['lastDrinkAgo'] = '等待猫咪饮水...'
                save_dashboard_data(data, skip_lock=True)
            return self._send_json(data)
        if path == '/api/events':
            data = load_dashboard_data()
            query = parse_qs(parsed.query)
            event_id = query.get('id', [None])[0]
            if event_id:
                event = find_event_with_archive(event_id, data)
                if event is None:
                    return self._send_json({'ok': False, 'error': 'Event Not Found'}, 404)
                return self._send_json({'event': ensure_event_fields(event)})
            include_archived = query.get('include_archived', ['0'])[0].lower() in ('1', 'true', 'yes')
            start_date = query.get('from', [''])[0]
            end_date = query.get('to', [''])[0]
            if start_date and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', start_date):
                return self._send_json({'ok': False, 'error': 'from must use YYYY-MM-DD'}, 400)
            if end_date and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', end_date):
                return self._send_json({'ok': False, 'error': 'to must use YYYY-MM-DD'}, 400)
            if start_date and end_date and start_date > end_date:
                return self._send_json({'ok': False, 'error': 'from must not be after to'}, 400)
            try:
                limit = max(1, min(5000, int(query.get('limit', ['1000'])[0])))
            except ValueError:
                return self._send_json({'ok': False, 'error': 'limit must be an integer'}, 400)
            events = list(data.get('events', []))
            if start_date:
                events = [event for event in events if str(event.get('start_time', ''))[:10] >= start_date]
            if end_date:
                events = [event for event in events if str(event.get('start_time', ''))[:10] <= end_date]
            if include_archived:
                events.extend(load_archived_events(start_date, end_date))
            unique = {}
            for event in events:
                if isinstance(event, dict) and event.get('id'):
                    unique[event['id']] = event
            events = sorted(unique.values(), key=lambda event: event.get('start_time', ''), reverse=True)[:limit]
            return self._send_json({
                'events': [ensure_event_fields(event) for event in events],
                'include_archived': include_archived,
                'archive': archive_status(),
            })
        if path == '/api/events/archive-status':
            return self._send_json(archive_status())
        if path.startswith('/api/events/'):
            data = load_dashboard_data()
            event_id = path.rsplit('/', 1)[-1]
            event = find_event_with_archive(event_id, data)
            if event is None:
                return self._send_json({'ok': False, 'error': 'Event Not Found'}, 404)
            return self._send_json({'event': ensure_event_fields(event)})
        if path == '/api/settings':
            return self._send_json(public_settings(load_settings()))
        if path == '/api/push-settings':
            return self._send_json(public_push(load_push()))
        if path == '/api/push-log':
            return self._send_json({'items': read_json(PUSH_LOG_FILE, [])})
        if path == '/api/volume-calibrations':
            history = read_json(VOLUME_CALIBRATIONS_FILE, {'version': 1, 'items': []})
            return self._send_json(history)
        if path == '/api/notification-status':
            return self._send_json(load_notification_status())
        if path == '/api/daily-audit-status':
            return self._send_json(load_daily_audit_status())
        if path == '/api/model-training-status':
            return self._send_json(load_model_training_status())
        if path == '/api/camera-status':
            return self._send_json(load_camera_status())
        if path == '/api/detector-status':
            return self._send_json(load_detector_status())
        if path == '/api/transcode-status':
            return self._send_json(load_transcode_status())
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
            query = parse_qs(parsed.query)
            event_id = query.get('id', [None])[0]
            source = query.get('source', [''])[0]
            archive_verification = query.get('verification', ['0'])[0] == '1'
            if not event_id:
                return self._send_json({'ok': False, 'error': 'missing event id'}, 400)
            data = load_dashboard_data()
            event = find_event_with_archive(event_id, data)
            if not event:
                return self._send_json({'ok': False, 'error': 'event not found'}, 404)
            local_clip = str(event.get('clip', '') or '')
            if source != 'archive' and local_clip.startswith('/media/events/'):
                clip_path = (BASE_DIR / local_clip.lstrip('/')).resolve()
                events_root = (MEDIA_DIR / 'events').resolve()
                if clip_path.is_relative_to(events_root) and clip_path.is_file():
                    file_size = clip_path.stat().st_size
                    self.send_response(200)
                    self.send_header('Content-Type', 'video/mp4')
                    self.send_header('Content-Length', str(file_size))
                    self.send_header('Accept-Ranges', 'bytes')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    with clip_path.open('rb') as video_file:
                        shutil.copyfileobj(video_file, self.wfile)
                    return
            try:
                clip_start, clip_end = event_clip_window(event, archive_verification)
            except ValueError as exc:
                return self._send_json({'ok': False, 'error': str(exc)}, 400)
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
                           '-vf', f'scale={1280 if archive_verification else 960}:-2', '-c:v', 'libx264',
                           '-preset', 'ultrafast', '-crf', '23', '-an',
                           '-movflags', '+faststart', tmp_path]
                    proc = subprocess.run(cmd, capture_output=True, timeout=120 if archive_verification else 30)
                else:
                    list_path = tmp_path + '.list.txt'
                    with open(list_path, 'w') as f:
                        for nf in needed_files:
                            f.write(f"file '{nf}'\n")
                    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path,
                           '-ss', f'{offset_sec:.3f}', '-t', f'{clip_duration:.3f}',
                           '-vf', f'scale={1280 if archive_verification else 960}:-2', '-c:v', 'libx264',
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
            data = load_dashboard_data()
            stats = rebuild_label_stats(data, load_label_stats())
            inventory = training_sample_inventory(data)
            stats['total_labeled'] = stats['confirmed'] + stats['false_positive']
            stats['available_confirmed'] = inventory['confirmed']
            stats['available_false_positive'] = inventory['false_positive']
            stats['missing_event_ids'] = inventory['missing_event_ids']
            stats['ready_to_train'] = (
                inventory['confirmed'] >= stats['threshold_confirmed'] and
                inventory['false_positive'] >= stats['threshold_false_positive']
            )
            stats['quality_by_trigger'] = label_quality_summary(data)
            return self._send_json(stats)
        if path == '/api/labels/manifest':
            data = load_dashboard_data()
            return self._send_json(build_training_manifest(data))
        if path == '/api/review-candidates':
            return self._send_json(load_review_candidates())
        if path == '/api/health':
            camera = load_camera_status()
            detector = load_detector_status()
            transcode = load_transcode_status()
            archive_verifier = load_archive_verifier_status()
            notifications = load_notification_status()
            daily_audit = load_daily_audit_status()
            model_training = load_model_training_status()
            pipeline_ok = bool(camera.get('ok') and detector.get('ok') and transcode.get('ok') and archive_verifier.get('ok'))
            overall_ok = bool(pipeline_ok and notifications.get('ok') and daily_audit.get('ok') and model_training.get('ok'))
            return self._send_json({
                'ok': overall_ok,
                'service': 'cat-water-dashboard-api',
                'pipeline_ok': pipeline_ok,
                'write_auth_enabled': bool(API_TOKEN),
                'services': {
                    'camera': {'ok': camera.get('ok', False), 'phase': camera.get('phase', ''), 'age_sec': camera.get('age_sec')},
                    'detector': {'ok': detector.get('ok', False), 'phase': detector.get('phase', ''), 'age_sec': detector.get('age_sec')},
                    'transcode': {'ok': transcode.get('ok', False), 'phase': transcode.get('phase', ''), 'age_sec': transcode.get('age_sec')},
                    'archive_verifier': {'ok': archive_verifier.get('ok', False), 'phase': archive_verifier.get('phase', ''), 'age_sec': archive_verifier.get('age_sec')},
                    'notifications': {'ok': notifications.get('ok', False), 'phase': notifications.get('phase', ''), 'configured': notifications.get('configured', False), 'age_sec': notifications.get('age_sec')},
                    'daily_audit': {'ok': daily_audit.get('ok', False), 'phase': daily_audit.get('phase', ''), 'date': daily_audit.get('date', ''), 'age_sec': daily_audit.get('age_sec')},
                    'model_training': {'ok': model_training.get('ok', False), 'phase': model_training.get('phase', ''), 'ready': model_training.get('ready', False), 'age_sec': model_training.get('age_sec')},
                },
            })
        if path == "/developer" or path == "/developer/" or path == "/dev.html":
            dev_path = Path("/app/data/dev.html")
            if not dev_path.exists():
                dev_path = Path("/app/data/developer.html")
            if dev_path.exists():
                content = dev_path.read_bytes()
                if b'/auth.js' not in content:
                    content = content.replace(
                        b'<head>', b'<head><script src="/auth.js"></script>', 1)
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
        if not self._require_auth():
            return
        if path == '/api/events':
            payload = self._read_request_json()
            if payload is None:
                return
            event, overview = upsert_event(payload)
            return self._send_json({'ok': True, 'event': event, 'overview': overview}, 201)

        if path == '/api/events/verify':
            payload = self._read_request_json()
            if payload is None:
                return
            event_id = str(payload.get('event_id', ''))
            if not event_id:
                return self._send_json({'ok': False, 'error': 'event_id is required'}, 400)
            try:
                event = update_event_verification(event_id, payload)
            except ValueError as exc:
                return self._send_json({'ok': False, 'error': str(exc)}, 400)
            except KeyError:
                return self._send_json({'ok': False, 'error': 'Event Not Found'}, 404)
            return self._send_json({'ok': True, 'event': event})

        if path == '/api/fountain-zone':
            payload = self._read_request_json()
            if payload is None:
                return
            with FILE_LOCK:
                settings = load_settings()
                try:
                    frame_width = _bounded_int(
                        payload.get('frameWidth', settings.get('frameWidth', 1280)),
                        'frameWidth', 2, 8192,
                    )
                    frame_height = _bounded_int(
                        payload.get('frameHeight', settings.get('frameHeight', 720)),
                        'frameHeight', 2, 8192,
                    )
                    zone = payload.get('fountainZone', settings.get('fountainZone'))
                    settings['fountainZone'] = normalize_zone(zone, frame_width, frame_height)
                except ValueError as exc:
                    return self._send_json({'ok': False, 'error': str(exc)}, 400)
                settings['frameWidth'] = frame_width
                settings['frameHeight'] = frame_height
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
            with FILE_LOCK:
                try:
                    current = normalize_settings_update(load_settings(), payload)
                except ValueError as exc:
                    return self._send_json({'ok': False, 'error': str(exc)}, 400)
                save_settings(current)
            return self._send_json({'ok': True, 'settings': public_settings(current)})

        if path == '/api/push-settings':
            payload = self._read_request_json()
            if payload is None:
                return
            try:
                current = normalize_push_update(load_push(), payload)
            except ValueError as exc:
                return self._send_json({'ok': False, 'error': str(exc)}, 400)
            save_push(current)
            return self._send_json({'ok': True, 'push': public_push(current)})

        if path == '/api/volume-calibration/preview':
            payload = self._read_request_json()
            if payload is None:
                return
            try:
                preview = calculate_volume_calibration(payload)
            except ValueError as exc:
                return self._send_json({'ok': False, 'error': str(exc)}, 400)
            return self._send_json({'ok': True, 'calibration': preview})

        if path == '/api/volume-calibration/apply':
            payload = self._read_request_json()
            if payload is None:
                return
            try:
                result = apply_volume_calibration(payload)
            except ValueError as exc:
                return self._send_json({'ok': False, 'error': str(exc)}, 400)
            return self._send_json(result)

        if path == '/api/integrations/openclaw/push':
            payload = self._read_request_json()
            if payload is None:
                return
            push = load_push()
            if not webhook_configured(push.get('openclawWebhook', '')):
                return self._send_json({'ok': False, 'error': '请先配置可访问的 Webhook 地址'}, 400)
            title = str(payload.get('title') or '猫咪饮水监控测试推送')[:120]
            content = payload.get('content') if isinstance(payload.get('content'), dict) else {}
            message = str(payload.get('message') or '')[:2000]
            if not message:
                message = (
                    f'{title}\n今日饮水次数：{content.get("drink_count", 0)}\n'
                    f'视频估算饮水量：{content.get("estimated_total_ml", 0)} ml\n'
                    f'最近一次饮水：{content.get("last_drink_time", "--:--")}'
                )
            event = {
                'type': str(payload.get('type') or 'test_push')[:60],
                'title': title,
                'message': message,
                'target': str(push.get('feishuTarget', ''))[:200],
                'data': content,
                'created_at': datetime.now(CST).isoformat(timespec='seconds'),
            }
            try:
                result = deliver_notification(push['openclawWebhook'], event)
                ok = True
                status = result.get('status', 200)
                error = ''
            except Exception as exc:
                ok = False
                status = 0
                error = str(exc)[:240]
            with FILE_LOCK:
                history = read_json(PUSH_LOG_FILE, [], skip_lock=True)
                history.insert(0, {
                    'type': event['type'], 'title': title, 'created_at': event['created_at'],
                    'ok': ok, 'status': status, 'error': error,
                })
                write_json(PUSH_LOG_FILE, history[:100], skip_lock=True)
            if not ok:
                return self._send_json({'ok': False, 'error': f'推送失败：{error}'}, 502)
            return self._send_json({'ok': True, 'delivered': True, 'message': '测试推送已送达 Webhook'})

        if path == '/api/review-candidates/label':
            payload = self._read_request_json()
            if payload is None:
                return
            sample_id = str(payload.get('sample_id', ''))
            label = str(payload.get('label', ''))
            if not sample_id:
                return self._send_json({'ok': False, 'error': 'sample_id is required'}, 400)
            try:
                annotation = label_review_candidate(sample_id, label)
                recovery = sync_review_candidate_recovery(sample_id, label)
            except ValueError as exc:
                return self._send_json({'ok': False, 'error': str(exc)}, 400)
            except KeyError:
                return self._send_json({'ok': False, 'error': 'Review Candidate Not Found'}, 404)
            return self._send_json({
                'ok': True,
                'sample_id': sample_id,
                'annotation': annotation,
                'recovery': recovery,
                'queue': load_review_candidates(),
            })

        if path == '/api/review-candidates/promote':
            payload = self._read_request_json()
            if payload is None:
                return
            sample_id = str(payload.get('sample_id', ''))
            labels = read_json(REVIEW_LABELS_FILE, {})
            if not sample_id or labels.get(sample_id, {}).get('label') != 'drinking':
                return self._send_json({'ok': False, 'error': 'sample must be labeled drinking'}, 400)
            try:
                recovery = sync_review_candidate_recovery(sample_id, 'drinking')
            except ValueError as exc:
                return self._send_json({'ok': False, 'error': str(exc)}, 400)
            except KeyError:
                return self._send_json({'ok': False, 'error': 'Review Candidate Not Found'}, 404)
            return self._send_json({'ok': True, 'sample_id': sample_id, 'recovery': recovery})

        if path == '/api/review-candidates/probe':
            payload = self._read_request_json()
            if payload is None:
                return
            sample_id = str(payload.get('sample_id', ''))
            if not sample_id:
                return self._send_json({'ok': False, 'error': 'sample_id is required'}, 400)
            try:
                recovery = sync_review_candidate_probe(sample_id)
            except ValueError as exc:
                return self._send_json({'ok': False, 'error': str(exc)}, 400)
            except KeyError:
                return self._send_json({'ok': False, 'error': 'Review Candidate Not Found'}, 404)
            return self._send_json({'ok': True, 'sample_id': sample_id, 'recovery': recovery})

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
                event['label'] = label
                event['labeled_at'] = datetime.now(CST).isoformat()
                sample_sync = sync_training_samples(data)
                stats = rebuild_label_stats(data, load_label_stats())
                stats['last_labeled_at'] = datetime.now(CST).isoformat()
                save_label_stats(stats)
                recalc_overview(data)
                save_dashboard_data(data, skip_lock=True)
                manifest = build_training_manifest(data)
                stats['total_labeled'] = stats['confirmed'] + stats['false_positive']
                stats['available_confirmed'] = sample_sync['confirmed']
                stats['available_false_positive'] = sample_sync['false_positive']
                stats['missing_event_ids'] = sample_sync['missing_event_ids']
                stats['ready_to_train'] = (
                    stats['available_confirmed'] >= stats['threshold_confirmed'] and
                    stats['available_false_positive'] >= stats['threshold_false_positive']
                )
            return self._send_json({
                'ok': True,
                'event': event,
                'stats': stats,
                'sample_sync': sample_sync,
                'training_records': len(manifest['records']),
            })

        if path == '/api/labels/rebuild':
            payload = self._read_request_json()
            if payload is None:
                return
            with FILE_LOCK:
                data = load_dashboard_data(skip_lock=True)
                sample_sync = sync_training_samples(data)
                stats = rebuild_label_stats(data, load_label_stats())
                stats['last_labeled_at'] = datetime.now(CST).isoformat()
                save_label_stats(stats)
                manifest = build_training_manifest(data)
            return self._send_json({
                'ok': True,
                'sample_sync': sample_sync,
                'training_records': len(manifest['records']),
            })

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
            try:
                resolve_model_path(model_name)
            except ValueError as exc:
                return self._send_json({'ok': False, 'error': str(exc)}, 400)
            except FileNotFoundError:
                return self._send_json({'ok': False, 'error': f'模型文件不存在: {model_name}'}, 404)
            with FILE_LOCK:
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
