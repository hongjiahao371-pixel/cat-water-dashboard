#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path

import cv2

import archive_verifier_worker as archive_verifier
import detection_worker as detector_module


DASHBOARD_API_URL = os.environ.get('DASHBOARD_API_URL', 'http://cat-water-dashboard:39095')
RECORD_ROOT = Path(os.environ.get(
    'DAILY_AUDIT_RECORD_ROOT',
    '/easynvr_rec/easynvr_rec/EASYNVR_CHANNEL_PLACEHOLDER/profile_1',
))
STATUS_FILE = Path(os.environ.get('DAILY_AUDIT_STATUS_FILE', '/app/data/daily_audit_status.json'))
STATE_FILE = Path(os.environ.get('DAILY_AUDIT_STATE_FILE', '/app/data/daily_audit_state.json'))
CANDIDATES_FILE = Path(os.environ.get(
    'DAILY_AUDIT_CANDIDATES_FILE', '/app/data/daily_audit_candidates.json'))
MEDIA_DIR = Path(os.environ.get('DAILY_AUDIT_MEDIA_DIR', '/app/media/audit_candidates'))
RUN_TIME = os.environ.get('DAILY_AUDIT_RUN_TIME', '03:30')
POLL_SEC = float(os.environ.get('DAILY_AUDIT_POLL_SEC', '300'))
PREFILTER_INTERVAL_SEC = float(os.environ.get('DAILY_AUDIT_PREFILTER_INTERVAL_SEC', '10'))
PREFILTER_MIN_CLASSIFIER = float(os.environ.get('DAILY_AUDIT_PREFILTER_MIN_CLASSIFIER', '0.15'))
PREFILTER_BATCH_FILES = int(os.environ.get('DAILY_AUDIT_PREFILTER_BATCH_FILES', '20'))
PREFILTER_MOTION_THRESHOLD = float(os.environ.get('DAILY_AUDIT_MOTION_THRESHOLD', '12.0'))
ANALYZE_FPS = float(os.environ.get('DAILY_AUDIT_ANALYZE_FPS', '2'))
QUICK_ANALYZE_FPS = float(os.environ.get('DAILY_AUDIT_QUICK_ANALYZE_FPS', '1'))
MAX_CANDIDATES = int(os.environ.get('DAILY_AUDIT_MAX_CANDIDATES', '200'))
CST = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(CST).isoformat(timespec='seconds')


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=path.parent,
        prefix=f'.{path.name}.', suffix='.tmp', delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_json(path, default):
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else json.loads(json.dumps(default))
    except (OSError, ValueError):
        return json.loads(json.dumps(default))


def write_status(**payload):
    payload.setdefault('ok', True)
    payload.setdefault('updated_at', now_iso())
    payload.setdefault('updated_unix', time.time())
    write_json_atomic(STATUS_FILE, payload)


def request_json(path, timeout=30):
    request = urllib.request.Request(
        f'{DASHBOARD_API_URL}{path}', headers={'User-Agent': 'cat-water-daily-audit/1.0'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def file_start_time(path):
    stem = path.name[:14]
    if len(stem) == 14 and stem.isdigit():
        return datetime.strptime(stem, '%Y%m%d%H%M%S').replace(tzinfo=CST)
    return datetime.fromtimestamp(path.stat().st_mtime, CST)


def recording_files(day):
    directory = RECORD_ROOT / day.strftime('%Y%m%d')
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob('*.mp4') if path.stat().st_size > 1024 * 1024)


def extract_prefilter_frames(paths):
    paths = [Path(path) for path in paths]
    directory = Path(tempfile.mkdtemp(prefix='daily-audit-frames-'))
    fps_filter = f'fps=1/{max(1.0, PREFILTER_INTERVAL_SEC):g}'
    frames = []
    failed_files = 0
    for source_index, path in enumerate(paths):
        pattern = directory / f'{source_index:03d}-frame-%03d.jpg'
        command = [
            'ffmpeg', '-y', '-loglevel', 'error', '-skip_frame', 'nokey',
            '-i', str(path),
            '-vf', f'{fps_filter},scale=960:-2', '-q:v', '4', str(pattern),
        ]
        try:
            result = subprocess.run(command, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            result = None
        produced = sorted(directory.glob(f'{source_index:03d}-frame-*.jpg'))
        if result is not None and result.returncode == 0 and produced:
            frames.extend((source_index, frame) for frame in produced)
        else:
            failed_files += 1
            for frame in produced:
                frame.unlink(missing_ok=True)
    if frames:
        return directory, frames, 'cpu_indexed', failed_files
    shutil.rmtree(directory, ignore_errors=True)
    return None, [], 'cpu_indexed', failed_files


def prefilter_batch(paths, detector, settings):
    directory, indexed_frames, decode_mode, failed_files = extract_prefilter_frames(paths)
    frames = [frame for _, frame in indexed_frames]
    if not frames:
        return set(), {
            'error': 'cannot_decode', 'decode_mode': decode_mode,
            'failed_files': failed_files, 'samples': 0,
        }
    max_classifier = 0.0
    max_cat = 0.0
    samples = 0
    inference_samples = 0
    interesting = set()
    background = None
    source_frame_counts = {}
    try:
        for source_index, frame_path in indexed_frames:
            source_frame_index = source_frame_counts.get(source_index, 0)
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            source_frame_counts[source_index] = source_frame_index + 1
            samples += 1
            height, width = frame.shape[:2]
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
            mask = detector_module.np.zeros(gray.shape, dtype=detector_module.np.uint8)
            fountain = detector_module.get_fountain_zone(settings, 320, 180)
            water = detector_module.get_water_zone(settings, 320, 180)
            fx1, fy1, fx2, fy2 = fountain
            mask[fy1:fy2, fx1:fx2] = 1
            wx1, wy1, wx2, wy2 = water
            mask[wy1:wy2, wx1:wx2] = 0
            if background is None:
                background = gray.astype(detector_module.np.float32)
                motion_score = 0.0
            else:
                diff = cv2.absdiff(gray, cv2.convertScaleAbs(background))
                motion_score = float(diff[mask > 0].mean()) if detector_module.np.any(mask) else 0.0
            force_check = source_frame_index == 0
            if not force_check and motion_score < PREFILTER_MOTION_THRESHOLD:
                cv2.accumulateWeighted(gray, background, 0.04)
                continue
            inference_samples += 1
            night_mode = detector_module.is_night_mode(frame)
            threshold = (detector_module.NIGHT_CAT_CROP_THRESHOLD if night_mode
                         else detector_module.CLOSEUP_CAT_SCORE_THRESHOLD)
            candidates = detector._infer_candidates(
                frame, threshold, night_mode=night_mode)
            water_zone = detector_module.get_water_zone(settings, width, height)
            candidates = [candidate for candidate in candidates
                          if detector_module.closeup_cat_near_water(
                              candidate['box'], water_zone, width, height)]
            best = max(candidates, key=lambda item: item['score']) if candidates else None
            if best:
                max_cat = max(max_cat, float(best.get('score', 0) or 0))
                cat_box = best['box']
                _, confidence = detector.classify_drinking(frame, cat_box)
                max_classifier = max(max_classifier, float(confidence or 0))
                if float(confidence or 0) >= PREFILTER_MIN_CLASSIFIER:
                    interesting.add(source_index)
    finally:
        if directory:
            shutil.rmtree(directory, ignore_errors=True)
    return interesting, {
        'samples': samples,
        'inference_samples': inference_samples,
        'max_classifier': round(max_classifier, 4), 'max_cat': round(max_cat, 4),
        'decode_mode': decode_mode, 'failed_files': failed_files,
    }


def prefilter_video(path, detector, settings):
    interesting, detail = prefilter_batch([path], detector, settings)
    return {'interesting': bool(interesting), **detail}


def event_archive_time(event):
    detail = event.get('archiveVerification') or {}
    start = archive_verifier.parse_event_time(event.get('start_time'))
    if event.get('auditRecovery') or event.get('triggerSource') == 'nightly_archive_audit':
        return (start, 45) if start else (None, 0)
    first_anchor = detail.get('first_anchor_sec')
    if start and isinstance(first_anchor, (int, float)) and first_anchor > 0:
        return start - timedelta(seconds=150) + timedelta(seconds=float(first_anchor)), 45
    return (start, 180) if start else (None, 0)


def covered_by_existing(candidate_time, events):
    for event in events:
        if event.get('label') == 'false_positive' or event.get('verificationStatus') == 'rejected':
            continue
        event_start = archive_verifier.parse_event_time(event.get('start_time'))
        if event_start and abs((event_start - candidate_time).total_seconds()) <= 60:
            return event.get('id', '')
        archive_time, tolerance = event_archive_time(event)
        if archive_time and abs((archive_time - candidate_time).total_seconds()) <= tolerance:
            return event.get('id', '')
    return ''


def nearby_audit_candidate(candidate_time, payload, tolerance_sec=45, active_sec=0):
    candidate_end = candidate_time + timedelta(seconds=max(0.0, float(active_sec or 0)))
    for item in payload.get('items', []):
        existing_time = archive_verifier.parse_event_time(item.get('created_at'))
        if not existing_time:
            continue
        existing_end = existing_time + timedelta(
            seconds=max(0.0, float(item.get('active_drink_sec', 0) or 0)))
        gap = max(
            0.0,
            (candidate_time - existing_end).total_seconds(),
            (existing_time - candidate_end).total_seconds(),
        )
        if gap <= tolerance_sec:
            return item.get('id', '')
    return ''


def audit_candidate_for_recording(path, payload):
    name = Path(path).name
    return next((item.get('id', '') for item in payload.get('items', [])
                 if name in _candidate_recording_files(item)), '')


def _candidate_recording_files(item):
    values = [item.get('recording_file', '')]
    values.extend(item.get('recording_files', []) or [])
    return {str(value) for value in values if value}


def merge_candidate_evidence(item, path, result):
    files = sorted(_candidate_recording_files(item) | {Path(path).name})
    active_sec = round(
        float(item.get('active_drink_sec', 0) or 0)
        + float(result.get('active_drink_sec', 0) or 0), 3)
    detail = dict(item.get('archive_verification') or {})
    detail['active_drink_sec'] = active_sec
    detail['merged_recording_files'] = files
    detail['merged_segments'] = int(detail.get('merged_segments', 1) or 1) + 1
    item['active_drink_sec'] = active_sec
    item['archive_verification'] = detail
    item['recording_files'] = files
    item['classifier_confidence'] = max(
        float(item.get('classifier_confidence', 0) or 0),
        float(result.get('max_classifier', 0) or 0),
    )
    item['cat_confidence'] = max(
        float(item.get('cat_confidence', 0) or 0),
        float(result.get('max_cat', 0) or 0),
    )


def merge_candidate_items(base, incoming):
    active_sec = round(
        float(base.get('active_drink_sec', 0) or 0)
        + float(incoming.get('active_drink_sec', 0) or 0), 3)
    files = sorted(_candidate_recording_files(base) | _candidate_recording_files(incoming))
    detail = dict(base.get('archive_verification') or {})
    detail['active_drink_sec'] = active_sec
    detail['merged_recording_files'] = files
    detail['merged_segments'] = (
        int(detail.get('merged_segments', 1) or 1)
        + int((incoming.get('archive_verification') or {}).get('merged_segments', 1) or 1)
    )
    base['active_drink_sec'] = active_sec
    base['archive_verification'] = detail
    base['recording_files'] = files
    base['classifier_confidence'] = max(
        float(base.get('classifier_confidence', 0) or 0),
        float(incoming.get('classifier_confidence', 0) or 0),
    )
    base['cat_confidence'] = max(
        float(base.get('cat_confidence', 0) or 0),
        float(incoming.get('cat_confidence', 0) or 0),
    )
    for singular, plural in (('snapshot_url', 'snapshot_urls'), ('crop_url', 'crop_urls')):
        values = []
        for value in (
            base.get(plural, []), base.get(singular, ''),
            incoming.get(plural, []), incoming.get(singular, ''),
        ):
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                if candidate and candidate not in values:
                    values.append(candidate)
        if values:
            base[plural] = values


def normalize_candidate_payload(payload, events):
    normalized = {'version': payload.get('version', 1), 'items': []}
    result = {'before': len(payload.get('items', [])), 'covered_removed': 0, 'merged': 0}
    items = sorted(
        (dict(item) for item in payload.get('items', []) if isinstance(item, dict)),
        key=lambda item: item.get('created_at', ''),
    )
    for item in items:
        candidate_time = archive_verifier.parse_event_time(item.get('created_at'))
        if not candidate_time:
            normalized['items'].append(item)
            continue
        if covered_by_existing(candidate_time, events):
            result['covered_removed'] += 1
            continue
        nearby_id = nearby_audit_candidate(
            candidate_time, normalized, active_sec=item.get('active_drink_sec', 0))
        if nearby_id:
            existing = next(existing for existing in normalized['items']
                            if existing.get('id') == nearby_id)
            merge_candidate_items(existing, item)
            result['merged'] += 1
        else:
            normalized['items'].append(item)
    payload['items'] = normalized['items'][-MAX_CANDIDATES:]
    payload['updated_at'] = now_iso()
    result['after'] = len(payload['items'])
    return result


def quick_result_needs_full_analysis(result):
    if result.get('status') == 'confirmed':
        return True
    anchor_floor = min(
        detector_module.CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD,
        detector_module.NIGHT_CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD,
    )
    return float(result.get('max_classifier', 0) or 0) >= anchor_floor


def save_evidence(path, result, candidate_id):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return '', ''
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(result.get('peak_classifier_sec', 0)) * 1000)
        ok, frame = capture.read()
        if not ok or frame is None:
            return '', ''
    finally:
        capture.release()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = MEDIA_DIR / f'{candidate_id}.jpg'
    crop = MEDIA_DIR / f'{candidate_id}-crop.jpg'
    cv2.imwrite(str(snapshot), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    box = [int(value) for value in result.get('peak_cat_box', [])]
    crop_url = ''
    if len(box) == 4:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1 and cv2.imwrite(str(crop), frame[y1:y2, x1:x2]):
            crop_url = f'/media/audit_candidates/{crop.name}'
    return f'/media/audit_candidates/{snapshot.name}', crop_url


def add_candidate(path, file_start, result, events, payload):
    candidate_time = file_start + timedelta(seconds=float(result.get('first_anchor_sec', 0)))
    covered_event = covered_by_existing(candidate_time, events)
    if covered_event:
        return 'covered', covered_event
    same_recording = audit_candidate_for_recording(path, payload)
    if same_recording:
        return 'existing', same_recording
    nearby_candidate = nearby_audit_candidate(
        candidate_time, payload, active_sec=result.get('active_drink_sec', 0))
    if nearby_candidate:
        existing = next(item for item in payload.get('items', [])
                        if item.get('id') == nearby_candidate)
        merge_candidate_evidence(existing, path, result)
        payload['updated_at'] = now_iso()
        return 'merged', nearby_candidate
    digest = hashlib.sha1(
        f'{path}:{candidate_time.isoformat()}'.encode('utf-8')).hexdigest()[:10]
    candidate_id = f'audit-{candidate_time:%Y%m%d-%H%M%S}-{digest}'
    if any(item.get('id') == candidate_id for item in payload.get('items', [])):
        return 'existing', candidate_id
    snapshot_url, crop_url = save_evidence(path, result, candidate_id)
    item = {
        'id': candidate_id,
        'created_at': candidate_time.isoformat(timespec='seconds'),
        'created_unix': candidate_time.timestamp(),
        'snapshot_url': snapshot_url,
        'crop_url': crop_url,
        'classifier_confidence': result.get('max_classifier', 0),
        'cat_confidence': result.get('max_cat', 0),
        'profile': 'night' if result.get('night_mode_seen') else 'day',
        'source': 'nightly_archive_audit',
        'active_drink_sec': result.get('active_drink_sec', 0),
        'archive_verification': result,
        'recording_file': path.name,
    }
    payload.setdefault('items', []).append(item)
    payload['items'] = payload['items'][-MAX_CANDIDATES:]
    payload['updated_at'] = now_iso()
    return 'candidate', candidate_id


def audit_date(day, max_files=0):
    files = recording_files(day)
    if max_files > 0:
        files = files[:max_files]
    detector = detector_module.CatDetector()
    settings = detector_module.read_settings()
    events = request_json('/api/events').get('events', [])
    candidates = read_json(CANDIDATES_FILE, {'version': 1, 'items': []})
    summary = {
        'date': day.isoformat(), 'files_total': len(files), 'files_scanned': 0,
        'prefilter_hits': 0, 'strong_hits': 0, 'new_candidates': 0,
        'covered_events': 0, 'errors': 0, 'prefilter_samples': 0,
        'prefilter_inferences': 0, 'quick_checks': 0, 'full_checks': 0,
        'merged_candidates': 0,
        'decode_modes': {}, 'last_error': '',
    }
    write_status(phase='scanning', **summary)
    analyzed_paths = set()
    batch_size = max(1, PREFILTER_BATCH_FILES)
    for batch_start in range(0, len(files), batch_size):
        batch = files[batch_start:batch_start + batch_size]
        try:
            interesting, detail = prefilter_batch(batch, detector, settings)
            summary['files_scanned'] += len(batch)
            summary['prefilter_hits'] += len(interesting)
            summary['prefilter_samples'] += int(detail.get('samples', 0) or 0)
            summary['prefilter_inferences'] += int(detail.get('inference_samples', 0) or 0)
            summary['errors'] += int(detail.get('failed_files', 0) or 0)
            decode_mode = detail.get('decode_mode') or 'unknown'
            summary['decode_modes'][decode_mode] = summary['decode_modes'].get(decode_mode, 0) + 1
            for batch_index in sorted(interesting):
                path = batch[batch_index]
                if path in analyzed_paths:
                    continue
                analyzed_paths.add(path)
                write_status(phase='analyzing', current_file=path.name, **summary)
                quick_result = archive_verifier.analyze_video(
                    path, detector=detector, sample_fps=QUICK_ANALYZE_FPS)
                summary['quick_checks'] += 1
                if not quick_result_needs_full_analysis(quick_result):
                    continue
                if audit_candidate_for_recording(path, candidates):
                    summary['strong_hits'] += 1
                    continue
                if quick_result.get('status') == 'confirmed':
                    quick_time = file_start_time(path) + timedelta(
                        seconds=float(quick_result.get('first_anchor_sec', 0)))
                    if covered_by_existing(quick_time, events):
                        summary['strong_hits'] += 1
                        summary['covered_events'] += 1
                        continue
                    if nearby_audit_candidate(quick_time, candidates):
                        summary['strong_hits'] += 1
                        continue
                result = archive_verifier.analyze_video(
                    path, detector=detector, sample_fps=ANALYZE_FPS)
                summary['full_checks'] += 1
                if result.get('status') != 'confirmed':
                    continue
                summary['strong_hits'] += 1
                action, _ = add_candidate(path, file_start_time(path), result, events, candidates)
                if action == 'candidate':
                    summary['new_candidates'] += 1
                    write_json_atomic(CANDIDATES_FILE, candidates)
                elif action == 'merged':
                    summary['merged_candidates'] += 1
                    write_json_atomic(CANDIDATES_FILE, candidates)
                elif action == 'covered':
                    summary['covered_events'] += 1
        except Exception as exc:
            summary['errors'] += len(batch)
            summary['last_error'] = str(exc)[:240]
        write_status(
            phase='scanning', current_file=batch[-1].name if batch else '', **summary)
    write_json_atomic(CANDIDATES_FILE, candidates)
    write_status(phase='idle', completed_at=now_iso(), **summary)
    return summary


def scheduled_day(now, state):
    hour, minute = [int(value) for value in RUN_TIME.split(':', 1)]
    if now.time() < dt_time(hour, minute):
        return None
    day = now.date() - timedelta(days=1)
    return None if day.isoformat() in state.get('completed_dates', {}) else day


def idle_status_payload(state):
    completed = state.get('completed_dates', {})
    if not completed:
        return {'phase': 'idle', 'detail': 'waiting for nightly audit'}
    last_date = sorted(completed)[-1]
    entry = completed[last_date]
    summary = dict(entry.get('summary') or {})
    return {
        **summary,
        'phase': 'idle',
        'detail': 'waiting for nightly audit',
        'completed_at': entry.get('completed_at', ''),
        'last_completed_date': last_date,
    }


def run_forever():
    state = read_json(STATE_FILE, {'completed_dates': {}})
    write_status(phase='idle', detail='waiting for nightly audit')
    while True:
        try:
            day = scheduled_day(datetime.now(CST), state)
            if day:
                summary = audit_date(day)
                state.setdefault('completed_dates', {})[day.isoformat()] = {
                    'completed_at': now_iso(), 'summary': summary,
                }
                state['completed_dates'] = dict(
                    sorted(state['completed_dates'].items())[-30:])
                write_json_atomic(STATE_FILE, state)
            else:
                write_status(**idle_status_payload(state))
        except Exception as exc:
            write_status(ok=False, phase='failed', error=str(exc)[:240])
        time.sleep(max(60, POLL_SEC))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date')
    parser.add_argument('--max-files', type=int, default=0)
    parser.add_argument('--normalize-candidates', action='store_true')
    args = parser.parse_args()
    if args.normalize_candidates:
        events = request_json('/api/events').get('events', [])
        candidates = read_json(CANDIDATES_FILE, {'version': 1, 'items': []})
        result = normalize_candidate_payload(candidates, events)
        write_json_atomic(CANDIDATES_FILE, candidates)
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.date:
        print(json.dumps(
            audit_date(date.fromisoformat(args.date), max_files=args.max_files),
            ensure_ascii=False,
        ))
        return
    run_forever()


if __name__ == '__main__':
    main()
