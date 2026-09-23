#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
import time
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

import detection_worker as detector_module


DASHBOARD_API_URL = os.environ.get('DASHBOARD_API_URL', 'http://cat-water-dashboard:39095')
API_TOKEN = os.environ.get('CAT_WATER_API_TOKEN', '').strip()
STATUS_FILE = Path(os.environ.get('ARCHIVE_VERIFY_STATUS_FILE', '/app/data/archive_verifier_status.json'))
STATE_FILE = Path(os.environ.get('ARCHIVE_VERIFY_STATE_FILE', '/app/data/archive_verifier_state.json'))
POLL_SEC = float(os.environ.get('ARCHIVE_VERIFY_POLL_SEC', '60'))
WAIT_SEC = float(os.environ.get('ARCHIVE_VERIFY_WAIT_SEC', '180'))
LOOKBACK_HOURS = float(os.environ.get('ARCHIVE_VERIFY_LOOKBACK_HOURS', '24'))
RECOVERY_LOOKBACK_HOURS = float(os.environ.get('ARCHIVE_VERIFY_RECOVERY_LOOKBACK_HOURS', '168'))
SAMPLE_FPS = float(os.environ.get('ARCHIVE_VERIFY_FPS', '2'))
REJECT_MAX_CONFIDENCE = float(os.environ.get('ARCHIVE_VERIFY_REJECT_MAX_CONFIDENCE', '0.35'))
ANCHOR_WINDOW_SEC = float(os.environ.get('ARCHIVE_VERIFY_ANCHOR_WINDOW_SEC', '15'))
ANCHOR_MIN_IOU = float(os.environ.get('ARCHIVE_VERIFY_ANCHOR_MIN_IOU', '0.20'))
MAX_ATTEMPTS = int(os.environ.get('ARCHIVE_VERIFY_MAX_ATTEMPTS', '3'))
AUTO_RECOVERY_ENABLED = os.environ.get('ARCHIVE_AUTO_RECOVERY_ENABLED', 'true').lower() == 'true'
AUTO_RECOVERY_MIN_CONFIDENCE = float(os.environ.get('ARCHIVE_AUTO_RECOVERY_MIN_CONFIDENCE', '0.15'))
AUTO_RECOVERY_MAX_PER_POLL = int(os.environ.get('ARCHIVE_AUTO_RECOVERY_MAX_PER_POLL', '3'))
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


def write_status(**payload):
    payload.setdefault('ok', True)
    payload.setdefault('updated_at', now_iso())
    payload.setdefault('updated_unix', time.time())
    write_json_atomic(STATUS_FILE, payload)


def request_json(path, payload=None, timeout=20):
    url = f'{DASHBOARD_API_URL}{path}'
    data = None
    method = 'GET'
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        method = 'POST'
        headers['Content-Type'] = 'application/json'
    if API_TOKEN:
        headers['X-API-Token'] = API_TOKEN
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def parse_event_time(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).rstrip('Z'))
        return parsed.replace(tzinfo=CST) if parsed.tzinfo is None else parsed.astimezone(CST)
    except ValueError:
        return None


def classify_verification(activated, anchors, max_classifier, classifier_available=True):
    if not classifier_available:
        return 'error'
    if activated and anchors >= 2:
        return 'confirmed'
    if anchors == 0 and max_classifier < REJECT_MAX_CONFIDENCE:
        return 'rejected'
    return 'uncertain'


class ArchiveAnchorGate:
    """Join sparse high-confidence archive hits from the same spatial target."""

    def __init__(self, window_sec=ANCHOR_WINDOW_SEC, min_iou=ANCHOR_MIN_IOU):
        self.window_sec = max(0.5, float(window_sec))
        self.min_iou = max(0.0, float(min_iou))
        self.hits = deque()
        self.confirmed_times = set()

    def observe(self, now, cat_box):
        while self.hits and now - self.hits[0][0] > self.window_sec:
            self.hits.popleft()
        matched_times = [
            previous_time for previous_time, previous_box in self.hits
            if detector_module.box_iou(cat_box, previous_box) >= self.min_iou
        ]
        matched = bool(matched_times)
        if matched:
            self.confirmed_times.update(matched_times)
            self.confirmed_times.add(float(now))
        self.hits.append((float(now), list(cat_box)))
        return matched


def dominant_anchor_cluster(times, window_sec=ANCHOR_WINDOW_SEC):
    ordered = sorted({float(value) for value in times})
    if not ordered:
        return 0.0, 0.0, 0.0
    clusters = [[ordered[0]]]
    for value in ordered[1:]:
        if value - clusters[-1][-1] <= window_sec:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    cluster = max(clusters, key=lambda values: (len(values), values[-1] - values[0]))
    return cluster[0], cluster[-1], max(0.0, cluster[-1] - cluster[0])


def total_anchor_cluster_duration(times, window_sec=ANCHOR_WINDOW_SEC, credit_sec=0.5):
    ordered = sorted({float(value) for value in times})
    if not ordered:
        return 0.0
    clusters = [[ordered[0]]]
    for value in ordered[1:]:
        if value - clusters[-1][-1] <= window_sec:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return sum(
        cluster[-1] - cluster[0] + credit_sec
        for cluster in clusters if len(cluster) >= 2
    )


def analyze_video(video_path, detector=None, sample_fps=SAMPLE_FPS):
    detector = detector or detector_module.CatDetector()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'cannot open video: {video_path}')
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    sample_every = max(1, int(round(source_fps / max(0.1, sample_fps))))
    settings = detector_module.read_settings()
    gates = {
        False: detector_module.TemporalClassifierGate(),
        True: detector_module.TemporalClassifierGate(
            threshold=detector_module.NIGHT_CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD),
    }
    archive_anchor_gate = ArchiveAnchorGate()
    frame_index = 0
    samples = 0
    candidates = 0
    anchors = 0
    anchor_times = []
    max_classifier = 0.0
    peak_classifier_sec = 0.0
    peak_cat_box = []
    max_cat = 0.0
    activated = False
    night_mode_seen = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_every:
                frame_index += 1
                continue
            sample_time = frame_index / source_fps
            frame_index += 1
            samples += 1
            height, width = frame.shape[:2]
            fountain_zone = detector_module.get_fountain_zone(settings, width, height)
            water_zone = detector_module.get_water_zone(settings, width, height)
            result = detector.detect(frame, force=True, fzone=fountain_zone, wzone=water_zone)
            night_mode = bool(result.get('night_mode'))
            night_mode_seen = night_mode_seen or night_mode
            cat_box = result.get('cat_box') or []
            max_cat = max(max_cat, float(result.get('cat_score', 0) or 0))
            confidence = 0.0
            if cat_box and result.get('closeup_water'):
                candidates += 1
                _, confidence = detector.classify_drinking(frame, cat_box)
                confidence = float(confidence or 0)
                if confidence >= max_classifier:
                    max_classifier = confidence
                    peak_classifier_sec = sample_time
                    peak_cat_box = list(cat_box)
                threshold = (detector_module.NIGHT_CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD
                             if night_mode else detector_module.CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD)
                if confidence >= threshold:
                    anchors += 1
                    anchor_times.append(sample_time)
                    activated = activated or archive_anchor_gate.observe(sample_time, cat_box)
            gate_active = gates[night_mode].observe(
                sample_time,
                bool(result.get('cat_detected')),
                bool(result.get('closeup_water')),
                confidence,
                cat_box,
            )
            activated = activated or gate_active
    finally:
        capture.release()
    classifier_available = detector._drink_clf_session is not None
    status = classify_verification(
        activated, anchors, max_classifier, classifier_available=classifier_available)
    # Spatial consistency decides whether this is a real event. Once confirmed,
    # all high-confidence near-water anchors define its active time; detector box
    # jitter must not collapse a long close-up drink to one IoU-matched pair.
    cluster_times = anchor_times if activated else []
    first_anchor, last_anchor, anchor_span = dominant_anchor_cluster(cluster_times)
    active_drink_sec = total_anchor_cluster_duration(cluster_times)
    return {
        'status': status,
        'samples': samples,
        'candidates': candidates,
        'anchors': anchors,
        'max_classifier': round(max_classifier, 4),
        'max_cat': round(max_cat, 4),
        'activated': activated,
        'night_mode_seen': night_mode_seen,
        'first_anchor_sec': round(first_anchor, 3),
        'last_anchor_sec': round(last_anchor, 3),
        'anchor_span_sec': round(anchor_span, 3),
        'active_drink_sec': round(active_drink_sec, 3),
        'peak_classifier_sec': round(peak_classifier_sec, 3),
        'peak_cat_box': [round(float(value), 1) for value in peak_cat_box],
        'reason': 'drinking_classifier_unavailable' if not classifier_available else '',
    }


def download_archive_video(event_id, destination):
    query = urllib.parse.urlencode({
        'id': event_id,
        'source': 'archive',
        'verification': '1',
    })
    request = urllib.request.Request(
        f'{DASHBOARD_API_URL}/api/event-video?{query}',
        headers={'User-Agent': 'cat-water-archive-verifier/1.0'},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        with open(destination, 'wb') as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    if Path(destination).stat().st_size == 0:
        raise RuntimeError('archive video is empty')


def load_state():
    try:
        payload = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else {'events': {}}
    except (OSError, ValueError):
        return {'version': 1, 'events': {}}


def eligible_events(events, state):
    now = datetime.now(CST)
    candidates = []
    attempts = state.setdefault('events', {})
    for event in events:
        if event.get('label') == 'false_positive':
            continue
        if event.get('verificationStatus') in ('confirmed', 'rejected', 'uncertain'):
            continue
        event_id = str(event.get('id', ''))
        if not event_id or int(attempts.get(event_id, {}).get('attempts', 0)) >= MAX_ATTEMPTS:
            continue
        end_time = parse_event_time(event.get('end_time') or event.get('start_time'))
        lookback_hours = RECOVERY_LOOKBACK_HOURS if event.get('recoveryCandidate') else LOOKBACK_HOURS
        oldest = now - timedelta(hours=lookback_hours)
        if not end_time or end_time < oldest or (now - end_time).total_seconds() < WAIT_SEC:
            continue
        candidates.append((end_time, event))
    return [event for _, event in sorted(candidates, key=lambda item: item[0])]


def eligible_review_candidates(queue):
    if not AUTO_RECOVERY_ENABLED:
        return []
    now = datetime.now(CST)
    eligible = []
    for candidate in queue.get('items', []):
        if (candidate.get('label') or candidate.get('recovery_status')
                or candidate.get('source') == 'nightly_archive_audit'):
            continue
        confidence = float(candidate.get('classifier_confidence', 0) or 0)
        if confidence < AUTO_RECOVERY_MIN_CONFIDENCE:
            continue
        created_at = parse_event_time(candidate.get('created_at'))
        if not created_at:
            continue
        age_sec = (now - created_at).total_seconds()
        if age_sec < WAIT_SEC or age_sec > RECOVERY_LOOKBACK_HOURS * 3600:
            continue
        eligible.append((created_at, candidate))
    return [candidate for _, candidate in sorted(eligible, key=lambda item: item[0])
            ][:max(0, AUTO_RECOVERY_MAX_PER_POLL)]


def create_automatic_probes(queue):
    results = []
    for candidate in eligible_review_candidates(queue):
        result = request_json('/api/review-candidates/probe', {
            'sample_id': candidate['id'],
        })
        results.append({
            'sample_id': candidate['id'],
            'action': result.get('recovery', {}).get('action', ''),
        })
    return results


def process_event(event, detector, state):
    event_id = str(event['id'])
    record = state.setdefault('events', {}).setdefault(event_id, {'attempts': 0})
    record['attempts'] = int(record.get('attempts', 0)) + 1
    record['last_attempt_at'] = now_iso()
    write_json_atomic(STATE_FILE, state)
    write_status(phase='processing', event_id=event_id, attempts=record['attempts'])
    temporary = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    video_path = Path(temporary.name)
    temporary.close()
    try:
        download_archive_video(event_id, video_path)
        result = analyze_video(video_path, detector=detector)
        request_json('/api/events/verify', {'event_id': event_id, **result}, timeout=20)
        record.update({'completed_at': now_iso(), 'result': result})
        write_json_atomic(STATE_FILE, state)
        print(f'VERIFIED {event_id}: {json.dumps(result, ensure_ascii=False)}', flush=True)
        return result
    except Exception as exc:
        reason = str(exc)[:240]
        record['last_error'] = reason
        write_json_atomic(STATE_FILE, state)
        if record['attempts'] >= MAX_ATTEMPTS:
            try:
                request_json('/api/events/verify', {
                    'event_id': event_id,
                    'status': 'error',
                    'reason': reason,
                })
            except Exception:
                pass
        raise
    finally:
        video_path.unlink(missing_ok=True)


def run_forever():
    detector = detector_module.CatDetector()
    state = load_state()
    write_status(phase='starting')
    while True:
        try:
            review_queue = request_json('/api/review-candidates', timeout=20)
            probes = create_automatic_probes(review_queue)
            events = request_json('/api/events', timeout=20).get('events', [])
            pending = eligible_events(events, state)
            for event in pending:
                try:
                    process_event(event, detector, state)
                except Exception as exc:
                    print(f'VERIFICATION FAILED {event.get("id")}: {exc}', flush=True)
                    write_status(ok=False, phase='event_failed', event_id=event.get('id'), error=str(exc)[:240])
            write_status(
                phase='idle', pending=0, tracked=len(state.get('events', {})),
                auto_probes=len(probes), auto_probe_results=probes[-5:],
            )
        except Exception as exc:
            print(f'POLL FAILED: {exc}', flush=True)
            write_status(ok=False, phase='poll_failed', error=str(exc)[:240])
        time.sleep(max(5, POLL_SEC))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video')
    args = parser.parse_args()
    if args.video:
        print(json.dumps(analyze_video(args.video), ensure_ascii=False))
        return
    run_forever()


if __name__ == '__main__':
    main()
