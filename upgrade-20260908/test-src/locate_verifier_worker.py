#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2


DASHBOARD_API_URL = os.environ.get('DASHBOARD_API_URL', 'http://cat-water-dashboard:39095')
STATUS_FILE = Path(os.environ.get(
    'LOCATE_STATUS_FILE', '/app/data/locate_verifier_status.json'))
STATE_FILE = Path(os.environ.get(
    'LOCATE_STATE_FILE', '/app/data/locate_verifier_state.json'))
RESULTS_FILE = Path(os.environ.get(
    'LOCATE_RESULTS_FILE', '/app/data/locate_verifier_results.json'))
EVIDENCE_DIR = Path(os.environ.get(
    'LOCATE_EVIDENCE_DIR', '/app/media/locate_verifier'))
SETTINGS_FILE = Path(os.environ.get('SETTINGS_FILE', '/app/data/settings.json'))
MODEL_PATH = Path(os.environ.get(
    'LOCATE_MODEL_PATH', '/models/locate-anything-q6_k.gguf'))
LIBRARY_PATH = os.environ.get(
    'LOCATE_LIBRARY_PATH', '/usr/local/lib/liblocate_anything.so')
MODEL_ID = os.environ.get(
    'LOCATE_MODEL_ID', 'locate-anything-q6_k:818ba69f0dd8')
STRATEGY_ID = os.environ.get(
    'LOCATE_STRATEGY_ID', 'mouth-contact-all-frames-v2')
LEGACY_STRATEGY_ID = 'broad-drinking-prompt-strict-replay-v1'
PROMPT = os.environ.get(
    'LOCATE_PROMPT',
    'Locate every cat mouth or muzzle that is visibly touching or immediately '
    'above the water surface while the cat is actively lapping from the pet '
    'water fountain. Ignore cats that are merely sitting, standing, walking, '
    'looking at, or passing near the fountain.')
POLL_SEC = float(os.environ.get('LOCATE_POLL_SEC', '60'))
MIN_INTERVAL_SEC = float(os.environ.get('LOCATE_MIN_INTERVAL_SEC', '1800'))
LOOKBACK_DAYS = float(os.environ.get('LOCATE_LOOKBACK_DAYS', '30'))
MAX_PER_POLL = int(os.environ.get('LOCATE_MAX_PER_POLL', '1'))
MAX_ATTEMPTS = int(os.environ.get('LOCATE_MAX_ATTEMPTS', '3'))
MAX_FRAMES = int(os.environ.get('LOCATE_MAX_FRAMES', '3'))
THREADS = int(os.environ.get('LOCATE_THREADS', '3'))
MODE = os.environ.get('LOCATE_MODE', 'slow').strip().lower()
IMAGE_MAX_EDGE = int(os.environ.get('LOCATE_IMAGE_MAX_EDGE', '672'))
MIN_BOX_AREA_RATIO = float(os.environ.get('LOCATE_MIN_BOX_AREA_RATIO', '0.003'))
MAX_BOX_AREA_RATIO = float(os.environ.get('LOCATE_MAX_BOX_AREA_RATIO', '0.92'))
ZONE_GAP_RATIO = float(os.environ.get('LOCATE_ZONE_GAP_RATIO', '0.05'))
MIN_HITS = int(os.environ.get('LOCATE_MIN_HITS', '2'))
MIN_HIT_RATIO = float(os.environ.get('LOCATE_MIN_HIT_RATIO', '0.5'))
MIN_COHERENT_IOU = float(os.environ.get('LOCATE_MIN_COHERENT_IOU', '0.08'))
MIN_COHERENT_PAIRS = int(os.environ.get('LOCATE_MIN_COHERENT_PAIRS', '2'))
MIN_COMPLETE_FRAMES = int(os.environ.get('LOCATE_MIN_COMPLETE_FRAMES', '3'))
MIN_LOCALIZED_HITS = int(os.environ.get('LOCATE_MIN_LOCALIZED_HITS', '2'))
MAX_LOCALIZED_BOX_AREA_RATIO = float(os.environ.get(
    'LOCATE_MAX_LOCALIZED_BOX_AREA_RATIO', '0.12'))
REQUIRE_ALL_FRAMES = os.environ.get(
    'LOCATE_REQUIRE_ALL_FRAMES', 'true').lower() != 'false'
MAX_LOAD = float(os.environ.get('LOCATE_MAX_LOAD', '3.5'))
RESULTS_MAX_ITEMS = int(os.environ.get('LOCATE_RESULTS_MAX_ITEMS', '500'))
EVIDENCE_MAX_EVENTS = int(os.environ.get('LOCATE_EVIDENCE_MAX_EVENTS', '200'))
SHADOW_MODE = os.environ.get('LOCATE_SHADOW_MODE', 'true').lower() != 'false'
CST = timezone(timedelta(hours=8))
MODE_IDS = {'hybrid': 0, 'slow': 1, 'fast': 2}


def now_iso():
    return datetime.now(CST).isoformat(timespec='seconds')


def read_json(path, default):
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        return payload if isinstance(payload, type(default)) else default
    except (OSError, ValueError, TypeError):
        return default


def write_json_atomic(path, payload):
    path = Path(path)
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
    stored = read_json(RESULTS_FILE, {'items': []})
    payload.setdefault('ok', True)
    payload.setdefault('shadow_mode', SHADOW_MODE)
    payload.setdefault('model', MODEL_ID)
    payload.setdefault('strategy_id', STRATEGY_ID)
    payload.setdefault('metrics', stored.get('metrics', {}))
    payload.setdefault('metrics_by_strategy', stored.get('metrics_by_strategy', {}))
    payload.setdefault('result_count', stored.get('current_result_count', 0))
    payload.setdefault('total_result_count', len(stored.get('items', [])))
    payload.setdefault('updated_at', now_iso())
    payload.setdefault('updated_unix', time.time())
    write_json_atomic(STATUS_FILE, payload)


def request_json(path, timeout=30):
    request = urllib.request.Request(
        f'{DASHBOARD_API_URL}{path}',
        headers={'User-Agent': 'cat-water-locate-verifier/1.0'},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def uses_direct_archive_timeline(event):
    return bool(
        event.get('auditRecovery')
        or event.get('triggerSource') == 'nightly_archive_audit'
        or event.get('recovery_trigger') == 'daily_archive_audit'
    )


def download_archive_video(event, destination):
    query = urllib.parse.urlencode({
        'id': event['id'],
        'source': 'archive',
        'verification': '0' if uses_direct_archive_timeline(event) else '1',
    })
    request = urllib.request.Request(
        f'{DASHBOARD_API_URL}/api/event-video?{query}',
        headers={'User-Agent': 'cat-water-locate-verifier/1.0'},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        with open(destination, 'wb') as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    if Path(destination).stat().st_size == 0:
        raise RuntimeError('archive video is empty')


def parse_event_time(value):
    try:
        parsed = datetime.fromisoformat(str(value).rstrip('Z'))
        return parsed.replace(tzinfo=CST) if parsed.tzinfo is None else parsed.astimezone(CST)
    except (TypeError, ValueError):
        return None


def truth_label(event):
    label = str(event.get('label') or '')
    if label == 'confirmed':
        return 'confirmed'
    if label == 'false_positive':
        return 'rejected'
    status = str(event.get('verificationStatus') or '')
    if status in ('confirmed', 'rejected', 'uncertain'):
        return status
    detail = event.get('archiveVerification')
    archive_status = str(detail.get('status') or '') if isinstance(detail, dict) else ''
    return archive_status if archive_status in ('confirmed', 'rejected', 'uncertain') else ''


def truth_source(event):
    if str(event.get('label') or '') in ('confirmed', 'false_positive'):
        return 'human'
    if truth_label(event):
        return 'automatic'
    return 'unknown'


def event_result_metadata(event):
    detail = event.get('archiveVerification')
    detail = detail if isinstance(detail, dict) else {}
    profile = str(event.get('profile') or '').strip().lower()
    return {
        'truth': truth_label(event),
        'truth_source': truth_source(event),
        'profile': profile if profile in ('day', 'night') else 'unknown',
        'trigger_source': str(event.get('triggerSource') or ''),
        'human_label': str(event.get('label') or ''),
        'verification_status': str(
            event.get('verificationStatus') or detail.get('status') or ''),
    }


def _scale_zone(zone, settings, frame_width, frame_height):
    if not isinstance(zone, dict) or not zone:
        return None
    settings_width = max(1, int(settings.get('frameWidth', frame_width) or frame_width))
    settings_height = max(1, int(settings.get('frameHeight', frame_height) or frame_height))
    scale_x = frame_width / settings_width
    scale_y = frame_height / settings_height
    x1 = max(0, min(frame_width - 1, int(float(zone.get('x1', 0)) * scale_x)))
    y1 = max(0, min(frame_height - 1, int(float(zone.get('y1', 0)) * scale_y)))
    x2 = max(x1 + 1, min(frame_width, int(float(zone.get('x2', frame_width)) * scale_x)))
    y2 = max(y1 + 1, min(frame_height, int(float(zone.get('y2', frame_height)) * scale_y)))
    return [x1, y1, x2, y2]


def box_iou(first, second):
    if not first or not second:
        return 0.0
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1.0, first_area + second_area - intersection)


def box_zone_gap(box, zone):
    dx = max(zone[0] - box[2], box[0] - zone[2], 0.0)
    dy = max(zone[1] - box[3], box[1] - zone[3], 0.0)
    return math.hypot(dx, dy)


def box_intersects_zone(box, zone):
    return min(box[2], zone[2]) > max(box[0], zone[0]) and min(box[3], zone[3]) > max(box[1], zone[1])


def sanitize_detections(detections, frame_width, frame_height, water_zone):
    sanitized = []
    frame_area = max(1, frame_width * frame_height)
    max_gap = max(frame_width, frame_height) * ZONE_GAP_RATIO
    for item in detections:
        raw_box = item.get('box') if isinstance(item, dict) else None
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            continue
        try:
            box = [float(value) for value in raw_box]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in box):
            continue
        box[0] = max(0.0, min(float(frame_width), box[0]))
        box[1] = max(0.0, min(float(frame_height), box[1]))
        box[2] = max(0.0, min(float(frame_width), box[2]))
        box[3] = max(0.0, min(float(frame_height), box[3]))
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        area_ratio = area / frame_area
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if area_ratio < MIN_BOX_AREA_RATIO or area_ratio > MAX_BOX_AREA_RATIO:
            continue
        if water_zone and not box_intersects_zone(box, water_zone) and box_zone_gap(box, water_zone) > max_gap:
            continue
        if any(box_iou(box, previous['box']) >= 0.92 for previous in sanitized):
            continue
        sanitized.append({
            'label': str(item.get('label') or '')[:160],
            'box': [round(value, 2) for value in box],
            'area_ratio': round(area_ratio, 4),
        })
    return sanitized


def coherent_pair_count(frame_results, box_filter=None):
    coherent_pairs = 0
    for index, current in enumerate(frame_results):
        current_boxes = [item['box'] for item in current.get('detections', [])
                         if box_filter is None or box_filter(item)]
        if not current_boxes:
            continue
        for previous in frame_results[:index]:
            previous_boxes = [item['box'] for item in previous.get('detections', [])
                              if box_filter is None or box_filter(item)]
            if any(box_iou(a, b) >= MIN_COHERENT_IOU for a in current_boxes for b in previous_boxes):
                coherent_pairs += 1
    return coherent_pairs


def classify_frame_results(frame_results, localized_enabled=True):
    positives = [item for item in frame_results if item.get('detections')]
    hit_ratio = len(positives) / max(1, len(frame_results))
    coherent_pairs = coherent_pair_count(positives)
    localized_filter = lambda item: (
        float(item.get('area_ratio', 1.0) or 1.0)
        <= MAX_LOCALIZED_BOX_AREA_RATIO
    )
    localized_frames = [
        frame for frame in frame_results
        if any(localized_filter(item) for item in frame.get('detections', []))
    ]
    localized_coherent_pairs = coherent_pair_count(
        localized_frames, box_filter=localized_filter)
    required_hits = min(max(1, MIN_HITS), max(1, len(frame_results)))
    legacy_support = (
        len(positives) >= required_hits
        and hit_ratio >= MIN_HIT_RATIO
        and coherent_pairs >= max(0, required_hits - 1)
    )
    enough_frames = len(frame_results) >= max(1, MIN_COMPLETE_FRAMES)
    all_frames_positive = len(positives) == len(frame_results)
    complete_support = (
        enough_frames
        and coherent_pairs >= max(0, MIN_COHERENT_PAIRS)
        and (all_frames_positive if REQUIRE_ALL_FRAMES else legacy_support)
    )
    localized_support = (
        localized_enabled
        and enough_frames
        and len(localized_frames) >= max(1, MIN_LOCALIZED_HITS)
        and localized_coherent_pairs >= max(0, MIN_LOCALIZED_HITS - 1)
    )
    strong_support = complete_support or localized_support
    if strong_support:
        verdict = 'supports_drinking'
    elif not positives:
        verdict = 'no_support'
    else:
        verdict = 'inconclusive'
    return {
        'verdict': verdict,
        'frames': len(frame_results),
        'positive_frames': len(positives),
        'hit_ratio': round(hit_ratio, 4),
        'coherent_pairs': coherent_pairs,
        'localized_frames': len(localized_frames),
        'localized_coherent_pairs': localized_coherent_pairs,
        'legacy_verdict': 'supports_drinking' if legacy_support else (
            'no_support' if not positives else 'inconclusive'),
        'decision_policy': (
            'complete_or_localized_mouth_evidence_v3'
            if localized_enabled else 'all_frames_and_spatial_pairs_v2'),
    }


def stratified_times(start, end, count):
    start = float(start)
    end = max(start, float(end))
    count = max(1, int(count))
    if count == 1 or end <= start:
        return [(start + end) / 2]
    span = end - start
    margin = min(1.0, span * 0.1)
    sample_start = start + margin
    sample_end = end - margin
    return [
        sample_start + (sample_end - sample_start) * index / (count - 1)
        for index in range(count)
    ]


def select_sample_times(duration_sec, event, count=MAX_FRAMES):
    duration_sec = max(0.0, float(duration_sec or 0))
    count = max(1, int(count))
    detail = event.get('archiveVerification') if isinstance(event, dict) else {}
    peak = float((detail or {}).get('peak_classifier_sec', 0) or 0)
    if uses_direct_archive_timeline(event):
        active_sec = float(event.get('activeDrinkSec', 0) or 0)
        active_sec = active_sec if active_sec > 0 else max(1.0, duration_sec - 25.0)
        start = min(duration_sec - 0.05, 15.0)
        end = min(duration_sec - 0.05, start + active_sec)
        times = stratified_times(start, end, count)
    elif 0 < peak < duration_sec:
        offsets = [0.0]
        step = 1.5
        while len(offsets) < count:
            distance = step * ((len(offsets) + 1) // 2)
            offsets.append(-distance if len(offsets) % 2 else distance)
        times = [max(0.0, min(duration_sec - 0.05, peak + offset)) for offset in offsets]
    elif detail:
        first_anchor = float(detail.get('first_anchor_sec', 0) or 0)
        last_anchor = float(detail.get('last_anchor_sec', 0) or 0)
        if 0 <= first_anchor <= last_anchor and last_anchor > 0:
            first_anchor = min(duration_sec - 0.05, first_anchor)
            last_anchor = min(duration_sec - 0.05, last_anchor)
            if last_anchor - first_anchor >= 4:
                times = stratified_times(first_anchor, last_anchor, count)
            else:
                center = (first_anchor + last_anchor) / 2
                start = max(0.0, center - 1.5)
                end = min(duration_sec - 0.05, center + 1.5)
                times = stratified_times(start, end, count)
        elif count == 1:
            times = [duration_sec / 2]
        else:
            start = min(duration_sec * 0.2, 20.0)
            end = max(start, duration_sec * 0.8)
            times = [start + (end - start) * index / (count - 1) for index in range(count)]
    elif count == 1:
        times = [duration_sec / 2]
    else:
        start = min(duration_sec * 0.2, 20.0)
        end = max(start, duration_sec * 0.8)
        times = [start + (end - start) * index / (count - 1) for index in range(count)]
    return sorted({round(value, 3) for value in times})


def crop_for_inference(frame, settings):
    height, width = frame.shape[:2]
    fountain_zone = _scale_zone(settings.get('fountainZone'), settings, width, height)
    water_zone = _scale_zone(settings.get('waterZone'), settings, width, height)
    if fountain_zone:
        pad_x = int((fountain_zone[2] - fountain_zone[0]) * 0.08)
        pad_y = int((fountain_zone[3] - fountain_zone[1]) * 0.18)
        crop_box = [
            max(0, fountain_zone[0] - pad_x),
            max(0, fountain_zone[1] - pad_y),
            min(width, fountain_zone[2] + pad_x),
            min(height, fountain_zone[3] + pad_y),
        ]
    else:
        crop_box = [0, 0, width, height]
    crop = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    scale = min(1.0, IMAGE_MAX_EDGE / max(crop.shape[:2]))
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return crop, crop_box, scale, water_zone


def map_detections_to_frame(detections, crop_box, scale):
    mapped = []
    for item in detections:
        box = item.get('box') if isinstance(item, dict) else None
        if not isinstance(box, list) or len(box) != 4:
            continue
        mapped.append({
            'label': item.get('label', ''),
            'box': [
                float(box[0]) / scale + crop_box[0],
                float(box[1]) / scale + crop_box[1],
                float(box[2]) / scale + crop_box[0],
                float(box[3]) / scale + crop_box[1],
            ],
        })
    return mapped


def annotate_frame(frame, detections, water_zone, verdict):
    output = frame.copy()
    if water_zone:
        cv2.rectangle(output, tuple(water_zone[:2]), tuple(water_zone[2:]), (255, 180, 0), 2)
    for item in detections:
        box = [int(round(value)) for value in item['box']]
        cv2.rectangle(output, tuple(box[:2]), tuple(box[2:]), (0, 220, 80), 3)
    cv2.putText(
        output, f'LocateAnything: {verdict}', (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2, cv2.LINE_AA,
    )
    return output


class LocateAnythingEngine:
    def __init__(self, library_path=LIBRARY_PATH, model_path=MODEL_PATH, threads=THREADS):
        self.library = ctypes.CDLL(str(library_path))
        self.library.la_capi_abi_version.restype = ctypes.c_int
        if self.library.la_capi_abi_version() != 1:
            raise RuntimeError('unsupported LocateAnything C ABI')
        self.library.la_capi_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.library.la_capi_load.restype = ctypes.c_void_p
        self.library.la_capi_free.argtypes = [ctypes.c_void_p]
        self.library.la_capi_locate_path.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
        ]
        self.library.la_capi_locate_path.restype = ctypes.c_void_p
        self.library.la_capi_free_string.argtypes = [ctypes.c_void_p]
        self.library.la_capi_last_error.argtypes = [ctypes.c_void_p]
        self.library.la_capi_last_error.restype = ctypes.c_char_p
        self.context = self.library.la_capi_load(
            str(model_path).encode('utf-8'), max(1, int(threads)))
        if not self.context:
            raise RuntimeError(f'cannot load LocateAnything model: {model_path}')

    def locate(self, image_path, prompt=PROMPT, mode=MODE):
        pointer = self.library.la_capi_locate_path(
            self.context,
            str(image_path).encode('utf-8'),
            prompt.encode('utf-8'),
            MODE_IDS.get(mode, 1),
        )
        if not pointer:
            detail = self.library.la_capi_last_error(self.context)
            raise RuntimeError((detail or b'LocateAnything inference failed').decode('utf-8', 'replace'))
        try:
            return json.loads(ctypes.string_at(pointer).decode('utf-8'))
        finally:
            self.library.la_capi_free_string(pointer)

    def close(self):
        if getattr(self, 'context', None):
            self.library.la_capi_free(self.context)
            self.context = None


def extract_frames(video_path, event, output_dir, sample_times=None):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'cannot open video: {video_path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else 0.0
    sample_times = (
        select_sample_times(duration, event)
        if sample_times is None else sorted({
            round(max(0.0, min(duration - 0.05, float(value))), 3)
            for value in sample_times
        })
    )
    settings = read_json(SETTINGS_FILE, {})
    samples = []
    try:
        for index, sample_time in enumerate(sample_times):
            capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            crop, crop_box, scale, water_zone = crop_for_inference(frame, settings)
            path = output_dir / f'input-{index:02d}.jpg'
            if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                continue
            samples.append({
                'index': index,
                'time_sec': round(sample_time, 3),
                'frame': frame,
                'input_path': path,
                'crop_box': crop_box,
                'scale': scale,
                'water_zone': water_zone,
            })
    finally:
        capture.release()
    if not samples:
        raise RuntimeError('no frames could be extracted from archive video')
    return samples


def process_event(
    event, engine, prompt=PROMPT, evidence_root=EVIDENCE_DIR,
    update_runtime_status=True, sample_times=None,
    strategy_id=STRATEGY_ID, evidence_url_prefix='/media/locate_verifier',
):
    event_id = str(event['id'])
    safe_id = re.sub(r'[^A-Za-z0-9_.-]+', '-', event_id)[:120]
    started = time.monotonic()
    temporary = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    video_path = Path(temporary.name)
    temporary.close()
    strategy_slug = re.sub(r'[^A-Za-z0-9_.-]+', '-', strategy_id)[:80]
    evidence_name = f'{safe_id}--{strategy_slug}'
    event_dir = Path(evidence_root) / evidence_name
    event_dir.mkdir(parents=True, exist_ok=True)
    try:
        download_archive_video(event, video_path)
        samples = extract_frames(
            video_path, event, event_dir, sample_times=sample_times)
        frame_results = []
        for sequence, sample in enumerate(samples, 1):
            if update_runtime_status:
                write_status(
                    phase='inferencing', event_id=event_id,
                    frame=sequence, frame_total=len(samples),
                )
            inference_started = time.monotonic()
            raw = engine.locate(sample['input_path'], prompt=prompt)
            raw_detections = raw.get('detections', []) if isinstance(raw, dict) else []
            mapped = map_detections_to_frame(
                raw_detections, sample['crop_box'], sample['scale'])
            frame = sample['frame']
            height, width = frame.shape[:2]
            detections = sanitize_detections(
                mapped, width, height, sample['water_zone'])
            frame_results.append({
                'time_sec': sample['time_sec'],
                'detections': detections,
                'raw_detection_count': len(raw_detections),
                'inference_sec': round(time.monotonic() - inference_started, 3),
            })
        summary = classify_frame_results(frame_results)
        for sample, detail in zip(samples, frame_results):
            annotated = annotate_frame(
                sample['frame'], detail['detections'], sample['water_zone'], summary['verdict'])
            evidence_path = event_dir / f'frame-{sample["index"]:02d}.jpg'
            cv2.imwrite(str(evidence_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
            detail['evidence_url'] = (
                f'{evidence_url_prefix.rstrip("/")}/'
                f'{evidence_name}/{evidence_path.name}')
            sample['input_path'].unlink(missing_ok=True)
        result = {
            'event_id': event_id,
            'event_time': event.get('start_time', ''),
            **event_result_metadata(event),
            **summary,
            'frame_results': frame_results,
            'model': MODEL_ID,
            'strategy_id': strategy_id,
            'mode': MODE,
            'prompt': prompt,
            'image_max_edge': IMAGE_MAX_EDGE,
            'threads': THREADS,
            'shadow_mode': SHADOW_MODE,
            'completed_at': now_iso(),
            'elapsed_sec': round(time.monotonic() - started, 3),
        }
        write_json_atomic(event_dir / 'result.json', result)
        return result
    finally:
        video_path.unlink(missing_ok=True)


def _safe_ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def _core_metrics(items):
    labeled = [item for item in items
               if item.get('truth') in ('confirmed', 'rejected')]
    tp = sum(item.get('truth') == 'confirmed'
             and item.get('verdict') == 'supports_drinking' for item in labeled)
    fp = sum(item.get('truth') == 'rejected'
             and item.get('verdict') == 'supports_drinking' for item in labeled)
    fn = sum(item.get('truth') == 'confirmed'
             and item.get('verdict') != 'supports_drinking' for item in labeled)
    tn = sum(item.get('truth') == 'rejected'
             and item.get('verdict') != 'supports_drinking' for item in labeled)
    inconclusive = sum(item.get('verdict') == 'inconclusive' for item in labeled)
    no_support = sum(item.get('verdict') == 'no_support' for item in labeled)
    supports = sum(item.get('verdict') == 'supports_drinking' for item in labeled)
    decisive = [item for item in labeled
                if item.get('verdict') in ('supports_drinking', 'no_support')]
    decisive_tp = sum(item.get('truth') == 'confirmed'
                      and item.get('verdict') == 'supports_drinking' for item in decisive)
    decisive_fp = sum(item.get('truth') == 'rejected'
                      and item.get('verdict') == 'supports_drinking' for item in decisive)
    decisive_fn = sum(item.get('truth') == 'confirmed'
                      and item.get('verdict') == 'no_support' for item in decisive)
    decisive_tn = sum(item.get('truth') == 'rejected'
                      and item.get('verdict') == 'no_support' for item in decisive)
    confirmed_total = sum(item.get('truth') == 'confirmed' for item in labeled)
    rejected_total = sum(item.get('truth') == 'rejected' for item in labeled)
    evaluated = len(labeled)
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'inconclusive': inconclusive,
        'supports_drinking': supports,
        'no_support': no_support,
        'evaluated': evaluated,
        'unlabeled': len(items) - evaluated,
        'precision': _safe_ratio(tp, tp + fp),
        'recall': _safe_ratio(tp, confirmed_total),
        'accuracy': _safe_ratio(tp + tn, evaluated),
        'decision_coverage': _safe_ratio(len(decisive), evaluated),
        'confirmed_support_rate': _safe_ratio(tp, confirmed_total),
        'rejected_false_support_rate': _safe_ratio(fp, rejected_total),
        'decisive': {
            'tp': decisive_tp,
            'fp': decisive_fp,
            'tn': decisive_tn,
            'fn': decisive_fn,
            'evaluated': len(decisive),
            'precision': _safe_ratio(decisive_tp, decisive_tp + decisive_fp),
            'recall': _safe_ratio(decisive_tp, decisive_tp + decisive_fn),
            'accuracy': _safe_ratio(
                decisive_tp + decisive_tn, len(decisive)),
        },
    }


def benchmark_metrics(items):
    metrics = _core_metrics(items)
    truth_sources = sorted({str(item.get('truth_source') or 'unknown') for item in items})
    profiles = sorted({str(item.get('profile') or 'unknown') for item in items})
    metrics['by_truth_source'] = {
        source: _core_metrics([
            item for item in items
            if str(item.get('truth_source') or 'unknown') == source
        ]) for source in truth_sources
    }
    metrics['by_profile'] = {
        profile: _core_metrics([
            item for item in items
            if str(item.get('profile') or 'unknown') == profile
        ]) for profile in profiles
    }
    return metrics


def refresh_stored_results(events):
    payload = read_json(RESULTS_FILE, {'version': 1, 'items': []})
    event_map = {str(event.get('id') or ''): event for event in events}
    changed = False
    for item in payload.get('items', []):
        before = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if not item.get('strategy_id'):
            item['strategy_id'] = LEGACY_STRATEGY_ID
        event = event_map.get(str(item.get('event_id') or ''))
        if event:
            item.update(event_result_metadata(event))
        if isinstance(item.get('frame_results'), list):
            item.update(classify_frame_results(
                item['frame_results'],
                localized_enabled=item.get('strategy_id') == STRATEGY_ID,
            ))
        changed = changed or before != json.dumps(item, ensure_ascii=False, sort_keys=True)
    items = payload.get('items', [])
    current_items = [item for item in items
                     if item.get('strategy_id') == STRATEGY_ID]
    metrics = benchmark_metrics(current_items)
    if payload.get('metrics') != metrics:
        payload['metrics'] = metrics
        changed = True
    strategies = sorted({str(item.get('strategy_id') or LEGACY_STRATEGY_ID)
                         for item in items})
    metrics_by_strategy = {
        strategy: benchmark_metrics([
            item for item in items if str(
                item.get('strategy_id') or LEGACY_STRATEGY_ID) == strategy
        ]) for strategy in strategies
    }
    if payload.get('metrics_by_strategy') != metrics_by_strategy:
        payload['metrics_by_strategy'] = metrics_by_strategy
        changed = True
    if payload.get('current_result_count') != len(current_items):
        payload['current_result_count'] = len(current_items)
        changed = True
    if changed:
        payload['current_strategy_id'] = STRATEGY_ID
        payload['decision_policy'] = 'complete_or_localized_mouth_evidence_v3'
        payload['updated_at'] = now_iso()
        write_json_atomic(RESULTS_FILE, payload)
    return metrics


def store_result(result):
    payload = read_json(RESULTS_FILE, {'version': 1, 'items': []})
    items = [item for item in payload.get('items', [])
             if not (
                 item.get('event_id') == result.get('event_id')
                 and item.get('strategy_id') == result.get('strategy_id')
             )]
    items.insert(0, result)
    current_items = [item for item in items
                     if item.get('strategy_id') == STRATEGY_ID]
    strategies = sorted({str(item.get('strategy_id') or LEGACY_STRATEGY_ID)
                         for item in items})
    payload.update({
        'version': 1,
        'model': MODEL_ID,
        'prompt': PROMPT,
        'current_strategy_id': STRATEGY_ID,
        'decision_policy': 'complete_or_localized_mouth_evidence_v3',
        'shadow_mode': SHADOW_MODE,
        'updated_at': now_iso(),
        'items': items[:max(1, RESULTS_MAX_ITEMS)],
        'current_result_count': len(current_items),
    })
    payload['metrics'] = benchmark_metrics(current_items)
    payload['metrics_by_strategy'] = {
        strategy: benchmark_metrics([
            item for item in payload['items']
            if str(item.get('strategy_id') or LEGACY_STRATEGY_ID) == strategy
        ]) for strategy in strategies
    }
    write_json_atomic(RESULTS_FILE, payload)
    cleanup_evidence(payload['items'])
    return payload['metrics']


def cleanup_evidence(items):
    keep = set()
    for item in items[:max(1, EVIDENCE_MAX_EVENTS)]:
        legacy = re.sub(
            r'[^A-Za-z0-9_.-]+', '-', str(item.get('event_id', '')))[:120]
        keep.add(legacy)
        for frame in item.get('frame_results', []):
            url = str(frame.get('evidence_url') or '')
            marker = '/media/locate_verifier/'
            if marker in url:
                keep.add(url.split(marker, 1)[1].split('/', 1)[0])
    if not EVIDENCE_DIR.exists():
        return
    for path in EVIDENCE_DIR.iterdir():
        if not path.is_dir() or path.name in keep:
            continue
        for child in path.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
        try:
            path.rmdir()
        except OSError:
            pass


def load_state():
    return read_json(STATE_FILE, {'version': 1, 'events': {}})


def eligible_events(events, state, limit=MAX_PER_POLL):
    completed = state.setdefault('events', {})
    cutoff = datetime.now(CST) - timedelta(days=max(1.0, LOOKBACK_DAYS))
    candidates = []
    for event in events:
        event_id = str(event.get('id') or '')
        truth = truth_label(event)
        event_time = parse_event_time(event.get('end_time') or event.get('start_time'))
        record = completed.get(event_id, {})
        if not event_id or not truth or not event_time or event_time < cutoff:
            continue
        if (
            record.get('model') == MODEL_ID
            and record.get('strategy_id') == STRATEGY_ID
            and record.get('completed_at')
        ):
            continue
        same_strategy = record.get('strategy_id') == STRATEGY_ID
        if same_strategy and int(record.get('attempts', 0) or 0) >= MAX_ATTEMPTS:
            continue
        human_labeled = event.get('label') in ('confirmed', 'false_positive')
        comparable = truth in ('confirmed', 'rejected')
        priority = 2 if human_labeled else 1 if comparable else 0
        candidates.append((priority, event_time, event))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [event for _, _, event in candidates[:max(0, int(limit))]]


def system_busy():
    try:
        daily = request_json('/api/daily-audit-status', timeout=10)
        if daily.get('phase') in ('scanning', 'analyzing'):
            return True, f'daily_audit_{daily.get("phase")}'
    except Exception:
        pass
    try:
        load_1m = os.getloadavg()[0]
        if load_1m > MAX_LOAD:
            return True, f'load_{load_1m:.2f}'
    except OSError:
        pass
    return False, ''


def run_once(engine, state, limit=MAX_PER_POLL, event_id=''):
    if not event_id:
        since_last_run = time.time() - float(state.get('last_run_unix', 0) or 0)
        if since_last_run < MIN_INTERVAL_SEC:
            remaining = max(1, int(math.ceil(MIN_INTERVAL_SEC - since_last_run)))
            write_status(phase='deferred', reason='cooldown', retry_after_sec=remaining)
            return 0
    busy, reason = system_busy()
    if busy:
        write_status(phase='deferred', reason=reason)
        return 0
    payload = request_json('/api/events?include_archived=1&limit=5000', timeout=30)
    events = payload.get('events', [])
    metrics = refresh_stored_results(events)
    if event_id:
        pending = [event for event in events if str(event.get('id') or '') == event_id]
        if not pending:
            raise ValueError(f'event not found: {event_id}')
    else:
        pending = eligible_events(events, state, limit=limit)
    processed = 0
    failed = 0
    owns_engine = engine is None and bool(pending)
    if owns_engine:
        write_status(phase='loading_model')
        engine = LocateAnythingEngine()
    try:
        for event in pending:
            event_id = str(event['id'])
            record = state.setdefault('events', {}).setdefault(event_id, {})
            if record.get('strategy_id') != STRATEGY_ID:
                record.clear()
                record['attempts'] = 0
            record['attempts'] = int(record.get('attempts', 0) or 0) + 1
            record['last_attempt_at'] = now_iso()
            record['model'] = MODEL_ID
            record['strategy_id'] = STRATEGY_ID
            write_json_atomic(STATE_FILE, state)
            try:
                write_status(phase='downloading', event_id=event_id)
                result = process_event(event, engine)
                metrics = store_result(result)
                record.update({
                    'completed_at': result['completed_at'],
                    'verdict': result['verdict'],
                    'elapsed_sec': result['elapsed_sec'],
                })
                record.pop('last_error', None)
                processed += 1
                print(f'LOCATE VERIFIED {event_id}: {json.dumps(result, ensure_ascii=False)}', flush=True)
            except Exception as exc:
                failed += 1
                record['last_error'] = str(exc)[:240]
                print(f'LOCATE FAILED {event_id}: {exc}', flush=True)
                write_status(ok=False, phase='event_failed', event_id=event_id, error=str(exc)[:240])
            write_json_atomic(STATE_FILE, state)
    finally:
        if owns_engine:
            engine.close()
    if pending:
        state['last_run_unix'] = time.time()
        state['last_run_at'] = now_iso()
        write_json_atomic(STATE_FILE, state)
    write_status(
        ok=failed == 0, phase='idle' if failed == 0 else 'idle_with_errors',
        pending=max(0, len(pending) - processed - failed), processed=processed,
        failed=failed, tracked=len(state.get('events', {})), metrics=metrics,
    )
    return processed


def representative_sample_time(result):
    frames = result.get('frame_results', []) if isinstance(result, dict) else []
    positive = [frame for frame in frames if frame.get('detections')]
    candidates = positive or frames
    if not candidates:
        return None
    return float(candidates[len(candidates) // 2].get('time_sec', 0) or 0)


def run_prompt_benchmark(
    event_ids, prompt, output_file, evidence_root,
    one_frame=False,
):
    payload = request_json('/api/events?include_archived=1&limit=5000', timeout=30)
    event_map = {str(event.get('id') or ''): event
                 for event in payload.get('events', [])}
    stored = read_json(RESULTS_FILE, {'items': []})
    source_map = {str(item.get('event_id') or ''): item
                  for item in stored.get('items', [])}
    missing = [event_id for event_id in event_ids if event_id not in event_map]
    if missing:
        raise ValueError(f'benchmark events not found: {", ".join(missing)}')
    engine = LocateAnythingEngine()
    results = []
    try:
        for index, event_id in enumerate(event_ids, 1):
            write_status(
                phase='benchmarking_prompt', event_id=event_id,
                benchmark_index=index, benchmark_total=len(event_ids),
            )
            selected_times = None
            source = source_map.get(event_id, {})
            if one_frame:
                selected_time = representative_sample_time(source)
                if selected_time is None:
                    raise ValueError(f'no stored frame time for benchmark event: {event_id}')
                selected_times = [selected_time]
            result = process_event(
                event_map[event_id], engine, prompt=prompt,
                evidence_root=Path(evidence_root), update_runtime_status=False,
                sample_times=selected_times,
                strategy_id='prompt-benchmark-mouth-contact-v2',
                evidence_url_prefix='/media/locate_prompt_benchmark',
            )
            result['source_prompt'] = source.get('prompt', '')
            result['source_verdict'] = source.get('verdict', '')
            result['source_legacy_verdict'] = source.get('legacy_verdict', '')
            results.append(result)
            print(
                f'LOCATE PROMPT BENCHMARK {event_id}: '
                f'{result["positive_frames"]}/{result["frames"]} positive',
                flush=True,
            )
    finally:
        engine.close()
    benchmark = {
        'version': 1,
        'model': MODEL_ID,
        'strategy_id': 'prompt-benchmark-mouth-contact-v2',
        'prompt': prompt,
        'one_frame': bool(one_frame),
        'created_at': now_iso(),
        'items': results,
        'metrics': benchmark_metrics(results),
    }
    write_json_atomic(Path(output_file), benchmark)
    write_status(
        phase='idle', detail='prompt benchmark completed',
        benchmark_output=str(output_file),
    )
    return benchmark


def run_forever():
    if MODE not in MODE_IDS:
        raise ValueError('LOCATE_MODE must be hybrid, slow, or fast')
    if not MODEL_PATH.is_file():
        write_status(ok=False, phase='waiting_model', error=f'model not found: {MODEL_PATH}')
        raise FileNotFoundError(MODEL_PATH)
    state = load_state()
    write_status(phase='idle', detail='model loads only for eligible events')
    while True:
        try:
            run_once(None, state)
        except Exception as exc:
            print(f'LOCATE POLL FAILED: {exc}', flush=True)
            write_status(ok=False, phase='poll_failed', error=str(exc)[:240])
        time.sleep(max(10, POLL_SEC))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--limit', type=int, default=MAX_PER_POLL)
    parser.add_argument('--event-id', default='')
    parser.add_argument('--refresh-results', action='store_true')
    parser.add_argument('--benchmark-event-id', action='append', default=[])
    parser.add_argument('--benchmark-prompt', default=PROMPT)
    parser.add_argument(
        '--benchmark-output', default='/app/data/locate_prompt_benchmark.json')
    parser.add_argument(
        '--benchmark-evidence-dir',
        default='/app/media/locate_prompt_benchmark')
    parser.add_argument('--benchmark-one-frame', action='store_true')
    args = parser.parse_args()
    if args.benchmark_event_id:
        run_prompt_benchmark(
            [str(value) for value in args.benchmark_event_id],
            str(args.benchmark_prompt), str(args.benchmark_output),
            str(args.benchmark_evidence_dir),
            one_frame=bool(args.benchmark_one_frame),
        )
        return
    if args.refresh_results:
        payload = request_json('/api/events?include_archived=1&limit=5000', timeout=30)
        metrics = refresh_stored_results(payload.get('events', []))
        write_status(phase='idle', detail='stored results refreshed', metrics=metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        return
    if not args.once:
        run_forever()
        return
    run_once(
        None, load_state(), limit=max(0, args.limit),
        event_id=str(args.event_id or ''),
    )


if __name__ == '__main__':
    main()
