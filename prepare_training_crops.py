#!/usr/bin/env python3
"""Create classifier-compatible cat crops for human-labeled events."""
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import cv2

import detection_worker as detector_mod


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'data' / 'mock.json'
REPORT_FILE = BASE_DIR / 'data' / 'training_crop_report.json'
MEDIA_DIR = BASE_DIR / 'media'
EVENT_DIR = MEDIA_DIR / 'events'
VALID_LABELS = {'confirmed', 'false_positive'}


def write_json_atomic(path, payload):
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=path.parent,
        prefix=f'.{path.name}.', suffix='.tmp', delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def write_jpeg_atomic(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f'.{path.stem}.{os.getpid()}.jpg')
    if not cv2.imwrite(str(temp_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        temp_path.unlink(missing_ok=True)
        return False
    os.replace(temp_path, path)
    return True


def local_media_path(url):
    if not isinstance(url, str) or not url.startswith('/media/'):
        return None
    candidate = (BASE_DIR / url.lstrip('/')).resolve()
    return candidate if candidate.is_relative_to(MEDIA_DIR.resolve()) else None


def event_crop_path(event_id):
    safe_id = str(event_id or '')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,180}', safe_id):
        return None
    return EVENT_DIR / f'crop-{safe_id}.jpg'


def detect_crop(detector, frame, settings):
    height, width = frame.shape[:2]
    fountain_zone = detector_mod.get_fountain_zone(settings, width, height)
    water_zone = detector_mod.get_water_zone(settings, width, height)
    detector._consecutive_cat = 0
    result = None
    for _ in range(max(1, detector_mod.CONSECUTIVE_FRAMES_REQUIRED)):
        result = detector.detect(frame, force=True, fzone=fountain_zone, wzone=water_zone)
    box = result.get('cat_box', []) if result else []
    if not box or len(box) != 4:
        return None, result or {}
    x1, y1, x2, y2 = [int(value) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, result
    crop = frame[y1:y2, x1:x2]
    return crop, {**result, 'cat_box': [x1, y1, x2, y2]}


def main():
    data = json.loads(DATA_FILE.read_text(encoding='utf-8'))
    settings = detector_mod.read_settings()
    detector = detector_mod.CatDetector()
    records = []
    for event in data.get('events', []):
        if event.get('label') not in VALID_LABELS:
            continue
        crop_path = event_crop_path(event.get('id'))
        source_path = local_media_path(event.get('img'))
        record = {'event_id': event.get('id'), 'label': event.get('label')}
        if crop_path is None:
            record['status'] = 'skipped_unsafe_event_id'
        elif crop_path.is_file():
            record['status'] = 'existing_crop'
            record['crop_url'] = f'/media/events/{crop_path.name}'
        elif source_path is None or not source_path.is_file():
            record['status'] = 'missing_snapshot'
        else:
            frame = cv2.imread(str(source_path))
            crop, result = detect_crop(detector, frame, settings) if frame is not None else (None, {})
            if crop is None:
                record['status'] = 'cat_not_found'
                record['cat_score'] = result.get('cat_score', 0)
            elif write_jpeg_atomic(crop_path, crop):
                record['status'] = 'created'
                record['crop_url'] = f'/media/events/{crop_path.name}'
                record['cat_box'] = result.get('cat_box', [])
                record['cat_score'] = result.get('cat_score', 0)
            else:
                record['status'] = 'write_failed'
        records.append(record)

    counts = {}
    for record in records:
        counts[record['status']] = counts.get(record['status'], 0) + 1
    report = {
        'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'records': records,
        'counts': counts,
    }
    write_json_atomic(REPORT_FILE, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
