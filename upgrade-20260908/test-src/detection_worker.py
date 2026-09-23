#!/usr/bin/env python3
"""Cat drinking detection worker - simplified.
YOLO detects cat. If cat stays in fountain zone > threshold, record drinking event.
Motion pre-filter skips inference when nothing changes, saving ~90% CPU.

Night mode: auto-detects IR/night footage via saturation analysis, applies CLAHE
contrast enhancement, and lowers detection thresholds for improved nighttime accuracy.
"""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
import onnxruntime as ort

# ── Config from env ──────────────────────────────────────────────
STREAM_URL = os.environ.get('CAMERA_ANALYSIS_STREAM_URL', 'http://cat-water-qsv-transcode:39100/live.flv')
SETTINGS_FILE = Path(os.environ.get('SETTINGS_FILE', '/app/data/settings.json'))
STATUS_FILE = Path(os.environ.get('DETECTOR_STATUS_FILE', '/app/data/detector_status.json'))
EVENT_DIR = Path(os.environ.get('DETECTOR_EVENT_DIR', '/app/media/events'))
DASHBOARD_API_URL = os.environ.get('DASHBOARD_API_URL', 'http://cat-water-dashboard:39095')
API_TOKEN = os.environ.get('CAT_WATER_API_TOKEN', '').strip()
MODEL_PATH = os.environ.get('CAT_DETECTOR_MODEL_PATH', '/app/models/yolov8n_640_opset21.onnx')

def get_active_model_path():
    """Read active model from settings.json, fallback to env var / default."""
    settings_path = Path('/app/data/settings.json')
    if settings_path.exists():
        try:
            settings_data = json.loads(settings_path.read_text())
            active = settings_data.get('activeModel', '')
            if active:
                candidate = Path('/app/models') / active
                if candidate.exists():
                    print(f"Model from settings: {candidate}")
                    return str(candidate)
        except Exception:
            pass
    # Fallback to env or default
    default = os.environ.get('CAT_DETECTOR_MODEL_PATH', '/app/models/yolov8n_640_opset21.onnx')
    print(f"Model from default: {default}")
    return default

# High-res RTSP for night mode (bypasses transcoded FLV to get original 2560x1440)
NIGHT_HIRES_RTSP_URL = os.environ.get('NIGHT_HIRES_RTSP_URL', '')
NIGHT_HIRES_ENABLED = os.environ.get('NIGHT_HIRES_ENABLED', 'true').lower() != 'false'
NIGHT_HIRES_TIMEOUT_SEC = float(os.environ.get('NIGHT_HIRES_TIMEOUT_SEC', '3.0'))
# Night hires crop: width ratio of frame (0.50 = 1280px → 2x zoom, covers half frame)
NIGHT_HIRES_CROP_WIDTH = float(os.environ.get('NIGHT_HIRES_CROP_WIDTH', '0.50'))
NIGHT_HIRES_CROP_HEIGHT = float(os.environ.get('NIGHT_HIRES_CROP_HEIGHT', '0.80'))
# Night false-positive filter: reject boxes larger than this ratio of frame area
NIGHT_MAX_BOX_AREA_RATIO = float(os.environ.get('NIGHT_MAX_BOX_AREA_RATIO', '0.15'))
# Night MOG2 background subtraction for water zone detection
NIGHT_MOG2_HISTORY = int(os.environ.get('NIGHT_MOG2_HISTORY', '300'))
NIGHT_MOG2_VAR_THRESHOLD = int(os.environ.get('NIGHT_MOG2_VAR_THRESHOLD', '16'))
NIGHT_WATER_BLOB_MIN = int(os.environ.get('NIGHT_WATER_BLOB_MIN_AREA', '80'))
NIGHT_WATER_BLOB_MAX = int(os.environ.get('NIGHT_WATER_BLOB_MAX_AREA', '800'))

# Water surface proximity: cat head within this many pixels of water line = at water
WATER_PROXIMITY_PX = float(os.environ.get('WATER_PROXIMITY_PX', '50'))
# Lower threshold for water zone crop YOLO (more sensitive than general cat detection)
WATER_ZONE_CROP_THRESHOLD = float(os.environ.get('WATER_ZONE_CROP_THRESHOLD', '0.18'))
# Night water zone: lower foreground pixel threshold (was 120)
NIGHT_WATER_FG_MIN = int(os.environ.get('NIGHT_WATER_FG_MIN', '80'))
# Night water zone: fewer background frames needed before detection starts (was 50)
NIGHT_WATER_BG_FRAMES = int(os.environ.get('NIGHT_WATER_BG_FRAMES', '30'))
# Prolonged near-presence is retained as diagnostic state only. It must never
# become a drinking event without classifier or geometric evidence.
PROLONGED_NEAR_SEC = float(os.environ.get('PROLONGED_NEAR_SEC', '15.0'))
NEAR_WATER_SCORE_MIN = float(os.environ.get('NEAR_WATER_SCORE_MIN', '0.40'))
# Body posture: drinking cats crouch → wider bbox. Standing/walking cats are taller.
DRINKING_POSTURE_MIN_RATIO = float(os.environ.get('DRINKING_POSTURE_MIN_RATIO', '0.65'))

CAT_SCORE_THRESHOLD = float(os.environ.get('CAT_SCORE_THRESHOLD', '0.40'))
CAT_CROP_SCORE_THRESHOLD = float(os.environ.get('CAT_CROP_SCORE_THRESHOLD', '0.25'))
DETECT_INTERVAL_SEC = float(os.environ.get('DETECTOR_INTERVAL_SEC', '1.0'))
CAT_MIN_BOX_AREA_RATIO = float(os.environ.get('CAT_MIN_BOX_AREA_RATIO', '0.005'))
BOTTOM_CROP_START_RATIO = float(os.environ.get('BOTTOM_CROP_START_RATIO', '0.45'))
RECONNECT_SEC = float(os.environ.get('DETECTOR_RECONNECT_SEC', '4'))
STREAM_OPEN_TIMEOUT_SEC = float(os.environ.get('DETECTOR_STREAM_OPEN_TIMEOUT_SEC', '10'))
STREAM_READ_TIMEOUT_SEC = float(os.environ.get('DETECTOR_STREAM_READ_TIMEOUT_SEC', '8'))

# Motion pre-filter
MOTION_THRESHOLD = float(os.environ.get('DETECTOR_MOTION_THRESHOLD', '6.0'))
MOTION_RESIZE = (320, 180)

# Drinking thresholds
CAT_NEAR_OVERLAP = float(os.environ.get('CAT_NEAR_OVERLAP_THRESHOLD', '0.03'))
CAT_NEAR_PROXIMITY = float(os.environ.get('CAT_NEAR_PROXIMITY_PX', '50'))
DRINK_MIN_SEC = float(os.environ.get('DRINK_MIN_DURATION_SEC', '3.0'))
NIGHT_DRINK_MIN_SEC = float(os.environ.get("NIGHT_DRINK_MIN_DURATION_SEC", "8.0"))
NIGHT_DRINK_CONF_THRESHOLD = float(os.environ.get("NIGHT_DRINK_CONF_THRESHOLD", "0.35"))
GEOMETRIC_DRINK_MIN_SEC = float(os.environ.get('GEOMETRIC_DRINK_MIN_DURATION_SEC', '6.0'))
NIGHT_GEOMETRIC_DRINK_MIN_SEC = float(os.environ.get('NIGHT_GEOMETRIC_DRINK_MIN_DURATION_SEC', '10.0'))
WATER_OCCUPIED_DRINK_MIN_SEC = float(os.environ.get('WATER_OCCUPIED_DRINK_MIN_DURATION_SEC', '5.0'))
NIGHT_WATER_OCCUPIED_DRINK_MIN_SEC = float(os.environ.get('NIGHT_WATER_OCCUPIED_DRINK_MIN_DURATION_SEC', '10.0'))
CLOSEUP_DRINK_MIN_SEC = float(os.environ.get('CLOSEUP_DRINK_MIN_DURATION_SEC', '5.0'))
NIGHT_CLOSEUP_DRINK_MIN_SEC = float(os.environ.get('NIGHT_CLOSEUP_DRINK_MIN_DURATION_SEC', '8.0'))
DRINK_END_GAP_SEC = float(os.environ.get('DRINK_END_GAP_SEC', '5.0'))
STALE_DETECTION_SEC = float(os.environ.get('STALE_DETECTION_SEC', '3.0'))
WATER_OCCUPIED_CLASSIFIER_FLOOR = float(os.environ.get('WATER_OCCUPIED_CLASSIFIER_FLOOR', '0.30'))
NIGHT_WATER_OCCUPIED_CLASSIFIER_FLOOR = float(os.environ.get('NIGHT_WATER_OCCUPIED_CLASSIFIER_FLOOR', '0.45'))

# Night mode
NIGHT_ENABLED = os.environ.get('NIGHT_MODE_ENABLED', 'true').lower() != 'false'
NIGHT_SAT_THRESHOLD = float(os.environ.get('NIGHT_SATURATION_THRESHOLD', '20'))
NIGHT_VAL_THRESHOLD = float(os.environ.get('NIGHT_BRIGHTNESS_THRESHOLD', '80'))
NIGHT_GRAY_DELTA_THRESHOLD = float(os.environ.get('NIGHT_GRAY_DELTA_THRESHOLD', '12'))
NIGHT_CAT_SCORE_THRESHOLD = float(os.environ.get('NIGHT_CAT_SCORE_THRESHOLD', '0.28'))
NIGHT_CAT_CROP_THRESHOLD = float(os.environ.get('NIGHT_CAT_CROP_SCORE_THRESHOLD', '0.15'))
NIGHT_MOTION_THRESHOLD = float(os.environ.get('NIGHT_MOTION_THRESHOLD', '4.0'))
NIGHT_FORCE_INFERENCE_SEC = float(os.environ.get('NIGHT_FORCE_INFERENCE_SEC', '3.0'))
NIGHT_CLAHE_CLIP = float(os.environ.get('NIGHT_CLAHE_CLIP_LIMIT', '2.0'))
# Night center-anchored crop: square crop size as ratio of frame width (0.50 = 480px square on 960px frame → 2x zoom after 640x640 letterbox)
NIGHT_CROP_SIZE_RATIO = float(os.environ.get('NIGHT_CROP_SIZE_RATIO', '0.50'))

# ONNX provider
PROVIDER_PREF = [s.strip().lower() for s in os.environ.get('CAT_DETECT_PROVIDER_PREFERENCE', 'cpu').split(',') if s.strip()]
OPENVINO_DEVICE = os.environ.get('CAT_DETECT_OPENVINO_DEVICE', 'AUTO')
YOLO_IMG_SIZE = 640
COCO_CAT_LABEL = 15
COCO_PERSON_LABEL = 0

# Person rejection: when YOLO sees both cat and person features, reject detections
# where person score dominates — prevents false triggers from people walking by.
PERSON_REJECTION_ENABLED = os.environ.get('PERSON_REJECTION_ENABLED', 'true').lower() != 'false'
PERSON_SCORE_REJECT_THRESHOLD = float(os.environ.get('PERSON_SCORE_REJECT_THRESHOLD', '0.30'))

# Consecutive frame confirmation: require N consecutive positive frames before
# reporting cat_detected=True. Filters single-frame false positives.
CONSECUTIVE_FRAMES_REQUIRED = int(os.environ.get('CONSECUTIVE_FRAMES_REQUIRED', '2'))
CAT_TRACK_MIN_IOU = float(os.environ.get('CAT_TRACK_MIN_IOU', '0.15'))
CLOSEUP_CAT_SCORE_THRESHOLD = float(os.environ.get('CLOSEUP_CAT_SCORE_THRESHOLD', '0.28'))
CLOSEUP_CAT_MIN_BOX_AREA_RATIO = float(os.environ.get('CLOSEUP_CAT_MIN_BOX_AREA_RATIO', '0.12'))
CLOSEUP_CAT_MAX_BOX_AREA_RATIO = float(os.environ.get('CLOSEUP_CAT_MAX_BOX_AREA_RATIO', '0.88'))
CLOSEUP_WATER_PROXIMITY_PX = float(os.environ.get('CLOSEUP_WATER_PROXIMITY_PX', '180'))

# Drinking classifier: when available, replaces geometric position-based drinking detection
# with a trained binary classifier (drinking / not_drinking) on the cat crop.
DRINKING_CLASSIFIER_PATH = os.environ.get('DRINKING_CLASSIFIER_PATH', '/app/models/cat_drinking_classifier.onnx')
DRINKING_CALIBRATION_PATH = Path(os.environ.get(
    'DRINKING_CALIBRATION_PATH', '/app/data/classifier_calibration.json'))
DRINKING_CLASSIFIER_THRESHOLD = float(os.environ.get('DRINKING_CLASSIFIER_THRESHOLD', '0.6'))
NIGHT_DRINKING_CLASSIFIER_THRESHOLD = float(os.environ.get('NIGHT_DRINKING_CLASSIFIER_THRESHOLD', '0.75'))
CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD = float(os.environ.get('CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD', '0.90'))
NIGHT_CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD = float(os.environ.get('NIGHT_CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD', '0.94'))
CLASSIFIER_TEMPORAL_WINDOW_SEC = float(os.environ.get('CLASSIFIER_TEMPORAL_WINDOW_SEC', '1.2'))
CLASSIFIER_TEMPORAL_MIN_HITS = int(os.environ.get('CLASSIFIER_TEMPORAL_MIN_HITS', '2'))
CLASSIFIER_TEMPORAL_MIN_IOU = float(os.environ.get('CLASSIFIER_TEMPORAL_MIN_IOU', '0.20'))
CLASSIFIER_TEMPORAL_HOLD_SEC = float(os.environ.get('CLASSIFIER_TEMPORAL_HOLD_SEC', '2.5'))
CLASSIFIER_TEMPORAL_MIN_SEC = float(os.environ.get('CLASSIFIER_TEMPORAL_MIN_DURATION_SEC', '1.5'))
NIGHT_CLASSIFIER_TEMPORAL_MIN_SEC = float(os.environ.get('NIGHT_CLASSIFIER_TEMPORAL_MIN_DURATION_SEC', '2.5'))
GEOMETRIC_CLASSIFIER_FLOOR = float(os.environ.get('GEOMETRIC_CLASSIFIER_FLOOR', '0.35'))
NIGHT_GEOMETRIC_CLASSIFIER_FLOOR = float(os.environ.get('NIGHT_GEOMETRIC_CLASSIFIER_FLOOR', '0.50'))
DRINKING_CLASSIFIER_SIZE = int(os.environ.get('DRINKING_CLASSIFIER_SIZE', '224'))
DRINK_CANDIDATE_GAP_SEC = float(os.environ.get('DRINK_CANDIDATE_GAP_SEC', '0.8'))
BEST_CANDIDATE_HOLD_SEC = float(os.environ.get('BEST_CANDIDATE_HOLD_SEC', '3.0'))
ACTIVE_LAPPING_MAX_GAP_SEC = float(os.environ.get('ACTIVE_LAPPING_MAX_GAP_SEC', '2.0'))
ACTIVE_LAPPING_MIN_IOU = float(os.environ.get('ACTIVE_LAPPING_MIN_IOU', '0.20'))
ACTIVE_LAPPING_MAX_CREDIT_SEC = float(os.environ.get('ACTIVE_LAPPING_MAX_CREDIT_SEC', '0.50'))

# Active-learning review queue. These samples never become training data until
# a human assigns a label through the review API.
REVIEW_SAMPLE_ENABLED = os.environ.get('REVIEW_SAMPLE_ENABLED', 'true').lower() != 'false'
REVIEW_SAMPLE_MIN_CONFIDENCE = float(os.environ.get('REVIEW_SAMPLE_MIN_CONFIDENCE', '0.15'))
REVIEW_SAMPLE_MAX_CONFIDENCE = float(os.environ.get('REVIEW_SAMPLE_MAX_CONFIDENCE', '1.0'))
REVIEW_SAMPLE_COOLDOWN_SEC = float(os.environ.get('REVIEW_SAMPLE_COOLDOWN_SEC', '180'))
REVIEW_SAMPLE_MAX_ITEMS = int(os.environ.get('REVIEW_SAMPLE_MAX_ITEMS', '300'))
NIGHT_REVIEW_CANDIDATE_CLF_MIN = float(os.environ.get('NIGHT_REVIEW_CANDIDATE_CLF_MIN', '0.3'))
NIGHT_REVIEW_CANDIDATE_CAT_MIN = float(os.environ.get('NIGHT_REVIEW_CANDIDATE_CAT_MIN', '0.4'))
REVIEW_SAMPLE_DIR = Path(os.environ.get(
    'REVIEW_SAMPLE_DIR', '/app/media/review_candidates'))
REVIEW_SAMPLE_FILE = Path(os.environ.get(
    'REVIEW_SAMPLE_FILE', '/app/data/review_candidates.json'))

# Event evidence: keep a small rolling buffer and turn it into an annotated MP4
# when a confirmed drinking event ends. The source stream is already reduced to
# 8 FPS, so this is bounded in CPU, memory, and disk use.
EVENT_EVIDENCE_ENABLED = os.environ.get('EVENT_EVIDENCE_ENABLED', 'true').lower() != 'false'
EVENT_EVIDENCE_FPS = float(os.environ.get('EVENT_EVIDENCE_FPS', '5'))
EVENT_PREBUFFER_SEC = float(os.environ.get('EVENT_PREBUFFER_SEC', '8'))
EVENT_POSTBUFFER_SEC = float(os.environ.get('EVENT_POSTBUFFER_SEC', '8'))
EVENT_EVIDENCE_JPEG_QUALITY = int(os.environ.get('EVENT_EVIDENCE_JPEG_QUALITY', '82'))

EVENT_DIR.mkdir(parents=True, exist_ok=True)
STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
_CALIBRATION_CACHE = {'mtime_ns': None, 'payload': {}, 'checked_at': 0.0}


def load_classifier_calibration():
    now = time.time()
    if now - _CALIBRATION_CACHE['checked_at'] < 30:
        return _CALIBRATION_CACHE['payload']
    _CALIBRATION_CACHE['checked_at'] = now
    try:
        mtime_ns = DRINKING_CALIBRATION_PATH.stat().st_mtime_ns
        if mtime_ns != _CALIBRATION_CACHE['mtime_ns']:
            payload = json.loads(DRINKING_CALIBRATION_PATH.read_text(encoding='utf-8'))
            threshold = float(payload.get('threshold', DRINKING_CLASSIFIER_THRESHOLD))
            if not 0.35 <= threshold <= 0.90:
                raise ValueError('calibrated threshold is out of range')
            _CALIBRATION_CACHE['payload'] = payload
            _CALIBRATION_CACHE['mtime_ns'] = mtime_ns
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        _CALIBRATION_CACHE['payload'] = {}
        _CALIBRATION_CACHE['mtime_ns'] = None
    return _CALIBRATION_CACHE['payload']


def effective_classifier_threshold(requested):
    calibration = load_classifier_calibration()
    if not calibration:
        return float(requested)
    calibrated = float(calibration['threshold'])
    profile_delta = max(0.0, float(requested) - DRINKING_CLASSIFIER_THRESHOLD)
    return min(0.98, calibrated + profile_delta)

DEFAULT_SETTINGS = {
    'roi': {'x1': 112, 'y1': 291, 'x2': 1132, 'y2': 620},
    'fountainZone': {'x1': 200, 'y1': 400, 'x2': 1080, 'y2': 700},
    'waterZone': {'x1': 350, 'y1': 350, 'x2': 650, 'y2': 480},
    'fpsSample': 2,
    'mlPerSec': 0.14,
    'mlPerActiveSec': 0.45,
    'mlPerActiveSecLow': 0.30,
    'mlPerActiveSecHigh': 0.70,
}

# ── Helpers ──────────────────────────────────────────────────────
_clahe = None

def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def is_night_mode(frame):
    """Detect IR/night black-white mode.

    IR footage is often bright but nearly grayscale, so brightness alone is not
    reliable. Treat low-saturation dark frames and bright grayscale frames as
    night profile candidates.
    """
    if not NIGHT_ENABLED:
        return False
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))
    b, g, r = cv2.split(frame.astype(np.float32))
    gray_delta = float(np.mean(np.maximum.reduce([
        np.abs(r - g), np.abs(r - b), np.abs(g - b)
    ])))
    return s_mean < NIGHT_SAT_THRESHOLD and (v_mean < NIGHT_VAL_THRESHOLD or gray_delta < NIGHT_GRAY_DELTA_THRESHOLD)

def enhance_night_frame(frame):
    """CLAHE contrast enhancement for night/IR frames."""
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=NIGHT_CLAHE_CLIP, tileGridSize=(8, 8))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def read_settings():
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            return DEFAULT_SETTINGS.copy()
        merged = DEFAULT_SETTINGS.copy()
        merged.update(data)
        return merged
    except Exception:
        return DEFAULT_SETTINGS.copy()

def write_status(payload):
    payload.setdefault('updated_at', now_iso())
    payload.setdefault('updated_unix', time.time())
    write_json_atomic(STATUS_FILE, payload)


def write_json_atomic(path, payload):
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=path.parent,
        prefix=f'.{path.name}.', suffix='.tmp', delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def open_video_stream(url, open_timeout_sec=STREAM_OPEN_TIMEOUT_SEC,
                      read_timeout_sec=STREAM_READ_TIMEOUT_SEC):
    """Open an FFmpeg-backed capture with bounded open and read waits."""
    params = []
    open_timeout_prop = getattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC', -1)
    read_timeout_prop = getattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC', -1)
    if open_timeout_prop >= 0:
        params.extend([open_timeout_prop, max(1000, int(open_timeout_sec * 1000))])
    if read_timeout_prop >= 0:
        params.extend([read_timeout_prop, max(1000, int(read_timeout_sec * 1000))])
    try:
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    except (TypeError, cv2.error):
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if open_timeout_prop >= 0:
            capture.set(open_timeout_prop, max(1000, int(open_timeout_sec * 1000)))
        if read_timeout_prop >= 0:
            capture.set(read_timeout_prop, max(1000, int(read_timeout_sec * 1000)))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture

def post_json(path, payload):
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if API_TOKEN:
        headers['X-API-Token'] = API_TOKEN
    req = urllib.request.Request(f'{DASHBOARD_API_URL}{path}', data=body,
                                 headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode('utf-8'))

def get_json(path):
    with urllib.request.urlopen(f'{DASHBOARD_API_URL}{path}', timeout=5) as resp:
        return json.loads(resp.read().decode('utf-8'))

def load_last_drink_at():
    try:
        data = get_json('/api/events')
        for event in data.get('events', []):
            if event.get('label') == 'false_positive':
                continue
            return event.get('end_time') or event.get('start_time')
    except Exception:
        return None
    return None


def annotate_evidence_frame(frame, fountain_zone, water_zone, cat_box, metadata):
    """Return a compact, self-explanatory frame for later event review."""
    output = frame.copy()
    for zone, color, name in (
        (fountain_zone, (78, 186, 111), 'fountain'),
        (water_zone, (229, 181, 75), 'water'),
    ):
        if zone and len(zone) == 4:
            x1, y1, x2, y2 = [int(v) for v in zone]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, name, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    if cat_box and len(cat_box) == 4:
        x1, y1, x2, y2 = [int(v) for v in cat_box]
        cv2.rectangle(output, (x1, y1), (x2, y2), (65, 200, 255), 2)
    text = (
        f"cat={metadata.get('cat_score', 0):.2f} "
        f"drink={metadata.get('clf_confidence', 0):.2f} "
        f"src={metadata.get('trigger_source', '-') or '-'}"
    )
    cv2.rectangle(output, (8, 8), (min(output.shape[1] - 8, 600), 38), (0, 0, 0), -1)
    cv2.putText(output, text, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return output


class EventEvidenceRecorder:
    """Build an annotated pre/post event clip without continuously writing video."""

    def __init__(self):
        self.enabled = EVENT_EVIDENCE_ENABLED and EVENT_EVIDENCE_FPS > 0
        self.interval = 1.0 / max(EVENT_EVIDENCE_FPS, 1.0)
        self.buffer = deque(maxlen=max(1, int(EVENT_PREBUFFER_SEC * EVENT_EVIDENCE_FPS)))
        self.last_capture_at = 0.0
        self.active = None

    def observe(self, frame, fountain_zone, water_zone, cat_box, metadata):
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_capture_at < self.interval:
            return
        self.last_capture_at = now
        item = {
            'captured_at': datetime.now().astimezone().isoformat(timespec='milliseconds'),
            'metadata': dict(metadata),
            'frame': annotate_evidence_frame(frame, fountain_zone, water_zone, cat_box, metadata),
        }
        self.buffer.append(item)
        if self.active is not None:
            self._write_item(self.active, item)

    def start(self, event_id, metadata):
        if not self.enabled:
            return
        if self.active is not None:
            self._finish_async(self.active)
        frame_dir = EVENT_DIR / f'.{event_id}-frames'
        frame_dir.mkdir(parents=True, exist_ok=True)
        self.active = {
            'event_id': event_id,
            'frame_dir': frame_dir,
            'items': [],
            'sequence': 0,
            'start_metadata': dict(metadata),
            'post_deadline': None,
        }
        for item in self.buffer:
            self._write_item(self.active, item)

    def stop(self, event_id, metadata, include_postbuffer=True):
        if self.active is None or self.active['event_id'] != event_id:
            return
        self.active['end_metadata'] = dict(metadata)
        if include_postbuffer:
            self.active['post_deadline'] = time.time() + EVENT_POSTBUFFER_SEC
            return
        active = self.active
        self.active = None
        self._finish_async(active)

    def tick(self):
        if self.active is None or self.active.get('post_deadline') is None:
            return
        if time.time() >= self.active['post_deadline']:
            active = self.active
            self.active = None
            self._finish_async(active)

    def _write_item(self, active, item):
        active['sequence'] += 1
        image_path = active['frame_dir'] / f"{active['sequence']:06d}.jpg"
        cv2.imwrite(str(image_path), item['frame'], [cv2.IMWRITE_JPEG_QUALITY, EVENT_EVIDENCE_JPEG_QUALITY])
        active['items'].append({
            'index': active['sequence'],
            'captured_at': item['captured_at'],
            **item['metadata'],
        })

    def _finish_async(self, active):
        Thread(target=self._encode, args=(active,), daemon=True).start()

    def _encode(self, active):
        event_id = active['event_id']
        frame_dir = active['frame_dir']
        clip_path = EVENT_DIR / f'{event_id}.mp4'
        evidence_path = EVENT_DIR / f'evidence-{event_id}.json'
        payload = {
            'event_id': event_id,
            'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'prebuffer_sec': EVENT_PREBUFFER_SEC,
            'postbuffer_sec': EVENT_POSTBUFFER_SEC,
            'fps': EVENT_EVIDENCE_FPS,
            'start': active.get('start_metadata', {}),
            'end': active.get('end_metadata', {}),
            'frames': active['items'],
        }
        write_json_atomic(evidence_path, payload)
        try:
            if active['sequence'] < 2:
                raise RuntimeError('not enough evidence frames')
            cmd = [
                '/usr/bin/ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-framerate', str(EVENT_EVIDENCE_FPS), '-i', str(frame_dir / '%06d.jpg'),
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(clip_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=90)
            if proc.returncode != 0 or not clip_path.exists() or clip_path.stat().st_size == 0:
                error = proc.stderr.decode(errors='replace')[-500:]
                raise RuntimeError(error or 'ffmpeg produced no clip')
            post_json('/api/events', {
                'event_id': event_id,
                'clip_url': f'/media/events/{event_id}.mp4',
                'evidence_url': f'/media/events/evidence-{event_id}.json',
            })
            print(f'EVENT EVIDENCE READY: {event_id} ({active["sequence"]} frames)', flush=True)
        except Exception as exc:
            print(f'EVENT EVIDENCE FAILED: {event_id}: {exc}', flush=True)
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

def box_iou(first, second):
    if not first or not second or len(first) != 4 or len(second) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1, first_area + second_area - inter)


def nms_suppress(items, iou_threshold=0.45):
    if not items:
        return []
    kept = []
    items = sorted(items, key=lambda x: -x.get('_combined_score', x.get('score', 0)))
    for item in items:
        bx1, by1, bx2, by2 = item['box']
        overlap = False
        for k in kept:
            if box_iou(item['box'], k['box']) > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(item)
    return kept

def _scale_zone(zone, settings_w, settings_h, frame_w, frame_h):
    """Scale a zone from settings coordinate space to actual frame dimensions."""
    if not zone:
        return None
    sx = frame_w / max(1, settings_w)
    sy = frame_h / max(1, settings_h)
    x1 = max(0, min(frame_w-1, int(zone.get('x1', 0) * sx)))
    y1 = max(0, min(frame_h-1, int(zone.get('y1', 0) * sy)))
    x2 = max(x1+1, min(frame_w, int(zone.get('x2', frame_w) * sx)))
    y2 = max(y1+1, min(frame_h, int(zone.get('y2', frame_h) * sy)))
    return (x1, y1, x2, y2)

def get_fountain_zone(settings, frame_w, frame_h):
    zone = settings.get('fountainZone') or {}
    if zone:
        sw = settings.get('frameWidth', frame_w)
        sh = settings.get('frameHeight', frame_h)
        return _scale_zone(zone, sw, sh, frame_w, frame_h)
    return (0, 0, frame_w, frame_h)

def get_water_zone(settings, frame_w, frame_h):
    zone = settings.get('waterZone') or {}
    if zone:
        sw = settings.get('frameWidth', frame_w)
        sh = settings.get('frameHeight', frame_h)
        return _scale_zone(zone, sw, sh, frame_w, frame_h)
    return None

def cat_in_fountain(cat_box, fzone):
    """Check if cat box overlaps fountain zone (overlap ratio or proximity)."""
    if not cat_box or len(cat_box) != 4:
        return False, 0.0
    bx1, by1, bx2, by2 = cat_box
    fx1, fy1, fx2, fy2 = fzone
    # Intersection
    ix1, iy1 = max(bx1, fx1), max(by1, fy1)
    ix2, iy2 = min(bx2, fx2), min(by2, fy2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    box_area = max(1, (bx2-bx1)*(by2-by1))
    overlap = inter / box_area
    # Proximity: cat bottom within N px of fountain top, with horizontal overlap
    near = (bx2 > fx1 and bx1 < fx2 and by2 >= fy1 - CAT_NEAR_PROXIMITY and by1 <= fy2)
    return (overlap >= CAT_NEAR_OVERLAP or near), overlap

def box_metrics(cat_box):
    x1, y1, x2, y2 = cat_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return width, height, width / height

def estimate_mouth_point(cat_box, wzone):
    """Approximate the drinking-side mouth point from the cat box and water zone.

    The detector returns a whole-cat box, not a head box. Picking the box bottom
    alone causes false positives when a sitting cat is merely behind the fountain,
    so this point is pulled toward the side of the box that is closest to water.
    """
    x1, y1, x2, y2 = cat_box
    wx1, wy1, wx2, wy2 = wzone
    width, height, _ = box_metrics(cat_box)
    water_cx = (wx1 + wx2) / 2
    if water_cx < (x1 + x2) / 2:
        mouth_x = x1 + 0.28 * width
    else:
        mouth_x = x2 - 0.28 * width
    mouth_y = y1 + 0.82 * height
    return mouth_x, mouth_y

def box_area_ratio(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return area / max(1, frame_w * frame_h)

def box_intersection_ratio(box, zone):
    x1, y1, x2, y2 = box
    zx1, zy1, zx2, zy2 = zone
    ix1, iy1 = max(x1, zx1), max(y1, zy1)
    ix2, iy2 = min(x2, zx2), min(y2, zy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    zone_area = max(1, (zx2 - zx1) * (zy2 - zy1))
    box_area = max(1, (x2 - x1) * (y2 - y1))
    return inter / zone_area, inter / box_area

def closeup_cat_near_water(cat_box, wzone, frame_w, frame_h):
    if not cat_box or len(cat_box) != 4 or not wzone:
        return False
    x1, y1, x2, y2 = cat_box
    wx1, wy1, wx2, wy2 = wzone
    area_ratio = box_area_ratio(cat_box, frame_w, frame_h)
    if area_ratio < CLOSEUP_CAT_MIN_BOX_AREA_RATIO:
        return False
    zone_overlap, box_overlap = box_intersection_ratio(cat_box, wzone)
    horizontally_near = x2 >= wx1 - CLOSEUP_WATER_PROXIMITY_PX and x1 <= wx2 + CLOSEUP_WATER_PROXIMITY_PX
    vertically_near = y2 >= wy1 - CLOSEUP_WATER_PROXIMITY_PX and y1 <= wy2 + CLOSEUP_WATER_PROXIMITY_PX
    # A large cat touching a frame edge is common with this camera placement and
    # does not imply that its mouth reaches the water. Keep this only as a
    # candidate cue; drinking still requires classifier or strict geometry.
    return zone_overlap >= 0.08 or box_overlap >= 0.015 or (horizontally_near and vertically_near and y2 >= wy1)

def trigger_min_duration(trigger_source, night_mode):
    if trigger_source.startswith('classifier_temporal'):
        return NIGHT_CLASSIFIER_TEMPORAL_MIN_SEC if night_mode else CLASSIFIER_TEMPORAL_MIN_SEC
    if trigger_source.startswith('classifier'):
        return NIGHT_DRINK_MIN_SEC if night_mode else DRINK_MIN_SEC
    if trigger_source.startswith('water_occupied'):
        return NIGHT_WATER_OCCUPIED_DRINK_MIN_SEC if night_mode else WATER_OCCUPIED_DRINK_MIN_SEC
    if trigger_source.startswith('closeup'):
        return NIGHT_CLOSEUP_DRINK_MIN_SEC if night_mode else CLOSEUP_DRINK_MIN_SEC
    if trigger_source.startswith('geometric'):
        return NIGHT_GEOMETRIC_DRINK_MIN_SEC if night_mode else GEOMETRIC_DRINK_MIN_SEC
    if trigger_source == 'prolonged_near':
        return 0.0
    return NIGHT_DRINK_MIN_SEC if night_mode else DRINK_MIN_SEC


class TemporalClassifierGate:
    """Require repeated high-confidence classifier evidence before a short hold."""

    def __init__(self, threshold=CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD,
                 min_hits=CLASSIFIER_TEMPORAL_MIN_HITS,
                 window_sec=CLASSIFIER_TEMPORAL_WINDOW_SEC,
                 hold_sec=CLASSIFIER_TEMPORAL_HOLD_SEC,
                 min_iou=CLASSIFIER_TEMPORAL_MIN_IOU):
        self.threshold = threshold
        self.min_hits = max(1, min_hits)
        self.window_sec = max(0.1, window_sec)
        self.hold_sec = max(0.0, hold_sec)
        self.min_iou = max(0.0, min_iou)
        self.hits = deque()
        self.hold_until = float('-inf')
        self.last_box = None
        self.last_confidence = 0.0

    def observe(self, now, cat_candidate, closeup_water, confidence, cat_box=None):
        while self.hits and now - self.hits[0] > self.window_sec:
            self.hits.popleft()
        if cat_candidate and closeup_water and confidence >= self.threshold:
            if (cat_box and self.last_box and
                    box_iou(cat_box, self.last_box) < self.min_iou):
                self.hits.clear()
            self.hits.append(now)
            self.last_confidence = float(confidence)
            if cat_box:
                self.last_box = list(cat_box)
            if len(self.hits) >= self.min_hits:
                self.hold_until = max(self.hold_until, now + self.hold_sec)
        if not self.hits and now > self.hold_until:
            self.last_box = None
            self.last_confidence = 0.0
        return now <= self.hold_until


class DrinkingCandidateGate:
    """Keep pre-event duration contiguous across only very short frame gaps."""

    def __init__(self, gap_sec=DRINK_CANDIDATE_GAP_SEC):
        self.gap_sec = max(0.0, gap_sec)
        self.started_at = None
        self.last_positive_at = None

    def observe(self, now, positive):
        if positive:
            if (self.last_positive_at is None or
                    now - self.last_positive_at > self.gap_sec):
                self.started_at = now
            self.last_positive_at = now
        elif (self.last_positive_at is not None and
              now - self.last_positive_at > self.gap_sec):
            self.reset()
        return self.started_at

    def reset(self):
        self.started_at = None
        self.last_positive_at = None


class ActiveDrinkEvidence:
    """Measure direct high-confidence lapping time without temporal hold."""

    def __init__(self, max_gap_sec=ACTIVE_LAPPING_MAX_GAP_SEC,
                 min_iou=ACTIVE_LAPPING_MIN_IOU,
                 max_credit_sec=ACTIVE_LAPPING_MAX_CREDIT_SEC):
        self.max_gap_sec = max(0.1, float(max_gap_sec))
        self.min_iou = max(0.0, float(min_iou))
        self.max_credit_sec = max(0.0, float(max_credit_sec))
        self.clusters = []

    def observe(self, now, direct_anchor, cat_box=None):
        if not direct_anchor or not cat_box or len(cat_box) != 4:
            return
        timestamp = float(now)
        box = list(cat_box)
        if self.clusters:
            cluster = self.clusters[-1]
            gap = timestamp - cluster['last']
            if 0 < gap <= self.max_gap_sec and box_iou(box, cluster['last_box']) >= self.min_iou:
                cluster['last'] = timestamp
                cluster['last_box'] = box
                cluster['gaps'].append(gap)
                cluster['count'] += 1
                return
        self.clusters.append({
            'first': timestamp,
            'last': timestamp,
            'last_box': box,
            'gaps': [],
            'count': 1,
        })

    def prune(self, cutoff):
        self.clusters = [cluster for cluster in self.clusters if cluster['last'] >= cutoff]

    @staticmethod
    def _median(values):
        ordered = sorted(values)
        size = len(ordered)
        if not size:
            return 0.25
        middle = size // 2
        if size % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2

    def duration(self):
        total = 0.0
        for cluster in self.clusters:
            if cluster['count'] < 2:
                continue
            credit = min(self.max_credit_sec, max(0.1, self._median(cluster['gaps'])))
            total += max(0.0, cluster['last'] - cluster['first']) + credit
        return total

    def anchor_count(self):
        return sum(cluster['count'] for cluster in self.clusters)

    def reset(self):
        self.clusters = []


class ReviewSampleRecorder:
    """Persist sparse, deduplicated uncertain crops for later human labeling."""

    def __init__(self):
        REVIEW_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        self.items = []
        self.last_capture_at = float('-inf')
        try:
            payload = json.loads(REVIEW_SAMPLE_FILE.read_text(encoding='utf-8'))
            self.items = payload.get('items', []) if isinstance(payload, dict) else []
            if self.items:
                self.last_capture_at = float(self.items[-1].get('created_unix', 0))
        except (OSError, ValueError, TypeError):
            self.items = []

    @staticmethod
    def _difference_hash(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (9, 8))
        value = 0
        for bit in (small[:, 1:] > small[:, :-1]).flatten():
            value = (value << 1) | int(bit)
        return f'{value:016x}'

    @staticmethod
    def _hash_distance(first, second):
        try:
            return bin(int(first, 16) ^ int(second, 16)).count('1')
        except (TypeError, ValueError):
            return 64

    def _save(self):
        write_json_atomic(REVIEW_SAMPLE_FILE, {
            'version': 1,
            'updated_at': now_iso(),
            'items': self.items,
        })

    def _remove_files(self, item):
        for key in ('snapshot_url', 'crop_url'):
            url = item.get(key, '')
            if isinstance(url, str) and url.startswith('/media/review_candidates/'):
                try:
                    (Path('/app') / url.lstrip('/')).unlink(missing_ok=True)
                except OSError:
                    pass

    def capture(self, frame, cat_box, confidence, cat_score, night_mode):
        if not REVIEW_SAMPLE_ENABLED or not cat_box or len(cat_box) != 4:
            return False
        if not (REVIEW_SAMPLE_MIN_CONFIDENCE <= confidence <= REVIEW_SAMPLE_MAX_CONFIDENCE):
            return False
        current = time.time()
        if current - self.last_capture_at < REVIEW_SAMPLE_COOLDOWN_SEC:
            return False
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(cat_box[0])))
        y1 = max(0, min(h - 1, int(cat_box[1])))
        x2 = max(x1 + 1, min(w, int(cat_box[2])))
        y2 = max(y1 + 1, min(h, int(cat_box[3])))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        fingerprint = self._difference_hash(crop)
        if any(self._hash_distance(fingerprint, item.get('fingerprint')) <= 2
               for item in self.items[-20:]):
            return False
        sample_id = f'review-{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}'
        snapshot_path = REVIEW_SAMPLE_DIR / f'{sample_id}.jpg'
        crop_path = REVIEW_SAMPLE_DIR / f'{sample_id}-crop.jpg'
        if not cv2.imwrite(str(snapshot_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            return False
        if not cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            snapshot_path.unlink(missing_ok=True)
            return False
        self.items.append({
            'id': sample_id,
            'created_at': now_iso(),
            'created_unix': current,
            'snapshot_url': f'/media/review_candidates/{snapshot_path.name}',
            'crop_url': f'/media/review_candidates/{crop_path.name}',
            'classifier_confidence': round(float(confidence), 4),
            'cat_confidence': round(float(cat_score), 4),
            'profile': 'night' if night_mode else 'day',
            'fingerprint': fingerprint,
            'source': 'uncertain_closeup',
        })
        while len(self.items) > max(1, REVIEW_SAMPLE_MAX_ITEMS):
            self._remove_files(self.items.pop(0))
        self.last_capture_at = current
        self._save()
        print(f'REVIEW SAMPLE: {sample_id}, clf={confidence:.3f}, cat={cat_score:.3f}')
        return True

    @property
    def count(self):
        return len(self.items)

# ── Cat Detector ─────────────────────────────────────────────────
def decode_yolo_layout(output):
    """Recognize supported raw YOLO and end-to-end ONNX detection layouts.

    Raw: (1, 4+classes, candidates), center/size + class scores.
    End-to-end: (1, candidates, 6), corners + score + class id.
    Fail explicitly instead of treating an unknown tensor as cat detections.
    """
    shape = output.shape
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError(f'Unsupported YOLO output shape: {shape}')
    if shape[2] == 6:
        return output[0], True
    if 5 <= shape[1] < 100:
        return output[0].T, False
    raise ValueError(f'Unsupported YOLO output shape: {shape}')


class CatDetector:
    def __init__(self):
        self.session = None
        self._current_model_path = None
        self.last_run_ts = 0.0
        self.last_result = {'cat_detected': False, 'cat_score': 0.0, 'cat_box': [],
                             'water_occupied': False}
        self._hires_cap = None  # Persistent RTSP connection for night hires grabs
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=NIGHT_MOG2_HISTORY, varThreshold=NIGHT_MOG2_VAR_THRESHOLD, detectShadows=False)
        self._mog2_warmed = False
        self._mog2_warm_frames = 0
        # Night water zone: running-average background model (works better than MOG2
        # on uniform fountain surfaces where MOG2's Gaussian model sees no variance).
        self._wz_bg = None       # Running-average background of water zone
        self._wz_bg_frames = 0   # Number of frames accumulated into background
        self._consecutive_cat = 0  # Consecutive frames with cat detected
        self._last_cat_box = None
        # Drinking classifier (optional - loaded when model file exists)
        self._drink_clf_session = None
        self._load_drinking_classifier()
        self._load_model()

    def _load_model(self):
        model_path = get_active_model_path()
        model = Path(model_path)
        if model.exists():
            try:
                self.session = self._build_session(model)
                self._current_model_path = model_path
                print(f"Model loaded: {model} (providers={self.session.get_providers()})")
            except Exception as e:
                print(f"Model load FAILED for {model}: {e}")
                if self.session is None:
                    raise  # No fallback, must crash
                # Keep existing session if hot-reload fails
        else:
            print(f"Model not found: {model}")

    def _load_drinking_classifier(self):
        """Load the drinking classifier ONNX model if available."""
        clf_path = Path(DRINKING_CLASSIFIER_PATH)
        if clf_path.exists():
            try:
                self._drink_clf_session = self._build_session(clf_path)
                print(f"Drinking classifier loaded: {clf_path} "
                      f"(threshold={DRINKING_CLASSIFIER_THRESHOLD}, "
                      f"providers={self._drink_clf_session.get_providers()})")
            except Exception as e:
                print(f"Drinking classifier load FAILED: {e}")
                self._drink_clf_session = None
        else:
            print(f"Drinking classifier not found at {clf_path} (using geometric fallback)")

    def classify_drinking(self, frame, cat_box, threshold=DRINKING_CLASSIFIER_THRESHOLD):
        """Run drinking classifier on the cat crop. Returns (is_drinking, confidence)."""
        if self._drink_clf_session is None or not cat_box or len(cat_box) != 4:
            return None, 0.0
        x1, y1, x2, y2 = cat_box
        # Clamp to frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, 0.0
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, 0.0
        # Preprocess: resize, normalize (same as training)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (DRINKING_CLASSIFIER_SIZE, DRINKING_CLASSIFIER_SIZE))
        img = crop_resized.astype(np.float32) / 255.0
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]
        logit = self._drink_clf_session.run(None, {'image': img})[0][0, 0]
        confidence = float(1.0 / (1.0 + np.exp(-logit)))  # sigmoid
        return confidence >= effective_classifier_threshold(threshold), confidence

    def reload_if_changed(self):
        """Check if active model changed in settings, hot-reload if so."""
        new_path = get_active_model_path()
        if new_path != self._current_model_path:
            print(f"Model change detected: {self._current_model_path} -> {new_path}")
            try:
                self._load_model()
                print("Model hot-reloaded successfully")
                return True
            except Exception as e:
                print(f"Model hot-reload FAILED: {e}")
                print(f"Keeping current model: {self._current_model_path}")
                return False
        return False

    def _build_session(self, model):
        available = ort.get_available_providers()
        candidates = []
        for item in PROVIDER_PREF:
            if item == 'openvino':
                candidates.append(('OpenVINOExecutionProvider', {'device_type': OPENVINO_DEVICE}))
            elif item == 'cpu':
                candidates.append('CPUExecutionProvider')
        for p in candidates:
            pname = p[0] if isinstance(p, tuple) else p
            if pname not in available:
                continue
            try:
                providers = [p, 'CPUExecutionProvider'] if isinstance(p, tuple) else [p]
                return ort.InferenceSession(str(model), providers=providers)
            except Exception as e:
                print(f"  {pname}: {e}")
        return ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])

    def _infer_candidates(self, frame, threshold, offset_x=0, offset_y=0, night_mode=False):
        h, w = frame.shape[:2]
        if night_mode:
            frame = enhance_night_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Letterbox
        scale = min(YOLO_IMG_SIZE / w, YOLO_IMG_SIZE / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (nw, nh))
        pad_x = (YOLO_IMG_SIZE - nw) // 2
        pad_y = (YOLO_IMG_SIZE - nh) // 2
        letterbox = np.ones((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8) * 114
        letterbox[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
        inp = letterbox.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis, :, :, :]
        output = self.session.run(None, {'images': inp})[0]
        predictions, end_to_end = decode_yolo_layout(output)
        # Filter scores in NumPy before Python iteration (raw heads have 8400 rows).
        if end_to_end or predictions.shape[1] == 5:
            score_column = 4
        elif predictions.shape[1] > 4 + COCO_CAT_LABEL:
            score_column = 4 + COCO_CAT_LABEL
        else:
            return []
        keep = np.isfinite(predictions).all(axis=1) & (predictions[:, score_column] >= threshold)
        if end_to_end:
            keep &= predictions[:, 5] == COCO_CAT_LABEL
        items = []
        for pred in predictions[keep]:
            if end_to_end:
                # YOLO26 one-to-one: x1,y1,x2,y2,score,class_id in input pixels.
                if float(pred[5]) != COCO_CAT_LABEL:
                    continue
                conf = float(pred[4])
                left, top, right, bottom = map(float, pred[:4])
            else:
                x, y, bw, bh = map(float, pred[:4])
                if len(pred) == 5:  # legacy single-class cat model
                    conf = float(pred[4])
                else:
                    if len(pred) <= 4 + COCO_CAT_LABEL:
                        continue
                    conf = float(pred[4 + COCO_CAT_LABEL])
                    if PERSON_REJECTION_ENABLED:
                        person_conf = float(pred[4 + COCO_PERSON_LABEL])
                        if person_conf > PERSON_SCORE_REJECT_THRESHOLD and person_conf > conf:
                            continue
                left, top, right, bottom = x - bw/2, y - bh/2, x + bw/2, y + bh/2
            if conf < threshold or right <= left or bottom <= top:
                continue
            x1 = (left - pad_x) / scale + offset_x
            y1 = (top - pad_y) / scale + offset_y
            x2 = (right - pad_x) / scale + offset_x
            y2 = (bottom - pad_y) / scale + offset_y
            items.append({'score': conf, 'box': [int(x1), int(y1), int(x2), int(y2)]})
        return items

    def _grab_hires_frame(self, timeout=NIGHT_HIRES_TIMEOUT_SEC):
        """Grab a single frame from the high-res RTSP camera.
        Returns (frame, scale_x, scale_y) or (None, 0, 0) on failure.
        scale factors map hires coords → transcode (960x540) coords.
        """
        if not NIGHT_HIRES_ENABLED:
            return None, 0, 0
        try:
            if self._hires_cap is None or not self._hires_cap.isOpened():
                self._hires_cap = open_video_stream(
                    NIGHT_HIRES_RTSP_URL, open_timeout_sec=timeout, read_timeout_sec=timeout)
            ret, frame = self._hires_cap.read()
            if not ret or frame is None:
                # Connection lost; close and retry next time
                if self._hires_cap:
                    self._hires_cap.release()
                self._hires_cap = None
                return None, 0, 0
            hires_h, hires_w = frame.shape[:2]
            scale_x = 960.0 / hires_w   # to transcode coords
            scale_y = 540.0 / hires_h
            return frame, scale_x, scale_y
        except Exception:
            self._hires_cap = None
            return None, 0, 0

    def detect(self, frame, force=False, fzone=None, wzone=None):
        now = time.time()
        if not self.session:
            return self.last_result
        if not force and now - self.last_run_ts < DETECT_INTERVAL_SEC:
            return self.last_result
        self.last_run_ts = now
        h, w = frame.shape[:2]
        frame_area = w * h

        night_mode = is_night_mode(frame)
        score_thresh = NIGHT_CAT_SCORE_THRESHOLD if night_mode else CAT_SCORE_THRESHOLD
        crop_thresh = NIGHT_CAT_CROP_THRESHOLD if night_mode else CAT_CROP_SCORE_THRESHOLD

        candidates = self._infer_candidates(frame, score_thresh, night_mode=night_mode)
        if CLOSEUP_CAT_SCORE_THRESHOLD < score_thresh:
            candidates.extend(self._infer_candidates(frame, CLOSEUP_CAT_SCORE_THRESHOLD, night_mode=night_mode))
        # Second pass on bottom crop for distant/small cats
        crop_start = max(0, min(h-1, int(h * BOTTOM_CROP_START_RATIO)))
        if crop_start > 0 and crop_start < h and (not candidates or max((c['score'] for c in candidates), default=0) < 0.55):
            crop = frame[crop_start:h, 0:w]
            if crop.size:
                candidates.extend(self._infer_candidates(crop, crop_thresh, 0, crop_start, night_mode=night_mode))
        # Night mode: extra detection passes for IR footage.
        if night_mode:
            # Pass 3: full frame with low threshold — catches marginal cats that
            # score between crop_thresh and score_thresh.
            candidates.extend(self._infer_candidates(frame, crop_thresh, night_mode=True))
            # Pass 4: high-res RTSP grab → wide crop at fountain bottom → YOLO.
            hires_frame, hsx, hsy = self._grab_hires_frame()
            if hires_frame is not None and fzone and len(fzone) == 4:
                hires_h, hires_w = hires_frame.shape[:2]
                fx1, fy1, fx2, fy2 = fzone
                fcx_h = int(((fx1 + fx2) / 2) / hsx) if hsx > 0 else hires_w // 2
                fy_btm_h = int(fy2 / hsy) if hsy > 0 else hires_h
                crop_w = int(hires_w * NIGHT_HIRES_CROP_WIDTH)
                crop_h_hires = int(hires_h * NIGHT_HIRES_CROP_HEIGHT)
                half_w = crop_w // 2
                hx1 = max(0, min(hires_w - crop_w, fcx_h - half_w))
                hy2 = min(hires_h, fy_btm_h + crop_h_hires // 3)
                hy1 = hy2 - crop_h_hires
                if hy1 < 0:
                    hy1, hy2 = 0, crop_h_hires
                hx2 = hx1 + crop_w
                hzcrop = hires_frame[hy1:hy2, hx1:hx2]
                if hzcrop.size:
                    hires_candidates = self._infer_candidates(hzcrop, crop_thresh, hx1, hy1, night_mode=True)
                    for c in hires_candidates:
                        bx1, by1, bx2, by2 = c['box']
                        c['box'] = [int(bx1 * hsx), int(by1 * hsy), int(bx2 * hsx), int(by2 * hsy)]
                    candidates.extend(hires_candidates)
        # ── Water zone detection ──────────────────────────────────
        water_occupied = False
        if wzone and len(wzone) == 4:
            wx1, wy1, wx2, wy2 = wzone
            wcrop = frame[wy1:wy2, wx1:wx2]
            if wcrop.size > 0:
                if night_mode:
                    gray = cv2.cvtColor(wcrop, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    if self._wz_bg is None or self._wz_bg.shape != gray.shape:
                        self._wz_bg = gray
                        self._wz_bg_frames = 1
                    else:
                        bg_diff = float(np.mean(cv2.absdiff(gray, self._wz_bg)))
                        if bg_diff < 6.0:
                            cv2.accumulateWeighted(gray, self._wz_bg, 0.02)
                            self._wz_bg_frames += 1
                    if self._wz_bg_frames > NIGHT_WATER_BG_FRAMES:
                        diff = cv2.absdiff(gray, self._wz_bg)
                        _, diff_th = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
                        diff_th = cv2.medianBlur(diff_th.astype(np.uint8), 3)
                        fg_px = np.count_nonzero(diff_th)
                        if NIGHT_WATER_FG_MIN < fg_px < 5000:
                            water_occupied = True
                else:
                    wz_thresh = WATER_ZONE_CROP_THRESHOLD if not night_mode else crop_thresh
                    wz_cands = self._infer_candidates(wcrop, wz_thresh, wx1, wy1, night_mode=False)
                    water_occupied = any(c['score'] >= wz_thresh for c in wz_cands)
        # Filter invalid boxes
        max_area_ratio = NIGHT_MAX_BOX_AREA_RATIO if night_mode else 0.6
        filtered = []
        for item in candidates:
            bx1, by1, bx2, by2 = item['box']
            box_area = (bx2-bx1)*(by2-by1)
            area_ratio = box_area / frame_area
            is_closeup_water_candidate = (
                wzone and
                item.get('score', 0) >= CLOSEUP_CAT_SCORE_THRESHOLD and
                area_ratio <= CLOSEUP_CAT_MAX_BOX_AREA_RATIO and
                closeup_cat_near_water(item['box'], wzone, w, h)
            )
            if box_area > max_area_ratio * frame_area and not is_closeup_water_candidate:
                continue
            if box_area < CAT_MIN_BOX_AREA_RATIO * frame_area:
                continue
            bw, bh = bx2-bx1, by2-by1
            if bh > 0 and bw/bh > 3.5 and not is_closeup_water_candidate:
                continue
            if bx1 < -50 or by1 < -50 or bx2 > w+50 or by2 > h+50:
                continue
            if bx1 < 5 and bx2 > w-5 and not is_closeup_water_candidate:
                continue
            if night_mode and by1 < 10:
                continue
            filtered.append({**item, '_combined_score': float(item['score']) + min(0.2, area_ratio*0.5)})
        filtered = nms_suppress(filtered)
        best = max(filtered, key=lambda x: x['_combined_score']) if filtered else None
        if best:
            if (self._last_cat_box is None or
                    box_iou(best['box'], self._last_cat_box) >= CAT_TRACK_MIN_IOU):
                self._consecutive_cat += 1
            else:
                self._consecutive_cat = 1
            self._last_cat_box = list(best['box'])
            closeup_water = closeup_cat_near_water(best['box'], wzone, w, h)
            cat_detected = self._consecutive_cat >= CONSECUTIVE_FRAMES_REQUIRED
            self.last_result = {
                'cat_detected': cat_detected, 'cat_score': round(float(best['score']), 3),
                'cat_box': best['box'],
                'night_mode': night_mode,
                'water_occupied': water_occupied,
                'closeup_water': closeup_water,
            }
        else:
            self._consecutive_cat = 0
            self._last_cat_box = None
            self.last_result = {'cat_detected': False, 'cat_score': 0.0, 'cat_box': [],
                                'night_mode': night_mode, 'water_occupied': water_occupied,
                                'closeup_water': False}
        return self.last_result

# ── Motion pre-filter ────────────────────────────────────────────
class MotionFilter:
    def __init__(self):
        self.prev_frame = None

    def has_motion(self, frame, threshold=None):
        """Check if frame has changed significantly from previous."""
        if threshold is None:
            threshold = MOTION_THRESHOLD
        small = cv2.resize(frame, MOTION_RESIZE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray.astype(np.float32)
            return True  # First frame always triggers
        diff = float(np.mean(cv2.absdiff(gray, cv2.convertScaleAbs(self.prev_frame))))
        # Update baseline
        cv2.accumulateWeighted(gray, self.prev_frame, 0.1)
        return diff >= threshold

# ── Main loop ────────────────────────────────────────────────────
def main():
    print("Starting simplified cat drinking detector")
    detector = CatDetector()
    motion = MotionFilter()
    settings = read_settings()
    print(f"Fountain zone: {settings.get('fountainZone')}")
    print(f"Water zone: {settings.get('waterZone')}")
    print(f"Night mode: enabled (sat<{NIGHT_SAT_THRESHOLD}, val<{NIGHT_VAL_THRESHOLD}, "
          f"score={NIGHT_CAT_SCORE_THRESHOLD}/{NIGHT_CAT_CROP_THRESHOLD}, "
          f"motion={NIGHT_MOTION_THRESHOLD}, clahe={NIGHT_CLAHE_CLIP})")
    print(f"Person rejection: {'on' if PERSON_REJECTION_ENABLED else 'off'} "
          f"(threshold={PERSON_SCORE_REJECT_THRESHOLD})")
    print(f"Consecutive frames required: {CONSECUTIVE_FRAMES_REQUIRED}")

    cap = None
    frame_count = 0
    last_fps_check = time.time()
    last_status = 0.0
    last_settings_reload = 0.0

    # Drinking state tracking
    cat_in_zone_since = None      # When cat first entered fountain zone
    last_cat_in_zone = None        # Last time cat was seen in zone
    last_cat_seen_at = None        # Timestamp of last cat detection (even without drinking)
    cat_near_since = None          # When cat first entered near-water zone (prolonged presence safety net)
    event_active = False
    current_event_id = None
    peak_cat_score = 0.0          # Max cat score during current event
    event_trigger_source = ''
    event_peak_clf_conf = 0.0
    event_profile = 'day'
    last_detection_result = None   # Cached result when skipping YOLO
    last_detection_time = 0.0
    last_drink_end_at = load_last_drink_at()  # Timestamp of last drink event end
    fps = 0.0                      # Current FPS
    night_mode = False             # Current day/night state
    last_forced_inference = 0.0    # Last forced YOLO run in night mode
    evidence_recorder = EventEvidenceRecorder()
    temporal_day_gate = TemporalClassifierGate()
    temporal_night_gate = TemporalClassifierGate(
        threshold=NIGHT_CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD,
    )
    candidate_gate = DrinkingCandidateGate()
    active_drink_evidence = ActiveDrinkEvidence()
    review_recorder = ReviewSampleRecorder()
    best_candidate_frame = None
    best_candidate_crop = None
    best_candidate_conf = 0.0
    best_candidate_at = float('-inf')

    def clear_drinking_state():
        nonlocal cat_in_zone_since, last_cat_in_zone, cat_near_since
        nonlocal event_active, current_event_id, peak_cat_score
        nonlocal event_trigger_source, event_peak_clf_conf, event_profile
        nonlocal best_candidate_frame, best_candidate_crop
        nonlocal best_candidate_conf, best_candidate_at
        cat_in_zone_since = None
        last_cat_in_zone = None
        cat_near_since = None
        event_active = False
        current_event_id = None
        peak_cat_score = 0.0
        event_trigger_source = ''
        event_peak_clf_conf = 0.0
        event_profile = 'day'
        best_candidate_frame = None
        best_candidate_crop = None
        best_candidate_conf = 0.0
        best_candidate_at = float('-inf')
        candidate_gate.reset()
        active_drink_evidence.reset()

    def finalize_active_event(reason, end_ts=None, include_postbuffer=True):
        nonlocal last_drink_end_at
        if not event_active or not current_event_id:
            clear_drinking_state()
            return
        actual_end = end_ts or last_cat_in_zone or time.time()
        actual_start = cat_in_zone_since or actual_end
        total_duration = max(0.0, actual_end - actual_start)
        active_drink_sec = round(active_drink_evidence.duration(), 1)
        ml_per_active_sec = float(settings.get('mlPerActiveSec', 0.45) or 0.45)
        estimated_ml = round(active_drink_sec * ml_per_active_sec, 1)
        print(f"DRINK END: {current_event_id}, total_duration={total_duration:.1f}s, "
              f"active_drink={active_drink_sec:.1f}s, estimated={estimated_ml:.1f}ml, reason={reason}")
        try:
            end_time = datetime.fromtimestamp(actual_end).astimezone().isoformat(timespec='seconds')
            start_time = datetime.fromtimestamp(actual_start).astimezone().isoformat(timespec='seconds')
            post_json('/api/events', {
                'event_id': current_event_id,
                'type': 'drink_end',
                'start_time': start_time,
                'end_time': end_time,
                'duration_sec': round(total_duration, 1),
                'confidence': event_peak_clf_conf if detector._drink_clf_session is not None else peak_cat_score,
                'cat_confidence': round(peak_cat_score, 3),
                'drinking_confidence': round(event_peak_clf_conf, 3),
                'active_drink_sec': active_drink_sec,
                'estimated_ml': estimated_ml,
                'ml_per_active_sec': ml_per_active_sec,
                'volume_method': 'active_lapping_v2',
                'snapshot_url': f'/media/events/snap-{current_event_id}.jpg',
                'trigger_source': event_trigger_source,
                'profile': event_profile,
                'clf_confidence': round(event_peak_clf_conf, 3),
                'end_reason': reason,
                'summary': (f'自动检测触发源：{event_trigger_source or "unknown"}，模式：{event_profile}，'
                            f'分类器峰值：{event_peak_clf_conf:.2f}，有效舔水：{active_drink_sec:.1f}秒，'
                            f'结束原因：{reason}'),
            })
        except Exception as exc:
            print(f"  Post failed: {exc}")
        evidence_recorder.stop(current_event_id, {
            'duration_sec': round(total_duration, 1),
            'cat_confidence': round(peak_cat_score, 3),
            'drinking_confidence': round(event_peak_clf_conf, 3),
            'active_drink_sec': active_drink_sec,
            'ml_per_active_sec': ml_per_active_sec,
            'volume_method': 'active_lapping_v2',
            'trigger_source': event_trigger_source,
            'profile': event_profile,
            'end_reason': reason,
        }, include_postbuffer=include_postbuffer)
        last_drink_end_at = datetime.now().astimezone().isoformat(timespec='seconds')
        clear_drinking_state()

    def handle_stream_interrupt(reason):
        nonlocal last_detection_result, last_detection_time
        finalize_active_event(reason, last_cat_in_zone, include_postbuffer=False)
        last_detection_result = None
        last_detection_time = 0.0
        motion.prev_frame = None
        detector._consecutive_cat = 0
        detector._last_cat_box = None

    while True:
        try:
            # Reconnect if needed
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = open_video_stream(STREAM_URL)
                if not cap.isOpened():
                    print("Failed to open stream, retrying...")
                    handle_stream_interrupt('stream_connect_failed')
                    write_status({'ok': False, 'phase': 'connect_failed'})
                    time.sleep(RECONNECT_SEC)
                    continue
                print("Stream connected")
                write_status({'ok': True, 'phase': 'running'})

            ret, frame = cap.read()
            if not ret:
                print("Read failed, reconnecting...")
                cap.release()
                handle_stream_interrupt('stream_read_failed')
                time.sleep(RECONNECT_SEC)
                continue

            frame_count += 1
            h, w = frame.shape[:2]
            fzone = get_fountain_zone(settings, w, h)
            wzone = get_water_zone(settings, w, h)

            # Reload settings periodically
            if time.time() - last_settings_reload > 60:
                settings = read_settings()
                last_settings_reload = time.time()
                # Check for model switch
                if detector.reload_if_changed():
                    print("Model hot-reloaded successfully")

            # Detect night mode once per frame (cheap HSV conversion)
            night_mode = is_night_mode(frame)

            # Motion check (use lower threshold at night) → decide whether to run YOLO
            motion_thresh = NIGHT_MOTION_THRESHOLD if night_mode else MOTION_THRESHOLD
            should_detect = motion.has_motion(frame, threshold=motion_thresh)

            # Night mode: force periodic YOLO inference even without motion.
            is_forced = False
            if night_mode and not should_detect:
                if time.time() - last_forced_inference >= NIGHT_FORCE_INFERENCE_SEC:
                    should_detect = True
                    is_forced = True
                    last_forced_inference = time.time()
            result = None

            if should_detect:
                result = detector.detect(frame, force=True, fzone=fzone, wzone=wzone)
                if result.get('cat_detected') or result.get('water_occupied') or not is_forced:
                    last_detection_result = result
                    last_detection_time = time.time()
            elif last_detection_result is not None and time.time() - last_detection_time < STALE_DETECTION_SEC:
                result = last_detection_result
            else:
                result = {'cat_detected': False, 'cat_score': 0.0, 'cat_box': [],
                          'night_mode': night_mode, 'water_occupied': False}

            now = time.time()
            if not event_active and now - best_candidate_at > BEST_CANDIDATE_HOLD_SEC:
                best_candidate_frame = None
                best_candidate_crop = None
                best_candidate_conf = 0.0
            cat_box = result.get('cat_box', [])
            cat_visible = result.get('cat_detected', False)
            if cat_visible:
                last_cat_seen_at = now_iso()
            water_occupied = result.get('water_occupied', False)
            closeup_water = result.get('closeup_water', False)
            cat_in_water_zone = bool((water_occupied or closeup_water) and cat_visible)
            water_motion = bool(water_occupied and not cat_visible and night_mode)
            cat_at_water = False
            classifier_drinking = False
            geometric_drinking = False
            trigger_source = ''
            drink_clf_conf = 0.0
            classifier_threshold = (NIGHT_DRINKING_CLASSIFIER_THRESHOLD if night_mode
                                    else DRINKING_CLASSIFIER_THRESHOLD)
            geometric_clf_floor = (NIGHT_GEOMETRIC_CLASSIFIER_FLOOR if night_mode
                                   else GEOMETRIC_CLASSIFIER_FLOOR)
            water_occupied_clf_floor = (NIGHT_WATER_OCCUPIED_CLASSIFIER_FLOOR if night_mode
                                        else WATER_OCCUPIED_CLASSIFIER_FLOOR)

            # ── Drinking detection: classifier-first, geometric fallback ──
            cat_candidate = bool(cat_box and len(cat_box) == 4 and wzone)
            temporal_gate = temporal_night_gate if night_mode else temporal_day_gate
            temporal_classifier_active = False
            if cat_candidate:
                if detector._drink_clf_session is not None:
                    is_drinking_clf, drink_clf_conf = detector.classify_drinking(
                        frame, cat_box, threshold=classifier_threshold)
                    if closeup_water and drink_clf_conf > best_candidate_conf:
                        bx1 = max(0, min(w - 1, int(cat_box[0])))
                        by1 = max(0, min(h - 1, int(cat_box[1])))
                        bx2 = max(bx1 + 1, min(w, int(cat_box[2])))
                        by2 = max(by1 + 1, min(h, int(cat_box[3])))
                        candidate_crop = frame[by1:by2, bx1:bx2]
                        if candidate_crop.size:
                            best_candidate_frame = frame.copy()
                            best_candidate_crop = candidate_crop.copy()
                            best_candidate_conf = float(drink_clf_conf)
                            best_candidate_at = now
                    mouth_x, mouth_y = estimate_mouth_point(cat_box, wzone)
                    mouth_near_water = (wzone[0] - 70 <= mouth_x <= wzone[2] + 70 and
                                        wzone[1] - 80 <= mouth_y <= wzone[3] + 90)
                    classifier_near_water = mouth_near_water or closeup_water
                    if cat_visible and is_drinking_clf and classifier_near_water:
                        cat_at_water = True
                        classifier_drinking = True
                        source = 'classifier' if mouth_near_water else 'classifier_closeup'
                        trigger_source = f'{source}({drink_clf_conf:.2f})'
                # Geometric fallback remains tied to confirmed cat detections.
                if cat_visible and not cat_at_water:
                    classifier_supports_geometry = (
                        detector._drink_clf_session is None or drink_clf_conf >= geometric_clf_floor)
                    water_surface_y = settings.get("waterSurfaceY")
                    if water_surface_y and classifier_supports_geometry:
                        sx = w / max(1, settings.get("frameWidth", w))
                        sy = h / max(1, settings.get("frameHeight", h))
                        scaled_water_y = water_surface_y * sy
                        mouth_x, mouth_y = estimate_mouth_point(cat_box, wzone)
                        in_water_x = wzone[0] - 30 <= mouth_x <= wzone[2] + 30
                        head_at_water = (scaled_water_y - WATER_PROXIMITY_PX * 0.4) <= mouth_y <= (scaled_water_y + WATER_PROXIMITY_PX)
                        _, _, posture_ratio = box_metrics(cat_box)
                        drinking_posture = posture_ratio >= DRINKING_POSTURE_MIN_RATIO
                        cat_at_water = in_water_x and head_at_water and drinking_posture
                        if cat_at_water:
                            geometric_drinking = True
                            trigger_source = 'geometric'
                    elif classifier_supports_geometry:
                        wx1, wy1, wx2, wy2 = wzone
                        ix1, iy1 = max(cat_box[0], wx1), max(cat_box[1], wy1)
                        ix2, iy2 = min(cat_box[2], wx2), min(cat_box[3], wy2)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        box_area = max(1, (cat_box[2]-cat_box[0]) * (cat_box[3]-cat_box[1]))
                        _, _, posture_ratio = box_metrics(cat_box)
                        drinking_posture = posture_ratio >= DRINKING_POSTURE_MIN_RATIO
                        cat_at_water = (inter / box_area) > 0.15 and drinking_posture
                        if cat_at_water:
                            geometric_drinking = True
                            trigger_source = 'geometric_overlap'
            temporal_classifier_active = temporal_gate.observe(
                now, cat_candidate, bool(closeup_water), drink_clf_conf, cat_box)
            active_anchor_threshold = (NIGHT_CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD
                                       if night_mode else CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD)
            direct_lapping_anchor = bool(
                cat_candidate and closeup_water and drink_clf_conf >= active_anchor_threshold)
            active_drink_evidence.observe(now, direct_lapping_anchor, cat_box)
            if not event_active and not temporal_classifier_active:
                active_drink_evidence.prune(
                    now - max(3.0, CLASSIFIER_TEMPORAL_WINDOW_SEC + 1.0))
            if temporal_classifier_active:
                cat_at_water = True
                classifier_drinking = True
                temporal_confidence = max(
                    float(drink_clf_conf), float(temporal_gate.last_confidence))
                trigger_source = f'classifier_temporal({temporal_confidence:.2f})'
            cat_score = result.get("cat_score", 0)
            cat_near_water = False
            water_occupied_supported = False
            if cat_visible and cat_box and len(cat_box) == 4 and wzone:
                mouth_x, mouth_y = estimate_mouth_point(cat_box, wzone)
                # FIX: add Y proximity check (was missing — only checked X before!)
                cat_near_water = (wzone[0] - 60 <= mouth_x <= wzone[2] + 60 and
                                  wzone[1] - 70 <= mouth_y <= wzone[3] + 90 and
                                  cat_score >= NEAR_WATER_SCORE_MIN)
                water_occupied_supported = (
                    classifier_drinking or
                    geometric_drinking or
                    (detector._drink_clf_session is not None and
                     drink_clf_conf >= classifier_threshold)
                )
            # Close-up is a candidate cue only. A drinking event needs classifier
            # confirmation or strict mouth-to-water geometry.
            drinking = (
                classifier_drinking or
                geometric_drinking or
                (water_occupied and cat_visible and cat_near_water and water_occupied_supported and cat_score >= 0.35) or
                False
            )
            if drinking and not trigger_source:
                if water_occupied and cat_visible and cat_near_water and water_occupied_supported:
                    trigger_source = 'water_occupied+cat'
                else:
                    trigger_source = 'unknown'

            night_review_gate = (
                not night_mode or
                (drink_clf_conf >= NIGHT_REVIEW_CANDIDATE_CLF_MIN and
                 result.get('cat_score', 0) >= NIGHT_REVIEW_CANDIDATE_CAT_MIN))
            if (cat_candidate and closeup_water and not drinking and
                    not event_active and detector._drink_clf_session is not None
                    and night_review_gate):
                try:
                    review_recorder.capture(
                        frame, cat_box, drink_clf_conf,
                        result.get('cat_score', 0), night_mode)
                except Exception as exc:
                    print(f'REVIEW SAMPLE FAILED: {exc}')

            if not event_active:
                previous_start = cat_in_zone_since
                cat_in_zone_since = candidate_gate.observe(now, drinking)
                if previous_start is not None and cat_in_zone_since != previous_start:
                    peak_cat_score = 0.0
                    event_peak_clf_conf = 0.0
                if cat_in_zone_since is None:
                    last_cat_in_zone = None

            # ── Prolonged near-presence diagnostic ─────────────
            if cat_near_water and not drinking:
                if cat_near_since is None:
                    cat_near_since = now
            else:
                cat_near_since = None

            evidence_recorder.observe(frame, fzone, wzone, cat_box, {
                'cat_score': round(float(cat_score), 3),
                'clf_confidence': round(float(drink_clf_conf), 3),
                'classifier_threshold': round(float(classifier_threshold), 3),
                'cat_visible': bool(cat_visible),
                'water_occupied': bool(water_occupied),
                'closeup_candidate': bool(closeup_water),
                'drinking_candidate': bool(drinking),
                'temporal_classifier_active': bool(temporal_classifier_active),
                'cat_box': [int(value) for value in cat_box] if len(cat_box) == 4 else [],
                'trigger_source': trigger_source,
                'profile': 'night' if night_mode else 'day',
            })

            # ── State machine ──────────────────────────────────
            if drinking:
                last_cat_in_zone = now

                if cat_score > peak_cat_score:
                    peak_cat_score = cat_score
                if drink_clf_conf > event_peak_clf_conf:
                    event_peak_clf_conf = drink_clf_conf
                if cat_in_zone_since is None:
                    cat_in_zone_since = now

                duration = now - cat_in_zone_since
                min_sec = trigger_min_duration(trigger_source, night_mode)
                if not event_active and duration >= min_sec:
                    event_active = True
                    current_event_id = f"event-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    event_trigger_source = trigger_source or 'unknown'
                    event_profile = 'night' if night_mode else 'day'
                    evidence_recorder.start(current_event_id, {
                        'model': get_active_model_path(),
                        'trigger_source': event_trigger_source,
                        'profile': event_profile,
                        'cat_score': round(float(peak_cat_score), 3),
                        'drinking_confidence': round(float(event_peak_clf_conf), 3),
                        'classifier_threshold': round(float(classifier_threshold), 3),
                    })
                    snap_path = str(EVENT_DIR / f'snap-{current_event_id}.jpg')
                    snapshot_frame = best_candidate_frame if best_candidate_frame is not None else frame
                    cv2.imwrite(snap_path, snapshot_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    crop_url = ''
                    if best_candidate_crop is not None and best_candidate_crop.size:
                        crop_path = EVENT_DIR / f'crop-{current_event_id}.jpg'
                        if cv2.imwrite(str(crop_path), best_candidate_crop,
                                      [cv2.IMWRITE_JPEG_QUALITY, 90]):
                            crop_url = f'/media/events/{crop_path.name}'
                    elif cat_box and len(cat_box) == 4:
                        x1 = max(0, min(w - 1, cat_box[0]))
                        y1 = max(0, min(h - 1, cat_box[1]))
                        x2 = max(x1 + 1, min(w, cat_box[2]))
                        y2 = max(y1 + 1, min(h, cat_box[3]))
                        crop = frame[y1:y2, x1:x2]
                        if crop.size:
                            crop_path = EVENT_DIR / f'crop-{current_event_id}.jpg'
                            if cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                                crop_url = f'/media/events/{crop_path.name}'
                    print(f"DRINK START: {current_event_id}, score={peak_cat_score:.3f}, "
                          f"duration={duration:.1f}s, min={min_sec:.1f}s, "
                          f"night={night_mode}, trigger={event_trigger_source}, clf={event_peak_clf_conf:.2f}")
                    try:
                        start_time = datetime.fromtimestamp(cat_in_zone_since).astimezone().isoformat(timespec='seconds')
                        post_json('/api/events', {
                            'event_id': current_event_id,
                            'type': 'drink_start',
                            'start_time': start_time,
                            'confidence': event_peak_clf_conf if detector._drink_clf_session is not None else peak_cat_score,
                            'cat_confidence': round(peak_cat_score, 3),
                            'drinking_confidence': round(event_peak_clf_conf, 3),
                            'snapshot_url': f'/media/events/snap-{current_event_id}.jpg',
                            'crop_url': crop_url,
                    'estimated_ml': 0.0,
                    'active_drink_sec': 0.0,
                    'ml_per_active_sec': float(settings.get('mlPerActiveSec', 0.45) or 0.45),
                    'volume_method': 'active_lapping_v2',
                            'trigger_source': event_trigger_source,
                            'profile': event_profile,
                            'clf_confidence': round(event_peak_clf_conf, 3),
                            'summary': f'自动检测触发源：{event_trigger_source}，模式：{event_profile}，分类器峰值：{event_peak_clf_conf:.2f}',
                        })
                    except Exception as e:
                        print(f"  Post failed: {e}")

            else:
                if cat_in_zone_since is not None and last_cat_in_zone is not None:
                    gap = now - last_cat_in_zone
                    if gap > DRINK_END_GAP_SEC:
                        finalize_active_event('no_drinking', last_cat_in_zone)

            # ── Status output ──────────────────────────────────
            if time.time() - last_fps_check >= 5:
                fps = frame_count / (time.time() - last_fps_check)
                duration = (now - cat_in_zone_since) if cat_in_zone_since and drinking else 0
                nflag = "NIGHT" if night_mode else "DAY"
                trig = trigger_source if trigger_source else '-'
                clf_str = f"clf={drink_clf_conf:.2f}" if detector._drink_clf_session else 'clf=off'
                print(f"FPS:{fps:.1f} cat_vis={cat_visible} wsurf={cat_at_water} "
                      f"water={water_occupied} closeup={closeup_water} drinking={event_active} dur={duration:.1f}s "
                      f"score={result.get('cat_score',0):.2f} {clf_str} trig={trig} [{nflag}]")
                frame_count = 0
                last_fps_check = time.time()

            if time.time() - last_status > 2:
                streak = (now - cat_in_zone_since) if cat_in_zone_since and drinking else 0.0
                write_status({
                    'ok': True, 'phase': 'running',
                    'cat_visible': cat_visible,
                    'cat_at_water_surface': cat_at_water,
                    'water_occupied': water_occupied,
                    'closeup_water': closeup_water,
                    'cat_in_water_zone': cat_in_water_zone,
                    'water_motion': water_motion,
                    'drink_active': event_active,
                    'active_drink_sec': round(active_drink_evidence.duration(), 1),
                    'cat_score': result.get('cat_score', 0),
                    'stream_url': STREAM_URL,
                    'fps': round(fps, 1),
                    'motion_detected': should_detect,
                    'streak_sec': round(streak, 1),
                    'last_drink_at': last_drink_end_at,
                    'detection_mode': 'simplified_v3',
                    'drinking_classifier': detector._drink_clf_session is not None,
                    'clf_confidence': round(drink_clf_conf, 3),
                    'clf_threshold': round(classifier_threshold, 3),
                    'geometric_clf_floor': round(geometric_clf_floor, 3),
                    'trigger_source': trigger_source,
                    'profile': 'night' if night_mode else 'day',
                    'night_mode': night_mode,
                    'cat_last_seen_at': last_cat_seen_at,
                    'review_queue_size': review_recorder.count,
                })
                last_status = time.time()

            evidence_recorder.tick()
            time.sleep(0.15)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

if __name__ == '__main__':
    main()
