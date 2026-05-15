#!/usr/bin/env python3
"""Cat drinking detection worker - simplified.
YOLO detects cat. If cat stays in fountain zone > threshold, record drinking event.
Motion pre-filter skips inference when nothing changes, saving ~90% CPU.

Night mode: auto-detects IR/night footage via saturation analysis, applies CLAHE
contrast enhancement, and lowers detection thresholds for improved nighttime accuracy.
"""
import json, os, sys, time, urllib.request
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

# ── Config from env ──────────────────────────────────────────────
STREAM_URL = os.environ.get('CAMERA_ANALYSIS_STREAM_URL', 'http://cat-water-qsv-transcode:39100/live.flv')
SETTINGS_FILE = Path(os.environ.get('SETTINGS_FILE', '/app/data/settings.json'))
STATUS_FILE = Path(os.environ.get('DETECTOR_STATUS_FILE', '/app/data/detector_status.json'))
EVENT_DIR = Path(os.environ.get('DETECTOR_EVENT_DIR', '/app/media/events'))
DASHBOARD_API_URL = os.environ.get('DASHBOARD_API_URL', 'http://cat-water-dashboard:39095')
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
# Prolonged near-presence: if cat stays near water zone with high score for this long, trigger drinking
PROLONGED_NEAR_SEC = float(os.environ.get('PROLONGED_NEAR_SEC', '8.0'))
NEAR_WATER_SCORE_MIN = float(os.environ.get('NEAR_WATER_SCORE_MIN', '0.40'))

CAT_SCORE_THRESHOLD = float(os.environ.get('CAT_SCORE_THRESHOLD', '0.40'))
CAT_CROP_SCORE_THRESHOLD = float(os.environ.get('CAT_CROP_SCORE_THRESHOLD', '0.25'))
DETECT_INTERVAL_SEC = float(os.environ.get('DETECTOR_INTERVAL_SEC', '1.0'))
CAT_MIN_BOX_AREA_RATIO = float(os.environ.get('CAT_MIN_BOX_AREA_RATIO', '0.005'))
BOTTOM_CROP_START_RATIO = float(os.environ.get('BOTTOM_CROP_START_RATIO', '0.45'))
RECONNECT_SEC = float(os.environ.get('DETECTOR_RECONNECT_SEC', '4'))

# Motion pre-filter
MOTION_THRESHOLD = float(os.environ.get('DETECTOR_MOTION_THRESHOLD', '6.0'))
MOTION_RESIZE = (320, 180)

# Drinking thresholds
CAT_NEAR_OVERLAP = float(os.environ.get('CAT_NEAR_OVERLAP_THRESHOLD', '0.03'))
CAT_NEAR_PROXIMITY = float(os.environ.get('CAT_NEAR_PROXIMITY_PX', '50'))
DRINK_MIN_SEC = float(os.environ.get('DRINK_MIN_DURATION_SEC', '3.0'))
NIGHT_DRINK_MIN_SEC = float(os.environ.get("NIGHT_DRINK_MIN_DURATION_SEC", "8.0"))
NIGHT_DRINK_CONF_THRESHOLD = float(os.environ.get("NIGHT_DRINK_CONF_THRESHOLD", "0.35"))
DRINK_END_GAP_SEC = float(os.environ.get('DRINK_END_GAP_SEC', '5.0'))
STALE_DETECTION_SEC = float(os.environ.get('STALE_DETECTION_SEC', '3.0'))

# Night mode
NIGHT_ENABLED = os.environ.get('NIGHT_MODE_ENABLED', 'true').lower() != 'false'
NIGHT_SAT_THRESHOLD = float(os.environ.get('NIGHT_SATURATION_THRESHOLD', '20'))
NIGHT_VAL_THRESHOLD = float(os.environ.get('NIGHT_BRIGHTNESS_THRESHOLD', '80'))
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

EVENT_DIR.mkdir(parents=True, exist_ok=True)
STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_SETTINGS = {
    'roi': {'x1': 112, 'y1': 291, 'x2': 1132, 'y2': 620},
    'fountainZone': {'x1': 200, 'y1': 400, 'x2': 1080, 'y2': 700},
    'waterZone': {'x1': 350, 'y1': 350, 'x2': 650, 'y2': 480},
    'fpsSample': 2,
    'mlPerSec': 0.14,
}

# ── Helpers ──────────────────────────────────────────────────────
_clahe = None

def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def is_night_mode(frame):
    """Detect IR night mode via low saturation + low brightness."""
    if not NIGHT_ENABLED:
        return False
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))
    return s_mean < NIGHT_SAT_THRESHOLD and v_mean < NIGHT_VAL_THRESHOLD

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
    STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

def post_json(path, payload):
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(f'{DASHBOARD_API_URL}{path}', data=body,
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode('utf-8'))

def nms_suppress(items, iou_threshold=0.45):
    if not items:
        return []
    kept = []
    items = sorted(items, key=lambda x: -x.get('_combined_score', x.get('score', 0)))
    for item in items:
        bx1, by1, bx2, by2 = item['box']
        overlap = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k['box']
            ix1, iy1 = max(bx1, kx1), max(by1, ky1)
            ix2, iy2 = min(bx2, kx2), min(by2, ky2)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            union = max(1, (bx2-bx1)*(by2-by1) + (kx2-kx1)*(ky2-ky1) - inter)
            if inter / union > iou_threshold:
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

# ── Cat Detector ─────────────────────────────────────────────────
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
        self._load_model()

    def _load_model(self):
        model_path = get_active_model_path()
        model = Path(model_path)
        if model.exists():
            try:
                self.session = self._build_session(model)
                self._current_model_path = model_path
                print(f"Model loaded: {model}")
            except Exception as e:
                print(f"Model load FAILED for {model}: {e}")
                if self.session is None:
                    raise  # No fallback, must crash
                # Keep existing session if hot-reload fails
        else:
            print(f"Model not found: {model}")

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
                return ort.InferenceSession(str(model), providers=[p, 'CPUExecutionProvider'])
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
        predictions = output[0].T
        num_channels = len(predictions[0]) if len(predictions) > 0 else 0
        is_coco = 5 < num_channels < 100
        items = []
        for pred in predictions:
            if len(pred) < 5:
                continue
            x, y, bw, bh = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
            if is_coco:
                if len(pred) <= 4 + COCO_CAT_LABEL:
                    continue
                conf = float(pred[4 + COCO_CAT_LABEL])
                # Person rejection: if person class score dominates cat score,
                # this detection is more likely a person than a cat.
                if PERSON_REJECTION_ENABLED and len(pred) > 4 + COCO_PERSON_LABEL:
                    person_conf = float(pred[4 + COCO_PERSON_LABEL])
                    if person_conf > PERSON_SCORE_REJECT_THRESHOLD and person_conf > conf:
                        continue
            else:
                conf = float(pred[4])
            if conf < threshold:
                continue
            # Map letterbox coords back to original
            x1 = (x - pad_x - bw/2) / scale + offset_x
            y1 = (y - pad_y - bh/2) / scale + offset_y
            x2 = (x - pad_x + bw/2) / scale + offset_x
            y2 = (y - pad_y + bh/2) / scale + offset_y
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
                self._hires_cap = cv2.VideoCapture(NIGHT_HIRES_RTSP_URL, cv2.CAP_FFMPEG)
                self._hires_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # OpenCV's default RTSP timeout is 30s; we want fail-fast
                self._hires_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
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
            if box_area > max_area_ratio * frame_area:
                continue
            if box_area < CAT_MIN_BOX_AREA_RATIO * frame_area:
                continue
            bw, bh = bx2-bx1, by2-by1
            if bh > 0 and bw/bh > 3.5:
                continue
            if bx1 < -50 or by1 < -50 or bx2 > w+50 or by2 > h+50:
                continue
            if bx1 < 5 and bx2 > w-5:
                continue
            if night_mode and by1 < 10:
                continue
            area_ratio = box_area / frame_area
            filtered.append({**item, '_combined_score': float(item['score']) + min(0.2, area_ratio*0.5)})
        filtered = nms_suppress(filtered)
        best = max(filtered, key=lambda x: x['_combined_score']) if filtered else None
        if best:
            self._consecutive_cat += 1
            cat_detected = self._consecutive_cat >= CONSECUTIVE_FRAMES_REQUIRED
            self.last_result = {
                'cat_detected': cat_detected, 'cat_score': round(float(best['score']), 3),
                'cat_box': best['box'],
                'night_mode': night_mode,
                'water_occupied': water_occupied,
            }
        else:
            self._consecutive_cat = 0
            self.last_result = {'cat_detected': False, 'cat_score': 0.0, 'cat_box': [],
                                'night_mode': night_mode, 'water_occupied': water_occupied}
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
    last_detection_result = None   # Cached result when skipping YOLO
    last_detection_time = 0.0
    last_drink_end_at = None       # Timestamp of last drink event end
    fps = 0.0                      # Current FPS
    night_mode = False             # Current day/night state
    last_forced_inference = 0.0    # Last forced YOLO run in night mode

    while True:
        try:
            # Reconnect if needed
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    print("Failed to open stream, retrying...")
                    write_status({'ok': False, 'phase': 'connect_failed'})
                    time.sleep(RECONNECT_SEC)
                    continue
                print("Stream connected")
                write_status({'ok': True, 'phase': 'running'})

            ret, frame = cap.read()
            if not ret:
                print("Read failed, reconnecting...")
                cap.release()
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

            cat_box = result.get('cat_box', [])
            cat_visible = result.get('cat_detected', False)
            if cat_visible:
                last_cat_seen_at = now_iso()
            water_occupied = result.get('water_occupied', False)
            cat_at_water = False
            if cat_visible and cat_box and len(cat_box) == 4 and wzone:
                water_surface_y = settings.get("waterSurfaceY")
                if water_surface_y:
                    sx = w / max(1, settings.get("frameWidth", w))
                    sy = h / max(1, settings.get("frameHeight", h))
                    scaled_water_y = water_surface_y * sy
                    cat_head_x = (cat_box[0] + cat_box[2]) / 2
                    cat_head_y = cat_box[3]
                    in_water_x = wzone[0] <= cat_head_x <= wzone[2]
                    spans_water = cat_box[1] <= scaled_water_y <= cat_box[3]
                    head_near_water = abs(cat_head_y - scaled_water_y) < WATER_PROXIMITY_PX * sy
                    cat_at_water = in_water_x and (head_near_water or spans_water)
                else:
                    cat_at_water, _ = cat_in_fountain(cat_box, fzone)
            cat_score = result.get("cat_score", 0)
            cat_near_water = False
            if cat_visible and cat_box and len(cat_box) == 4 and wzone:
                cat_head_x_near = (cat_box[0] + cat_box[2]) / 2
                cat_near_water = wzone[0] <= cat_head_x_near <= wzone[2] and cat_score >= NEAR_WATER_SCORE_MIN
            drinking = cat_at_water or water_occupied

            now = time.time()

            # ── Prolonged near-presence safety net ─────────────
            if cat_near_water and not drinking:
                if cat_near_since is None:
                    cat_near_since = now
                elif now - cat_near_since >= PROLONGED_NEAR_SEC:
                    drinking = True
            else:
                cat_near_since = None

            # ── State machine ──────────────────────────────────
            if drinking:
                last_cat_in_zone = now

                if cat_score > peak_cat_score:
                    peak_cat_score = cat_score
                if cat_in_zone_since is None:
                    cat_in_zone_since = now

                duration = now - cat_in_zone_since
                min_sec = NIGHT_DRINK_MIN_SEC if night_mode else DRINK_MIN_SEC
                if not event_active and duration >= min_sec:
                    event_active = True
                    current_event_id = f"event-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    snap_path = str(EVENT_DIR / f'snap-{current_event_id}.jpg')
                    cv2.imwrite(snap_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if cat_box and len(cat_box) == 4:
                        crop = frame[cat_box[1]:cat_box[3], cat_box[0]:cat_box[2]]
                        if crop.size:
                            cv2.imwrite(str(EVENT_DIR / f'crop-{current_event_id}.jpg'),
                                       crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    print(f"DRINK START: {current_event_id}, score={peak_cat_score:.3f}, "
                          f"duration={duration:.1f}s, night={night_mode}")
                    try:
                        start_time = datetime.fromtimestamp(cat_in_zone_since).astimezone().isoformat(timespec='seconds')
                        post_json('/api/events', {
                            'event_id': current_event_id,
                            'type': 'drink_start',
                            'start_time': start_time,
                            'confidence': peak_cat_score,
                            'snapshot_url': f'/media/events/snap-{current_event_id}.jpg',
                            'estimated_ml': 0.0,
                        })
                    except Exception as e:
                        print(f"  Post failed: {e}")

            else:
                if cat_in_zone_since is not None and last_cat_in_zone is not None:
                    gap = now - last_cat_in_zone
                    if gap > DRINK_END_GAP_SEC:
                        if event_active:
                            total_duration = last_cat_in_zone - cat_in_zone_since
                            print(f"DRINK END: {current_event_id}, total_duration={total_duration:.1f}s, "
                                  f"gap={gap:.1f}s, night={night_mode}")
                            try:
                                end_time = datetime.fromtimestamp(last_cat_in_zone).astimezone().isoformat(timespec='seconds')
                                start_time = datetime.fromtimestamp(cat_in_zone_since).astimezone().isoformat(timespec='seconds')
                                post_json('/api/events', {
                                    'event_id': current_event_id,
                                    'type': 'drink_end',
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'duration_sec': round(total_duration, 1),
                                    'confidence': peak_cat_score,
                                    'estimated_ml': round(total_duration * settings.get('mlPerSec', 0.14), 1),
                                    'snapshot_url': f'/media/events/snap-{current_event_id}.jpg',
                                })
                            except Exception as e:
                                print(f"  Post failed: {e}")
                            last_drink_end_at = datetime.now().astimezone().isoformat(timespec='seconds')
                        cat_in_zone_since = None
                        event_active = False
                        current_event_id = None
                        peak_cat_score = 0.0

            # ── Status output ──────────────────────────────────
            if time.time() - last_fps_check >= 5:
                fps = frame_count / (time.time() - last_fps_check)
                duration = (now - cat_in_zone_since) if cat_in_zone_since and drinking else 0
                nflag = "NIGHT" if night_mode else "DAY"
                print(f"FPS:{fps:.1f} cat_vis={cat_visible} wsurf={cat_at_water} "
                      f"water={water_occupied} drinking={event_active} dur={duration:.1f}s "
                      f"score={result.get('cat_score',0):.2f} skip={not should_detect} [{nflag}]")
                frame_count = 0
                last_fps_check = time.time()

            if time.time() - last_status > 2:
                streak = (now - cat_in_zone_since) if cat_in_zone_since and drinking else 0.0
                write_status({
                    'ok': True, 'phase': 'running',
                    'cat_visible': cat_visible,
                    'cat_at_water_surface': cat_at_water,
                    'water_occupied': water_occupied,
                    'drink_active': event_active,
                    'cat_score': result.get('cat_score', 0),
                    'stream_url': STREAM_URL,
                    'fps': round(fps, 1),
                    'motion_detected': should_detect,
                    'streak_sec': round(streak, 1),
                    'last_drink_at': last_drink_end_at,
                    'detection_mode': 'simplified_v3',
                    'night_mode': night_mode,
                    'cat_last_seen_at': last_cat_seen_at,
                })
                last_status = time.time()

            time.sleep(0.15)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

if __name__ == '__main__':
    main()
