#!/usr/bin/env python3
"""Verify drinking detection fix: old (OR) vs new (water_occupied requires cat_visible).
Scans EasyNVR night recordings, simulates both logics side-by-side."""

import json, os, sys, time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

# Import detection logic from the actual worker
sys.path.insert(0, '/app')
# We'll inline the needed config/functions instead

SETTINGS_FILE = Path('/app/data/settings.json')
YOLO_IMG_SIZE = 640
COCO_CAT_LABEL = 15
COCO_PERSON_LABEL = 0

# Load settings
try:
    settings = json.loads(SETTINGS_FILE.read_text())
except:
    settings = {}

frame_w_ref = settings.get('frameWidth', 1280)
frame_h_ref = settings.get('frameHeight', 720)

def scale_zone(zone, sw, sh, fw, fh):
    if not zone: return None
    sx = fw / max(1, sw)
    sy = fh / max(1, sh)
    return (int(zone['x1']*sx), int(zone['y1']*sy), int(zone['x2']*sx), int(zone['y2']*sy))

fzone = scale_zone(settings.get('fountainZone', {}), frame_w_ref, frame_h_ref, 960, 540)
wzone = scale_zone(settings.get('waterZone', {}), frame_w_ref, frame_h_ref, 960, 540)
waterSurfaceY = settings.get('waterSurfaceY', 243)
sy_scale = 540 / max(1, frame_h_ref)
scaled_water_y = waterSurfaceY * sy_scale

# Thresholds (matching detection_worker.py)
WATER_PROXIMITY_PX = 50 * sy_scale
NIGHT_WATER_FG_MIN = 80
NIGHT_WATER_BG_FRAMES = 30

# Load ONNX model
import onnxruntime as ort
model_path = '/app/models/yolov8n_640_opset21.onnx'
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size/w, size/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh))
    pad = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size-nw)//2, (size-nh)//2
    pad[dy:dy+nh, dx:dx+nw] = resized
    return pad, scale, dx, dy

def yolo_detect(frame, thresh=0.40):
    lb, scale, dx, dy = letterbox(frame)
    blob = lb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    out = session.run(None, {input_name: blob})[0][0]
    h, w = frame.shape[:2]
    cats = []
    for row in out:
        cx, cy, bw, bh = row[:4]
        cls_scores = 1.0 / (1.0 + np.exp(-row[4:]))
        cat_idx = 15
        person_idx = 0
        if cls_scores[cat_idx] >= thresh and cls_scores[cat_idx] > cls_scores[person_idx]:
            ix = int((cx - dx) / scale)
            iy = int((cy - dy) / scale)
            iw = int(bw / scale)
            ih = int(bh / scale)
            bx1, by1 = max(0, ix-iw//2), max(0, iy-ih//2)
            bx2, by2 = min(w, ix+iw//2), min(h, iy+ih//2)
            cats.append({'box': [bx1, by1, bx2, by2], 'score': float(cls_scores[cat_idx])})
    return cats

def is_night(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:,:,1])) < 20 and float(np.mean(hsv[:,:,2])) < 80

# Water zone background for night mode
wz_bg = None
wz_bg_frames = 0

def check_water_occupied(frame, night_mode):
    global wz_bg, wz_bg_frames
    if not wzone:
        return False
    wx1, wy1, wx2, wy2 = wzone
    wcrop = frame[wy1:wy2, wx1:wx2]
    if wcrop.size == 0:
        return False
    if night_mode:
        gray = cv2.cvtColor(wcrop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if wz_bg is None or wz_bg.shape != gray.shape:
            wz_bg = gray
            wz_bg_frames = 1
            return False
        bg_diff = float(np.mean(cv2.absdiff(gray, wz_bg)))
        if bg_diff < 6.0:
            cv2.accumulateWeighted(gray, wz_bg, 0.02)
            wz_bg_frames += 1
        if wz_bg_frames > NIGHT_WATER_BG_FRAMES:
            diff = cv2.absdiff(gray, wz_bg)
            _, diff_th = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            diff_th = cv2.medianBlur(diff_th.astype(np.uint8), 3)
            fg_px = np.count_nonzero(diff_th)
            if NIGHT_WATER_FG_MIN < fg_px < 5000:
                return True
        return False
    else:
        # Day: YOLO on water zone crop with low threshold
        wz_cands = yolo_detect(wcrop, thresh=0.18)
        return any(c['score'] >= 0.18 for c in wz_cands)

def check_cat_at_water(cat_box, wzone):
    if not cat_box or len(cat_box) != 4 or not wzone:
        return False
    cat_head_x = (cat_box[0] + cat_box[2]) / 2
    cat_head_y = cat_box[3]
    in_water_x = wzone[0] <= cat_head_x <= wzone[2]
    spans_water = cat_box[1] <= scaled_water_y <= cat_box[3]
    head_near_water = abs(cat_head_y - scaled_water_y) < WATER_PROXIMITY_PX
    return in_water_x and (head_near_water or spans_water)

# ── Scan recordings ──
REC_BASE = '/easynvr_rec/easynvr_rec/EASYNVR_CHANNEL_PLACEHOLDER/profile_1'

def scan_recordings(date_str, hour_start, hour_end, sample_sec=3):
    """Scan recordings for a date range, sample frames, run both logics."""
    global wz_bg, wz_bg_frames
    rec_dir = f'{REC_BASE}/{date_str}'
    if not os.path.isdir(rec_dir):
        print(f'  No recordings for {date_str}')
        return
    
    files = sorted([f for f in os.listdir(rec_dir) if f.endswith('.mp4')])
    
    # Filter by hour range
    target_files = []
    for f in files:
        # Format: 20260518000038-60928.mp4
        ts = f[:14]  # 20260518000038
        hour = int(ts[8:10])
        if hour_start <= hour_end:
            if hour_start <= hour < hour_end:
                target_files.append(f)
        else:  # wraps midnight
            if hour >= hour_start or hour < hour_end:
                target_files.append(f)
    
    if not target_files:
        print(f'  No files in hour range {hour_start}-{hour_end}')
        return
    
    print(f'  Found {len(target_files)} recordings in {date_str} {hour_start:02d}:00-{hour_end:02d}:00')
    
    old_events = []  # (timestamp, score, cat_vis, water_occ, reason)
    new_events = []
    
    # State machine for old logic
    old_in_zone_since = None
    old_event_active = False
    old_last_in_zone = None
    old_event_id = None
    old_peak_score = 0
    
    # State machine for new logic
    new_in_zone_since = None
    new_event_active = False
    new_last_in_zone = None
    new_event_id = None
    new_peak_score = 0
    
    frame_count = 0
    
    for fname in target_files:
        fpath = os.path.join(rec_dir, fname)
        ts_str = fname[:14]
        rec_time = datetime.strptime(ts_str, '%Y%m%d%H%M%S')
        
        cap = cv2.VideoCapture(fpath)
        if not cap.isOpened():
            continue
        
        fps_rec = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_skip = max(1, int(fps_rec * sample_sec))
        
        fidx = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Scale to transcode size
            frame = cv2.resize(frame, (960, 540))
            frame_count += 1
            night = is_night(frame)
            
            # Run YOLO (full frame)
            cats = yolo_detect(frame, thresh=0.28 if night else 0.40)
            # Also bottom crop
            h, w = frame.shape[:2]
            crop_start = int(h * 0.45)
            crop = frame[crop_start:h, 0:w]
            if crop.size:
                crop_thresh = 0.15 if night else 0.25
                cats.extend(yolo_detect(crop, thresh=crop_thresh))
            
            cat_vis = len(cats) > 0
            best = max(cats, key=lambda c: c['score']) if cats else None
            cat_score = best['score'] if best else 0.0
            cat_box = best['box'] if best else []
            
            water_occ = check_water_occupied(frame, night)
            cat_at_water = check_cat_at_water(cat_box, wzone) if cat_vis else False
            
            # OLD logic: drinking = cat_at_water or water_occupied
            old_drinking = cat_at_water or water_occ
            
            # NEW logic: water_occupied requires cat_visible
            new_drinking = cat_at_water or (water_occ and cat_vis)
            
            now = time.time()
            nflag = "NIGHT" if night else "DAY"
            
            # Track both state machines
            for logic, drinking, in_zone_since, last_in_zone, event_active, event_id, peak_score, events_list, label in [
                ("OLD", old_drinking, old_in_zone_since, old_last_in_zone, old_event_active, old_event_id, old_peak_score, old_events, "old"),
                ("NEW", new_drinking, new_in_zone_since, new_last_in_zone, new_event_active, new_event_id, new_peak_score, new_events, "new"),
            ]:
                pass  # We'll track separately
            
            # OLD state machine
            if old_drinking:
                if old_in_zone_since is None:
                    old_in_zone_since = rec_time
                old_last_in_zone = rec_time
                if cat_score > old_peak_score:
                    old_peak_score = cat_score
                dur = (rec_time - old_in_zone_since).total_seconds()
                if not old_event_active and dur >= 3.0:
                    old_event_active = True
                    old_event_id = f"old-{rec_time.strftime('%H%M%S')}"
                    old_events.append({
                        'id': old_event_id, 'start': str(old_in_zone_since),
                        'trigger': f'cat_vis={cat_vis} water={water_occ} at_water={cat_at_water}',
                        'night': night, 'score': old_peak_score
                    })
            else:
                if old_in_zone_since and old_last_in_zone:
                    gap = (rec_time - old_last_in_zone).total_seconds()
                    if gap > 5.0:
                        if old_event_active:
                            total = (old_last_in_zone - old_in_zone_since).total_seconds()
                            old_events[-1]['end'] = str(old_last_in_zone)
                            old_events[-1]['duration'] = total
                        old_in_zone_since = None
                        old_event_active = False
                        old_peak_score = 0
            
            # NEW state machine
            if new_drinking:
                if new_in_zone_since is None:
                    new_in_zone_since = rec_time
                new_last_in_zone = rec_time
                if cat_score > new_peak_score:
                    new_peak_score = cat_score
                dur = (rec_time - new_in_zone_since).total_seconds()
                if not new_event_active and dur >= 3.0:
                    new_event_active = True
                    new_event_id = f"new-{rec_time.strftime('%H%M%S')}"
                    new_events.append({
                        'id': new_event_id, 'start': str(new_in_zone_since),
                        'trigger': f'cat_vis={cat_vis} water={water_occ} at_water={cat_at_water}',
                        'night': night, 'score': new_peak_score
                    })
            else:
                if new_in_zone_since and new_last_in_zone:
                    gap = (rec_time - new_last_in_zone).total_seconds()
                    if gap > 5.0:
                        if new_event_active:
                            total = (new_last_in_zone - new_in_zone_since).total_seconds()
                            new_events[-1]['end'] = str(new_last_in_zone)
                            new_events[-1]['duration'] = total
                        new_in_zone_since = None
                        new_event_active = False
                        new_peak_score = 0
            
            fidx += frame_skip
        
        cap.release()
    
    print(f'  Sampled {frame_count} frames')
    print(f'  OLD logic events: {len(old_events)}')
    for e in old_events:
        print(f'    {e["id"]} | start={e["start"]} | night={e["night"]} | {e["trigger"]} | score={e["score"]:.3f} | dur={e.get("duration","?"):.1f}s')
    print(f'  NEW logic events: {len(new_events)}')
    for e in new_events:
        print(f'    {e["id"]} | start={e["start"]} | night={e["night"]} | {e["trigger"]} | score={e["score"]:.3f} | dur={e.get("duration","?"):.1f}s')
    
    # Analysis
    old_night = [e for e in old_events if e['night']]
    new_night = [e for e in new_events if e['night']]
    old_day = [e for e in old_events if not e['night']]
    new_day = [e for e in new_events if not e['night']]
    print(f'\n  Summary:')
    print(f'    Night: OLD={len(old_night)} events → NEW={len(new_night)} events (eliminated {len(old_night)-len(new_night)} false positives)')
    print(f'    Day:   OLD={len(old_day)} events → NEW={len(new_day)} events')

# ── Run tests ──
print("=" * 70)
print("Drinking Detection Fix Verification")
print("Comparing OLD (or) vs NEW (water_occupied requires cat_visible)")
print("=" * 70)

# Test 1: Last night (May 17, 21:00 → May 18, 06:00) - should be IR mode
print("\n--- Test 1: Night recordings May 17 21:00 - May 18 06:00 ---")
wz_bg = None; wz_bg_frames = 0
scan_recordings('20260517', 21, 24, sample_sec=3)

print("\n--- Test 2: Night recordings May 18 00:00 - 06:00 ---")
wz_bg = None; wz_bg_frames = 0
scan_recordings('20260518', 0, 6, sample_sec=3)

# Test 2: Daytime recordings (May 17, 10:00-16:00) - should be color
print("\n--- Test 3: Day recordings May 17 10:00 - 16:00 ---")
wz_bg = None; wz_bg_frames = 0
scan_recordings('20260517', 10, 16, sample_sec=3)

print("\n" + "=" * 70)
print("Done!")
