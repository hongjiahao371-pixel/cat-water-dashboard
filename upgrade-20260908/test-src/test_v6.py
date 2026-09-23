#!/usr/bin/env python3
"""Corrected verification: use actual detection_worker.py inference logic."""
import json, os, cv2, numpy as np
from pathlib import Path
import onnxruntime as ort

SETTINGS_FILE = Path('/app/data/settings.json')
settings = json.loads(SETTINGS_FILE.read_text()) if SETTINGS_FILE.exists() else {}
fw = settings.get('frameWidth', 1280)
fh = settings.get('frameHeight', 720)
REC_BASE = '/easynvr_rec/easynvr_rec/EASYNVR_CHANNEL_PLACEHOLDER/profile_1'
YOLO_IMG_SIZE = 640
COCO_CAT = 15
COCO_PERSON = 0

def sz(zone, sw, sh, fw2, fh2):
    if not zone: return None
    sx, sy = fw2/max(1,sw), fh2/max(1,sh)
    return (int(zone['x1']*sx), int(zone['y1']*sy), int(zone['x2']*sx), int(zone['y2']*sy))

fzone = sz(settings.get('fountainZone',{}), fw, fh, 960, 540)
wzone = sz(settings.get('waterZone',{}), fw, fh, 960, 540)
wsy = settings.get('waterSurfaceY', 243) * (540/max(1,fh))
prox = 50 * (540/max(1,fh))

session = ort.InferenceSession('/app/models/yolov8n_640_opset21.onnx', providers=['CPUExecutionProvider'])

# CLAHE for night
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def enhance_night(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def yolo_detect(frame, threshold, night_mode=False):
    """Exact same logic as CatDetector._infer_candidates"""
    h, w = frame.shape[:2]
    if night_mode:
        frame = enhance_night(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scale = min(YOLO_IMG_SIZE / w, YOLO_IMG_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(rgb, (nw, nh))
    pad_x = (YOLO_IMG_SIZE - nw) // 2
    pad_y = (YOLO_IMG_SIZE - nh) // 2
    letterbox = np.ones((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8) * 114
    letterbox[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    inp = letterbox.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[np.newaxis, :, :, :]
    output = session.run(None, {'images': inp})[0]
    predictions = output[0].T  # (8400, 84) — THIS WAS THE BUG
    cats = []
    for pred in predictions:
        if len(pred) < 5 + COCO_CAT: continue
        x, y, bw, bh = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
        conf = float(pred[4 + COCO_CAT])  # sigmoid already fused, no need to apply again
        person_conf = float(pred[4 + COCO_PERSON])
        if person_conf > 0.30 and person_conf > conf: continue  # person rejection
        if conf < threshold: continue
        x1 = int((x - pad_x - bw/2) / scale)
        y1 = int((y - pad_y - bh/2) / scale)
        x2 = int((x - pad_x + bw/2) / scale)
        y2 = int((y - pad_y + bh/2) / scale)
        cats.append({'score': conf, 'box': [max(0,x1), max(0,y1), min(w,x2), min(h,y2)]})
    return cats

def is_night(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:,:,1]))<20 and float(np.mean(hsv[:,:,2]))<80

def analyze_recording(fpath, label, sample_interval=3):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  File: {os.path.basename(fpath)}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(fpath)
    if not cap.isOpened():
        print("  CANNOT OPEN"); return
    fps_r = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps_r
    skip = max(1, int(fps_r * sample_interval))
    
    wz_bg = None; wz_frames = 0
    
    print(f"  Duration: {duration:.0f}s, sampling every {sample_interval}s")
    print(f"  {'Time':>5} | {'Ngt':>3} | {'CatV':>4} | {'Score':>5} | {'Wtr':>3} | {'AtW':>3} | OLD | NEW")
    print(f"  {'-'*55}")
    
    old_y=0; new_y=0; total=0
    
    fi = 0
    while fi < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (960,540))
        total += 1
        sec = fi / fps_r
        is_n = is_night(frame)
        
        score_thresh = 0.28 if is_n else 0.40
        crop_thresh = 0.15 if is_n else 0.25
        
        cats = yolo_detect(frame, score_thresh, night_mode=is_n)
        # Bottom crop pass
        h2,w2 = frame.shape[:2]; cs = int(h2*0.45)
        cr = frame[cs:h2,0:w2]
        if cr.size:
            crop_cats = yolo_detect(cr, crop_thresh, night_mode=is_n)
            for c in crop_cats:
                c['box'] = [c['box'][0], c['box'][1]+cs, c['box'][2], c['box'][3]+cs]
            cats.extend(crop_cats)
        
        vis = len(cats)>0
        best = max(cats, key=lambda c:c['score']) if cats else None
        sc = best['score'] if best else 0
        box = best['box'] if best else []
        
        # Water zone
        wo = False
        if wzone:
            wx1,wy1,wx2,wy2 = wzone
            wc = frame[wy1:wy2,wx1:wx2]
            if wc.size:
                if is_n:
                    g = cv2.cvtColor(wc,cv2.COLOR_BGR2GRAY).astype(np.float32)
                    if wz_bg is None or wz_bg.shape!=g.shape:
                        wz_bg=g; wz_frames=1
                    else:
                        bd = float(np.mean(cv2.absdiff(g,wz_bg)))
                        if bd<6.0:
                            cv2.accumulateWeighted(g,wz_bg,0.02); wz_frames+=1
                        if wz_frames>30:
                            df = cv2.absdiff(g,wz_bg)
                            _,dt = cv2.threshold(df,20,255,cv2.THRESH_BINARY)
                            dt = cv2.medianBlur(dt.astype(np.uint8),3)
                            fg = np.count_nonzero(dt)
                            if 80<fg<5000: wo=True
                else:
                    wc_cands = yolo_detect(wc, 0.18)
                    wo = any(c['score']>=0.18 for c in wc_cands)
        
        atw = False
        if vis and box and len(box)==4 and wzone:
            hx = (box[0]+box[2])/2; hy = box[3]
            inx = wzone[0]<=hx<=wzone[2]
            spans = box[1]<=wsy<=box[3]
            near = abs(hy-wsy)<prox
            atw = inx and (near or spans)
        
        old_drink = atw or wo
        new_drink = atw or (wo and vis)
        
        if old_drink: old_y += 1
        if new_drink: new_y += 1
        
        # Print all frames with any signal
        marker = ""
        if old_drink and not new_drink: marker = " <<< ELIMINATED"
        elif old_drink and new_drink: marker = " <<< BOTH"
        if vis or wo or atw or old_drink:
            print(f"  {sec:4.0f}s | {'Y' if is_n else 'n':>3} | {'Y' if vis else '-':>4} | {sc:.3f} | {'Y' if wo else '-':>3} | {'Y' if atw else '-':>3} | {'Y' if old_drink else '-':>3} | {'Y' if new_drink else '-':>3}{marker}")
        
        fi += skip
    
    cap.release()
    elim = old_y - new_y
    pct = 100*elim/max(1,old_y) if old_y > 0 else 0
    print(f"\n  >>> Total: {total} frames | OLD: {old_y} triggers | NEW: {new_y} triggers | Eliminated: {elim} ({pct:.0f}%)")

# Pick recordings
recordings = [
    # Night known false-positive periods
    (f'{REC_BASE}/20260517/20260517232505-60928.mp4', 'Night May17 23:25 (false positive period)'),
    (f'{REC_BASE}/20260517/20260517220051-60928.mp4', 'Night May17 22:00 (deep IR)'),
    (f'{REC_BASE}/20260518/20260518010034-60928.mp4', 'Night May18 01:00 (deep IR)'),
    (f'{REC_BASE}/20260518/20260518030033-60928.mp4', 'Night May18 03:00 (deep IR)'),
    # Day with known real events
    (f'{REC_BASE}/20260517/20260517120045-60800.mp4', 'Day May17 12:00 (color mode)'),
    # Today before fix
    (f'{REC_BASE}/20260518/20260518103026-60928.mp4', 'Day May18 10:30 (before fix)'),
]

print("Model loaded. Starting verification...")
for fpath, label in recordings:
    if os.path.exists(fpath):
        analyze_recording(fpath, label, sample_interval=2)
    else:
        # Find closest
        d = os.path.dirname(fpath)
        prefix = os.path.basename(fpath)[:12]
        if os.path.isdir(d):
            files = sorted(f for f in os.listdir(d) if f.endswith('.mp4') and f.startswith(prefix))
            if files:
                analyze_recording(os.path.join(d, files[0]), label+f' [{files[0]}]', sample_interval=2)
            else:
                print(f"\n  SKIP {label}: no matching file")
        else:
            print(f"\n  SKIP {label}: dir not found")

print(f"\n{'='*60}")
print("DONE - Fix eliminates water_occupied-only false triggers")
print("OLD = atw or water_occupied (any water signal triggers drinking)")
print("NEW = atw or (water_occupied AND cat_visible) (water only confirms when cat detected)")
print(f"{'='*60}")
