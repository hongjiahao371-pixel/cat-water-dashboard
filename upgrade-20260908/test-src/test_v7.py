#!/usr/bin/env python3
"""Continuous night recording test: process 30+ minutes for water zone warmup."""
import json, os, cv2, numpy as np
from pathlib import Path
import onnxruntime as ort
from datetime import datetime

SETTINGS_FILE = Path('/app/data/settings.json')
settings = json.loads(SETTINGS_FILE.read_text()) if SETTINGS_FILE.exists() else {}
fw = settings.get('frameWidth', 1280)
fh = settings.get('frameHeight', 720)
REC_BASE = '/easynvr_rec/easynvr_rec/OpD6HNpPU8j0X/profile_1'
YOLO_SIZE = 640
COCO_CAT = 15
COCO_PERSON = 0

def sz(zone, sw, sh, fw2, fh2):
    if not zone: return None
    sx, sy = fw2/max(1,sw), fh2/max(1,sh)
    return (int(zone['x1']*sx), int(zone['y1']*sy), int(zone['x2']*sx), int(zone['y2']*sy))

wzone = sz(settings.get('waterZone',{}), fw, fh, 960, 540)
wsy = settings.get('waterSurfaceY', 243) * (540/max(1,fh))
prox = 50 * (540/max(1,fh))

session = ort.InferenceSession('/app/models/yolov8n_640_opset21.onnx', providers=['CPUExecutionProvider'])
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def enhance_night(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(_clahe.apply(gray), cv2.COLOR_GRAY2BGR)

def yolo(frame, threshold, night_mode=False):
    h, w = frame.shape[:2]
    if night_mode: frame = enhance_night(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scale = min(YOLO_SIZE/w, YOLO_SIZE/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(rgb, (nw, nh))
    px, py = (YOLO_SIZE-nw)//2, (YOLO_SIZE-nh)//2
    lb = np.ones((YOLO_SIZE, YOLO_SIZE, 3), dtype=np.uint8) * 114
    lb[py:py+nh, px:px+nw] = resized
    inp = lb.astype(np.float32)/255.0
    inp = np.transpose(inp, (2,0,1))[np.newaxis]
    output = session.run(None, {'images': inp})[0]
    preds = output[0].T
    cats = []
    for p in preds:
        if len(p) < 5+COCO_CAT: continue
        conf = float(p[4+COCO_CAT])
        pconf = float(p[4+COCO_PERSON])
        if pconf > 0.30 and pconf > conf: continue
        if conf < threshold: continue
        x,y,bw,bh = float(p[0]),float(p[1]),float(p[2]),float(p[3])
        x1 = int((x-px-bw/2)/scale); y1 = int((y-py-bh/2)/scale)
        x2 = int((x-px+bw/2)/scale); y2 = int((y-py+bh/2)/scale)
        cats.append({'score':conf,'box':[max(0,x1),max(0,y1),min(w,x2),min(h,y2)]})
    return cats

def is_night(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:,:,1]))<20 and float(np.mean(hsv[:,:,2]))<80

def run_continuous(date, h_start, h_end, label, skip_every=5):
    """Process continuous recordings from h_start to h_end, ~1 frame/sec."""
    d = f'{REC_BASE}/{date}'
    if not os.path.isdir(d):
        print(f"  No dir {d}"); return
    files = sorted(f for f in os.listdir(d) if f.endswith('.mp4'))
    sel = [f for f in files if h_start <= int(f[8:10]) < h_end]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {date} {h_start:02d}:00-{h_end:02d}:00 ({len(sel)} files)")
    print(f"{'='*60}")
    
    wz_bg = None; wz_frames = 0
    total=0; old_y=0; new_y=0
    last_old_streak=0; last_new_streak=0
    old_streaks=[]; new_streaks=[]
    old_in_streak=False; new_in_streak=False
    interesting = []
    
    for fname in sel:
        fpath = os.path.join(d, fname)
        cap = cv2.VideoCapture(fpath)
        if not cap.isOpened(): continue
        fps_r = cap.get(cv2.CAP_PROP_FPS) or 25
        tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fi = skip_every  # skip first frame
        while fi < tf:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (960,540))
            total += 1
            sec = fi/fps_r
            is_n = is_night(frame)
            
            # Only run YOLO every ~2 seconds to save time, otherwise use cached
            run_yolo = (total % 2 == 1) or total <= 5
            
            if run_yolo:
                score_thresh = 0.28 if is_n else 0.40
                crop_thresh = 0.15 if is_n else 0.25
                cats = yolo(frame, score_thresh, night_mode=is_n)
                h2,w2 = frame.shape[:2]; cs = int(h2*0.45)
                cr = frame[cs:h2,0:w2]
                if cr.size:
                    for c in yolo(cr, crop_thresh, night_mode=is_n):
                        c['box'] = [c['box'][0], c['box'][1]+cs, c['box'][2], c['box'][3]+cs]
                        cats.append(c)
                vis = len(cats)>0
                best = max(cats, key=lambda c:c['score']) if cats else None
                sc = best['score'] if best else 0
                box = best['box'] if best else []
                last_vis, last_sc, last_box = vis, sc, box
            else:
                vis, sc, box = last_vis, last_sc, last_box
            
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
                        wc_cands = yolo(wc, 0.18)
                        wo = any(c['score']>=0.18 for c in wc_cands)
            
            atw = False
            if vis and box and len(box)==4 and wzone:
                hx = (box[0]+box[2])/2; hy = box[3]
                inx = wzone[0]<=hx<=wzone[2]
                spans = box[1]<=wsy<=box[3]
                near = abs(hy-wsy)<prox
                atw = inx and (near or spans)
            
            old = atw or wo
            new = atw or (wo and vis)
            
            if old: old_y += 1
            if new: new_y += 1
            
            # Track streaks
            if old:
                if not old_in_streak: old_in_streak=True; old_streaks.append(1)
                else: old_streaks[-1] += 1
            else: old_in_streak=False
            
            if new:
                if not new_in_streak: new_in_streak=True; new_streaks.append(1)
                else: new_streaks[-1] += 1
            else: new_in_streak=False
            
            # Record interesting frames
            if old and not new:
                ts = fname[:14]
                interesting.append(f"  {ts} sec={sec:.0f}s vis={vis} water={wo} atw={atw} score={sc:.3f} {'[N]' if is_n else '[D]'}")
            
            fi += skip_every
        cap.release()
    
    elim = old_y - new_y
    pct = 100*elim/max(1,old_y) if old_y else 0
    print(f"\n  Processed: {total} frames | wz_bg warmed: {wz_frames} frames")
    print(f"  OLD logic: {old_y} drinking frames ({len(old_streaks)} streaks, max {max(old_streaks) if old_streaks else 0})")
    print(f"  NEW logic: {new_y} drinking frames ({len(new_streaks)} streaks, max {max(new_streaks) if new_streaks else 0})")
    print(f"  >>> Eliminated: {elim} frames ({pct:.0f}% of OLD triggers)")
    if interesting:
        print(f"\n  Frames eliminated by fix (OLD=yes, NEW=no):")
        for line in interesting[:20]:
            print(line)
        if len(interesting)>20:
            print(f"  ... and {len(interesting)-20} more")

print("Model loaded. Running continuous night test...")
# 30-minute night stretch on May 17
run_continuous('20260517', 22, 23, "Night May17 22:00-23:00 (continuous 1hr, IR mode)")
# 30-minute night stretch on May 18
run_continuous('20260518', 1, 3, "Night May18 01:00-03:00 (continuous 2hr, IR mode)")
# Day stretch for comparison
run_continuous('20260517', 11, 13, "Day May17 11:00-13:00 (continuous 2hr, color mode)")

print(f"\n{'='*60}")
print("CONCLUSION:")
print("  Fix changes: drinking = cat_at_water or water_occupied")
print("            → drinking = cat_at_water or (water_occupied AND cat_visible)")
print("  water_occupied alone (from fountain water motion) no longer triggers events")
print(f"{'='*60}")
