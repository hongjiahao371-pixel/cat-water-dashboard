#!/usr/bin/env python3
"""Precise verification: specific night false-positive recordings, frame-by-frame."""
import json, os, cv2, numpy as np
from pathlib import Path
import onnxruntime as ort

SETTINGS_FILE = Path('/app/data/settings.json')
settings = json.loads(SETTINGS_FILE.read_text()) if SETTINGS_FILE.exists() else {}
fw = settings.get('frameWidth', 1280)
fh = settings.get('frameHeight', 720)
REC_BASE = '/easynvr_rec/easynvr_rec/OpD6HNpPU8j0X/profile_1'

def sz(zone, sw, sh, fw2, fh2):
    if not zone: return None
    sx, sy = fw2/max(1,sw), fh2/max(1,sh)
    return (int(zone['x1']*sx), int(zone['y1']*sy), int(zone['x2']*sx), int(zone['y2']*sy))

fzone = sz(settings.get('fountainZone',{}), fw, fh, 960, 540)
wzone = sz(settings.get('waterZone',{}), fw, fh, 960, 540)
wsy = settings.get('waterSurfaceY', 243) * (540/max(1,fh))
prox = 50 * (540/max(1,fh))

session = ort.InferenceSession('/app/models/yolov8n_640_opset21.onnx', providers=['CPUExecutionProvider'])
inp_name = session.get_inputs()[0].name

def lb(img, s=640):
    h,w = img.shape[:2]; sc = min(s/w,s/h)
    nw,nh = int(w*sc),int(h*sc); r = cv2.resize(img,(nw,nh))
    p = np.full((s,s,3),114,dtype=np.uint8); dx,dy=(s-nw)//2,(s-nh)//2
    p[dy:dy+nh,dx:dx+nw]=r; return p,sc,dx,dy

def yolo(frame, th=0.40):
    lbimg,sc,dx,dy = lb(frame)
    b = lbimg.astype(np.float32)/255.0; b = b.transpose(2,0,1)[np.newaxis]
    out = session.run(None,{inp_name:b})[0][0]
    h,w = frame.shape[:2]; cats=[]
    for row in out:
        cx,cy,bw2,bh2 = row[:4]
        cls = 1.0/(1.0+np.exp(-row[4:]))
        if cls[15]>=th and cls[15]>cls[0]:
            ix,iy = int((cx-dx)/sc), int((cy-dy)/sc)
            iw,ih = int(bw2/sc), int(bh2/sc)
            cats.append({'box':[max(0,ix-iw//2),max(0,iy-ih//2),min(w,ix+iw//2),min(h,iy+ih//2)],'score':float(cls[15])})
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
    
    print(f"  Duration: {duration:.0f}s, FPS: {fps_r:.0f}, sampling every {sample_interval}s")
    print(f"  {'Time':>6} | {'Night':>5} | {'CatVis':>6} | {'Score':>5} | {'Water':>5} | {'AtWater':>7} | {'OLD':>3} | {'NEW':>3}")
    print(f"  {'-'*60}")
    
    old_yes = 0; new_yes = 0; total = 0
    
    fi = 0
    while fi < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (960, 540))
        total += 1
        sec = fi / fps_r
        
        is_n = is_night(frame)
        
        # YOLO
        cats = yolo(frame, 0.28 if is_n else 0.40)
        h2,w2 = frame.shape[:2]; cs = int(h2*0.45); cr = frame[cs:h2,0:w2]
        if cr.size: cats.extend(yolo(cr, 0.15 if is_n else 0.25))
        
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
                    wc_cands = yolo(wc, 0.18)
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
        
        if old_drink: old_yes += 1
        if new_drink: new_yes += 1
        
        # Only print interesting frames (where something triggers)
        if vis or wo or atw:
            print(f"  {sec:5.0f}s | {'YES' if is_n else 'no':>5} | {str(vis):>6} | {sc:.3f} | {str(wo):>5} | {str(atw):>7} | {'YES' if old_drink else 'no':>3} | {'YES' if new_drink else 'no':>3}")
        
        fi += skip
    
    cap.release()
    print(f"\n  Summary: {total} frames | OLD triggers: {old_yes} ({100*old_yes/max(1,total):.0f}%) | NEW triggers: {new_yes} ({100*new_yes/max(1,total):.0f}%) | Eliminated: {old_yes-new_yes}")

# Specific recordings to test:
# Night false-positive periods from event data:
# May 7 23:25-23:27, May 8 01:45, May 8 21:01, May 8 21:18
# May 17 night, May 18 early morning

recordings = [
    # Night recordings (IR mode expected)
    (f'{REC_BASE}/20260517/20260517232500-60928.mp4', 'Night May17 23:25 (known false positive period)'),
    (f'{REC_BASE}/20260517/20260517220000-60928.mp4', 'Night May17 22:00 (deep night IR)'),
    (f'{REC_BASE}/20260518/20260518010000-60928.mp4', 'Night May18 01:00 (deep night IR)'),
    (f'{REC_BASE}/20260518/20260518030000-60928.mp4', 'Night May18 03:00 (deep night IR)'),
    # Day recordings (should have fewer false triggers)
    (f'{REC_BASE}/20260517/20260517120000-60864.mp4', 'Day May17 12:00 (color, known real events)'),
    (f'{REC_BASE}/20260518/20260518100000-60928.mp4', 'Day May18 10:00 (today, before fix)'),
]

print("Loading YOLO model...")

for fpath, label in recordings:
    if os.path.exists(fpath):
        analyze_recording(fpath, label, sample_interval=2)
    else:
        # Try to find a nearby file
        d = os.path.dirname(fpath)
        if os.path.isdir(d):
            prefix = os.path.basename(fpath)[:12]  # e.g., 202605172325
            files = sorted(f for f in os.listdir(d) if f.startswith(prefix))
            if files:
                actual = os.path.join(d, files[0])
                analyze_recording(actual, label + f' [actual: {files[0]}]', sample_interval=2)
            else:
                # Just pick any file from that hour
                hour = os.path.basename(fpath)[8:10]
                files = [f for f in os.listdir(d) if f.endswith('.mp4') and f[8:10]==hour]
                if files:
                    actual = os.path.join(d, files[len(files)//2])
                    analyze_recording(actual, label + f' [actual: {os.path.basename(actual)}]', sample_interval=2)
                else:
                    print(f"\n  SKIP {label}: no files found")
        else:
            print(f"\n  SKIP {label}: dir not found")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
