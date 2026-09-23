#!/usr/bin/env python3
"""Focused test: pick specific recordings from key time periods."""
import json, os, cv2, numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import onnxruntime as ort

SETTINGS_FILE = Path('/app/data/settings.json')
settings = json.loads(SETTINGS_FILE.read_text()) if SETTINGS_FILE.exists() else {}
fw = settings.get('frameWidth', 1280)
fh = settings.get('frameHeight', 720)
REC_BASE = '/easynvr_rec/easynvr_rec/EASYNVR_CHANNEL_PLACEHOLDER/profile_1'

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

# Per-frame analysis, no state machine - just show raw triggers
def analyze_files(files_with_time, label):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    
    wz_bg = None; wz_frames = 0
    old_triggers = 0  # frames where old logic would say drinking
    new_triggers = 0  # frames where new logic would say drinking
    cat_vis_frames = 0
    water_occ_frames = 0
    total = 0
    details = []
    
    for fpath, base_t in files_with_time:
        cap = cv2.VideoCapture(fpath)
        if not cap.isOpened(): continue
        fps_r = cap.get(cv2.CAP_PROP_FPS) or 25
        fi = 0
        skip = max(1, int(fps_r*3))  # sample every 3s
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (960,540))
            total += 1
            is_n = is_night(frame)
            
            cats = yolo(frame, 0.28 if is_n else 0.40)
            h2,w2 = frame.shape[:2]; cs = int(h2*0.45); cr = frame[cs:h2,0:w2]
            if cr.size: cats.extend(yolo(cr, 0.15 if is_n else 0.25))
            
            vis = len(cats)>0
            best = max(cats, key=lambda c:c['score']) if cats else None
            sc = best['score'] if best else 0
            box = best['box'] if best else []
            
            if vis: cat_vis_frames += 1
            
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
            
            if wo: water_occ_frames += 1
            
            atw = False
            if vis and box and len(box)==4 and wzone:
                hx = (box[0]+box[2])/2; hy = box[3]
                inx = wzone[0]<=hx<=wzone[2]
                spans = box[1]<=wsy<=box[3]
                near = abs(hy-wsy)<prox
                atw = inx and (near or spans)
            
            old_drink = atw or wo
            new_drink = atw or (wo and vis)
            
            if old_drink: old_triggers += 1
            if new_drink: new_triggers += 1
            
            # Record any frame where old/new disagree
            if old_drink != new_drink:
                t = base_t + timedelta(seconds=fi/fps_r)
                details.append(f"  {t.strftime('%H:%M:%S')} OLD=yes NEW=no | vis={vis} water={wo} atw={atw} score={sc:.3f} {'[NIGHT]' if is_n else '[DAY]'}")
            
            fi += skip
        cap.release()
    
    print(f"Frames: {total} | cat_vis: {cat_vis_frames} ({100*cat_vis_frames/max(1,total):.1f}%) | water_occ: {water_occ_frames} ({100*water_occ_frames/max(1,total):.1f}%)")
    print(f"OLD logic drinking triggers: {old_triggers} ({100*old_triggers/max(1,total):.1f}%)")
    print(f"NEW logic drinking triggers: {new_triggers} ({100*new_triggers/max(1,total):.1f}%)")
    print(f"Eliminated: {old_triggers-new_triggers} false trigger frames")
    if details:
        print(f"\nDisagreement details (OLD=yes, NEW=no):")
        for d in details[:30]:
            print(d)
        if len(details)>30:
            print(f"  ... and {len(details)-30} more")

# Pick specific files
def pick_files(date, hours, count=10):
    d = f'{REC_BASE}/{date}'
    if not os.path.isdir(d): return []
    files = sorted(f for f in os.listdir(d) if f.endswith('.mp4'))
    sel = [f for f in files if int(f[8:10]) in hours]
    # Pick evenly spaced files
    if len(sel) > count:
        step = len(sel) // count
        sel = sel[::step][:count]
    return [(os.path.join(d, f), datetime.strptime(f[:14], '%Y%m%d%H%M%S')) for f in sel]

# Night test: May 17, 22:00-23:30 (deep night, definitely IR)
print("Loading model...")
night_files = pick_files('20260517', list(range(22,24)), count=12)
print(f"Night files: {len(night_files)}")
analyze_files(night_files, "NIGHT: May 17 22:00-23:59 (IR black-and-white)")

# Early morning: May 18, 01:00-05:00
early_files = pick_files('20260518', list(range(1,6)), count=15)
print(f"\nEarly morning files: {len(early_files)}")
analyze_files(early_files, "EARLY MORNING: May 18 01:00-05:59 (IR black-and-white)")

# Day test: May 17, 11:00-13:00 (color)
day_files = pick_files('20260517', list(range(11,14)), count=10)
print(f"\nDay files: {len(day_files)}")
analyze_files(day_files, "DAY: May 17 11:00-13:59 (color)")

# Evening transition: May 17, 19:00-21:00 (may have both)
eve_files = pick_files('20260517', list(range(19,22)), count=10)
print(f"\nEvening files: {len(eve_files)}")
analyze_files(eve_files, "EVENING: May 17 19:00-21:59 (transition)")
