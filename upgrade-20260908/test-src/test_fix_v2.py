#!/usr/bin/env python3
"""Quick verification: scan specific night hours, compare old vs new logic."""
import json, os, sys, time, cv2, numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import onnxruntime as ort

# Config
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

# Model
session = ort.InferenceSession('/app/models/yolov8n_640_opset21.onnx', providers=['CPUExecutionProvider'])
inp = session.get_inputs()[0].name

def lb(img, s=640):
    h,w = img.shape[:2]; sc = min(s/w,s/h)
    nw,nh = int(w*sc),int(h*sc); r = cv2.resize(img,(nw,nh))
    p = np.full((s,s,3),114,dtype=np.uint8); dx,dy=(s-nw)//2,(s-nh)//2
    p[dy:dy+nh,dx:dx+nw]=r; return p,sc,dx,dy

def yolo(frame, th=0.40):
    lbimg,sc,dx,dy = lb(frame)
    b = lbimg.astype(np.float32)/255.0; b = b.transpose(2,0,1)[np.newaxis]
    out = session.run(None,{inp:b})[0][0]
    h,w = frame.shape[:2]; cats=[]
    for row in out:
        cx,cy,bw2,bh2 = row[:4]
        cls = 1.0/(1.0+np.exp(-row[4:]))
        if cls[15]>=th and cls[15]>cls[0]:
            ix,iy = int((cx-dx)/sc), int((cy-dy)/sc)
            iw,ih = int(bw2/sc), int(bh2/sc)
            cats.append({'box':[max(0,ix-iw//2),max(0,iy-ih//2),min(w,ix+iw//2),min(h,iy+ih//2)],'score':float(cls[15])})
    return cats

def night(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:,:,1]))<20 and float(np.mean(hsv[:,:,2]))<80

# State trackers for old/new
class State:
    def __init__(self, name):
        self.name = name
        self.in_zone = None; self.last_in = None; self.active = False; self.peak = 0
        self.events = []
    def process(self, drinking, t, cat_vis, water_occ, at_water, score, is_night):
        if drinking:
            if self.in_zone is None: self.in_zone = t
            self.last_in = t
            if score > self.peak: self.peak = score
            dur = (t-self.in_zone).total_seconds()
            if not self.active and dur >= 3.0:
                self.active = True
                self.events.append({'start':str(self.in_zone),'dur':0,'score':self.peak,
                    'night':is_night,'trigger':f'vis={cat_vis} water={water_occ} atw={at_water}'})
        else:
            if self.in_zone and self.last_in:
                gap = (t-self.last_in).total_seconds()
                if gap > 5.0:
                    if self.active:
                        self.events[-1]['dur'] = round((self.last_in-self.in_zone).total_seconds(),1)
                        self.events[-1]['end'] = str(self.last_in)
                    self.in_zone = None; self.active = False; self.peak = 0

def scan(date, h_start, h_end, sample=4):
    d = f'{REC_BASE}/{date}'
    if not os.path.isdir(d): 
        print(f'  No dir {d}'); return [],[]
    files = sorted(f for f in os.listdir(d) if f.endswith('.mp4'))
    sel = []
    for f in files:
        h = int(f[8:10])
        if h_start<=h_end:
            if h_start<=h<h_end: sel.append(f)
        else:
            if h>=h_start or h<h_end: sel.append(f)
    print(f'  {date} {h_start:02d}-{h_end:02d}h: {len(sel)} files')
    if not sel: return [],[]
    
    old = State('OLD'); new = State('NEW')
    wz_bg = None; wz_frames = 0
    
    for fname in sel:
        cap = cv2.VideoCapture(os.path.join(d, fname))
        if not cap.isOpened(): continue
        fps_r = cap.get(cv2.CAP_PROP_FPS) or 25
        skip = max(1, int(fps_r*sample))
        t = datetime.strptime(fname[:14], '%Y%m%d%H%M%S')
        fi = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (960,540))
            is_n = night(frame)
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
            
            # cat_at_water
            atw = False
            if vis and box and len(box)==4 and wzone:
                hx = (box[0]+box[2])/2; hy = box[3]
                inx = wzone[0]<=hx<=wzone[2]
                spans = box[1]<=wsy<=box[3]
                near = abs(hy-wsy)<prox
                atw = inx and (near or spans)
            
            old_drink = atw or wo
            new_drink = atw or (wo and vis)
            
            now = datetime.strptime(fname[:14]+'00','%Y%m%d%H%M%S') + timedelta(seconds=fi/int(fps_r))
            old.process(old_drink, now, vis, wo, atw, sc, is_n)
            new.process(new_drink, now, vis, wo, atw, sc, is_n)
            fi += skip
        cap.release()
    return old.events, new.events

print("="*70)
print("Detection Fix Verification: OLD vs NEW logic")
print("="*70)

# Test night: May 17 night (21-24) + May 18 early morning (0-6)
print("\n[TEST 1] May 17 night 21:00-24:00 (IR mode expected)")
o1,n1 = scan('20260517', 21, 24, sample=4)

print("\n[TEST 2] May 18 early morning 00:00-06:00 (IR mode expected)")
o2,n2 = scan('20260518', 0, 6, sample=4)

print("\n[TEST 3] May 17 daytime 10:00-14:00 (color mode)")
o3,n3 = scan('20260517', 10, 14, sample=4)

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

for label, oe, ne in [("Night May17 21-24h", o1,n1), ("Night May18 00-06h", o2,n2), ("Day May17 10-14h", o3,n3)]:
    print(f"\n  {label}:")
    print(f"    OLD logic: {len(oe)} events")
    for e in oe:
        print(f"      {e['start'][:19]} dur={e['dur']}s night={e['night']} {e['trigger']} score={e['score']:.3f}")
    print(f"    NEW logic: {len(ne)} events")
    for e in ne:
        print(f"      {e['start'][:19]} dur={e['dur']}s night={e['night']} {e['trigger']} score={e['score']:.3f}")
    elim = len(oe)-len(ne)
    print(f"    >>> Eliminated {elim} events ({'ALL false positives!' if elim==len(oe) else 'some reduced'})")
