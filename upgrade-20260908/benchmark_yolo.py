"""Read-only replay of saved evidence with production detector and archive verifier.
No API calls, event writes, training or notification delivery.
"""
import argparse,json,os,time,statistics
from pathlib import Path
import cv2,numpy as np,onnxruntime as ort
ROOT=Path(__file__).resolve().parent
os.environ.update({'DETECTOR_EVENT_DIR':str(ROOT/'eval-runtime/events'),'DETECTOR_STATUS_FILE':str(ROOT/'eval-runtime/status.json'),'SETTINGS_FILE':str(ROOT/'baseline/data/settings.json'),'DRINKING_CLASSIFIER_PATH':str(ROOT/'baseline/models/cat_drinking_classifier.onnx'),'DRINKING_CALIBRATION_PATH':str(ROOT/'baseline/data/classifier_calibration.json'),'NIGHT_HIRES_ENABLED':'false','CONSECUTIVE_FRAMES_REQUIRED':'2','CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD':'0.80','NIGHT_CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD':'0.85','CLASSIFIER_TEMPORAL_MIN_HITS':'2','CLASSIFIER_TEMPORAL_MIN_IOU':'0.10','NIGHT_DRINKING_CLASSIFIER_THRESHOLD':'0.75','CAT_DETECT_PROVIDER_PREFERENCE':'cpu'})
import detection_worker as d
import archive_verifier_worker as av
parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=['v8n','26n','26s']);parser.add_argument('--mode',choices=['snapshots','clips','segments'],default='clips');parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
models={'v8n':ROOT/'baseline/models/yolov8n_640_opset21.onnx','26n':ROOT/'models/yolo26n.onnx','26s':ROOT/'models/yolo26s.onnx'}
models.update({'26n-e2e':ROOT/'models/yolo26n-e2e.onnx','26s-e2e':ROOT/'models/yolo26s-e2e.onnx'})
manifest=json.loads((ROOT/'eval_manifest.json').read_text());settings=d.read_settings();build=d.CatDetector._build_session
sessions={}
def session_for(self,path):
 key=str(path)
 if key not in sessions:
  opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
  sessions[key]=ort.InferenceSession(key,sess_options=opts,providers=['CPUExecutionProvider'])
 return sessions[key]
d.CatDetector._build_session=session_for
out=ROOT/'benchmark-results';out.mkdir(exist_ok=True)
for name in args.models:
 d.get_active_model_path=lambda: str(models[name])
 results=[];times=[]
 entries=manifest[args.mode];entries=entries[:args.limit] if args.limit else entries
 for n,row in enumerate(entries):
  detector=d.CatDetector();t=time.perf_counter()
  if args.mode in ('clips','segments'):
   result=av.analyze_video(ROOT/row['clip'],detector=detector,sample_fps=2)
  else:
   f=cv2.imread(str(ROOT/row['image']))
   if f is None:raise RuntimeError('Missing '+row['image'])
   h,w=f.shape[:2];fz=d.get_fountain_zone(settings,w,h);wz=d.get_water_zone(settings,w,h)
   result=detector.detect(f,force=True,fzone=fz,wzone=wz)
   _,confidence=detector.classify_drinking(f,result.get('cat_box'))
   result={**result,'classifier':float(confidence),'anchor':bool(result.get('cat_box') and result.get('closeup_water') and confidence >= (d.NIGHT_CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD if result.get('night_mode') else d.CLASSIFIER_TEMPORAL_ANCHOR_THRESHOLD))}
  result={'id':row['id'],'label':row['label'],'profile':row.get('profile'),**result,'elapsed_sec':round(time.perf_counter()-t,4)}
  results.append(result)
  (out/f'{name}-{args.mode}.json').write_text(json.dumps(results,indent=2))
  if args.mode in ('clips','segments') or n%20==0:print(name,args.mode,n+1,'/',len(entries),row['id'],result.get('status',result.get('anchor')),result.get('max_classifier',result.get('classifier')),result['elapsed_sec'],flush=True)
 print('FINISHED',name,args.mode,len(results),flush=True)
