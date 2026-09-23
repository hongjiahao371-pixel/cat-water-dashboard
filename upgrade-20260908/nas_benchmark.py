import os,time,json,statistics
from pathlib import Path
os.environ['DETECTOR_EVENT_DIR']='/tmp/yolo-bench/events';os.environ['DETECTOR_STATUS_FILE']='/tmp/yolo-bench/status.json';os.environ['SETTINGS_FILE']='/app/data/settings.json';os.environ['NIGHT_HIRES_ENABLED']='false';os.environ['DRINKING_CLASSIFIER_PATH']='/app/models/cat_drinking_classifier.onnx';os.environ['CONSECUTIVE_FRAMES_REQUIRED']='2'
import cv2,numpy as np,onnxruntime as ort
import detection_worker as d
cv2.setNumThreads(1)
def build(self,path):
 opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
 return ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])
d.CatDetector._build_session=build
c=cv2.VideoCapture('/evidence/event-20260712-084921.mp4');c.set(cv2.CAP_PROP_POS_MSEC,9000);ok,f=c.read();c.release();assert ok
settings=d.read_settings();h,w=f.shape[:2];fz=d.get_fountain_zone(settings,w,h);wz=d.get_water_zone(settings,w,h)
results=[]
for name,path in [('v8n','/app/models/yolov8n_640_opset21.onnx'),('26n','/audit/yolo26n.onnx'),('26s','/audit/yolo26s.onnx')]:
 d.get_active_model_path=lambda:path
 detector=d.CatDetector();latencies=[]
 for i in range(15):
  t=time.perf_counter();v=detector.detect(f,force=True,fzone=fz,wzone=wz);_,conf=detector.classify_drinking(f,v.get('cat_box'));elapsed=time.perf_counter()-t
  if i>=3:latencies.append(elapsed)
 result={'model':name,'frame_size':[w,h],'samples':len(latencies),'detector_plus_classifier_median_ms':round(statistics.median(latencies)*1000,2),'p95_ms':round(float(np.percentile(latencies,95))*1000,2),'cat_score':v['cat_score'],'classifier':conf,'output_shape':detector.session.get_outputs()[0].shape,'providers':detector.session.get_providers()}
 results.append(result);print(json.dumps(result),flush=True)
Path('/results/nas-benchmark.json').write_text(json.dumps(results,indent=2))
