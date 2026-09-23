import os,tempfile,importlib.util,json
from pathlib import Path
import cv2,onnxruntime as ort
r=Path(__file__).resolve().parent
os.environ['DETECTOR_EVENT_DIR']=tempfile.mkdtemp();os.environ['DETECTOR_STATUS_FILE']=tempfile.mktemp()
import detection_worker as new
sp=importlib.util.spec_from_file_location('old',r/'baseline/detection_worker.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
o=ort.SessionOptions();o.intra_op_num_threads=2;s=ort.InferenceSession(str(r/'baseline/models/yolov8n_640_opset21.onnx'),sess_options=o)
class Cached:
 output=None
 def run(self,*a):
  if self.output is None:self.output=s.run(*a)
  return self.output
entries=json.loads((r/'eval_manifest.json').read_text())['snapshots'];cases=0;boxes=0
for row in entries[::7]:
 f=cv2.imread(str(r/row['image']))
 for night in [False,True]:
  c=Cached();a=new.CatDetector.__new__(new.CatDetector);a.session=c;b=old.CatDetector.__new__(old.CatDetector);b.session=c
  x=a._infer_candidates(f,.12,20,30,night);y=b._infer_candidates(f,.12,20,30,night)
  assert x==y,(row['id'],night,x,y)
  cases+=1;boxes+=len(x)
print('PASS',cases,'real-image day/night parity cases,',boxes,'identical candidates')
