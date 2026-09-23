import os,json,time,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torchvision.models import mobilenet_v3_small
R=Path(__file__).resolve().parent
os.environ.update(NIGHT_HIRES_ENABLED='false',DRINKING_CLASSIFIER_PATH=str(R/'models/cat_drinking_classifier.onnx'),SETTINGS_FILE=str(R/'data/settings.json'))
import detection_worker as d
from roi_classifier import RoiDrinkingClassifier
m=mobilenet_v3_small(weights=None);m.classifier[3]=nn.Linear(1024,1);m.load_state_dict(torch.load(R/'training/best.pt',map_location='cpu',weights_only=True));m.eval();torch.set_num_threads(2)
f=R/'models/roi_drinking_20260908.onnx'
torch.onnx.export(m,torch.zeros(1,3,224,224),str(f),input_names=['image'],output_names=['logit'],opset_version=17,dynamo=False)
c={'enabled':True,'model':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'threshold':.9,'roi':[.08,.12,.94,.82],'threshold_selection':'largest tested threshold retaining 23/27 validation positives with <=1 false positive among 18 negatives; no holdout scores used'}
(R/'models/roi_drinking.json').write_text(json.dumps(c,indent=2))
roi=RoiDrinkingClassifier(R/'models/roi_drinking.json');assert roi.session is not None
labels=json.loads((R/'training/holdout-labels-frozen.json').read_text())['rows'];records={x['id']:x for x in json.loads((R/'dataset_manifest.json').read_text())['records']}
d.get_active_model_path=lambda:str(R/'models/yolov8n_640_opset21.onnx');det=d.CatDetector();settings=d.read_settings();out=[];deltas=[]
for a in labels:
 im=cv2.imread(str(R/records[a['id']]['image']));h,w=im.shape[:2];r=det.detect(im,force=True,fzone=d.get_fountain_zone(settings,w,h),wzone=d.get_water_zone(settings,w,h));_,old=det.classify_drinking(im,r.get('cat_box'))
 score=roi.probability(im);x1,y1,x2,y2=roi.box(im);rgb=cv2.cvtColor(cv2.resize(im[y1:y2,x1:x2],(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)/255;xx=((rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None]
 with torch.no_grad():pt=float(torch.sigmoid(m(torch.from_numpy(xx))).item())
 deltas.append(abs(pt-score));night=r.get('night_mode',False)
 out.append(dict(a,new_score=score,new_pred=score>=.9,old_score=old,old_pred=bool(r.get('cat_box') and r.get('closeup_water')) and old>=(.85 if night else .8),cat_box=r.get('cat_box'),closeup=r.get('closeup_water'),night=night))
def metrics(key):
 a=[x for x in out if x['label'] is not None];tp=sum(x[key] and x['label']==1 for x in a);fp=sum(x[key] and x['label']==0 for x in a);fn=sum(not x[key] and x['label']==1 for x in a);tn=sum(not x[key] and x['label']==0 for x in a);return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn),precision=tp/(tp+fp) if tp+fp else 0,accuracy=(tp+tn)/len(a))
report={'threshold':.9,'old':metrics('old_pred'),'new':metrics('new_pred'),'onnx_torch_max_delta':max(deltas),'rows':out}
(R/'training/holdout-results.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
