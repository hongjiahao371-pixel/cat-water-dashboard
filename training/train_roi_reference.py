"""Train only on pre-August-5 human labels; tune only on Aug 5-9 labels.
Recent unlabeled holdout never enters fit, augmentation, checkpoint selection or tuning.
"""
import json,re,random,os,hashlib,time
from pathlib import Path
import numpy as np,torch,cv2
from torch import nn
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from torchvision import transforms as T
from PIL import Image
R=Path(__file__).resolve().parent;OUT=R/'training';OUT.mkdir(exist_ok=True)
SEED=20260908;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
ROI=(.08,.12,.94,.82)
rows=json.loads((R/'dataset_manifest.json').read_text())['records']
for x in rows:x['date']=re.search(r'(2026\d{4})',x['id']).group(1)
train=[x for x in rows if x['label'] is not None and x['date']<'20260805'];val=[x for x in rows if x['label'] is not None and x['date']>='20260805'];hold=[x for x in rows if x['label'] is None]
print('SPLITS',[(name,len(a),sum(x['label']==1 for x in a),sum(x['label']==0 for x in a)) for name,a in [('train',train),('validation',val),('holdout',hold)]] ,flush=True)
assert len(train)>50 and len(val)>10 and all(x['date']>='20260810' for x in hold)
(OUT/'split.json').write_text(json.dumps({'seed':SEED,'roi':ROI,'train':[x['id'] for x in train],'validation':[x['id'] for x in val],'holdout':[x['id'] for x in hold]},indent=2))
normalize=T.Normalize([.485,.456,.406],[.229,.224,.225]);aug=T.Compose([T.Resize((240,240)),T.RandomCrop((224,224)),T.ColorJitter(brightness=.2,contrast=.15,saturation=.15),T.ToTensor(),normalize]);plain=lambda im: normalize(torch.from_numpy(cv2.resize(np.asarray(im),(224,224),interpolation=cv2.INTER_LINEAR).copy()).permute(2,0,1).float()/255.)
class Data(torch.utils.data.Dataset):
 def __init__(self,a,tf):
  self.a=a;self.tf=tf;self.images=[]
  for x in a:
   im=Image.open(R/x['image']).convert('RGB');w,h=im.size;self.images.append(im.crop((round(w*ROI[0]),round(h*ROI[1]),round(w*ROI[2]),round(h*ROI[3]))))
 def __len__(self):return len(self.a)
 def __getitem__(self,i):return self.tf(self.images[i]),float(self.a[i]['label'] or 0)
loader=torch.utils.data.DataLoader(Data(train,aug),batch_size=16,shuffle=True,num_workers=0)
vloader=torch.utils.data.DataLoader(Data(val,plain),batch_size=16,num_workers=0)
device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
m=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT);m.classifier[3]=nn.Linear(1024,1)
# Fine-tune late features plus classifier; retain early generic visual features.
for p in m.features[:8].parameters():p.requires_grad=False
m.to(device);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=0.0003,weight_decay=.01)
pos=sum(x['label'] for x in train);lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(train)-pos)/pos],device=device));best=float('inf');history=[]
for epoch in range(30):
 m.train();m.features[:8].eval();losses=[]
 for xx,yy in loader:
  xx=xx.to(device);yy=yy.float().to(device).view(-1,1);opt.zero_grad();loss=lossfn(m(xx),yy);loss.backward();opt.step();losses.append(loss.item())
 m.eval();scores=[];vl=[]
 with torch.no_grad():
  for xx,yy in vloader:
   logits=m(xx.to(device));vl.append(nn.functional.binary_cross_entropy_with_logits(logits,yy.float().to(device).view(-1,1)).item());scores.extend(torch.sigmoid(logits).cpu().flatten().tolist())
 vloss=float(np.mean(vl));history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_loss':vloss})
 if vloss<best:
  best=vloss;torch.save(m.cpu().state_dict(),OUT/'best.pt');m.to(device);(OUT/'validation-scores.json').write_text(json.dumps([dict(x,score=float(s)) for x,s in zip(val,scores)],indent=2))
 print('epoch',epoch+1,'train',round(np.mean(losses),4),'validation',round(vloss,4),'best',round(best,4),flush=True)
(OUT/'history.json').write_text(json.dumps(history,indent=2))
print('TRAINING DONE',flush=True)
