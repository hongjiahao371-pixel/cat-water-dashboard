import json,collections,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent
summary={'scope':'Existing labeled full-frame snapshots and separately reviewed saved event clips; not an independent test set or online accuracy estimate.','snapshots':{},'clips':{},'segments':{},'decision':'Keep YOLOv8n active; stage YOLO26n and YOLO26s as candidates. No demonstrated overall accuracy gain from direct replacement.'}
for name in ['v8n','26n','26s','26n-e2e','26s-e2e']:
 for kind in ['snapshots','clips','segments']:
  p=r/'benchmark-results'/f'{name}-{kind}.json'
  if not p.exists():continue
  a=json.loads(p.read_text())
  if kind=='snapshots':
   summary[kind][name]={'positive_images':sum(x['label']=='confirmed' for x in a),'positive_anchors':sum(x['label']=='confirmed' and x['anchor'] for x in a),'negative_images':sum(x['label']=='false_positive' for x in a),'negative_anchors':sum(x['label']=='false_positive' and x['anchor'] for x in a)}
  else:summary[kind][name]={'total':len(a),'decisions':dict(collections.Counter(x['status'] for x in a))}
(r/'benchmark-results/summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
