import pathlib,json,hashlib,tarfile
r=pathlib.Path('/volume1/docker/cat-water-dashboard');out=r/'accuracy-20260908';out.mkdir(exist_ok=True)
labels=json.loads((r/'data/review_candidate_labels.json').read_text());events=json.loads((r/'data/mock.json').read_text())['events'];records=[];seen={};missing=[];conflicts=[]
def add(id,label,url,source,profile=None):
 if label not in ('drinking','not_drinking','no_cat','confirmed','false_positive'):return
 target=1 if label in ('drinking','confirmed') else 0
 p=r/url.lstrip('/')
 if not p.is_file():missing.append({'id':id,'url':url});return
 sha=hashlib.sha256(p.read_bytes()).hexdigest()
 if sha in seen:
  if seen[sha]['label']!=target:conflicts.append({'first':seen[sha]['id'],'second':id,'sha':sha})
  return
 row={'id':id,'label':target,'original_label':label,'source':source,'profile':profile,'image':str(p.relative_to(r)),'sha256':sha};records.append(row);seen[sha]=row
for id,v in labels.items():
 lab=v.get('label')
 if not lab:continue
 folder='audit_candidates' if id.startswith('audit-') else 'review_candidates'
 add(id,lab,f'/media/{folder}/{id}.jpg','explicit_review_label')
for e in events:
 if e.get('label') and e.get('img'):add(e['id'],e['label'],e['img'],'explicit_event_label',e.get('profile'))
manifest={'records':records,'missing':missing,'conflicts':conflicts}
(out/'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
paths={r/x['image'] for x in records}
with tarfile.open(out/'dataset.tar.gz','w:gz') as t:
 for p in sorted(paths):t.add(p,arcname=str(p.relative_to(r)))
 for f in ['data/settings.json','data/review_candidate_labels.json','data/mock.json','data/review_candidates.json','data/daily_audit_candidates.json','models/cat_drinking_classifier.onnx','models/yolov8n_640_opset21.onnx','detection_worker.py','archive_verifier_worker.py','classifier_calibration_worker.py','server.py']:
  t.add(r/f,arcname=f)
 t.add(out/'dataset_manifest.json',arcname='dataset_manifest.json')
print(json.dumps({'records':len(records),'positive':sum(x['label'] for x in records),'negative':sum(not x['label'] for x in records),'missing':len(missing),'conflicts':conflicts,'archive_bytes':(out/'dataset.tar.gz').stat().st_size}))
