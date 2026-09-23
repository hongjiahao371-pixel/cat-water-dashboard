import json,pathlib,tarfile
r=pathlib.Path('/volume1/docker/cat-water-dashboard');out=r/'upgrade-20260908'
es=json.loads((r/'data/mock.json').read_text())['events'];records=[];paths=set()
for e in es:
 if e.get('label') and e.get('clip'):
  p=r/e['clip'].lstrip('/')
  if p.is_file():
   records.append({'id':e['id'],'label':e['label'],'profile':e.get('profile'),'clip':str(p.relative_to(r)),'bytes':p.stat().st_size});paths.add(p)
# full snapshots with existing human labels, used only as image-level diagnostics
snaps=[]
for e in es:
 if e.get('label') and e.get('img'):
  p=r/e['img'].lstrip('/')
  if p.is_file():
   snaps.append({'id':e['id'],'label':e['label'],'profile':e.get('profile'),'image':str(p.relative_to(r))});paths.add(p)
(out/'eval_manifest.json').write_text(json.dumps({'clips':records,'snapshots':snaps},indent=2))
with tarfile.open(out/'eval_media.tar.gz','w:gz') as t:
 for p in sorted(paths):t.add(p,arcname=str(p.relative_to(r)))
 t.add(out/'eval_manifest.json',arcname='eval_manifest.json')
print('clips',len(records),'clip_bytes',sum(x['bytes'] for x in records),'snapshots',len(snaps),'tar_bytes',(out/'eval_media.tar.gz').stat().st_size)
