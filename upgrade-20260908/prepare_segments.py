import json,cv2
from pathlib import Path
from PIL import Image,ImageDraw
r=Path(__file__).resolve().parent;p=r/'eval_manifest.json';m=json.loads(p.read_text());out=r/'eval-segments';out.mkdir(exist_ok=True)
rows=[]
for n,start,end in [(3,0,4.5),(4,0,4),(7,0,6),(9,0,6),(11,0,4),(13,0,4)]:
 x=m['clips'][n];c=cv2.VideoCapture(str(r/x['clip']));fps=c.get(cv2.CAP_PROP_FPS);w=int(c.get(3));h=int(c.get(4));dst=out/f'negative-{n:02d}.mp4';v=cv2.VideoWriter(str(dst),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));images=[];i=0
 while True:
  ok,f=c.read()
  if not ok or i/fps>=end:break
  if i/fps>=start:
   v.write(f)
   if i%max(1,round(fps*0.5))==0:images.append((i/fps,f.copy()))
  i+=1
 v.release();c.release()
 sheet=Image.new('RGB',(1200,260*((len(images)+3)//4)),'white');d=ImageDraw.Draw(sheet)
 for k,(t,f) in enumerate(images):
  im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));im.thumbnail((300,225));xx=k%4*300;yy=k//4*260;sheet.paste(im,(xx,yy));d.text((xx,yy+230),f'clip {n} t={t:.1f}',fill='black')
 sheet.save(r/'contact-sheets'/f'negative-{n:02d}.jpg')
 rows.append({'id':f'negative-{n:02d}','label':'false_positive','profile':x['profile'],'clip':str(dst.relative_to(r)),'source_clip':x['clip'],'start_sec':start,'end_sec':end,'truth_source':'assistant_visual_review_pending'})
m['segments']=rows;p.write_text(json.dumps(m,indent=2))
