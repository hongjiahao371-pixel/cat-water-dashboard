import pathlib,hashlib,shutil,json,subprocess
r=pathlib.Path('/volume1/docker/cat-water-dashboard');u=r/'upgrade-20260908'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(r/'detection_worker.py')=='2bd59e8883778f9ad707f3def55782837780dae7dd1b47adcd53fe1be6e2d2f2','Production source changed since baseline; abort'
assert sha(u/'test-src/detection_worker.py')=='6bb1410bf0ec35a2c71349c24f73e2386c262b61f420c3bb080ffdfd0d576317','Candidate source checksum mismatch'
assert not json.loads((r/'data/detector_status.json').read_text()).get('drink_active'),'Active drinking event; defer restart'
for n in ['n','s']:
 src=u/f'yolo26{n}.onnx';dst=r/'models'/f'yolo26{n}_640_opset17.onnx'
 if dst.exists():assert sha(dst)==sha(src),'Existing model differs'
 else:shutil.copyfile(src,dst);dst.chmod(0o644)
shutil.copyfile(u/'test-src/detection_worker.py',r/'detection_worker.py')
shutil.copyfile(u/'test-src/tests/test_yolo_formats.py',r/'tests/test_yolo_formats.py')
(r/'detection_worker.py').chmod(0o755);(r/'tests/test_yolo_formats.py').chmod(0o644)
readme=r/'README.md';text=readme.read_text();tag='## 2026-09-08 YOLO26 compatibility and evaluation'
if tag not in text:
 readme.write_text(text+'\n\n'+tag+'\n\nYOLO26n and YOLO26s (640, ONNX opset17) are installed as candidates. YOLOv8n remains active: replay did not demonstrate a consistent overall drinking-accuracy improvement. The detector accepts raw and end-to-end ONNX outputs. See `upgrade-20260908/YOLO26升级实测报告.md` and `upgrade-20260908/benchmark-results/summary.json`. Restore point: `.codex-backup-20260908-yolo26/baseline.tar.gz`.\n')
print('Active model:',json.loads((r/'data/settings.json').read_text())['activeModel'])
print('Source sha256:',sha(r/'detection_worker.py'))
for service in ['cat-water-detector','cat-water-archive-verifier','cat-water-daily-audit']:
 subprocess.run(['docker','restart',service],check=True)
