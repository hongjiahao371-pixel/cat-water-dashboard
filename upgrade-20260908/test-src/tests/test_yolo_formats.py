import sys
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DETECTOR_EVENT_DIR", tempfile.mkdtemp(prefix="cat-yolo-test-"))
os.environ.setdefault("DETECTOR_STATUS_FILE", str(Path(tempfile.mkdtemp(prefix="cat-yolo-status-"))/"status.json"))
import detection_worker as d

class Session:
 def __init__(self, output):self.output=output
 def run(self,*args):return [self.output]

class YoloFormatTests(unittest.TestCase):
 def infer(self,output,ox=0,oy=0):
  detector=d.CatDetector.__new__(d.CatDetector);detector.session=Session(output)
  return detector._infer_candidates(np.zeros((720,1280,3),dtype=np.uint8),0.4,ox,oy)
 def test_end_to_end_filters_person_and_maps_letterbox(self):
  o=np.zeros((1,300,6),dtype=np.float32)
  o[0,0]=[100,190,300,290,.9,15];o[0,1]=[100,190,300,290,.99,0]
  r=self.infer(o);self.assertEqual(len(r),1);self.assertEqual(r[0]['box'],[200,100,600,300])
 def test_raw_coco_matches_end_to_end_cat_geometry(self):
  raw=np.zeros((1,84,8400),dtype=np.float32);raw[0,:4,0]=[200,240,200,100];raw[0,19,0]=.9
  end=np.zeros((1,300,6),dtype=np.float32);end[0,0]=[100,190,300,290,.9,15]
  self.assertEqual(self.infer(raw,30,40),self.infer(end,30,40))
 def test_raw_person_rejection_preserved(self):
  raw=np.zeros((1,84,8400),dtype=np.float32);raw[0,:4,0]=[200,240,200,100];raw[0,19,0]=.6;raw[0,4,0]=.95
  self.assertEqual(self.infer(raw),[])
 def test_legacy_single_cat(self):
  raw=np.zeros((1,5,8400),dtype=np.float32);raw[0,:,0]=[200,240,200,100,.9]
  self.assertEqual(self.infer(raw)[0]['box'],[200,100,600,300])
 def test_invalid_and_nonfinite_boxes_ignored(self):
  o=np.array([[[100,190,300,290,.9,15],[300,190,100,290,.9,15],[100,190,300,290,np.nan,15],[100,190,300,290,.9,15.1]]],dtype=np.float32)
  self.assertEqual(len(self.infer(o)),1)
 def test_empty_end_to_end(self):self.assertEqual(self.infer(np.empty((1,0,6))),[])
 def test_unknown_layout_rejected(self):
  for shape in [(300,6),(2,300,6),(1,8400,84)]:
   with self.assertRaises(ValueError):d.decode_yolo_layout(np.zeros(shape))
if __name__=='__main__':unittest.main()
