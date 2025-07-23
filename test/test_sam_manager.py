import os
import sys
import json
import pathlib
import numpy as np

import unittest
import torch

# Add SAM2 to path
SAM2_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, SAM2_PATH)

from sam2.sam_manager.sam_manager import SamManager
from sam2.server.utils import Point, BBox
from sam2.utils.amg import rle_to_mask

# to change in config when available with other size possible
os.environ['TINY_SAM2_CHECKPOINT']=r'//opt/sam2_lv/checkpoints/sam2.1_hiera_tiny.pt'
os.environ['TINY_MODEL_CFG']=r'//opt/sam2_lv/sam2/configs/sam2.1/sam2.1_hiera_t.yaml'

class SamManagerTest(unittest.TestCase):
    def test_bedroom_video(self):
        this_file_path = pathlib.Path(__file__).parent.resolve()


        sm = SamManager()
        video_path = os.path.join(this_file_path,"assets/bedroom.mp4")
        sm.init_model(video_path)
        
        points = []
        points.append(Point(frame_id=20, target_id=0, coord=[210, 350], label=1))
        points.append(Point(frame_id=20, target_id=0, coord=[250, 220], label=1))
    
        fm = sm.add_points_and_bbox(points=points)
        fm = sm.propagate_in_video(max_frame_num_to_track=5)
        
        sm.reset_state()
        sm.init_model(video_path)
        
        points = []
        points.append(Point(frame_id=0, target_id=0, coord=[210, 350], label=1))
        points.append(Point(frame_id=0, target_id=0, coord=[250, 220], label=1))
    
        bboxs = []
        bboxs.append(BBox(frame_id=0, target_id=1, coord=[300, 0, 500, 400]))
    
        fm = sm.add_points_and_bbox(points=points, bboxs=bboxs)
        fm = sm.propagate_in_video()    
        
        last_id = len(fm)-1
        
        with open(os.path.join(this_file_path, "assets", f"frame{last_id}_mask.json"), "r") as fd:
            ground_truth_rle = json.load(fd)

        for key, value in ground_truth_rle.items():
            mask = fm[last_id].masks.get_binary(int(key))
            ground_truth_mask = torch.tensor(rle_to_mask(value), dtype=torch.uint8)

            inter = torch.logical_and(ground_truth_mask, mask).sum()
            union = torch.logical_or(ground_truth_mask, mask).sum()
            iou = inter / union

            self.assertGreaterEqual(iou,0.98)


if __name__ == "__main__":
    unittest.main()