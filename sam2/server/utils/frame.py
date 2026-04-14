from typing import List

from .bbox import BBox
from .point import Point
from .mask import FrameMask

import logging
log = logging.getLogger(__name__)

class Frame():

    def __init__(self, idx) -> None:    
        self.idx:int=idx
        self.points:list=[]
        self.bboxs:list=[]
        self.masks:FrameMask=None

    def update(self, point:Point=None, bbox:BBox=None, masks:FrameMask=None):
        if point:
            self.points.append(point)
        if bbox:
            self.bboxs.append(bbox)
        if masks:
            self.masks = masks
        return self
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{[p.__repr__() for p in self.points]}_{[b.__repr__() for b in self.bboxs]}_{self.masks}"

    @classmethod
    def from_dict(cls, data: dict):
        idx = data.get('idx', 0)
        frame = cls(idx=idx)
        frame.points = [Point(frame_id=frame.idx, target_id=point.get('target_id'), coord=point.get('coord'), label=point.get('label')) for point in data.get('points', [])]
        frame.bboxs = [BBox(frame_id=frame.idx, target_id=bbox.get('target_id'), coord=bbox.get('coord')) for bbox in data.get('bboxs', [])]
        frame.masks = FrameMask.from_dict(data.get('masks', {}))
            
        return frame
    
    @classmethod
    def to_dict(cls, frame):
        ret = {}
        ret['idx'] = frame.idx
        if frame.points:
            ret['points'] = [Point.to_dict(point) for point in frame.points]
        if frame.bboxs:
            ret['bboxs'] = [BBox.to_dict(bbox) for bbox in frame.bboxs]
        if frame.masks:
            ret['masks'] = FrameMask.to_dict(frame.masks)
        return ret



class FrameManager():
    """
    By design frames will be a dict where the id == to the frame idx
    """
    frames:dict = {}

    def update_or_create(self, frame_idx:int, point:Point=None, bbox:BBox=None, masks:FrameMask=None):
        """Add information to a frame. If the frane has not been created yet, it is.
            Point and bbox are appended to the existing info.
            Mask are overriden
        Args:
            frame_idx (int): id of the frame (start by 0)
            point (Point): one point
            bbox (BBox): one bbox
            mask (FrameMask): the mask of the frame
        """
        if frame_idx not in self.frames:
            self.frames[frame_idx] = Frame(idx=frame_idx)
        self.frames[frame_idx] = self.frames[frame_idx].update(point=point, bbox=bbox, masks=masks)
        return self
    

    def load_frames(self, frames: List[Frame]):
        self.frames = {frame.idx: frame for frame in frames}
        return

    def __getitem__(self, id):
        return self.frames[id]
    def __len__(self):
        return len(self.frames)
        
    def __repr__(self) -> str:
        return f"{[f for f in self.frames]}"
    
    def get_frames_for_http_request(self):
        return {frame.idx: Frame.to_dict(frame) for frame in self.frames.values()}
    
    def clean_frames(self):
        self.frames = {}
        return self

