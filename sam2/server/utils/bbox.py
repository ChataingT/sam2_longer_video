class BBox():
    def __init__(self, frame_id:int, target_id:int, coord:list) -> None:
        self.frame_id = frame_id
        self.target_id = target_id
        self.coord = coord

    def to_dict(self):
        return {
            "frame_id": str(self.frame_id),
            "target_id": str(self.target_id),
            "coord": self.coord
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[fid:{self.frame_id}][{self.target_id}-{self.coord}]"

    def scale(self, width:int, height:int):
        self.coord = [int(self.coord[0]*width), int(self.coord[1]*height), int(self.coord[2]*width), int(self.coord[3]*height)]
        return self
    
    @classmethod
    def from_dict(cls, bbox_dict:dict):
        frame_id = int(bbox_dict.get("frame_id"))
        target_id = int(bbox_dict.get("target_id"))
        coord = bbox_dict.get("coord")
        return cls(frame_id=frame_id, target_id=target_id, coord=coord)