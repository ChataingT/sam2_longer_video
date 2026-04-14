class Point():
    def __init__(self, frame_id:int, target_id:int, coord:list, label:int) -> None:
        self.frame_id = frame_id
        self.target_id = target_id
        self.coord = coord
        self.label = label

    def to_dict(self):
        return {
            "frame_id": str(self.frame_id),
            "target_id": str(self.target_id),
            "coord": self.coord,
            "label": self.label
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[fid:{self.frame_id}][{self.target_id}-{self.coord}_{self.label}]"

    def scale(self, width:int, height:int):
        self.coord = [int(self.coord[0]*width), int(self.coord[1]*height)]
        return self
    
    @classmethod
    def from_dict(cls, point_dict:dict):
        frame_id = int(point_dict.get("frame_id"))
        target_id = int(point_dict.get("target_id"))
        coord = point_dict.get("coord")
        label = int(point_dict.get("label"))
        return cls(frame_id=frame_id, target_id=target_id, coord=coord, label=label)
