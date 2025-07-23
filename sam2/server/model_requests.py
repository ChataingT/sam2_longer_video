"""
Request models for SAM2 Model Service
Pydantic models for request validation and documentation
"""

from pydantic import BaseModel
from typing import List

# -------------------
# SAM2 MODEL SERVICE REQUEST MODELS
# -------------------

class InitializeRequest(BaseModel):
    """Request model for initializing the SAM2 model"""
    video_path: str
    model_size: str = "tiny"
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True

class AddPointsAndBboxRequest(BaseModel):
    """Request model for adding points and bounding boxes to the SAM2 model"""
    points: List[dict] = None  # List of points with frame_id, target_id, coord, label
    bboxs: List[dict] = None    # List of bounding boxes with frame_id, target_id, coord

class PropagateRequest(BaseModel):
    """Request model for video segmentation initialization"""
    start_frame_idx : int = 0
    max_frame_num_to_track: int = None
    reverse: bool = False
    print_gpumem_every: int = 0

# -------------------
# LEGACY REQUEST MODELS (for compatibility with main webapp)
# -------------------

class VideoRequest(BaseModel):
    session: str

class SetRunConfigRequest(BaseModel):
    video_name: str
    video_resolution: str
    mdl_size: str
    frames: List[dict] = None

class InitializeRequestLegacy(BaseModel):
    video_name: str
    mdl_size: str

class DetectRequest(BaseModel):
    video_name: str
    frames: List[dict] = None

class PropagateRequestLegacy(BaseModel):
    video_name: str
    frames: List[dict] = None
    propframes: int = None