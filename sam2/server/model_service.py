"""
SAM2 Model Service
A standalone FastAPI service that provides SAM2 model inference capabilities.
This service will be consumed by the main webapp container.
"""

import os
import sys
import logging

import uvicorn
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException

# Add SAM2 to path
SAM2_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, SAM2_PATH)

from sam2.sam_manager.sam_manager import SamManager
from sam2.server.utils import FrameManager

from model_requests import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['HYDRA_FULL_ERROR'] = '1'

# ThreadPoolExecutor for running tasks in the background
executor = ThreadPoolExecutor(max_workers=1)

this_file_path = os.path.dirname(os.path.abspath(__file__))
OUTPUT_LOCAL = os.path.join(this_file_path, "../../data/output")
INPUT_VIDEO_DIR = os.path.join(this_file_path, "../../data/input")

MODEL_CHECKPOINTS_DIR = os.path.join(SAM2_PATH, "checkpoints")
MODEL_CONFIG_DIR = os.path.join(SAM2_PATH, "sam2", "configs", "sam2_1")

app = FastAPI(
    title="SAM2 Model Service",
    description="Standalone SAM2 model inference service",
    version="1.0.0"
)

# Global vars
sam_manager = None
initialization_result = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global initialization_result
    return {
        "status": "healthy",
        "models_loaded": initialization_result,
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "SAM2 Model Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Health check endpoint",
            "/initialize": "Initialize the SAM2 model",
            "/add_points_and_bbox": "Add points and bounding boxes for segmentation",
            "/propagate": "Propagate segmentation masks through the video",
            "/reset_state": "Reset the inference state",        }
    }

@app.post("/initialize")
async def initialize(request: InitializeRequest):
    """
    Initialize model and inference state on selected video
    
    Args:
        request: Initialize request with video_path, model_size, and offload options
    
    Returns:
        JSON response with initialization status
    """
    global sam_manager, initialization_result

    def init_and_store_result(video_path):
        try:
            result = sam_manager.init_model(video_path)
            logger.info(f"Model initialization result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error during model initialization: {e}")

    try:
        logger.info(f"Loading SAM2 models with size: {request.model_size}")

        sam_manager = SamManager(model_size=request.model_size,
                                 offload_video_to_cpu=True,
                                 offload_state_to_cpu=True)
        logger.info(f"Model instantiated with size {request.model_size} and video path {request.video_path} and offload_video_to_cpu {True} and offload_state_to_cpu {True}")


        # Submit the task to the executor
        future = executor.submit(init_and_store_result, request.video_path)
        initialization_result = future.result()  # Wait for the result

        # create local directory if it does not exist
        if not os.path.exists(OUTPUT_LOCAL):
            os.makedirs(OUTPUT_LOCAL)

        content = {"message": "Model initialized", "result": initialization_result}
        return JSONResponse(content=content, status_code=201)

    except Exception as e:
        logger.error(f"Error initializing SAM2 model: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize SAM2 model")
    
@app.post("/add_points_and_bbox")
async def add_points_and_bbox(request: AddPointsAndBboxRequest):
    """
    Add points and bounding boxes to the video segmentation
    
    Args:
        points: List of points to add
        bboxs: List of bounding boxes to add
    
    Returns:
        JSON response with updated frames
    """
    global sam_manager

    if not sam_manager:
        raise HTTPException(status_code=400, detail="Model not initialized")

    try:
        frame_manager = sam_manager.add_points_and_bbox(points=request.points, bboxs=request.bboxs)
        response = {}
        response["message"] = "Detection Finished!"
        response["frames"] = frame_manager.get_frames_for_http_request()
        return JSONResponse(content= response, status_code=200)
        
    except Exception as e:
        logger.error(f"Error adding points and bounding boxes: {e}")
        raise HTTPException(status_code=500, detail="Failed to add points and bounding boxes")

@app.post("/propagate")
async def propagate(request: PropagateRequest):
    """
    Propagate segmentation masks through the video
    
    Args:
        request: Propagate request with start_frame_idx, max_frame_num_to_track, and reverse options
    Returns:
        JSON response with propagation results
    """
    global sam_manager

    if not sam_manager:
        raise HTTPException(status_code=400, detail="Model not initialized")

    try:
        frame_manager = sam_manager.propagate_in_video(
            start_frame_idx=request.start_frame_idx,
            max_frame_num_to_track=request.max_frame_num_to_track,
            reverse=request.reverse
        )
        response = {}
        response["message"] = "Propagation Finished!"
        response["frames"] = frame_manager.get_frames_for_http_request()
        return JSONResponse(content=response, status_code=200)
        
    except Exception as e:
        logger.error(f"Error during propagation: {e}")
        raise HTTPException(status_code=500, detail="Failed to propagate segmentation masks")


@app.delete("/reset_state")
async def reset_state():
    """Reset the inference state"""
    global sam_manager

    if not sam_manager:
        raise HTTPException(status_code=400, detail="Model not initialized")

    reset_true = sam_manager.reset_state()
    if not reset_true:
        raise HTTPException(status_code=500, detail="Failed to reset inference state")
    else :
        logger.info("SAM2 state reset successfully")
        return JSONResponse(content={"message": "SAM2 state reset successfully"}, status_code=200)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )
