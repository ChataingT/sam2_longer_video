import os
import sys
import torch 
import logging

# Add SAM2 to path
SAM2_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, SAM2_PATH)

from sam2.build_sam import build_sam2_video_predictor

from ..server.utils import *
log = logging.getLogger('myapp')

this_file_path = os.path.dirname(os.path.abspath(__file__))
MODEL_CHECKPOINTS_DIR = os.getenv("MODEL_CHECKPOINTS_DIR", os.path.join(this_file_path, "../../checkpoints"))
MODEL_CONFIG_DIR = os.getenv("MODEL_CONFIG_DIR", os.path.join(this_file_path, "../configs/sam2_1"))

class SamManager():

    predictor = None
    inference_state = None
    video_path:str = None

    def __init__(self, 
                 model_size:str="tiny", 
                 offload_video_to_cpu=False, 
                 offload_state_to_cpu=False,
                 ) -> None:
                # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        log.info(f"using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            log.warning(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
        log.info(f"Build SAM2 — size: {model_size}")

        # Select the model configuration and checkpoint based on the model size
        model_cfg, sam2_checkpoint = self._select_model(model_size)
        log.info(f"Using model config: {model_cfg}")
        log.info(f"Using model checkpoint: {sam2_checkpoint}")

        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        self.frames = FrameManager()

    def _select_model(self, model_size:str):
        """
        Select the model configuration and checkpoint based on the model size
        """
        if model_size == "tiny":
            model_cfg = os.path.join(MODEL_CONFIG_DIR, "sam2.1_hiera_t.yaml")
            sam2_checkpoint = os.path.join(MODEL_CHECKPOINTS_DIR, "sam2.1_hiera_tiny.pt")
        elif model_size == "small":
            model_cfg = os.path.join(MODEL_CONFIG_DIR, "sam2.1_hiera_s.yaml")
            sam2_checkpoint = os.path.join(MODEL_CHECKPOINTS_DIR, "sam2_small.pt")
        elif model_size == "base_plus":
            model_cfg = os.path.join(MODEL_CONFIG_DIR, "sam2.1_hiera_b+.yaml")
            sam2_checkpoint = os.path.join(MODEL_CHECKPOINTS_DIR, "sam2.1_hiera_base_plus.pt")
        elif model_size == "large":
            model_cfg = os.path.join(MODEL_CONFIG_DIR, "sam2.1_hiera_l.yaml")
            sam2_checkpoint = os.path.join(MODEL_CHECKPOINTS_DIR, "sam2.1_hiera_large.pt")
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        return model_cfg, sam2_checkpoint

    def init_model(self, video_path:str):
        """
        Build then Init sam model from the model size chosen
        video_path is expected to be absolute
        """

        log.info(f"Init sam2 model from video path {video_path} and offload_video_to_cpu {self.offload_video_to_cpu} and offload_state_to_cpu {self.offload_state_to_cpu}")
        self.video_path = video_path
        self.inference_state = self.predictor.init_state(video_path=video_path,
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=True,
            )  
        return True
    
    def add_points_and_bbox(self,points:list=None, bboxs:list=None):
        log.debug(f"add points and bbox {(points, bboxs)}")
        if bboxs:
            for bbox in bboxs:
                if not isinstance(bbox, BBox):
                    bbox = BBox.from_dict(bbox)
                log.debug(f"add bboxs {bbox}")
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=bbox.frame_id,
                    obj_id=bbox.target_id,
                    box=bbox.coord
                )
                masks = FrameMask(target_ids=out_obj_ids, mask_logits=out_mask_logits)
                self.frames = self.frames.update_or_create(frame_idx=bbox.frame_id, bbox=bbox, masks=masks)

        if points:
            for point in points:
                if not isinstance(point, Point):
                    point = Point.from_dict(point)
                log.debug(f"add points {point}")
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=point.frame_id,
                    obj_id=point.target_id,
                    points=[point.coord],
                    labels=[point.label]
                )
                masks = FrameMask(target_ids=out_obj_ids, mask_logits=out_mask_logits)
                self.frames = self.frames.update_or_create(frame_idx=point.frame_id, point=point, masks=masks)
        return self.frames

    def propagate_in_video(self, start_frame_idx=None,
                                 max_frame_num_to_track=None,
                                 reverse=False,
                                 print_gpumem_every=0):
        """
        Propagate sam inference until the idx_frame
        """
        log.info(f"Propagate till {max_frame_num_to_track}")
        propagater = self.predictor.propagate_in_video(self.inference_state, 
                                                        start_frame_idx=start_frame_idx,
                                                        max_frame_num_to_track=max_frame_num_to_track,
                                                        reverse=reverse,
                                                        print_gpumem_every=print_gpumem_every
                                                        )
                
        for frame_idx, object_ids, mask_logits in propagater:
            masks = FrameMask(target_ids=object_ids, mask_logits=mask_logits)
            self.frames.update_or_create(frame_idx=frame_idx,  masks=masks)

        return self.frames
    
    def reset_state(self):
        """
        Reset SAM2 state
        """
        log.info("Reset SAM state")
        self.predictor.reset_state(self.inference_state)
        if not self.inference_state["temp_output_dict_per_obj"]:  # Check if the state is reset
            log.info("SAM2 state reset successfully")
            return True


if __name__ == "__main__":

    samM = SamManager()
    test_video_path = os.path.join(this_file_path, "../../test/assets/bedroom.mp4")
    samM.init_model(test_video_path)
    