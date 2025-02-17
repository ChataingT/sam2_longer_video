# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict

import torch
import gc
import cv2

from sympy.physics.units import current
from tqdm import tqdm
import math

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames, AsyncVideoFrameLoader
import subprocess  # TODO; Temporarily used for development to check memory usage, will be removed later

class SAM2VideoPredictor(SAM2Base):
    """Predictor class for handling user interactions and managing inference state."""

    def __init__(
        self,
        fill_hole_area=0,
        # Whether to apply non-overlapping constraints to the output object masks
        non_overlap_masks=False,
        # Whether to clear non-conditional memory around frames with correction clicks
        # Note: This only applies to *single object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True
        clear_non_cond_mem_around_input=False,
        # Whether to also clear non-conditional memory around frames for multiple objects (only effective when `clear_non_cond_mem_around_input` is True)
        clear_non_cond_mem_for_multi_obj=False,
        add_all_frames_to_correct_as_cond=False,

        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.add_all_frames_to_correct_as_cond=add_all_frames_to_correct_as_cond

        # memory monitoring
        self.l_used =[]
        self.l_free =[]
        

    # Initialize inference_state
    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu=True,  # True to transfer video frames to CPU memory, reducing GPU memory usage by 0.025G per frame, saving 2.5G per 100 frames
        offload_state_to_cpu=False,  # Set to False for speed, True to save GPU memory! When True, all tensor device transfers involving storage_device need to set non_blocking=False to avoid mask output misalignment; stores tensors in ['output_dict] and ['output_dict_per_obj'] on CPU, increasing time overhead by about 22%
        async_loading_frames=False,  # False to implement AsyncVideoFrameLoader compatible with nd array format input, but asynchronous loading seems to have no impact on final GPU and memory usage
    ):
        print(f"Initialization: offload_video_to_cpu:{offload_video_to_cpu}, offload_state_to_cpu:{offload_state_to_cpu}")
        """Initialize inference state."""
        compute_device = self.device  # Device where the model is located
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
        inference_state = {}
        inference_state["images"] = images  # Store video frame images (N, 3, 1024, 1024)
        # Number of video frames N, may serve as a marker for the maximum frame index, needs to always be the total number of frames ever loaded rather than the current number of frames
        inference_state["num_frames"] = len(images)  # (In specific cases, old frames may be removed to free memory, so the total number of frames ever loaded may not equal the current number of frames)
        # Record the mapping of tensor indices in images to actual video frame indices [0,1,4,5,6,9,10,11,...] where the number K at index N indicates that the Nth tensor in images corresponds to the Kth frame of the actual video
        inference_state["images_idx"] = list(range(len(images)))
        # Whether to transfer video frames to CPU memory
        # Enabling this option can save GPU memory with very little overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # Whether to transfer inference state to CPU memory
        # Enabling this option can save GPU memory but will reduce tracking FPS
        # (e.g., in tests with the 768x768 model, FPS dropped from 27 to 24 when tracking one object, and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # Original video height and width, used to adjust final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device  # Compute device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")  # Set storage device to CPU
        else:
            inference_state["storage_device"] = compute_device  # Set storage device to compute device
        # Inputs for each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # Visual features of a few recently accessed frames for quick interaction
        inference_state["cached_features"] = {}
        # Values that remain constant across all frames (so we only need to save one copy)
        inference_state["constants"] = {}
        # Mapping between client object IDs and model object indices
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Store tracking results and states of the model on each frame. Save overhead:
        # The tensors in the dictionary ["cond_frame_outputs"]["maskmem_features"] and ["pred_masks"] are stored on the storage_device
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # Dictionary containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # Dictionary containing {frame_idx: <out>}
        }
        # Slices (views) of tracking results for each object, sharing the same memory with "output_dict". Ensured that ["maskmem_features"] are stored on the storage_device
        inference_state["output_dict_per_obj"] = {}
        # Temporary storage, new outputs are stored here when users interact with frames (e.g., adding clicks or masks)
        # Will be merged into "output_dict" before propagation begins
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already contain merged outputs from click or mask inputs
        # (we directly use their merged outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # Set containing frame indices
            "non_cond_frame_outputs": set(),  # Set containing frame indices
        }
        # Metadata for each tracking frame (e.g., tracking direction)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Indices of all conditional frames and non-conditional frames in the preloaded memory
        inference_state["preloading_memory_cond_frame_idx"] = None
        inference_state["preloading_memory_non_cond_frames_idx"] = None
        # Maximum update length when updating historical frame information if a new client ID appears during tracking
        inference_state["max_update_length_for_new_obj_id"] = 100
        # Warm up the vision backbone network and cache the image features of frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    # Reset inference_state settings in the preloaded memory (e.g., memory optimization settings)
    def init_preloading_state(
        self,
        inference_state,
        offload_video_to_cpu = True,  # True to transfer video frames to CPU memory, reducing GPU memory usage by 0.025G per frame, saving 2.5G per 100 frames
        offload_state_to_cpu = True,  # ! When True, all tensor device transfers involving storage_device need to set non_blocking=False to avoid mask output misalignment; stores tensors in ['output_dict] and ['output_dict_per_obj'] on CPU, increasing time overhead by about 22%
    ):
        '''
        This method is only called when externally loading the preloaded memory, used to synchronize related settings (e.g., storage optimization parameters) of the preloaded memory to the latest settings for this inference
        '''
        if offload_video_to_cpu:
            inference_state["images"] = inference_state["images"].to("cpu")  # Transfer video frames to CPU memory

        compute_device = self.device  # Device where the model is located
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")  # Set storage device to CPU
        else:
            inference_state["storage_device"] = compute_device  # Set storage device to compute device
        device = inference_state["storage_device"]

        for frame_idx in range(inference_state["num_frames"]-1):
            # print(f"Processing preloaded memory for {frame_idx}")
            # Only conditional frames should exist in the memory
            current_cond_frame = inference_state['output_dict']['cond_frame_outputs'][frame_idx]

            current_cond_frame['maskmem_features'] = current_cond_frame['maskmem_features'].to(device,non_blocking=False)
            current_cond_frame['pred_masks'] = current_cond_frame['pred_masks'].to(device,non_blocking=False)

            for obj_idx in inference_state["obj_idx_to_id"].keys():
                current_cond_obj = inference_state['output_dict_per_obj'][obj_idx]['cond_frame_outputs'][frame_idx]

                current_cond_obj['maskmem_features'] = current_cond_obj['maskmem_features'].to(device,non_blocking=False)
                current_cond_obj['pred_masks'] = current_cond_obj['pred_masks'].to(device,non_blocking=False)

        print("Preloaded memory processing completed")

    # Add new frames to an existing inference_state
    @torch.inference_mode()
    def update_state(
        self,
        video_path,
        inference_state,  # Add new frames, update to inference_state
        async_loading_frames=False,
    ):
        '''
        Use init_state method for the first time to add frames, then use this method for subsequent additions
        '''
        # Load new video frames
        new_images, new_video_height, new_video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=inference_state["offload_video_to_cpu"],
            async_loading_frames=async_loading_frames,
            compute_device=self.device,
        )

        # Get the original video height and width
        video_height = inference_state["video_height"]
        video_width = inference_state["video_width"]
        assert video_height == new_video_height and video_width == new_video_width, "Ensure new image frames have the same height and width as existing frames"

        # Merge new and existing frame index mapping lists
        last_frame_index = inference_state["images_idx"][-1]
        new_frame_indices = list(range(last_frame_index+1, last_frame_index+1 + len(new_images)))
        inference_state["images_idx"].extend(new_frame_indices)

        # Merge new and existing frames
        images = inference_state["images"]  # Get previous video frame images
        assert images.shape[1:] == new_images.shape[1:], "Ensure new image frames have the same dimensions (channels, height, width)"
        # Check if images and new_images are instances of AsyncVideoFrameLoader for asynchronous loading
        if isinstance(images, AsyncVideoFrameLoader):
            images = images.to_tensor()
        if isinstance(new_images, AsyncVideoFrameLoader):
            new_images = new_images.to_tensor()
        combined_images = torch.cat((images, new_images), dim=0)  # torch.Size([N frames, 3, 1024, 1024])

        #  print("---update_state:combined_images.shape:",combined_images.shape)

        # Update inference state
        inference_state["images"] = combined_images  # Store video frame images
        inference_state["num_frames"] += len(new_images)  # Number of video frames, needs to always be the total number of frames ever loaded rather than the current number of frames, otherwise it may cause errors
        # print("Total frames",inference_state["num_frames"])

        return inference_state


    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2VideoPredictor":
        """
        Load a pre-trained model from the Hugging Face hub.

        Parameters:
          model_id (str): The ID of the Hugging Face repository.
          **kwargs: Additional parameters to pass to the model constructor.
        Returns:
          (SAM2VideoPredictor): The loaded model instance.
        """
        from sam2.build_sam import build_sam2_video_predictor_hf

        sam_model = build_sam2_video_predictor_hf(model_id, **kwargs)  # Load and build the model
        return sam_model  # Return the loaded model instance

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client object ID to model object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)  # Get object index
        if obj_idx is not None:
            return obj_idx  # If object ID already exists, return the corresponding object index

        # This is a new object ID that has not been sent to the server before. We only allow adding new objects before tracking starts.
        allow_new_object = not inference_state["tracking_has_started"]  # Determine if adding a new object during non-tracking
        if allow_new_object:  # Add new category when tracking has not started
            # Get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])  # Assign a new object index
            inference_state["obj_id_to_idx"][obj_id] = obj_idx  # Update the mapping from object ID to index
            inference_state["obj_idx_to_id"][obj_idx] = obj_id  # Update the mapping from object index to ID
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])  # Update the list of object IDs
            # Set input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}  # Initialize point input structure for this object
            inference_state["mask_inputs_per_obj"][obj_idx] = {}  # Initialize mask input structure for this object
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # Dictionary containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # Dictionary containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # Dictionary containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # Dictionary containing {frame_idx: <out>}
            }
            return obj_idx  # Return the newly assigned object index
        else:  # Add new category when tracking has started
            # Get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])  # Assign a new object index
            inference_state["obj_id_to_idx"][obj_id] = obj_idx  # Update the mapping from object ID to index
            inference_state["obj_idx_to_id"][obj_idx] = obj_id  # Update the mapping from object index to ID
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])  # Update the list of object IDs
            # Set input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}  # Initialize point input structure for this object
            inference_state["mask_inputs_per_obj"][obj_idx] = {}  # Initialize mask input structure for this object
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # Dictionary containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # Dictionary containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # Dictionary containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # Dictionary containing {frame_idx: <out>}
            }

            # Print function
            def print_consolidated_out(consolidated_out):
                for key, value in consolidated_out.items():
                    if isinstance(value, torch.Tensor):
                        print(f"Key: {key}, Value: Tensor, Shape: {value.shape}")
                    elif isinstance(value, dict):
                        print(f"Key: {key}, Value: dict")
                        print_consolidated_out(value)  # Recursively print nested dictionaries
                    elif isinstance(value, list):
                        print(f"Key: {key}, Value: list, Length: {len(value)}")
                    else:
                        print(f"Key: {key}, Value: {value}")

            preloading_memory_cond_frame_idx = inference_state["preloading_memory_cond_frame_idx"]  # List of conditional frame indices in the preloaded memory
            max_update_length = inference_state["max_update_length_for_new_obj_id"]  # Get the maximum length to update
            print(f"A new client ID appeared during tracking, updating the information of the last {max_update_length} frames and the preloaded memory (if any) with the latest ID mapping standard")

            output_dict = inference_state["output_dict"]
            cond_frame_indices = sorted(output_dict["cond_frame_outputs"].keys())  # Get all conditional frame indices and sort them in chronological order
            # Only select the most recent max_update_length frame indices
            if max_update_length > 0:
                cond_frame_indices = cond_frame_indices[-max_update_length:]
            # Add conditional frames in the preloaded memory to the update queue
            if preloading_memory_cond_frame_idx is not None:
                for t in preloading_memory_cond_frame_idx:
                    if t not in cond_frame_indices:
                        cond_frame_indices.append(t)
            # Update all historical conditional frames with the new mapping standard
            for cond_frame_idx in tqdm(cond_frame_indices, desc=f"Updating the last {max_update_length} frames and conditional frames in the preloaded memory"):
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx=cond_frame_idx,
                    is_cond=True,
                    run_mem_encoder=True,
                    consolidate_at_video_res=False,
                )
                # print_consolidated_out(consolidated_out)
                # Merge them into the "output_dict"

                output_dict["cond_frame_outputs"][cond_frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, cond_frame_idx, consolidated_out, storage_key="cond_frame_outputs"
                )
            # # Update all historical non-conditional frames with the new mapping standard TODO: Is it necessary to update non-conditional frames? Commenting out this step seems to work, if unnecessary, this step can be omitted to reduce computational overhead
            # non_cond_frame_outputs = inference_state["output_dict"]["non_cond_frame_outputs"]
            # for non_cond_frame_idx in tqdm(non_cond_frame_outputs.keys(),desc="Updating all historical non-conditional frames"):
            #     consolidated_out = self._consolidate_temp_output_across_obj(
            #         inference_state,
            #         frame_idx=non_cond_frame_idx,
            #         is_cond=False,
            #         run_mem_encoder=True,
            #         consolidate_at_video_res=False,
            #     )
            #     # print_consolidated_out(consolidated_out)
            #     # Merge them into the "output_dict"
            #     output_dict["non_cond_frame_outputs"][non_cond_frame_idx] = consolidated_out
            #     self._add_output_per_object(
            #         inference_state, non_cond_frame_idx, consolidated_out, storage_key="non_cond_frame_outputs"
            #     )
            return obj_idx  # Return the newly assigned object index

            # raise RuntimeError(
            #     f"Cannot add new object ID {obj_id} after tracking has started. "
            #     f"All existing object IDs: {inference_state['obj_ids']}."
            #     f"Please call 'reset_state' to start over."
            # )  # If tracking has started, throw an error

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model object index to client object ID."""
        return inference_state["obj_idx_to_id"][obj_idx]  # Return the client object ID corresponding to the given object index

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object IDs received in this session so far."""
        return len(inference_state["obj_idx_to_id"])  # Return the total number of object IDs
    
    @torch.inference_mode()  # Run in inference mode, disabling gradient computation for efficiency
    def add_new_points_or_box(
        self,
        inference_state,  # Inference state dictionary containing all information during tracking
        frame_idx,  # Index of the current frame
        obj_id,  # Object ID provided by the client
        points=None,  # Points to add (coordinates), default is None
        labels=None,  # Labels for each point (e.g., [1,0,0,0,1] etc., 1 positive prompt, 0 negative prompt), default is None
        clear_old_points=True,  # Whether to clear old points, default is True
        normalize_coords=True,  # Whether to normalize point coordinates, default is True
        box=None,  # Box to add (coordinates), default is None
    ):
        """Add new point prompts to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)  # Map client object ID to model object index
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]  # Get point inputs for the corresponding frame
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]  # Get mask inputs for the corresponding frame

        if (points is not None) != (labels is not None):  # If points and labels are not provided together, raise an exception
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:  # If neither points nor box are provided, raise an exception
            raise ValueError("At least one of points or box must be provided as input")

        if points is None:  # If no points are provided, create an empty point tensor
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):  # If points are not a tensor, convert them to a tensor
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:  # If no labels are provided, create an empty label tensor
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):  # If labels are not a tensor, convert them to a tensor
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:  # If the point tensor is 2-dimensional, add a batch dimension
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:  # If the label tensor is 1-dimensional, add a batch dimension
            labels = labels.unsqueeze(0)  # add batch dimension

        # If a box is provided, add it as the first two points and set labels to 2 and 3
        # This is consistent with how SAM 2 is trained
        if box is not None:
            if not clear_old_points:  # If not clearing old points, raise an exception because the box must be added before points
                raise ValueError(
                    "Cannot add box without clearing old points, as box prompts must be provided before point prompts "
                    "(please use clear_old_points=True)"
                )
            if not isinstance(box, torch.Tensor):  # If the box is not a tensor, convert it to a tensor
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)  # Reshape the box to (1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)  # Create box labels
            box_labels = box_labels.reshape(1, 2)  # Reshape the labels
            points = torch.cat([box_coords, points], dim=1)  # Concatenate box points with other points
            labels = torch.cat([box_labels, labels], dim=1)  # Concatenate box labels with other labels

        if normalize_coords:  # If coordinates need to be normalized
            video_H = inference_state["video_height"]  # Get video height
            video_W = inference_state["video_width"]  # Get video width
            points = points / torch.tensor([video_W, video_H]).to(points.device)  # Normalize points
        # Scale coordinates according to the model's internal image size
        points = points * self.image_size
        points = points.to(inference_state["device"])  # Move points to the inference device
        labels = labels.to(inference_state["device"])  # Move labels to the inference device

        if not clear_old_points:  # If not clearing old points, get the current frame's point inputs
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None  # Otherwise, clear point inputs
        point_inputs = concat_points(point_inputs, points, labels)  # Concatenate new and old point inputs

        point_inputs_per_frame[frame_idx] = point_inputs  # Update the frame's point inputs
        mask_inputs_per_frame.pop(frame_idx, None)  # Remove old mask inputs

        # If this frame has not been tracked before, treat it as an initial condition frame,
        # meaning the input points will be used to generate segmentation on this frame without using memory from other frames (as in SAM).
        # Otherwise (if already tracked), the input points will be used to correct the already tracked mask.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # Whether to track in reverse chronological order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]  # Get the object's output dictionary
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]  # Get the object's temporary output dictionary

        # If it is an initial condition frame or the model treats all frames receiving clicks/masks as condition frames, add the frame to condition outputs
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get the previously predicted mask logits on this object and input them to the SAM mask decoder along with the new clicks
        prev_sam_mask_logits = None
        # First look in the temporary output dictionary, which contains the latest outputs
        # (if not found, look in condition and non-condition frame outputs)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        # If there is a previous output and there are predicted masks
        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = inference_state["device"]
            # Move the previous mask logits to the correct device and avoid blocking
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            # Limit the scale of prev_sam_mask_logits to avoid rare numerical issues
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)

        # Run single frame inference, returning the current frame's output
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # Process the output slice for a single object
            frame_idx=frame_idx,
            batch_size=1,  # Process the output slice for a single object
            is_init_cond_frame=is_init_cond_frame,  # Whether it is an initial condition frame
            point_inputs=point_inputs,  # Input points for the current frame
            mask_inputs=None,  # Mask inputs for the current frame are empty
            reverse=reverse,  # Whether to track in reverse chronological order
            # Skip the memory encoder when adding clicks or masks. We run the memory encoder at the beginning of `propagate_in_video` (after the user finishes clicking).
            # This allows us to enforce non-overlapping constraints on all objects before encoding them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,  # Previous mask logits as input for the current inference
        )
        # Add the output to the output dictionary (for future use as memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Adjust the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,  # Current frame index
            is_cond=is_cond,  # Whether it is a condition frame
            run_mem_encoder=False,  # Do not run the memory encoder
            consolidate_at_video_res=True,  # Consolidate output at video resolution
        )
        # Get the output mask at the original video resolution
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )

        # Return the current frame index, list of object IDs, and masks at video resolution
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
    ):
        """Add a new mask to a specific frame."""
        # Get the corresponding object index based on the object ID
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        # Get the point inputs and mask inputs for the current frame
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        # If the mask is not a tensor, convert it to a boolean tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        # Ensure the mask is 2-dimensional
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape

        mask_inputs_orig = mask[None, None]  # Add batch and channel dimensions to the mask
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])  # Convert the mask to float and move it to the specified device

        # If the mask size does not match the model's image size, resize it
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # Use anti-aliasing downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        # Store the resized mask input in the mask input dictionary for the current frame
        mask_inputs_per_frame[frame_idx] = mask_inputs
        # Remove the point inputs for the current frame from the point input dictionary
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame has not been tracked before, treat it as an initial condition frame,
        # meaning the input points will be used to generate segmentation on this frame without using memory from other frames (as in SAM).
        # Otherwise (if already tracked), the input points will be used to correct the already tracked mask.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # Determine whether to track in reverse chronological order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        # Get the output dictionary and temporary output dictionary for the current object
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # If it is an initial condition frame or the model treats all frames receiving clicks/masks as condition frames, add the frame to condition outputs.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        # Choose the storage key based on whether it is a condition frame
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Run single frame inference, generating the output for the current frame
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # Process the output slice for a single object
            frame_idx=frame_idx,
            batch_size=1,  # Process the output slice for a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or masks. We run the memory encoder at the beginning of `propagate_in_video` (after the user finishes clicking).
            # This allows us to enforce non-overlapping constraints on all objects before encoding them into memory.
            run_mem_encoder=False,
        )
        # Add the current output to the temporary output dictionary (for future use as memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Adjust the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        # Get the output mask adjusted to the original video resolution
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        # Return the frame index, list of object IDs, and masks at video resolution
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Adjust object scores to the original video resolution (video_res_masks),
        and apply non-overlapping constraints for the final output.
        """
        device = inference_state["device"]  # Get device information
        video_H = inference_state["video_height"]  # Get video height
        video_W = inference_state["video_width"]  # Get video width
        any_res_masks = any_res_masks.to(device, non_blocking=True)  # Move masks to the device, non-blocking
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks  # If the mask resolution already matches the video, use it directly
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),  # Resize the mask to the video's width and height
                mode="bilinear",  # Use bilinear interpolation
                align_corners=False,  # Do not align corners
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)  # Apply non-overlapping constraints if needed

        # Move video_res_mask and any_res_masks to storage_device:
        video_res_masks = video_res_masks.to(inference_state["storage_device"], non_blocking=False)  # non_blocking=False
        any_res_masks = any_res_masks.to(inference_state["storage_device"], non_blocking=False)  # non_blocking=False
        return any_res_masks, video_res_masks  # Return the masks before and after resizing to video resolution

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate the temporary outputs `temp_output_dict_per_obj` for each object at a specific frame into a single output for all objects, including the following steps:
        1) Fill in any missing objects: if they exist in `output_dict_per_obj`,
            use the contents from `output_dict_per_obj`; if not, leave them as placeholders.
        2) Optionally, re-run the memory encoder after applying non-overlapping constraints to the object scores.

        The consolidated_out['maskmem_features'] should be on inference_state['storage_device'] to save resources.
        """
        batch_size = self._get_obj_num(inference_state)  # Get the number of objects
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"  # Choose the output dictionary key based on the condition
        # Optionally, allow consolidating temporary outputs at the original video resolution (for a better mask hint editing experience).
        if consolidate_at_video_res:
            assert not run_mem_encoder, "Memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]  # Get video height
            consolidated_W = inference_state["video_width"]  # Get video width
            consolidated_mask_key = "pred_masks_video_res"  # Set the key for consolidated masks
        else:
            consolidated_H = consolidated_W = self.image_size // 4  # Set the resolution for consolidated masks
            consolidated_mask_key = "pred_masks"  # Set the key for consolidated masks

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added after re-running the memory encoder on the object scores.
        # Its "pred_masks" is pre-filled with a large negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,  # Use NO_OBJ_SCORE as the fill value for missing objects
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,  # Use NO_OBJ_SCORE as the fill value for missing objects
                dtype=torch.float32,
                device=inference_state["device"],
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                # By default, set object_score_logits to 10.0, assuming the object exists, as sigmoid(10)=1, consistent with the `MaskDecoder`'s `predict_masks` setting.
                fill_value=10.0,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
        }
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]  # Get the temporary output dictionary for each object
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]  # Get the output dictionary for each object
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)  # Get the temporary output for the corresponding frame
            # If the object is not in "temp_output_dict_per_obj", fall back to its previous output in
            # "output_dict_per_obj".
            # We look in both "cond_frame_outputs" and "non_cond_frame_outputs" to find the previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object is also not in "output_dict_per_obj", skip it
            # and leave its mask score as the default score (i.e., the NO_OBJ_SCORE placeholder above),
            # and set its object pointer to a dummy pointer.
            if out is None:
                # Fill in dummy object pointers for objects that have no input or tracking results on this frame
                # (only do this when `run_mem_encoder=True`, i.e., when we need to build memory for tracking).
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(
                            inference_state, frame_idx
                        )
                    # Fill the object pointer with a dummy pointer based on an empty mask
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue
            # Add the temporary object output mask to the consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask  # If the mask resolution matches, add it directly
            else:
                # If the temporary object mask has a different resolution, resize it first
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],  # Resize to the consolidated mask resolution
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]  # Update the object pointer
            consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out[
                "object_score_logits"
            ]

        # Optionally, apply non-overlapping constraints on the consolidated scores and re-run the memory encoder
        if run_mem_encoder:
            device = inference_state["device"]
            high_res_masks = torch.nn.functional.interpolate(
                consolidated_out["pred_masks"].to(device, non_blocking=True),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.non_overlap_masks_for_mem_enc:
                high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                object_score_logits=consolidated_out["object_score_logits"],
                is_mask_from_pts=True,  # These frames are the result of user interaction
            )

            consolidated_out["maskmem_features"] = maskmem_features  # Update memory encoder features, already on storage_device
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc  # Update memory encoder position encoding

        return consolidated_out  # Return the consolidated output

    def _get_empty_mask_ptr(self, inference_state, frame_idx):
        """Get a dummy object pointer based on an empty mask on the current frame."""
        # Create a dummy (empty) mask with only one object
        batch_size = 1
        mask_inputs = torch.zeros(
            (batch_size, 1, self.image_size, self.image_size),  # Mask size matches image size
            dtype=torch.float32,
            device=inference_state["device"],  # Use the device from inference state
        )

        # Get current image features
        (
            _,
            _,
            current_vision_feats,  # Current image features
            current_vision_pos_embeds,  # Current image position embeddings
            feat_sizes,  # Feature sizes
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # Input the empty mask and the above image features to get a dummy object pointer
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,  # Indicate this is an initial condition frame
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,  # No point inputs
            mask_inputs=mask_inputs,  # Use the empty mask
            output_dict={},
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,  # Do not run the memory encoder
            prev_sam_mask_logits=None,
            preloading_memory_cond_frame_idx=None,  # Do not pass preloaded memory condition frame indices
        )
        return current_out["obj_ptr"]  # Return the dummy object pointer

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference state and consolidate temporary outputs before tracking."""
        # Tracking has started, no new objects can be added until the session is reset.
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)  # Get the number of objects to track
        # print(f"Batch size in propagate_in_video_preflight: {batch_size}")

        # Consolidate temporary outputs in "temp_output_dict_per_obj" for each object and add them to "output_dict".
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" contains indices of frames with consolidated temporary outputs (either in the current call or previous calls to `propagate_in_video_preflight`).
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]

        for is_cond in [False, True]:
            # Consolidate conditional and non-conditional temporary outputs separately
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all frames with temporary outputs for any object
            # (these should be frames that just received clicks or mask inputs via `add_new_points_or_box` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            # Collect and update indices of temporary frame inference results for each object, and merge these indices into the global frame index record
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # print(f"{'Conditional' if is_cond else 'Non-conditional'} frames to consolidate: {sorted(temp_frame_inds)}")
            # # Non-conditional frames [], Conditional frames [90,105]

            # Consolidate temporary outputs for all objects on these frames
            for frame_idx in temp_frame_inds:
                # print(f"Frame index: {frame_idx}")
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
                )

                # def print_consolidated_out(consolidated_out):  # Print consolidated output
                #     for key, value in consolidated_out.items():
                #         if isinstance(value, torch.Tensor):
                #             print(f"Key: {key}, Value: Tensor, Shape: {value.shape}")
                #         elif isinstance(value, dict):
                #             print(f"Key: {key}, Value: dict")
                #             print_consolidated_out(value)  # Recursively print nested dictionaries
                #         elif isinstance(value, list):
                #             print(f"Key: {key}, Value: list, Length: {len(value)}")
                #         else:
                #             print(f"Key: {key}, Value: {value}")
                # print_consolidated_out(consolidated_out)

                # Merge them into "output_dict" and create slices for each object
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    # Clear non-conditional memory around the frame
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            # Clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # Edge case: If outputs were added to "cond_frame_outputs", remove any previous "non_cond_frame_outputs" on the same frame
        # (we do not want a frame to be both conditional and non-conditional).
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Ensure frame indices in "consolidated_frame_inds" are exactly those frames with point or mask inputs (should be true under correct workflow).
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())

        # I tried to implement release_old_frames() to remove old conditional and non-conditional frames to save memory, but this assertion blocked me.
        # Commenting out this assertion, I personally think it's fine, releasing old conditional frames will inevitably clear the indices in consolidated_frame_inds, and in fact, these indices will not be used in subsequent propagation.
        # assert all_consolidated_frame_inds == input_frames_inds  # Ensure all consolidated frame indices match input frame indices

    def print_gpu_memory(self, print_method=print, print_now=True):  # TODO: Temporarily used for development to check memory usage, will not be used in production
        try:
            # Use nvidia-smi to get memory information
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"])
            result = result.decode("utf-8").strip().split("\n")
            # Memory usage (used, free) for each GPU
            gpu_memory = [tuple(map(int, line.split(", "))) for line in result]
            if gpu_memory:
                for idx, (used, free) in enumerate(gpu_memory):
                    self.l_used.append(used)
                    self.l_free.append(free)
                    if print_now:
                        print_method(f"GPU Memory - Used: {min(self.l_used)} < {sum(self.l_used)/len(self.l_used)} < {max(self.l_used)} MB, Free: {min(self.l_free)} < {sum(self.l_free)/len(self.l_free)} < {max(self.l_free)} MB")
                        self.l_used, self.l_free = [], []
        except Exception as e:
            print_method(f"Error in getting GPU memory: {e}")
            return None
        
        


    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        print_gpumem_every=0
    ):
        """Propagate input points throughout the video for tracking."""

        # Prepare inference state and consolidate temporary outputs before tracking
        self.propagate_in_video_preflight(inference_state)  # Consolidate temporary outputs, clear unnecessary memory, and ensure consistency of inference state

        # Extract information from the processed inference_state
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # print(f"Extracted information from processed inference_state:")
        # cond_frame_outputs_count = len(output_dict.get("cond_frame_outputs", {}))
        # non_cond_frame_outputs_count = len(output_dict.get("non_cond_frame_outputs", {}))
        # print(f"Conditional frames in output_dict: {cond_frame_outputs_count}, Non-conditional frames: {non_cond_frame_outputs_count}")

        # # output_dict contains conditional frames from all previous and current inference sequences, only contains non-conditional frames from previous sequences, not current sequence
        # consolidated_cond_frame_count = len(consolidated_frame_inds["cond_frame_outputs"])
        # consolidated_non_cond_frame_count = len(consolidated_frame_inds["non_cond_frame_outputs"])
        # print(f"Conditional frames in consolidated_frame_inds: {consolidated_cond_frame_count}, Non-conditional frames: {consolidated_non_cond_frame_count}")

        # print(f"List of object IDs to track: {obj_ids}")
        # print(f"Total number of frames in video: {num_frames}")

        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points provided; please add points first")

        # Decide whether to clear non-conditional memory based on settings
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )
        # print(f"Clear non-conditional memory: {clear_non_cond_mem}")

        # Set start index, end index, and processing order
        if start_frame_idx is None:
            # Default to start from the first frame with input points
            start_frame_idx = min(output_dict["cond_frame_outputs"])
            # print(f"Start frame (from first conditional frame): {start_frame_idx}")
        if max_frame_num_to_track is None:
            # print(f"max_frame_num_to_track is None, tracking all frames in video")
            # Default to track all frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            # If tracking in reverse, calculate the end frame index
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track + 1, 0)  # This +1 is added to ensure accurate propagation length
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # Skip reverse tracking if starting from frame 0
        else:
            # If tracking forward, calculate the end frame index
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        # print(f"End frame (determined by start frame, tracking direction, and max tracking length): {end_frame_idx}")

        # Iterate through each frame in the processing order
        for frame_idx in tqdm(processing_order, desc=f"propagate in video start:{start_frame_idx},end:{end_frame_idx}"):
        # for frame_idx in processing_order:
            # Skip frames that already have consolidated outputs (these frames have already received input clicks or masks).
            # Note that we cannot directly perform batch inference because the number of clicks on each object may vary.
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                # print(f"Frame index {frame_idx}, already has conditional frame output, skipping inference")
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    # Clear non-conditional memory around the frame
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                # print(f"Frame index {frame_idx}, already has non-conditional frame output, skipping inference")
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                # print(f"Frame index {frame_idx}, not processed, performing single frame inference!")
                # For unprocessed frames, perform single frame inference
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )

                output_dict[storage_key][frame_idx] = current_out  # current_out['maskmem_features'] and ['pred_masks'] are on storage_device

            # Create output slices for each object for subsequent interactions
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )  # Split inference results for a frame by object and add to inference_state

            # Record whether a frame has been tracked and save the tracking direction for that frame
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Adjust output masks to original video resolution (directly use mask scores on GPU to avoid intermediate CPU conversion)
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )

            # if print_gpumem_every:
                # self.print_gpu_memory(print_method=tqdm.write, print_now=((frame_idx % print_gpumem_every) == 0))
                    
            yield frame_idx, obj_ids, video_res_masks  # Return current frame index, object IDs, and video resolution masks

    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        """
        Split multi-object output into per-object output slices and add them to `output_dict_per_obj`.
        The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            # Create slices for each object
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            # If there are maskmem_features, add them to the object output
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]  # On storage_device
            # If there are maskmem_pos_enc, add them to the object output
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]   # On GPU
            # Add the object's output to `output_dict_per_obj`
            obj_output_dict[storage_key][frame_idx] = obj_out

    @torch.inference_mode()
    def clear_all_prompts_in_frame(
            self, inference_state, frame_idx, obj_id, need_output=True
    ):
        """Clear all input points or masks for a given object in a specified frame."""
        # Get the object index based on the object ID
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Clear all conditional information on this frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Check if the current frame still has inputs for any object
        batch_size = self._get_obj_num(inference_state)
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break

        # If the frame no longer has inputs for any object, further clear the conditional state of the frame
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

            # Remove the conditional output of the frame (may downgrade to non-conditional frame output)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                # Since the frame no longer receives inputs, it is no longer a conditional frame, downgrade it to non-conditional frame output
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)

            # Similarly handle the slice output for each object
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out

            # If all conditional frames are removed, also clear the tracking output
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

        if not need_output:
            return

        # Finally, output the updated masks for each object (after removing inputs above)
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Delete all input points or masks for all frames in the video."""
        # Reset tracking results
        self._reset_tracking_results(inference_state)
        # Reset tracking results
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results in the video."""
        # Clear point inputs for each object
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        # Clear mask inputs for each object
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        # Clear conditional and non-conditional frame outputs in the output dictionary for each object
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        # Clear conditional and non-conditional frame outputs in the temporary output dictionary for each object
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        # Clear conditional and non-conditional frame outputs in the total output dictionary
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        # Clear consolidated frame indices
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        # Reset tracking state
        inference_state["tracking_has_started"] = False
        # Clear already tracked frames
        inference_state["frames_already_tracked"].clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute image features on a given frame."""
        # First look in the cache
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will perform inference on a single image
            device = inference_state["device"]
            # Get the image of the current frame from the cache, transfer to the device, convert to float, and add a batch dimension
            target_idx = inference_state["images_idx"].index(frame_idx) # images tensor does not continuously record video frames, need to map through images_idx to know which position in images corresponds to the actual frame
            # print(f"Trying to get actual frame {frame_idx}, position in images_idx {target_idx}")
            image = inference_state["images"][target_idx].to(device).float().unsqueeze(0)
            # Compute image features through forward propagation
            backbone_out = self.forward_image(image)
            # Cache the features of the most recent frame (for repeated interactions on the same frame; LRU cache can be used in the future to handle more frames)
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # Expand features to match the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        # Expand each feature map to match the batch size
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        # Expand each position encoding to match the batch size
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        # Prepare backbone features
        features = self._prepare_backbone_features(expanded_backbone_out)
        # Return a tuple containing the expanded image and prepared features
        features = (expanded_image,) + features
        return features

    # Release old frames to save GPU memory and memory
    def release_old_frames(self, inference_state, frame_idx, max_inference_state_frames, pre_frames, release_images=False):
        '''
        Clear frames that will no longer be inferred, generally max_inference_state_frames is greater than the propagation length
        :param
        inference_state: Inference memory library
        frame_idx: Current frame index
        max_inference_state_frames: Maximum number of frames to retain
        pre_frames: Number of frames in the preloaded memory library, need to ensure that frames in the preloaded memory library are not cleared. (pre_frames-1) is the maximum frame index of the preloaded memory library
        vis_frame_stride: If -1 means no visualization, old video frames can be cleared
        '''
        # print(f"Current frame {frame_idx}, maximum number of frames to retain {max_inference_state_frames}")

        # Set the oldest allowed frame index to `frame_idx - max_inference_state_frames`, i.e., only retain the most recent max_inference_state_frames frames
        oldest_allowed_idx = frame_idx - max_inference_state_frames

        # Get all frame indices stored in inference_state['output_dict']
        all_cond_frames_idx = inference_state['output_dict']['cond_frame_outputs'].keys()
        all_non_cond_frames_idx = inference_state['output_dict']['non_cond_frame_outputs'].keys()
        old_cond_frames_idx = [idx for idx in all_cond_frames_idx if (pre_frames - 1) < idx <= oldest_allowed_idx]  # Frame indices less than oldest_allowed_idx and greater than the preloaded memory library (pre_frames-1)
        old_non_cond_frames_idx = [idx for idx in all_non_cond_frames_idx if (pre_frames - 1) < idx <= oldest_allowed_idx]  # Frame indices less than oldest_allowed_idx and greater than the preloaded memory library (pre_frames-1)
        # print(f"old_cond_frames_idx:{old_cond_frames_idx}")
        # print(f"old_non_cond_frames_idx:{old_non_cond_frames_idx}")

        for old_idx in old_non_cond_frames_idx:
            # Delete old non-conditional frames in 'output_dict'
            inference_state['output_dict']['non_cond_frame_outputs'].pop(old_idx,None)
            # Delete old non-conditional frames in 'output_dict_per_obj'
            for obj in inference_state['output_dict_per_obj'].keys():
                inference_state['output_dict_per_obj'][obj]['non_cond_frame_outputs'].pop(old_idx,None)

        for old_idx in old_cond_frames_idx:
            # Simultaneously delete old conditional frames in 'output_dict' and 'consolidated_frame_inds'
            inference_state['output_dict']['cond_frame_outputs'].pop(old_idx,None)
            inference_state['consolidated_frame_inds']['cond_frame_outputs'].discard(old_idx)
            # Delete old conditional frames in 'output_dict_per_obj'
            for obj in inference_state['output_dict_per_obj'].keys():
                inference_state['output_dict_per_obj'][obj]['cond_frame_outputs'].pop(old_idx,None)

        if release_images: # Clear old video frames
            old_image_indices = [idx for idx in inference_state["images_idx"] if (pre_frames - 1) < idx <= oldest_allowed_idx]
            # print(f"Information of image frames to be deleted: {old_image_indices}")
            image_idx_to_remove = []
            for old_idx in old_image_indices:
                old_image_idx = inference_state["images_idx"].index(old_idx)  # Convert the actual video frame index to be deleted to the corresponding index in images
                image_idx_to_remove.append(old_image_idx)
            # print(f"Information of image frames to be deleted corresponding to images index: {image_idx_to_remove}")

            # Perform deletion operations on both images and images_idx
            # Use torch.index_select to delete unnecessary frames and retain other frames
            mask = torch.tensor([i for i in range(inference_state["images"].size(0)) if i not in image_idx_to_remove])
            inference_state["images"] = torch.index_select(inference_state["images"], dim=0, index=mask)
            inference_state["images_idx"] = [idx for idx in inference_state["images_idx"] if idx not in old_image_indices]
            # print(f"images_idx after deletion: {inference_state['images_idx']}")
            # print(f"Length of images after deletion: {len(inference_state['images'])}")

            assert len(inference_state["images"]) == len(inference_state["images_idx"])  # Ensure the lengths of images and images_idx are consistent

        # Actively call garbage collection after batch deletion
        gc.collect()

        # print(f"Conditional frame indices in output_dict: {inference_state['output_dict']['cond_frame_outputs'].keys()}")
        # print(f"Non-conditional frame indices in output_dict: {inference_state['output_dict']['non_cond_frame_outputs'].keys()}")
        # print(f"Conditional frame indices in consolidated_frame_inds: {inference_state['consolidated_frame_inds']['cond_frame_outputs']}")

    # Perform single frame inference
    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""

        # Get image features of the current frame
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # Ensure point inputs and mask inputs do not appear on the same frame simultaneously
        assert point_inputs is None or mask_inputs is None

        # Print track_step inputs
        # print(f"is_init_cond_frame: {is_init_cond_frame}")
        # print(f"current_vision_feats: {current_vision_feats[0].shape}")  # torch.Size([65536, 1, 32]
        # print(f"current_vision_pos_embeds: {current_vision_pos_embeds[0].shape}")  # torch.Size([65536, 1, 256])
        # print(f"feat_sizes: {feat_sizes}")  # [(256, 256), (128, 128), (64, 64)]
        # print(f"track_in_reverse: {reverse}")
        # print(f"prev_sam_mask_logits: {prev_sam_mask_logits}")  # None

        # Run tracking step with current frame features and inputs
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            preloading_memory_cond_frame_idx=inference_state["preloading_memory_cond_frame_idx"], # Pass preloaded memory condition frame indices to ensure they participate in memory attention calculation
        )

        # Optionally, transfer output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            # Convert features to bfloat16 to save memory
            maskmem_features = maskmem_features.to(torch.bfloat16)
            # Transfer features to storage device (e.g., CPU)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=False)  # non_blocking=False

        pred_masks_gpu = current_out["pred_masks"]
        # If needed, fill holes in predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        # Transfer predicted masks to storage device
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=False)  # non_blocking=False

        # "maskmem_pos_enc" is the same across all frames, so only one copy needs to be stored
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # Object pointer is a small tensor, so it is always kept in GPU memory for quick access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]

        # Create a compact version of the current frame output to reduce state size
        compact_current_out = {
            "maskmem_features": maskmem_features,  # Located on storage_device
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,  # Located on storage_device
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }

        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts
    ):
        """
        Run memory encoder on `high_res_masks`. This is usually done after applying non-overlapping constraints to object scores.
        Since their scores have changed, their memory also needs to be recalculated through the memory encoder.
        """
        # Get image features of the current frame
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        # Encode new memory
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        # Optionally, transfer output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        # Convert memory features to bfloat16 to save memory
        maskmem_features = maskmem_features.to(torch.bfloat16)
        # Transfer features to storage device
        maskmem_features = maskmem_features.to(storage_device, non_blocking=False)  # non_blocking=False
        # print("maskmem_features:", maskmem_features.device)  # CPU
        # "maskmem_pos_enc" is the same across all frames, so only one copy needs to be stored
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across all frames and objects, so we cache it as
        a constant to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]

        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                # Ensure the output is a list of tensors
                assert isinstance(out_maskmem_pos_enc, list)

                # Take a slice of one object since it is the same across all objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                # Cache maskmem_pos_enc
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                # Use cached maskmem_pos_enc
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]

            # Expand cached maskmem_pos_enc to actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    @torch.inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        """
        Remove an object ID from the tracking state. If strict is True, check if the object ID exists,
        and raise an error if it does not.
        """
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []

        # Check if the object ID to be removed exists, and raise an error based on the strict setting.
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object ID {obj_id} because it does not exist."
                f"All existing object IDs: {inference_state['obj_ids']}."
            )

        # If this is the only remaining object ID, reset the state directly.
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.reset_state(inference_state)
            return inference_state["obj_ids"], updated_frames

        # There are other remaining objects after removing this object ID. In this case,
        # we need to remove the object's storage from the inference state tensors.
        # Step 0: Clear inputs on frames with points or mask inputs for this object
        # (Note this step is necessary because it may downgrade conditional frames to non-conditional frames).
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_prompts_in_frame(
                inference_state, frame_idx, obj_id, need_output=False
            )

        # Step 1: Update object ID mappings (Note this step must be done after Step 0,
        # because Step 0 still needs to use the old object ID mappings in the inference state).
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))

        # Build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        # Step 2: For each object's tensor storage, move their object indices to dictionary keys.
        # (Note "consolidated_frame_inds" does not need to be updated in this step because it was already handled in Step 0).
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        # Step 3: For packed tensor storage, index the remaining IDs and rebuild slices for each object.
        def _slice_state(output_dict, storage_key):
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = out["maskmem_features"][remain_old_obj_inds]
                out["maskmem_pos_enc"] = [
                    x[remain_old_obj_inds] for x in out["maskmem_pos_enc"]
                ]
                # "maskmem_pos_enc" is the same across frames, so we only need to store one copy
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(inference_state, out)
                out["pred_masks"] = out["pred_masks"][remain_old_obj_inds]
                out["obj_ptr"] = out["obj_ptr"][remain_old_obj_inds]
                out["object_score_logits"] = out["object_score_logits"][
                    remain_old_obj_inds
                ]
                # Also need to update slices for each object
                self._add_output_per_object(
                    inference_state, frame_idx, out, storage_key
                )

        _slice_state(inference_state["output_dict"], "cond_frame_outputs")
        _slice_state(inference_state["output_dict"], "non_cond_frame_outputs")

        # Step 4: Further collect outputs on frames in `obj_input_frames_inds`,
        # which may show updated masks for objects occluded by the removed object.
        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=False,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames
    
    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove non-conditional memory around input frames. When users provide correction clicks, 
        non-conditional memory of surrounding frames may still contain outdated object appearance information, 
        which may confuse the model.

        This method clears non-conditional memory of surrounding frames to avoid providing the model with both old and new information about objects.
        """
        r = self.memory_temporal_stride_for_eval
        # Calculate the start and end indices of frames to be cleared
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        output_dict = inference_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]

        for t in range(frame_idx_begin, frame_idx_end + 1):
            # Remove non-conditional outputs of the specified frame from the output dictionary
            non_cond_frame_outputs.pop(t, None)

            # Remove non-conditional outputs of the specified frame from each object's output dictionary
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)

    # def full_propagation(self,
    #                     inference_state,
    #                     prompts,
    #                     video_path,
    #                     offload_video_to_cpu=True,
    #                     offload_state_to_cpu=True,
    #                     async_loading_frames=True,
    #                     max_frame_num_to_track=-1
    #                     reverse=False):
        
    #     cap = cv2.VideoCapture(video_path)
    #     if not cap.isOpened():
    #         print(f"---Unable to open video file: {video_path}")
    #         return

    #     frame_idx = 0  # Initialize the frame index counter for the current video stream
    #     while True:
    #         ret, frame = cap.read()

    #         if not ret:  # End of video, perform inference on the remaining accumulated video frames in the buffer
    #             break

    #         # Convert BGR image to RGB
    #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # shape = (1080, 1920, 3)
    #         self.inference_state = self.init_state(frame_rgb,
    #                                         offload_video_to_cpu=self.offload_video_to_cpu,
    #                                         offload_state_to_cpu=self.offload_state_to_cpu,
    #                                         async_loading_frames=True,)

    #         frame_idx += 1

    #     cap.release()
