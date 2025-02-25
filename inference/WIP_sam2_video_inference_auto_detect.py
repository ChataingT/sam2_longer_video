import os
import psutil
import subprocess  # TODO: Package for viewing resource usage during development, should be removed later
# from pympler import asizeof  # TODO: Package for viewing resource usage during development, should be removed later
import cv2
import sys
import gc
import time
import torch
from sam2.build_sam import build_sam2_video_predictor
from frames2video import frames_to_video
from sam2.utils.misc import tensor_to_frame_rgb
import ultralytics
from IPython import display
display.clear_output()  # clearing the output
ultralytics.checks()  # running checks
from ultralytics import YOLO  # importing YOLO
from IPython.display import display, Image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    '''
    This is a pipeline that receives video streams and automatically adds conditional prompts to SAM2 through a detection model.
    '''
    def __init__(
        self,
        output_dir,
        sam2_checkpoint,
        model_cfg,
        # detect_model_weights,
        # detect_confidence=0.85,  # YOLO model confidence
        # skip_classes={11, 14, 15, 19},  # In this model's YOLO output: 0: black ball 1: blue ball 2: blue ball-t 3: brown ball 4: brown ball-t 5: green ball 6: green ball-t 7: orange ball 8: orange ball-t 9: pink ball 10: pink ball-t 11: pocket 12: red ball 13: red ball-t 14: table 15: triballs 16: white ball 17: yellow ball 18: yellow ball-t 19: Plam_cue
        vis_frame_stride=-1,  # Render segmentation results every few frames, -1 means no rendering
        visualize_prompt=False, # Whether to visualize interactive frames
        frame_buffer_size=30,  # Perform inference after accumulating a certain number of frames
        detect_interval=-1,  # Perform detection every few frames (conditional frame prompt), -1 means no detection (no conditional frame prompt) but there must be at least one conditional frame in all frames (can be a system prompt frame)
        max_frame_num_to_track=60,  # The maximum length of reverse tracking inference from the last frame during propagation (propagate_in_video). This limit can effectively reduce the repetitive computation overhead of long videos
        max_inference_state_frames=60,  # Should not be less than max_frame_num_to_track; limit the number of frames in inference_state, and release old frames when exceeding this number, -1 means no limit
        load_inference_state_path=None,  # If preloading memory library is needed, pass the path of the preloading memory library (.pkl)
        save_inference_state_path=None,  # If saving the inferred memory library is needed, pass the save path (.pkl)
    ):
        self.output_dir = output_dir  # Path to save rendered results
        self.sam2_checkpoint = sam2_checkpoint  # SAM2 model weights
        self.model_cfg = model_cfg  # SAM2 model configuration file
        # self.detect_model_weights = detect_model_weights  # YOLO model weights
        # self.detect_confidence = detect_confidence  # YOLO model confidence
        # self.skip_classes = skip_classes  # Classes to ignore from YOLO detection output to SAM conditional input
        self.vis_frame_stride = vis_frame_stride  # Render segmentation results every few frames
        self.visualize_prompt = visualize_prompt  # Whether to visualize interactive frames
        # Video stream cumulative inference and detection model interval inference
        self.frame_buffer_size = frame_buffer_size  # Perform inference after accumulating a certain number of frames
        self.detect_interval = detect_interval  # Perform detection every few frames
        self.frame_buffer = []  # Used to accumulate frames
        # Maximum tracking length during propagation (propagate_in_video)
        self.max_frame_num_to_track = max_frame_num_to_track  # Limit the number of frames in inference_state, and release old frames when exceeding this number, -1 means no limit
        self.max_inference_state_frames = max_inference_state_frames  # Ensure that the cleared frames will not be used again, max_inference_state_frames generally needs to be greater than or equal to the maximum propagation length max_frame_num_to_track
        # Paths for preloading memory library and saving inferred memory library
        self.load_inference_state_path = load_inference_state_path
        self.save_inference_state_path = save_inference_state_path
        self.pre_frames = 0  # Initialize the video frame length of the preloaded memory library, if there is no preloading, this value is always 0 and does not affect any calculations

        if save_inference_state_path is not None:
            assert max_inference_state_frames == -1, "If you need to save the inferred memory library to create a preloaded memory library, you should not release any old frames, max_inference_state_frames needs to be set to -1"

        # Perform separate SAM2 analysis on special classes of the detection model (such as pockets) and save them in a dictionary
        self.special_classes = 11  # The index of special classes in the detection model, here 11 is the pocket in the billiard scene. That is, multiple objects of the same class will appear, and SAM cannot assign the same ID to multiple objects
        self.special_classes_detection = []  # Used to store detection results of special classes

        print(
            f"---Maximum accumulated frames:{self.frame_buffer_size},"
            f"Detection interval:{self.detect_interval},"
            f"Maximum propagation length:{self.max_frame_num_to_track},"
            f"Inference retained frames:{self.max_inference_state_frames},"
            f"Preloaded memory library:{self.load_inference_state_path},"
        )

        # Build SAM2 model
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        # Load YOLOv8n model
        # self.detect_model = YOLO(detect_model_weights)

        # video_segments contains the segmentation results of each frame
        self.video_segments = {}
        # Global inference_state placeholder, inference_state is officially initialized in the first frame of process_frame
        self.inference_state = None

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Select computing device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # print(f"---using device: {device}")
        if device.type == "cuda":
            # Using bfloat16 for the entire script can double the inference speed
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # Enable tfloat32 for Ampere architecture GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    # Get GPU memory usage
    def print_gpu_memory(self):  # TODO: View resource usage during development, should be removed eventually
        try:
            # Use nvidia-smi to get memory information
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"])
            result = result.decode("utf-8").strip().split("\n")
            # Memory usage (used, free) for each GPU
            gpu_memory = [tuple(map(int, line.split(", "))) for line in result]
            if gpu_memory:
                for idx, (used, free) in enumerate(gpu_memory):
                    print(f"GPU{idx} memory - Used: {used} MB, Free: {free} MB")

        except Exception as e:
            print(f"Error in getting GPU memory: {e}")
            return None

    def calculate_tensor_size(self,value):  # TODO: View resource usage during development, should be removed eventually
        """
        Calculate the memory usage (MB) of a tensor.
        """
        if isinstance(value, torch.Tensor):
            size = value.element_size() * value.nelement()
            return size
        return 0
    def calculate_object_size(self,value):  # TODO: View resource usage during development, should be removed eventually
        """
        Recursively calculate the memory usage of non-tensor objects (dict, list, etc.), considering memory usage if they contain tensors.
        """
        total_size = 0
        if isinstance(value, dict):
            for k, v in value.items():
                total_size += self.calculate_tensor_size(v)
                total_size += self.calculate_object_size(v)
        elif isinstance(value, list):
            for item in value:
                total_size += self.calculate_tensor_size(item)
                total_size += self.calculate_object_size(item)
        return total_size
    def print_size_of(self, inference_state):  # TODO: View resource usage during development, should be removed eventually
        """
        Print the memory usage of all tensors and other objects in inference_state.
        """
        total_size = 0
        for key, value in inference_state.items():
            tensor_size = self.calculate_tensor_size(value)
            if tensor_size > 0:
                total_size += tensor_size
                print(f"{key}: {value.size() if isinstance(value, torch.Tensor) else type(value)}, "
                      f"{value.dtype if isinstance(value, torch.Tensor) else ''}, "
                      f"{tensor_size / (1024 ** 2):.2f} MB")
            else:
                object_size = self.calculate_object_size(value)
                if object_size > 0:
                    total_size += object_size
                    print(f"{key}: {type(value)}, size: {object_size / (1024 ** 2):.2f} MB")

        print(f"Total size: {total_size / (1024 ** 2):.2f} MB")

    # Get CPU memory usage
    def print_cpu_memory(self):  # TODO: View resource usage during development, should be removed eventually
        memory_info = psutil.virtual_memory()
        # Used and total memory, in GB
        cpu_used = memory_info.used / (1024 ** 3)
        cpu_total = memory_info.total / (1024 ** 3)
        print(f"CPU memory - Used/Total: {cpu_used:.2f}/{cpu_total:.2f}GB")

    def calculate_video_segments_memory(self, video_segments):   # TODO: View resource usage during development, should be removed eventually
        total_memory = 0

        for frame_idx, objects in video_segments.items():
            for obj_id, mask_array in objects.items():
                if isinstance(mask_array, np.ndarray):
                    total_memory += mask_array.nbytes  # NumPy array memory usage
                else:
                    print(f"Warning: Object {obj_id} in frame {frame_idx} is not a NumPy array.")

        return total_memory


    def clear(self):
        """
        Clear all content related to this video inference while retaining the instantiation.
        """
        self.frame_buffer = []  # Clear video accumulation buffer
        self.pre_frames = 0  # Reset preloaded video frame length
        self.special_classes_detection = []  # Clear special class detection results

        self.video_segments = {}
        self.inference_state = None  # Restore global inference_state placeholder


    def detect_predict(self, images, past_num_frames):
        """
        Use the YOLO model to perform interval detection on multiple images (select frames that meet self.detect_interval from images (self.frame_buffer))
        and return the detection results of the corresponding frame indices (calculated from past_num_frames, images, and detect_interval).
        """

        selected_frames = []  # Select frames that need to be detected from images
        absolute_indices = []  # Absolute indices of frames that need to be detected in the entire video
        detection_results = {}  # Initialize result dictionary

        if self.detect_interval == -1:  # If detect_interval is set to -1, do not perform detection inference
            return detection_results  # Return empty result dictionary

        logger.warning(f"---Performing detection inference on {len(images)} frames")
        # Traverse the frames in frame_buffer and select frames that meet the detect_interval
        for i, image in enumerate(images):
            # Calculate the absolute index of each frame in the video
            frame_idx = past_num_frames + i  # past_num_frames + 1 = the Nth frame of this prediction at the beginning, past_num_frames + 1 - 1 = the frame index of this prediction at the beginning (starting from 0).

            # If the frame meets the detect_interval detection frequency, select the frame and convert it back to the original BGR format read by cv2
            if frame_idx % self.detect_interval == 0:
                selected_frames.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Frames that need to be detected
                absolute_indices.append(frame_idx)  # Absolute indices of frames that need to be detected

        if len(selected_frames) == 0:  # If there are no frames to be detected in this accumulated frame
            return detection_results  # Return empty result dictionary

        # Infer a list of images in selected_frames
        results = self.detect_model(selected_frames, stream=True, conf=self.detect_confidence, iou=0.1, verbose=False)  # conf=0.85

        # Process the results object generator
        for i, result in enumerate(results):
            frame_detections = []  # Used to store the detection results of the current frame
            boxes = result.boxes  # Get the Boxes object of bounding box output
            if boxes is not None:
                for box in boxes:
                    coords = box.xyxy[0].cpu().numpy()  # Get the coordinates of the top-left and bottom-right corners (x1, y1, x2, y2)
                    cls = box.cls.cpu().numpy()  # Get the object class
                    conf = box.conf.cpu().numpy()  # Get the confidence

                    frame_detections.append({
                        "coordinates": coords,
                        "class": cls,
                        "confidence": conf
                    })
                    # print(f"Frame index (excluding preloading) {absolute_indices[i]-self.pre_frames}: Coordinates: {coords}, Class: {cls}, Confidence: {conf}")

                # Ensure that the collected special class detection results are the maximum
                if not self.special_classes_detection:
                    self.special_classes_count = 0  # If there are no special class detection results, initialize the special class counter
                special_classes_count = sum(
                    [1 for detection in frame_detections if detection['class'] == self.special_classes])
                if special_classes_count > self.special_classes_count: # If the current number of special classes is greater than the existing number of special classes
                    # Clear existing special class detection results
                    self.special_classes_detection = []
                    # Update special class detection results
                    for detection in frame_detections:
                        if detection['class'] == self.special_classes:
                            self.special_classes_detection.append(detection["coordinates"])
                    # Update the recorded number of special classes
                    self.special_classes_count = special_classes_count

            # Add the detection results of the current frame to the total result dictionary according to absolute_indices
            detection_results[f"frame_{absolute_indices[i]}"] = frame_detections

        return detection_results

    def Detect_2_SAM2_Prompt(self, detection_results_json):
        """
        Pass YOLO detection results as conditions to SAM.

        /!\ detection_results_json is a dictionary, that shoulbd be empty because we are not using the detection model in this example.
        """
        # If the passed detection_results_json is empty, directly return the current inference_state
        if not detection_results_json:
            # print(f"---detection_results_json dictionary is empty, there are no conditional frames in the accumulated video stream length this round")
            return self.inference_state

        # Traverse the detection results of each frame
        for frame_idx, detections in detection_results_json.items():
            ann_frame_idx = int(frame_idx.replace('frame_', ''))  # Get frame index

            if self.visualize_prompt:  # If visualizing interactive frames
                # Create an image window and display the current frame
                plt.figure(figsize=(9, 6))
                plt.title(f"frame {ann_frame_idx}")
                ann_img_tensor = self.inference_state["images"][ann_frame_idx:ann_frame_idx + 1]  # Get the current frame from inference_state, dimension is (1, 3, 1024, 1024)
                ann_frame_rgb = tensor_to_frame_rgb(ann_img_tensor)  # Convert tensor to RGB image
                plt.imshow(ann_frame_rgb)  # Display the current frame image

            for detection in detections:
                # Get the object's class and coordinates from the detection results, ensure 'class' is a scalar or extract the first element from the array
                obj_class = int(detection['class'][0]) if isinstance(detection['class'], np.ndarray) else int(detection['class'])
                # Check if the object class is in the skipped classes
                # if obj_class in self.skip_classes:
                    # continue  # Skip the current detection result and continue processing the next one

                coordinates = detection['coordinates']
                coordinates = np.array(coordinates, dtype=np.float32)
                
                # Pass the data to the predictor
                box = None
                points = None
                labels=None
                if coordinates.shape == (4,):
                    box = coordinates
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=obj_class,  # Use object class as ID
                        box=box,
                    )
                else :
                    points = np.array(coordinates[:, :2], dtype=np.float32)
                    labels = np.array(coordinates[:, -1], dtype=np.float32)
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=obj_class,  # Use object class as ID
                        points=points,
                        labels=labels
                    )

                if self.visualize_prompt:  # If visualizing interactive frames
                    # Draw detection box
                    if box is not None:
                        self.show_box(box, plt.gca(), obj_class)
                    if points is not None:
                        self.show_points(points, labels, plt.gca(), obj_class)
                    # Draw mask
                    self.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

            if self.visualize_prompt:  # If visualizing interactive frames
                # Save image to specified directory
                save_path = os.path.join("temp_output/prompt_results", f"frame_{ann_frame_idx}.png")
                plt.savefig(save_path)
                plt.close()  # Close the figure to release memory

        return self.inference_state

    def render_frame(self, out_frame_idx, frame_rgb, video_segments):
        """
        Render segmentation results for a single frame.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(f"frame {out_frame_idx-self.pre_frames}")  # The frame index here does not include preloaded frames
        ax.imshow(frame_rgb)

        ax.axis('off')  # Remove the axis

        # Render on the segmentation results
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            self.show_mask(out_mask, ax, obj_id=out_obj_id)

        # Save the rendered result to output_dir
        save_path = os.path.join(self.output_dir, f"frame_{out_frame_idx:05d}.png")

        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)  # Use the figure object to save, removing extra whitespace
        plt.close(fig)  # Close the figure to release memory

        # If you want to display the image, you can call plt.show() here, but you need to redraw:
        # plt.show()


    def Detect_and_SAM2_inference(self, frame_idx, manual_prompt={}):
        """
        Perform conditional detection (detect_predict will internally determine if it needs to be executed),
        update the new buffer to inference_state,
        perform SAM2 inference with self.Detect_2_SAM2_Prompt,
        and propagate in all existing frames with propagate_in_video.
        """
        # If self.inference_state exists, get it from self.inference_state, otherwise it is the first inference, and the historical frame is 0
        past_num_frames = self.inference_state["num_frames"] if self.inference_state else 0
        # Pass self.frame_buffer to the detection model for prediction, and perform interval processing internally in self.detect_predict
        detection_results_json = self.detect_predict(self.frame_buffer, past_num_frames)
        # print(detection_results_json)

        focused_manual_prompt = self.extract_prompt_by_frame_idx(manual_prompt, frame_idx_min=past_num_frames, frame_idx_max=frame_idx)
        detection_results_json.update(focused_manual_prompt)
            

        # Update inference_state
        if self.inference_state is None:  # If it is the first time, initialize it using init_state()
            self.inference_state = self.predictor.init_state(video_path=self.frame_buffer)
        else:  # If it is not the first time, use update_state()
            self.inference_state = self.predictor.update_state(
                video_path=self.frame_buffer,
                inference_state=self.inference_state
            )

    
        # print("Current specific keys in inference_state:")
        # print(f"obj_id_to_idx: {self.inference_state['obj_id_to_idx']}")  # OrderedDict([(16, 0), (10, 1)])
        # print(f"obj_idx_to_id: {self.inference_state['obj_idx_to_id']}")  # OrderedDict([(0, 16), (1, 10)])
        # print(f"obj_ids: {self.inference_state['obj_ids']}")  # [16, 10]
        # print(f"point_inputs_per_obj: {self.inference_state['point_inputs_per_obj']}")  # {Object index 0: {Frame index N: Prompt type and coordinates, Frame index M: Prompt type and coordinates}, Object index 1: {Frame index K: Prompt type and coordinates}}
        # # {0: {90: {'point_coords': tensor([[[544.9968, 531.2661],[564.3727, 565.0934]]], device='cuda:0'),'point_labels': tensor([[2, 3]], device='cuda:0', dtype=torch.int32)},
        # #      105: {'point_coords': tensor([[[535.3818, 440.6492],[554.4316, 474.4221]]], device='cuda:0'),'point_labels': tensor([[2, 3]], device='cuda:0', dtype=torch.int32)}},
        # #  1: {105: {'point_coords': tensor([[[533.5795, 200.6569],[552.9704, 247.3801]]], device='cuda:0'),'point_labels': tensor([[2, 3]], device='cuda:0', dtype=torch.int32)}}
        # print(f"mask_inputs_per_obj: {self.inference_state['mask_inputs_per_obj']}")  # {0: {}, 1: {}}
        # print(f"output_dict_per_obj: {self.inference_state['output_dict_per_obj'].keys()}")  # dict_keys([0, 1])
        # print(f"temp_output_dict_per_obj: {self.inference_state['temp_output_dict_per_obj']}")
        # # {0: {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}},
        # #  1: {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}}

        # SAM2 inference
        try:
            self.inference_state = self.Detect_2_SAM2_Prompt(detection_results_json)
        except RuntimeError as e:   # Normally, this branch will not be reached after supporting the function of adding new object IDs online, but just in case.
            if "reset_state" in str(e):
                print("---Unable to add new object ID online, resetting state")
                self.predictor.reset_state(self.inference_state)
                self.inference_state = self.Detect_2_SAM2_Prompt(detection_results_json)

        # Perform tracking inference (propagation operation)
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=frame_idx,  # In the propagation function, start tracking inference in reverse order from the last frame
            max_frame_num_to_track=self.max_frame_num_to_track,  # Maximum length of tracking inference
            reverse=True,  # In the propagation function, set to track inference in reverse order from the starting frame
        ):
            if out_frame_idx >= self.pre_frames:  # Do not store preloaded frames in the segmentation result dictionary
                # print(f"Rendering frame {out_frame_idx} into video_segments")
                self.video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        # print("Before cleanup:")
        # processor.print_cpu_memory()

        if self.max_inference_state_frames != -1 :  # Whether to try to release old frames
            self.predictor.release_old_frames(
                self.inference_state,
                frame_idx,
                self.max_inference_state_frames,
                self.pre_frames,
                release_images=True if self.vis_frame_stride == -1 else False,  # Only release old image tensors when not visualizing
            )  # Release old frames

        # print("After cleanup:")
        # self.print_cpu_memory()
        # print(f"Memory usage of video_segments: {self.calculate_video_segments_memory(self.video_segments) / (1024 ** 2):.2f} MB")


        # processor.print_gpu_memory()
        # processor.print_size_of(self.inference_state)

    def process_frame(self, frame_idx, frame, manual_prompt={}):
        """
        Perform detection and segmentation after accumulating a certain number of frames.
        """
        # Accumulate frames to buffer
        self.frame_buffer.append(frame)

        # Perform YOLO detection and SAM2 inference when a certain number of frames are accumulated
        if len(self.frame_buffer) >= self.frame_buffer_size:
            # If conditions are met, perform inference on the accumulated buffer of video stream
            self.Detect_and_SAM2_inference(frame_idx, manual_prompt=manual_prompt)
            # Clear the buffer
            self.frame_buffer.clear()

        return self.inference_state


    # (Below) Some visualization tools, basically the same as the official examples

    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        # If random_color is True, randomly generate colors (including RGB colors and alpha channel transparency)
        if random_color:
            # Generate random RGB color values and add an alpha channel with a value of 0.6 for partial transparency
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            # Use the 'tab10' colormap in matplotlib, which provides 10 different colors
            cmap = plt.get_cmap("tab20")
            # If obj_id is None, use the default index 0, otherwise use obj_id as the index for the colormap
            cmap_idx = 0 if obj_id is None else obj_id
            # Get the RGB color from the colormap and add an alpha channel with a value of 0.6
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        # Get the height and width of the mask
        h, w = mask.shape[-2:]
        # Reshape the mask to a 2D image, multiply by the color vector to overlay color information on the mask
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        # Display the processed mask image on the provided matplotlib axis
        ax.imshow(mask_image)

        # # Display the mask image separately
        # # Create a new figure window
        # plt.figure(figsize=(6, 4))
        # plt.title(f"Object ID: {obj_id}")
        # # Display the mask image
        # plt.imshow(mask_image)
        # plt.axis('off')  # Turn off the axis
        # plt.show()  # Show the image

    def show_box(self, box, ax, obj_class):
        # show_box can only create a visualization box for one object, it cannot create visualization boxes for multiple boxes at the same time
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=2))

        # Display the class ID above the box
        text_x = x0 + w / 2  # Center position of the box
        text_y = y0 - 10  # Position above the box
        ax.text(text_x, text_y, str(obj_class), fontsize=10, color='white', ha='center', va='bottom')

    def show_points(self, coords, labels, ax, marker_size=200):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)

    # (Below) Methods for saving and loading inference_state (memory library)

    def save_inference_state(self, save_path):
        # Get the directory path and ensure the directory exists
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save inference_state
        with open(save_path, 'wb') as f:
            pickle.dump(self.inference_state, f)
        print(f"---inference_state saved to {save_path}, total {self.inference_state['num_frames']} frames")

    def load_inference_state(self, load_path):
        with open(load_path, 'rb') as f:
            pre_inference_state = pickle.load(f)
        print(f"---inference_state loaded from {load_path}, total {pre_inference_state['num_frames']} frames")
        return pre_inference_state

    # (Below) Methods for loading frames from a folder

    def load_frames_from_folder(self, folder_path):
        frames = []
        # Get all file names in the folder, sorted by file name to ensure order
        frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

        for frame_file in frame_files:
            frame_path = os.path.join(folder_path, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"---Unable to read frame file: {frame_path}")
                continue

            # If needed, convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        return frames

    def extract_prompt_by_frame_idx(self, prompt, frame_idx_min, frame_idx_max):
        ret = {}
        for i in range(frame_idx_min, frame_idx_max):
            frame_prompt = prompt.get(f'frame_{i}', {})
            if frame_prompt:
                ret.update({f'frame_{i}': frame_prompt})
        return ret

    def run(
            self,
            video_path=None, # Input video
            frame_dir=None, # Or frame folder
            prompt=None
            # output_video_segments_pkl_path="/root/autodl-tmp/temp_output/video_segments.pkl", # Path to save video segmentation results as pkl
            # output_special_classes_detection_pkl_path="/root/autodl-tmp/temp_output/special_classes_detection.pkl", # Path to save special class detection results as pkl
    ):
        """
        Run the video processing pipeline.
        """
        # run_start_time = time.time()

        # If the path to load the memory library is not empty, preload the specified memory library
        if self.load_inference_state_path is not None:
            self.inference_state = self.load_inference_state(self.load_inference_state_path)
            # Get the conditional frame and non-conditional frame indices in the preloaded memory library, and save the records in the inference_state dictionary under "preloading_memory_cond_frame_idx" and "preloading_memory_non_cond_frames_idx"
            preloading_memory_cond_frame_idx = list(self.inference_state["output_dict"]["cond_frame_outputs"].keys())
            preloading_memory_non_cond_frames_idx = list(
                self.inference_state["output_dict"]["non_cond_frame_outputs"].keys())
            self.inference_state["preloading_memory_cond_frame_idx"] = preloading_memory_cond_frame_idx
            self.inference_state["preloading_memory_non_cond_frames_idx"] = preloading_memory_non_cond_frames_idx
            # Get the number of existing frames in the preloaded memory library, update self.pre_frames, and save it in inference_state["preloading_memory_frames"]
            self.pre_frames = self.inference_state["num_frames"]
            self.predictor.init_preloading_state(self.inference_state)  # Move some tensors in the preloaded memory library to the CPU

            # self.print_cpu_memory()
            # processor.print_gpu_memory()
            # processor.print_size_of(self.inference_state)

        if prompt is None:
            logger.error("Automatic detection is not supported. Manual prompt are needed")
            return None

        # If the target video path exists
        if video_path is not None:
            # Load video stream from video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"---Unable to open video file: {video_path}")
                return

            frame_idx = 0  # Initialize the frame index counter for the current video stream
            while True:
                ret, frame = cap.read()

                # get prompt associated to the current frame idx
                # current_frame_prompt = self.extract_prompt_by_frame_idx(prompt, frame_idx)

                if not ret:  # End of video, perform inference on the remaining accumulated video frames in the buffer
                    if self.frame_buffer is not None and len(self.frame_buffer) > 0:
                        # print(f"---End of video, inferring remaining frames: {len(self.frame_buffer)}")
                        self.Detect_and_SAM2_inference(frame_idx=self.pre_frames + frame_idx - 1, manual_prompt=prompt)  # Index of the last frame, note that it needs to be -1 when the loop ends
                    break

                # Convert BGR image to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # shape = (1080, 1920, 3)
                self.inference_state = self.process_frame(self.pre_frames + frame_idx, frame_rgb, manual_prompt=prompt)

                frame_idx += 1

            cap.release()

        # If the target video frame folder exists
        elif frame_dir is not None:
            frames = self.load_frames_from_folder(frame_dir)

            if not frames:
                print(f"---No valid frame files found: {frame_dir}")
                return

            total_frames = len(frames)
            frame_idx = 0  # Initialize the frame index counter for the current video stream

            while frame_idx < total_frames:
                frame_rgb = frames[frame_idx]  # Frame read from the folder
                self.inference_state = self.process_frame(self.pre_frames + frame_idx, frame_rgb)

                frame_idx += 1

            # At the end of the video, process the remaining frames in the buffer
            if self.frame_buffer is not None and len(self.frame_buffer) > 0:
                # print(f"---Processing remaining frames: {len(self.frame_buffer)}")
                self.Detect_and_SAM2_inference(frame_idx=self.pre_frames + frame_idx - 1)

        # Neither a complete video nor a frame folder
        else:
            print("---No valid video or frame folder path provided")

        # run_end_time = time.time()
        # print("---Total inference time:", run_end_time - run_start_time, "seconds")

        # Support post-processing operations, save the results as pkl for post-processing
        # Save the dictionary of segmentation results in self.video_segments, the saved self.video_segments should not contain preloaded frames
        # self.video_segments = {idx - self.pre_frames: segment for idx, segment in self.video_segments.items() if idx >= self.pre_frames}
        # with open(output_video_segments_pkl_path, 'wb') as file:
        #     pickle.dump(self.video_segments, file)
        # print(f"---self.video_segments segmentation results saved to {output_video_segments_pkl_path}")
        # # Save the dictionary of special class detection results needed for post-processing in self.special_classes_detection
        # if self.special_classes_detection is None:
        #     print(f"---{self.special_classes_detection} does not meet the collection conditions, collection failed")
        # else:
        #     with open(output_special_classes_detection_pkl_path, 'wb') as file:
        #         pickle.dump(self.special_classes_detection, file)
        #     print(f"---self.special_classes_detection special class detection results saved to {output_special_classes_detection_pkl_path}")

        # Whether to save the memory library information of this inference
        # if self.save_inference_state_path is not None:  # If the save path is not empty, save inference_state
        #     self.save_inference_state(self.save_inference_state_path)

        if self.vis_frame_stride == -1:
            print("---No frames will be rendered, inference completed")
        else:
            # First, clear all existing files in the self.output_dir folder
            for file in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # print(self.video_segments.keys())
            # Render all frames of this video inference uniformly
            images_tensor = self.inference_state["images"]
            for i in tqdm(range(self.pre_frames, images_tensor.shape[0]),desc="Mask visualization rendering progress"):  # Start rendering from the frame after the preloaded memory library
                if i % self.vis_frame_stride == 0:  # i is the index of the frame in this video inference
                    # print("Rendering frame index",i)
                    tensor_img = images_tensor[i:i + 1]  # Get self.frames from inferenced_state["images"] for rendering. Dimension is (1, 3, 1024, 1024)
                    frame_rgb = tensor_to_frame_rgb(tensor_img)  # Current RGB frame ndarray
                    self.render_frame(i-self.pre_frames, frame_rgb, self.video_segments)
            print(f"---Rendered every {self.vis_frame_stride} frames, rendering completed")
            frames_to_video(
                frames_folder=self.output_dir,
                output_video_path='/root/autodl-tmp/temp_output/output_video.mp4',
                fps=2
            )  # Create a video from all frames in the frame folder




if __name__ == '__main__':
    # video_path = 'videos/video中.mp4'
    video_path = r'D:\idtracking\sam2_longer_video\test\assets\bedroom.mp4' # /Det-SAM2评估集/videos/video5.mp4  # /长视频/5min.mp4
    # rtsp_url = 'rtsp://175.178.18.243:19699/'
    # frame_dir = '/root/autodl-tmp/data/预加载内存库10帧'  # For creating preloaded memory library
    output_dir = r'.\temp_output\det_sam2_RT_output'
    # sam2_checkpoint = '../checkpoints/sam2.1_hiera_large.pt' # '../checkpoints/sam2.1_hiera_base_plus.pt'
    sam2_checkpoint = r'D:\idtracking\sam2_longer_video\checkpoints\sam2.1_hiera_tiny.pt' # '../checkpoints/sam2.1_hiera_base_plus.pt'
    # model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml' # 'configs/sam2.1/sam2.1_hiera_b+.yaml'
    model_cfg = r'configs\sam2.1\sam2.1_hiera_t.yaml' # 'configs/sam2.1/sam2.1_hiera_b+.yaml'
    # detect_model_weights = 'det_weights/train_referee12_960.pt'
    # load_inference_state_path = '/root/autodl-tmp/output_inference_state/inference_state_10frames.pkl'
    # save_inference_state_path = '/root/autodl-tmp/output_inference_state/inference_state_10frames.pkl'

    manual_prompt = {
        'frame_0': [
            {
                        "coordinates": [220, 150, 290, 220],
                        "class": '0',
                        "confidence": 1
                    },
                    
            {
                        "coordinates": [[210, 350, 1], [250, 220, 1]],
                        "class": '1',
                        "confidence": 1
                    },
                    
        ]
    }
    # Initialize the VideoProcessor class
    processor = VideoProcessor(
        output_dir=output_dir,
        sam2_checkpoint=sam2_checkpoint,
        model_cfg=model_cfg,
        vis_frame_stride=30
        visualize_prompt=True
        # detect_model_weights=detect_model_weights,
        # load_inference_state_path=load_inference_state_path,  # If not provided or None, do not preload memory library
        # save_inference_state_path=save_inference_state_path,  # If not provided or None, do not save memory library
    )

    # processor.print_cpu_memory()
    # processor.print_gpu_memory()

    processor.run(
        video_path=video_path,  # Provide video path (choose either this or frame folder)
        # frame_dir=frame_dir,  # Provide frame folder (choose either this or video path)
        prompt=manual_prompt
    )
