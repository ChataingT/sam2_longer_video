# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import logging
import decord

logger = logging.getLogger(__name__)

def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # Use Flash Attention only on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled because it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # Retain math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only available in PyTorch 2.2+,
        # and Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__}, which does not support Flash Attention v2."
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (potentially faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on


def get_connected_components(mask):
    """
    Get connected components of a binary mask (8-connectivity), shape (N, 1, H, W).

    Input:
    - mask: Binary mask tensor of shape (N, 1, H, W), where 1 indicates foreground and 0 indicates background.

    Output:
    - labels: Tensor of shape (N, 1, H, W) containing connected component labels for foreground pixels, background pixels are 0.
    - counts: Tensor of shape (N, 1, H, W) containing the area of connected components for foreground pixels, background pixels are 0.
    """
    from sam2 import _C

    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())


def mask_to_box(masks: torch.Tensor):
    """
    Compute bounding boxes from input masks.

    Input:
    - masks: Masks of shape [B, 1, H, W], dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], containing (x, y) coordinates of bounding boxes, i.e., top-left and bottom-right coordinates, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)

    return bbox_coords


def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is the expected data type for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image data type: {img_np.dtype} in {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # Original video dimensions
    return img, video_height, video_width


class AsyncVideoFrameLoader:
    """
    Class for asynchronously loading a set of video frames without blocking session startup.

    It is expected to asynchronously load and be compatible with different img_paths formats, including:
    1. List of paths to multi-frame image files; 2. List of single-frame image paths; 3. List of multi-frame np arrays; 4. Single-frame np array
    """

    def __init__(
        self,
        img_paths,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
        video_height=None,
        video_width=None
    ):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # Items in `self.images` will be loaded asynchronously
        self.images = [None] * len(img_paths)
        # Capture and raise any exceptions in the async loading thread
        self.exception = None
        # Video height and width will be populated when the first image is loaded
        self.video_height = video_height
        self.video_width = video_width
        self.compute_device = compute_device

        self._num_frames = len(img_paths)
        if self._num_frames > 0:
            # Load the first frame to populate video height and width, and cache it (as this is the most likely place the user will click)
            first_image = self.__getitem__(0)
            channels, height, width = first_image.shape
            self._shape = (self._num_frames, channels, height, width)
        else:
            self._shape = (0, 0, 0, 0)  # Default shape is 0

        # Asynchronously load the remaining frames without blocking session startup
        def _load_frames():
            try:
                # for n in tqdm(range(len(self.images)), desc="Loading frames"):
                for n in range(len(self.images)):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        # self.thread = Thread(target=_load_frames, daemon=True)  # TODO: Commented out to try to save more resources
        # self.thread.start()  # TODO: Commented out to try to save more resources

    @property
    def shape(self):
        """Return a tuple of (num_frames, channels, height, width)."""
        return self._shape

    def to_tensor(self):
        """Convert all image frames to a Tensor and return a Tensor of shape (num_frames, channels, height, width)."""
        frame_list = [self.__getitem__(i).clone().detach() for i in range(self._num_frames)]
        return torch.stack(frame_list)

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        img_path = self.img_paths[index]
        if isinstance(img_path, str):
            # img_path is an image path
            img, video_height, video_width = _load_img_as_tensor(img_path, self.image_size)
        elif isinstance(img_path, np.ndarray):
            # img_path is a frame in np.ndarray format
            img_np = cv2.resize(img_path, (self.image_size, self.image_size)) / 255.0
            img = torch.from_numpy(img_np).permute(2, 0, 1).float()
            video_height, video_width = img_path.shape[:2]
        elif isinstance(img_path, decord.ndarray.NDArray):
            numpy_array = img_path.asnumpy()
            # Convert the NumPy array to a PyTorch tensor
            img = torch.from_numpy(numpy_array)
            img = img.permute(2, 0, 1)
            img = img.float() / 255.0
            video_height, video_width = img_path.shape[:2]
        elif isinstance(img_path, torch.Tensor):
            img = img_path.permute(2, 0, 1)
            img = img.float() / 255.0
            video_height, video_width = img_path.shape[:2]
        else:
            raise TypeError(f"Unsupported img_paths type: {type(img_path)}")
        
        if not self.video_width:
            self.video_width = video_width
        if not self.video_height:
            self.video_height = video_height

        # img, video_height, video_width = _load_img_as_tensor(
        #     self.img_paths[index], self.image_size
        # )
        # self.video_height = video_height
        # self.video_width = video_width

        # Normalize by mean and standard deviation
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        # self.images[index] = img  # TODO: Commented out to try to save more resources
        return img

    def __len__(self):
        return len(self.images)

# Restore img_tensor to frame_rgb
def tensor_to_frame_rgb(
    tensor_img,
    original_size=(1920, 1080),
    img_mean=(0.485, 0.456, 0.406),  # Normalization parameters need to be consistent with load_video_frames!
    img_std=(0.229, 0.224, 0.225),  # Normalization parameters need to be consistent with load_video_frames!
):
    '''
    Convert a normalized tensor image back to an RGB format NumPy array.

    Note that we expect tensor_to_frame_rgb to achieve the inverse operation of load_video_frames as much as possible, but there is inevitably precision loss during the resize and normalization and denormalization processes.
    We can only hope it does not significantly affect the visual effect.
    '''
    # Convert mean and standard deviation to tensors
    device = tensor_img.device
    img_mean = torch.tensor(img_mean, dtype=torch.float32, device=device)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32, device=device)[:, None, None]

    # Remove normalization by mean and standard deviation
    tensor_img = tensor_img * img_std + img_mean

    # Convert image dimensions from (1, 3, 1024, 1024) to (3, 1024, 1024), then to (1024, 1024, 3)
    tensor_img = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Resize image from 1024x1024 back to 1080x1920
    frame_rgb = cv2.resize(tensor_img, original_size)

    # Convert image from 0-1 range back to 0-255 and convert to uint8 format
    frame_rgb = np.clip(frame_rgb * 255, 0, 255).astype(np.uint8)

    return frame_rgb


# Load all frames from a video at once
def load_video_frames(
    video_path,  # Can pass a folder containing image frames or a list of image frame paths
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,  # Whether to load frames asynchronously
    compute_device=torch.device("cuda"),
):
    """
    Load video frames from a directory of JPEG files (format "<frame_index>.jpg").

    Frames are resized to image_size x image_size and loaded to GPU (if `offload_video_to_cpu` is `False`)
    or loaded to CPU (if `offload_video_to_cpu` is `True`).

    Frames can be loaded asynchronously by setting `async_loading_frames` to `True`.
    """

    if isinstance(video_path, str) and os.path.isdir(video_path):
        # print("If the input is a folder directory of image frames")
        # If the input is a folder directory of image frames
        jpg_folder = video_path
        frame_names = [
            p
            for p in os.listdir(jpg_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"No images found in {jpg_folder}")
        img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]

    elif isinstance(video_path, list) and all(os.path.isfile(p) for p in video_path):
        # print("If the input is a list of image file paths")
        # If the input is a list of image file paths
        img_paths = video_path
        num_frames = len(img_paths)

    elif isinstance(video_path, np.ndarray):
        # # Frames already loaded from video stream, assuming each frame is an RGB format np.ndarray
        frame_rgb = video_path
        num_frames = 1  # Process one frame

    elif isinstance(video_path, list) and all(isinstance(p, np.ndarray) for p in video_path):
        # print(f"Multiple frames accumulated in the video stream stored as a list, assuming each frame is an RGB format np.ndarray")
        # Multiple frames accumulated in the video stream stored as a list, assuming each frame is an RGB format np.ndarray
        frame_rgb_list = video_path
        num_frames = len(frame_rgb_list)

    elif isinstance(video_path, str) and os.path.isfile(video_path):
        # ND no support for signel file image
        # print("If the input is a single image file path")
        # If the input is a single image file path
        # img_paths = [video_path]
        # num_frames = 1
        import decord
        logger.info(f"Loading video frames from video file {video_path}")
        # Get the original video height and width
        decord.bridge.set_bridge("torch")
        video_height, video_width, _ = decord.VideoReader(video_path).next().shape
        # Iterate over all frames in the video
        images = []

        img_paths = decord.VideoReader(video_path, width=image_size, height=image_size)

    else:
        print("Unsupported frame format passed in")
        raise NotImplementedError(
            "Currently only JPEG frames are supported. For video files, you can use ffmpeg (https://ffmpeg.org/) "
            "to extract frames to a JPEG folder, for example:\n"
            "```\n"
            "ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'\n"
            "```\n"
            "where `-q:v` generates high-quality JPEG frames, and `-start_number 0` requires "
            "ffmpeg to start JPEG files from 00000.jpg."
            "Subsequently, video_path supports passing in folder paths, lists of image paths, or single image paths"
        )

    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    # At this point, img_paths may have formats: 1. List of paths to multi-frame image files; 2. List of single-frame image paths; 3. List of multi-frame np arrays; 4. Single-frame np array
    if async_loading_frames:
        if "frame_rgb" in locals():
            async_frames = frame_rgb
        elif "frame_rgb_list" in locals():
            async_frames = frame_rgb_list
        else:
            async_frames = img_paths

        lazy_images = AsyncVideoFrameLoader(
            async_frames,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device,
            video_height=video_height,
            video_width=video_width
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    # The original official code here converts to FP32 precision, you can try to load it with FP16 precision to save half the memory overhead
    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float16)  # dtype=torch.float32

    if "frame_rgb" in locals():
        # Process frames from the video stream, manually process and normalize
        img_np = cv2.resize(frame_rgb, (image_size, image_size)) / 255.0
        img = torch.from_numpy(img_np).permute(2, 0, 1)
        images[0] = img
        video_height, video_width = frame_rgb.shape[:2]
    elif "frame_rgb_list" in locals():
        # Process multiple frames accumulated in the video stream, manually process and normalize
        for n, frame_rgb in enumerate(frame_rgb_list):
            # print(frame_rgb.shape) # (1080, 1920, 3)
            img_np = cv2.resize(frame_rgb, (image_size, image_size)) / 255.0
            img = torch.from_numpy(img_np).permute(2, 0, 1)
            images[n] = img
        video_height, video_width = frame_rgb_list[0].shape[:2]
    else:
        if len(img_paths) == 0:
            print("Incorrect image frames passed in sam2.utils.misc.load_video_frames()")
            return None, None, None
        # Process list of image paths or frames in the folder, use SAM2's built-in _load_img_as_tensor
        for n, img_path in enumerate(tqdm(img_paths, desc="Loading frames")):
            images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)

    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)

    # Normalize by mean and standard deviation
    images -= img_mean
    images /= img_std

    # print("images", images.shape)  # torch.Size([1, 3, 1024, 1024])
    # print(video_height, video_width)  # 1080 1920
    return images, video_height, video_width

def fill_holes_in_mask_scores(mask, max_area):
    """
    Post-processing step to fill small holes in the mask scores that are smaller than `max_area`.
    """
    # Holes are areas of connected components in the background with area <= self.max_area
    # (Background areas are regions where mask scores <= 0)
    assert max_area > 0, "max_area must be positive"

    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        # Fill holes with small positive mask scores (0.1), turning them into foreground.
        mask = torch.where(is_hole, 0.1, mask)

    # except Exception as e:
    #     # If CUDA kernel fails, skip the post-processing step of filling small holes
    #     warnings.warn(
    #         f"{e}\n\nSkipping post-processing step due to the above error. You can still use SAM 2, "
    #         "ignoring the above error is fine, although some post-processing functionality may be limited (this does not affect results in most cases; see "
    #         "https://github.com/facebookresearch/segment-anything-2/blob/main/INSTALL.md).",
    #         category=UserWarning,
    #         stacklevel=2,
    #     )
    except Exception:

        mask = input_mask

    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to the previous point inputs (append to the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}


if __name__ == '__main__':
    '''
    Verify whether tensor_to_frame_rgb and _load_img_as_tensor are approximately inverse operations
    '''

    # Define the size and normalization parameters for the original image loading
    image_size = 1024
    original_size = (1920, 1080)
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)

    # Generate a random image and convert it to uint8 format
    original_frame_rgb = np.random.randint(0, 256, (original_size[0], original_size[1], 3), dtype=np.uint8)

    # Simulate normalization and resizing using load_video_frames logic
    original_frame_resized = cv2.resize(original_frame_rgb, (image_size, image_size)) / 255.0
    img_tensor = torch.from_numpy(original_frame_resized).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # Save the image to disk for load_video_frames to use
    test_image_path = 'test_random_image.jpg'
    cv2.imwrite(test_image_path, cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR))

    # Call load_video_frames to load and normalize
    images, video_height, video_width = load_video_frames(
        video_path=[test_image_path],
        image_size=image_size,
        offload_video_to_cpu=True,
        img_mean=img_mean,
        img_std=img_std
    )

    # Restore the tensor to an RGB format image
    restored_frame_rgb = tensor_to_frame_rgb(
        images[0].unsqueeze(0),
        original_size=original_size,
        img_mean=img_mean,
        img_std=img_std
    )

    # Calculate the difference between the original image and the restored image
    original_frame_resized = cv2.resize(original_frame_rgb, original_size)
    difference = np.abs(original_frame_resized.astype(np.float32) - restored_frame_rgb.astype(np.float32))
    mean_difference = np.mean(difference)
    max_difference = np.max(difference)

    print(f"Mean pixel difference: {mean_difference:.2f}")  # 32.09
    print(f"Max pixel difference: {max_difference:.2f}")  # 180.00

    # Delete the test image file
    os.remove(test_image_path)