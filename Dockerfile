# Use PyTorch official image with CUDA 12.6 and cuDNN 9 development environment
# This provides a complete ML/DL stack with GPU support for deep learning workloads
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

# Configure NVIDIA GPU access within the container
# NVIDIA_VISIBLE_DEVICES=all makes all GPUs available to the container
# NVIDIA_DRIVER_CAPABILITIES specifies which GPU capabilities to expose
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Prevent interactive prompts during package installations
# This ensures automated builds don't hang waiting for user input
ENV DEBIAN_FRONTEND=noninteractive

# Define supported CUDA compute capabilities for PyTorch compilation
# Covers GPU architectures from Pascal (6.0) to Ampere (8.6) and future compatibility (+PTX)
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Set the working directory where the application will be installed and run
WORKDIR /opt/chataint

# Install essential build tools and utilities
# - build-essential: GCC compiler and build tools for compiling extensions
# - curl: For downloading files from URLs
# - git: Version control system for cloning repositories
RUN apt-get update && \
    apt-get -y install build-essential curl git

    # Install system dependencies required for OpenCV and GUI applications
# These packages are essential for computer vision and image processing:
# - git: Version control (additional install for completeness)
# - libgl1: OpenGL library for graphics rendering
# - libglib2.0-0: GLib library for low-level system functionality
# - libsm6, libxext6, libxrender-dev: X11 libraries for display and rendering
# Clean up package cache to reduce image size
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*
 
# Clone the SAM2 (Segment Anything Model 2) repository for longer video processing
# This is a fork with enhancements for processing longer video sequences
RUN git clone https://github.com/ChataingT/sam2_longer_video.git

# Download pre-trained model checkpoints
# These are the neural network weights required for SAM2 inference
RUN cd sam2_longer_video/checkpoints/ && ./download_ckpts.sh

# Install SAM2 package in development mode with notebook dependencies
# -e flag installs in editable mode for development
# [notebooks] includes Jupyter notebook dependencies
# --extra-index-url specifies PyTorch's CUDA 12.6 wheel repository
RUN cd sam2_longer_video && python -m pip install -e ".[notebooks]" --extra-index-url https://download.pytorch.org/whl/cu126

# Alternative method to download SAM2 weights directly (currently commented out)
# This would download the large model weights from Facebook's official repository
# RUN curl -O https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# Set the default command when the container starts
# Opens a bash shell for interactive use of the container
CMD ["bash"]