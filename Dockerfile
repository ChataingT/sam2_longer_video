# ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
ARG MODEL_SIZE=tiny

FROM ${BASE_IMAGE}

# Gunicorn environment variables
# ENV GUNICORN_WORKERS=1
# ENV GUNICORN_THREADS=2
# ENV GUNICORN_PORT=5000

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME="/usr/local/cuda"
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="${PATH}:/home/user/.local/bin"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.5 8.0 8.6+PTX 8.7 8.9"


# SAM 2 environment variables
ENV APP_ROOT=/opt/sam2_lv
ENV PYTHONPATH=":${APP_ROOT}"

ENV PYTHONUNBUFFERED=1
# ENV SAM2_BUILD_CUDA=1
ENV SAM2_BUILD_CUDA=0
ENV SAM2_BUILD_ALLOW_ERRORS=0
# ENV MODEL_SIZE=${MODEL_SIZE}
ENV MODEL_SIZE=tiny

# Install system requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    build-essential \
    libffi-dev \
    ninja-build


RUN mkdir ${APP_ROOT}
WORKDIR ${APP_ROOT}

# Copy SAM 2 inference files
COPY sam2 ./sam2
COPY test ./test


COPY setup.py .
COPY README.md .


# Download SAM 2.1 checkpoints
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_tiny.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_small.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_base_plus.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_large.pt



RUN pip install --upgrade pip setuptools
RUN pip install -e ".[dev]" -v
# RUN pip install -e ".[dev]"
# RUN python setup.py build_ext --inplace

# https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/issues/69#issuecomment-1826764707
# RUN rm /opt/conda/bin/ffmpeg && ln -s /bin/ffmpeg /opt/conda/bin/ffmpeg



# Copy backend server files
# COPY demo/backend/server ${APP_ROOT}/server




# https://pythonspeed.com/articles/gunicorn-in-docker/
# CMD gunicorn --worker-tmp-dir /dev/shm \
#     --worker-class gthread app:app \
#     --log-level info \
#     --access-logfile /dev/stdout \
#     --log-file /dev/stderr \
#     --workers ${GUNICORN_WORKERS} \
#     --threads ${GUNICORN_THREADS} \
#     --bind 0.0.0.0:${GUNICORN_PORT} \
#     --timeout 60