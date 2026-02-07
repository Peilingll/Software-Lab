# 1. Base Image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 2. Work Directory
WORKDIR /app

# 3. System Dependencies
# Install basic tools and graphics libraries required by Open3D
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# 5. Configure Conda
# Add conda-forge, set priority to flexible (fixes solver errors), and accept ToS
RUN conda config --add channels conda-forge \
    && conda config --set channel_priority flexible \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 6. Environment Setup
COPY environment.yml .
RUN conda env create -f environment.yml

# 7. Shell Config
# Ensure all subsequent commands run inside the 'sonata2' environment
SHELL ["conda", "run", "-n", "sonata2", "/bin/bash", "-c"]

# 7.5 Install torch-scatter (needs torch available, so must be after conda env)
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

# 8. Copy Files
# Note: huge data folders are excluded via .dockerignore
COPY . .

# 9. Permissions & Entry
RUN chmod +x pipeline_runner.py
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "sonata2", "python", "pipeline_runner.py"]