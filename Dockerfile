# 1. Base Image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 2. Work Directory
WORKDIR /app

# 3. System Dependencies
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

# 5. Environment Setup
COPY environment.yml .
RUN conda env create -f environment.yml

# 6. Shell Config
SHELL ["conda", "run", "-n", "sonata2", "/bin/bash", "-c"]

# 7. Copy Files (data/ is ignored by .dockerignore)
COPY . .

# 8. Permissions & Entry
RUN chmod +x pipeline_runner.py
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "sonata2", "python", "pipeline_runner.py"]