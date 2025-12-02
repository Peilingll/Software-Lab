# Real-Time Point Cloud to BIM Pipeline 

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Docker Image Version](https://img.shields.io/docker/v/peilingll/sonata-pipeline?sort=semver&label=Docker%20Image&logo=docker&logoColor=white)](https://hub.docker.com/r/peilingll/sonata-pipeline)

This project implements an **automated pipeline** for processing Point Cloud data. It leverages the Sonata framework to convert raw scans into structured BIM information.

>  **Docker Hub**: The pre-built image is available at [peilingll/sonata-pipeline](https://hub.docker.com/r/peilingll/sonata-pipeline).

## 🌟 Key Features
- **Automatic Processing**: Fully automated pipeline converting raw `.las` point clouds to IFC models.
- **Deep Learning Integration**: Uses `sonata_full_pipeline.py` (located in the `sonata/` folder) for intelligent processing.
- **Real-time capable**: Designed for efficient processing of point cloud scans.
- **Dockerized**: Includes a containerized environment for easy deployment across different machines.

## 📂 Project Structure

```text
Software-Lab/
├── batches/                # 📂 Output folder for processed results
├── ckpt/                   # ⚠️ Model weights
│   ├── sonata.pth
│   └── sonata_linear_prob_head_sc.pth
├── data/
│   └── scans/              # 📂 Place your input .las files here
├── logs/                   # Execution logs (Auto-generated)
├── script/                 # Helper scripts for geometry processing
├── sonata/                 # Sonata deep learning submodule
│   └── sonata_full_pipeline.py
├── environment.yml         # Conda environment configuration
├── Dockerfile              # Docker configuration
├── .dockerignore           # Docker build ignore list
├── pipeline_runner.py      # Main entry script
└── README.md

````

## ⚠️ Prerequisites

### For Docker Users:

The pre-built image (`v1`) already includes these weights(ckpt/). You only need your input data (`.las` files).

### For Local (Conda) Users:

You **must download the model weights first**.

1.  **Download** `sonata.pth` and `sonata_linear_prob_head_sc.pth`:
    **[Google Drive: Checkpoints & Test Data](https://drive.google.com/drive/folders/1IMTsD6btyR7csem7WaTh6m9fsp7lszRK?usp=sharing)**
2.  **Place the files** into the `ckpt/` directory.

-----

## 🐳 Option A: Docker Usage (Recommended)

This is the easiest way to run the project without configuring environments manually.

### 1\. Run the Pipeline (Quick Start)

You can run the pipeline immediately using the pre-built image from Docker Hub. Docker will automatically download it if it's not on your computer.

**Command:**
You must mount your local data and output directories so the container can access your files.

```bash
# Replace '/path/to/your/data' with your actual local path containing the 'scans' folder
docker run --gpus all \
  -v /path/to/your/data:/app/data \
  -v $(pwd)/batches_output:/app/batches \
  peilingll/sonata-pipeline:v1
```

  * **`--gpus all`**: Enables GPU support (Required).
  * **`-v ...:/app/data`**: Maps your local input scans to the container.
  * **`-v ...:/app/batches`**: Maps the container's output to your local folder.
  * **`peilingll/sonata-pipeline:v1`**: The official image name.

### 2\. Build from Source 

Only follow this step if you are a developer modifying the `Dockerfile`.

```bash
# 1. Clone repo and switch branch
git clone [https://github.com/Peilingll/Software-Lab.git](https://github.com/Peilingll/Software-Lab.git)
cd Software-Lab
git checkout docker-dev

# 2. Build manually (Ensure ckpt/ contains .pth files first)
docker build -t sonata-pipeline .
```

-----

## 🛠️ Option B: Local Installation (Conda)

If you prefer to run locally without Docker, follow these steps.

### 1\. Setup Environment

The environment is managed via Conda.

```bash
# Clone the repository
git clone [https://github.com/Peilingll/Software-Lab.git](https://github.com/Peilingll/Software-Lab.git)
cd Software-Lab

# Create the environment from the provided config
conda env create -f environment.yml

# Activate the environment
conda activate sonata2
```

### 2\. Run Pipeline

Ensure your `.las` files are in `data/scans/` and checkpoints are in `ckpt/`.

```bash
# Ensure environment is active
conda activate sonata2

# Run the pipeline
python pipeline_runner.py
```

## 📊 Outputs

  - **Processed Models**: Check the `batches/` folder (or your mounted output folder if using Docker).
  - **Logs**: Check `logs/` for detailed processing information or error tracing.

<!-- end list -->

```
```