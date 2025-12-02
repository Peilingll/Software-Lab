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

### 🐳 Option A: Docker Usage (Recommended)

This is the easiest way to run the project. No need to install Python or download weights manually.

### 1. File Preparation (Important)

Before running, create a workspace folder on your computer with the following structure.
**Note:** The subfolder name `scans` is mandatory.

```text
my_workspace/           # You can name this root folder whatever you want
├── scans/              # ⚠️ MUST create a folder named 'scans' here!
│   ├── area1.las       # Place your .las files inside
│   └── area2.las
└── results/            # 📂 Create an empty folder for outputs

### 2. Run the Pipeline

Open your terminal, navigate to your workspace folder (cd my_workspace), and run the following command.

Docker will automatically download the image (peilingll/sonata-pipeline:v1) if it is missing.

```bash
# Docker will map the current folder "$(pwd)" to the container
docker run --gpus all \
  -v $(pwd):/app/data \
  -v $(pwd)/results:/app/batches \
  peilingll/sonata-pipeline:v1
```

  * **`--gpus all`**: Enables GPU support (Required).
  * **`-v $(pwd):/app/data`**: Maps your current folder (containing the scans subfolder) to the container.
  * **`-v $(pwd)/results:/app/batches`**: Maps your local results folder to receive the output files.
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