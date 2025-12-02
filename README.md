

-----

````markdown
# Real-Time Point Cloud to BIM Pipeline 

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Docker Image Version](https://img.shields.io/docker/v/peilingll/sonata-pipeline?sort=semver&label=Docker%20Image&logo=docker&logoColor=white)](https://hub.docker.com/r/peilingll/sonata-pipeline)

This project implements an **automated pipeline** for processing Point Cloud data. It leverages the Sonata framework to convert raw scans into structured BIM information.

> 🐳 **Docker Hub**: The pre-built image is ready to use at [peilingll/sonata-pipeline](https://hub.docker.com/r/peilingll/sonata-pipeline).

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

-----

## 🐳 Option A: Docker Usage (Recommended)

This is the easiest way to run the project. You do **not** need to install Python or download model weights manually—they are included in the image.

### 1\. File Preparation (Important)

Before running, create a workspace folder on your computer with the following specific structure.
**Note:** The subfolder name `scans` is mandatory.

```text
my_workspace/           # You can name this root folder whatever you want
├── scans/              # ⚠️ MUST create a folder named 'scans' here!
│   ├── area1.las       # Place your .las files inside
│   └── area2.las
└── results/            # 📂 Create an empty folder for outputs
```

### 2\. Run the Pipeline

Open your terminal, **navigate to your workspace folder** (`cd my_workspace`), and run the command below. Docker will automatically download the image (`peilingll/sonata-pipeline:v1`) if it is missing.

```bash
docker run --gpus all \
  -v $(pwd):/app/data \
  -v $(pwd)/results:/app/batches \
  peilingll/sonata-pipeline:v1
```

**Explanation:**

  * **`--gpus all`**: Enables NVIDIA GPU support (Required).
  * **`-v $(pwd):/app/data`**: Maps your current folder (containing the `scans` subfolder) to the container.
  * **`-v $(pwd)/results:/app/batches`**: Maps your local `results` folder to receive the output files.
  * **`peilingll/sonata-pipeline:v1`**: The official pre-built image.

-----

## 🛠️ Option B: Local Installation (Conda)

If you prefer to run locally without Docker, follow these steps.

### 1\. Download Weights (Crucial)

You **must download the model weights** manually for local execution.

1.  **Download** `sonata.pth` and `sonata_linear_prob_head_sc.pth`:
    👉 **[Google Drive: Checkpoints & Test Data](https://drive.google.com/drive/folders/1IMTsD6btyR7csem7WaTh6m9fsp7lszRK?usp=sharing)**
2.  **Place the files** into the `ckpt/` directory inside the project.

> **Note**: The pipeline will fail immediately if these files are missing.

### 2\. Setup Environment

The environment is managed via Conda.

```bash
# Clone the repository
git clone https://github.com/Peilingll/Software-Lab.git
cd Software-Lab

# Create the environment from the provided config
conda env create -f environment.yml

# Activate the environment
conda activate sonata2
```

### 3\. Run Pipeline

Ensure your `.las` files are in `data/scans/`.

```bash
# Ensure environment is active
conda activate sonata2

# Run the pipeline
python pipeline_runner.py
```

-----

## 🏗️ Advanced: Build Docker from Source

Only follow this step if you are a developer modifying the `Dockerfile`. Regular users should use **Option A**.

```bash
# 1. Clone repo and switch to the docker branch
git clone https://github.com/Peilingll/Software-Lab.git
cd Software-Lab
git checkout docker-dev

# 2. Build manually (Ensure ckpt/ contains .pth files first!)
docker build -t sonata-pipeline .
```

## 📊 Outputs

  - **Processed Models**: Check the `batches/` folder (or your mounted output folder if using Docker).
  - **Logs**: Check `logs/` for detailed processing information or error tracing.

<!-- end list -->

```
```