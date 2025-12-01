````markdown
# Real-Time Point Cloud to BIM Pipeline 

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](https://www.docker.com/)

This project implements an **automated pipeline** for processing Point Cloud data. It leverages the Sonata framework to convert raw scans into structured BIM information.

## 🌟 Key Features
- **Automatic Processing**: Fully automated pipeline converting raw `.las` point clouds to IFC models.
- **Deep Learning Integration**: Uses `sonata_full_pipeline.py` (located in the `sonata/` folder) for intelligent processing.
- **Real-time capable**: Designed for efficient processing of point cloud scans.
- **Dockerized**: Includes a containerized environment for easy deployment across different machines.

## 📂 Project Structure

```text
Software-Lab/
├── batches/                # 📂 Output folder for processed results
├── ckpt/                   # ⚠️ Model weights (Download Required) 
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

## ⚠️ Prerequisites (Crucial)

Regardless of whether you use Docker or Conda, **you must download the model weights first**.

1.  **Download** the `sonata.pth` and `sonata_linear_prob_head_sc.pth` from the link below:
    👉 **[Google Drive: Checkpoints & Test Data](https://drive.google.com/drive/folders/1IMTsD6btyR7csem7WaTh6m9fsp7lszRK?usp=sharing)**
2.  **Place the `.pth` files** into the `ckpt/` directory.

> **Note**: The pipeline will fail immediately if these files are missing.

-----

## 🐳 Option A: Docker Usage (Recommended)

This project includes a Docker setup to ensure a consistent environment with CUDA 12.4 support. This is the best way to run the project on a new machine.

### 1\. Get the Code

Clone the repository and switch to the development branch containing the Docker configuration.

```bash
git clone [https://github.com/Peilingll/Software-Lab.git](https://github.com/Peilingll/Software-Lab.git)
cd Software-Lab

# Switch to the branch with Docker support
git checkout docker-dev
```

### 2\. Build the Image

Ensure you have placed the checkpoints in `ckpt/` before building.

```bash
# Run this in the project root
docker build -t sonata-pipeline .
```

### 3\. Run the Pipeline

You must mount your local data and output directories so the container can access files and save results to your disk.

```bash
# Replace '/path/to/your/data' with your actual local path containing the 'scans' folder
docker run --gpus all \
  -v /path/to/your/data:/app/data \
  -v $(pwd)/batches_output:/app/batches \
  sonata-pipeline
```

  * **`--gpus all`**: Enables GPU support (Required).
  * **`-v ...:/app/data`**: Maps your local input scans to the container.
  * **`-v ...:/app/batches`**: Maps the container's output to your local folder.

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

Ensure your `.las` files are in `data/scans/`.

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