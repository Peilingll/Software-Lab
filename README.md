# Real-Time Point Cloud to BIM Pipeline 

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

This project implements an **automated pipeline** for processing Point Cloud data. It leverages the Sonata framework to convert raw scans into structured BIM information.

## 🌟 Key Features
- **Automatic Processing**: Fully automated pipeline converting raw `.las` point clouds to IFC models.
- **Deep Learning Integration**: Uses `sonata_full_pipeline.py` (located in the `sonata/` folder) for intelligent processing.
- **Real-time capable**: Designed for efficient processing of point cloud scans.

## 📂 Project Structure

```text
Software-Lab/
├── batches/                # 📂 Output folder for processed results
├── ckpt/                   # ⚠️ Model weights (Download required) 
├── data/
│   └── scans/              # 📂 Place your input .las files here
├── logs/                   # Execution logs (Auto-generated)
├── script/                 # Helper scripts for geometry processing
├── sonata/                 # Sonata deep learning submodule
│   └── sonata_full_pipeline.py
├── environment.yml         # Conda environment configuration
├── pipeline_runner.py      # 🚀 Main entry script
└── README.md
````

## 🛠️ Installation & Environment

The environment is managed via Conda. Please ensure you have [Anaconda](https://www.anaconda.com/) or Miniconda installed.

### 1\. Clone the repository

```bash
git clone https://github.com/Peilingll/Software-Lab.git
cd Software-Lab
```

### 2\. Setup Environment

This project uses a dedicated Conda environment named `sonata2`.

```bash
# Create the environment from the provided config
conda env create -f environment.yml

# Activate the environment
conda activate sonata2
```

### 3\. Download Checkpoints (Crucial)

The model weights are too large for GitHub and must be downloaded separately.

1.  **Download** the `sonata.pth` and testing data from the link below:
    👉 **[Google Drive: Checkpoints & Test Data](https://drive.google.com/drive/folders/1IMTsD6btyR7csem7WaTh6m9fsp7lszRK?usp=sharing)**
2.  **Place the `.pth` files** into the `ckpt/` directory.

> ⚠️ **Warning**: The pipeline will fail immediately if `ckpt/sonata.pth` and `sonata_linear_prob_head_sc.pth` are missing.

## 🚀 Usage

### Step 1: Prepare Data

Ensure your point cloud scans are in **`.las`** format and place them in the input directory:

```text
data/scans/
```

### Step 2: Run Pipeline

Execute the main runner script. This will sequentially trigger the segmentation and reconstruction modules.

```bash
# Ensure environment is active
conda activate sonata2

# Run the pipeline
python pipeline_runner.py
```

### Step 3: Check Results

  - **Processed Models**: Check the `batches/` folder for output files.
  - **Logs**: Check `logs/` for detailed processing information or error tracing.

<!-- end list -->

```
```