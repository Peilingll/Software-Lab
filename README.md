# Real-Time Point Cloud to BIM Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Docker Image Version](https://img.shields.io/docker/v/peilingll/sonata-pipeline?sort=semver&label=Docker%20Image&logo=docker&logoColor=white)](https://hub.docker.com/r/peilingll/sonata-pipeline)

An automated pipeline for converting raw `.las` point cloud scans into IFC/BIM models using the Sonata deep learning framework.

> **Docker Hub**: Pre-built image at [peilingll/sonata-pipeline](https://hub.docker.com/r/peilingll/sonata-pipeline).

## Prerequisites

- NVIDIA GPU with drivers installed
- Docker Engine
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Quick Start

### 1. Prepare your data

```text
my_workspace/
├── scans/              # MUST be named 'scans'
│   ├── area1.las
│   └── area2.las
└── results/            # Empty folder for outputs
```

### 2. Run

```bash
cd my_workspace

docker run --gpus all \
  -v $(pwd):/app/data \
  -v $(pwd)/results:/app/batches \
  peilingll/sonata-pipeline:v1
```

Docker will automatically pull the image on first run. No Python installation or manual weight downloads needed -- everything is included in the image.

### 3. Check results

- **Output files**: `results/` folder
- **Logs**: printed to terminal (or use `docker logs` if running detached)

---

## Build from Source (For Developers)

If you are modifying the source code and need to rebuild the image:

```bash
git clone https://github.com/Peilingll/Software-Lab.git
cd Software-Lab
git checkout docker-dev

# Place .las files in data/scans/
mkdir -p data/scans batches

# Build and run
docker compose up --build
```

Other useful commands:

```bash
docker compose build            # Build only
docker compose up --build -d    # Run in background
docker compose logs -f          # View logs
```

### Push updated image to Docker Hub

```bash
docker compose build
docker tag softwarelab-pipeline peilingll/sonata-pipeline:v2
docker push peilingll/sonata-pipeline:v2
```

---

## Project Structure

```text
Software-Lab/
├── batches/                # Output folder for processed results
├── ckpt/                   # Model weights (baked into Docker image)
├── data/
│   └── scans/              # Input .las files
├── script/                 # Helper scripts for geometry processing
├── sonata/                 # Sonata deep learning module
├── Dockerfile
├── docker-compose.yml
├── pipeline_runner.py      # Main entry script
└── README.md
```
