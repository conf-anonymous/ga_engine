# RunPod GPU Setup for CliffordPointNet Experiments

This guide explains how to set up a RunPod GPU instance to run CliffordPointNet experiments on the REAL ModelNet40 dataset.

## Recommended GPU

For optimal performance, use:
- **RTX 4090 / RTX 5090** - Best value for training
- **A100 / H100** - For production/paper results

Expected speedup vs CPU: **50-100x** (hours → minutes)

## Quick Start

### 1. Create RunPod Instance

1. Go to [runpod.io](https://runpod.io)
2. Select a GPU pod with:
   - **GPU**: RTX 4090/5090, A100, or H100
   - **Image**: `runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04`
   - **Disk**: At least 20GB

### 2. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

### 3. Clone Repository

```bash
git clone https://github.com/YOUR_ORG/ga_engine_anonymous.git
cd ga_engine_anonymous
```

### 4. Download ModelNet40 Dataset

```bash
mkdir -p data
cd data

# Download from Princeton (original source, ~1.9GB)
wget http://modelnet.cs.princeton.edu/ModelNet40.zip

# Extract
unzip ModelNet40.zip

cd ..
```

Verify the dataset structure:
```bash
ls data/ModelNet40/
# Should show: airplane bathtub bed bench bookshelf ...
```

### 5. Build and Run Experiments

```bash
# Build with release optimizations
cargo build --release

# Run the plaintext ModelNet40 experiment
MODELNET_PATH=data/ModelNet40 EPOCHS=200 POINTS=1024 HIDDEN=128 LR=0.001 SEED=42 \
  cargo run --release --example experiment_modelnet40
```

## Dataset Information

### ModelNet40

| Property | Value |
|----------|-------|
| Source | http://modelnet.cs.princeton.edu/ModelNet40.zip |
| Size | ~1.9 GB (compressed) |
| Format | OFF mesh files |
| Classes | 40 (airplane, bathtub, bed, bench, ...) |
| Train samples | 9,843 |
| Test samples | 2,468 |

The loader automatically:
- Parses OFF mesh format
- Samples 1024 points per mesh using farthest point sampling
- Normalizes to unit sphere

### Optional: ScanObjectNN (More Challenging)

For harder real-world experiments:

```bash
# ScanObjectNN (requires registration at https://hkust-vgd.ust.hk/scanobjectnn/)
# Download and extract to data/ScanObjectNN/
```

### Optional: ShapeNet Part Segmentation

For part segmentation experiments:

```bash
# ShapeNet Part (shapenet.cs.stanford.edu)
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip -d data/
```

## Experiment Commands

### E1: Plaintext Baseline (ModelNet40)

```bash
MODELNET_PATH=data/ModelNet40 \
EPOCHS=200 \
POINTS=1024 \
HIDDEN=128 \
LR=0.001 \
SEED=42 \
cargo run --release --example experiment_modelnet40
```

Expected output:
- Overall Accuracy (OA): Target ~85-92%
- Mean Class Accuracy (mAcc): Target ~82-88%
- Training time: ~10-20 minutes on RTX 4090/5090

### Quick Test (Verify Setup)

```bash
# Quick test with subset
MODELNET_PATH=data/ModelNet40 \
MAX_SAMPLES=10 \
EPOCHS=5 \
cargo run --release --example experiment_modelnet40
```

### E3: Encrypted Inference (CUDA)

After training, run encrypted inference:

```bash
# TODO: Add CUDA encrypted inference example
cargo run --release --example encrypted_inference_cuda
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELNET_PATH` | `data/ModelNet40` | Path to extracted dataset |
| `POINTS` | `1024` | Points per sample |
| `HIDDEN` | `128` | Hidden layer dimension |
| `EPOCHS` | `200` | Training epochs |
| `LR` | `0.001` | Learning rate |
| `MAX_SAMPLES` | `0` (all) | Limit samples per class (for testing) |
| `SEED` | `42` | Random seed |

## Saving Results

The experiment will output:
1. Console logs with accuracy metrics
2. Weight file: `data/modelnet40_weights_seed{SEED}.json`
3. Formatted results for EXPERIMENTS.md

Copy the final output table to your paper/documentation.

## Troubleshooting

### CUDA not found
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version
```

### Out of memory
```bash
# Reduce batch size or points
POINTS=512 cargo run --release --example experiment_modelnet40
```

### Download failed
```bash
# Alternative mirror (if Princeton is slow)
# Try shapenet.cs.stanford.edu mirrors or academic network
```

## Expected Results

On REAL ModelNet40 with default settings:

| Metric | Expected Range |
|--------|----------------|
| Overall Accuracy (OA) | 85-92% |
| Mean Class Accuracy (mAcc) | 82-88% |
| Training Time (RTX 4090) | ~10-20 min |
| Training Time (A100) | ~5-10 min |

For reference, PointNet achieves ~89.2% OA on ModelNet40.
