# NVIDIA GPU Benchmark Setup Guide

This document provides instructions for running Clifford FHE benchmarks on NVIDIA GPUs.

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA Compute Capability 7.0+ (RTX 20 series or newer)
- Recommended: RTX 3090, RTX 4090, A100, or H100 for best performance

### Software
- CUDA Toolkit 12.0+ (12.3 recommended, as configured in Cargo.toml)
- Rust toolchain (stable, 1.70+)
- Linux (Ubuntu 20.04+ recommended) or Windows with CUDA support

## Environment Setup

### 1. Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check GPU is detected
nvidia-smi

# Verify CUDA libraries are available
ls /usr/local/cuda/lib64/libcudart.so*
```

### 2. Set Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 3. Clone and Build

```bash
git clone <repository-url>
cd ga_engine_anonymous

# Build with CUDA support (IMPORTANT: disable lattice-reduction to avoid blas issues)
cargo build --release --no-default-features --features f64,nd,v2,v2-gpu-cuda
```

**Note:** We disable `lattice-reduction` because it requires BLAS/LAPACK which can conflict with CUDA on cloud instances. Lattice reduction is only needed for security analysis, not FHE operations.

## Quick Start (Recommended)

Use the provided scripts for a streamlined experience:

```bash
# 1. Setup environment (run once after spinning up pod)
chmod +x scripts/setup_runpod_cuda.sh
./scripts/setup_runpod_cuda.sh

# 2. Run benchmarks
chmod +x scripts/run_cuda_benchmarks.sh
./scripts/run_cuda_benchmarks.sh quick    # Essential benchmarks (~5-10 min)
./scripts/run_cuda_benchmarks.sh full     # All benchmarks (~30-60 min)
```

Results are saved to `results/cuda_benchmarks_<timestamp>/`.

## Manual Benchmark Commands

### Quick Sanity Check

```bash
# Test CUDA device detection
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda \
    --example test_cuda_ntt_roundtrip
```

### Core Operations Benchmark

```bash
# Comprehensive benchmark of all homomorphic operations
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda \
    --example bench_cuda_all_ops
```

### Geometric Product Benchmark (Most Important)

```bash
# V4 packed geometric product (best performance)
cargo run --release --no-default-features --features f64,nd,v2,v4,v2-gpu-cuda \
    --example bench_v4_cuda_geometric

# Quick version for sanity check
cargo run --release --no-default-features --features f64,nd,v2,v4,v2-gpu-cuda \
    --example bench_v4_cuda_geometric_quick
```

### Bootstrapping Benchmark (If V3 enabled)

```bash
cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda \
    --example bench_cuda_bootstrap
```

## Expected Results

Based on M3 Max CPU results, we expect the following on a high-end NVIDIA GPU:

| Operation | M3 Max CPU | Expected NVIDIA GPU | Target Speedup |
|-----------|------------|---------------------|----------------|
| Key Generation | 5ms | ~2-5ms | 1-2× |
| Encryption (32 cts) | 93ms | ~5-10ms | 10-20× |
| Geometric Product | 959ms | **50-100ms** | **10-20×** |
| Decryption | 7ms | ~2-5ms | 1-3× |

**Key metric:** If geometric product achieves <100ms, we demonstrate practical encrypted 3D inference.

## Troubleshooting

### CUDA Not Found

```bash
# Check if cudarc can find CUDA
cargo build --release --no-default-features --features f64,nd,v2,v2-gpu-cuda 2>&1 | grep -i cuda

# Common fix: ensure CUDA_HOME is set
export CUDA_HOME=/usr/local/cuda-12.3  # Adjust version as needed
```

### Out of GPU Memory

```bash
# Check GPU memory usage
nvidia-smi

# For large parameters (N=8192), ensure 16GB+ VRAM
# For N=4096, 8GB should suffice
# For N=1024 (test params), 4GB is enough
```

### BLAS/LAPACK Conflicts

If you see errors about netlib-src, blas-src, or lapack:

```bash
# Solution: disable lattice-reduction feature
cargo build --release --no-default-features --features f64,nd,v2,v2-gpu-cuda
```

### cudarc Version Mismatch

The project uses cudarc 0.11 with CUDA 12.3. If your CUDA version differs:

1. Check your CUDA version: `nvcc --version`
2. Modify `Cargo.toml` if needed:
   ```toml
   cudarc = { version = "0.11", optional = true, features = ["cuda-12000"] }  # For CUDA 12.0
   ```

## Recording Results

After running benchmarks, update `docs/BENCHMARK_RESULTS.md` with:

1. GPU model and VRAM
2. CUDA version
3. All timing results
4. Any errors or anomalies

## Comparison Points

When analyzing results, compare:

1. **vs V1 CPU baseline** (~13s per geometric product)
2. **vs V2 M3 Max CPU** (959ms per geometric product)
3. **vs V2 Metal GPU** (TBD for full geometric product)

The key question: **Can we achieve <100ms geometric product on NVIDIA GPU?**

If yes, encrypted 3D point cloud inference becomes practical for real applications.
