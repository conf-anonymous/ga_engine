#!/bin/bash
# scripts/setup_runpod_cuda.sh
#
# Setup script for RunPod CUDA environment
# Run this ONCE after spinning up a new pod
#
# Usage:
#   chmod +x scripts/setup_runpod_cuda.sh
#   ./scripts/setup_runpod_cuda.sh

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   RunPod CUDA Environment Setup for Clifford FHE             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're on RunPod
if [ ! -d "/workspace" ]; then
    echo "WARNING: /workspace not found. Are you on RunPod?"
fi

# Check NVIDIA GPU
echo "[1/6] Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo ""
else
    echo "ERROR: nvidia-smi not found. Is CUDA installed?"
    exit 1
fi

# Check CUDA version
echo "[2/6] Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "WARNING: nvcc not found. Installing CUDA toolkit..."
    apt-get update && apt-get install -y cuda-toolkit-12-0
fi
echo ""

# Install Rust if not present
echo "[3/6] Setting up Rust..."
if command -v rustc &> /dev/null; then
    echo "Rust already installed: $(rustc --version)"
else
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi
echo ""

# Ensure we have the right Rust version
echo "[4/6] Updating Rust to stable..."
rustup default stable
rustup update
echo ""

# Install build dependencies
echo "[5/6] Installing build dependencies..."
apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    git \
    2>/dev/null || echo "Some packages may already be installed"
echo ""

# Set up environment variables
echo "[6/6] Setting up environment variables..."
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Add to bashrc for persistence
cat >> ~/.bashrc << 'EOF'
# CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Setup Complete!                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Environment:"
echo "  - Rust: $(rustc --version)"
echo "  - CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'Check manually')"
echo "  - GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Check manually')"
echo ""
echo "Next steps:"
echo "  1. cd /workspace/ga_engine"
echo "  2. ./scripts/run_cuda_benchmarks.sh"
echo ""
