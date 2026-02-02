#!/bin/bash
# V4 CUDA Quick Test - Fixed Index Out of Bounds Issue
# Run this on the CUDA server

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   V4 CUDA Quick Test - Fixed Strided Layout Issue            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Set up CUDA environment
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

echo "Step 1: Setting CUDA environment..."
echo "  LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "  CUDA toolkit path = /usr/local/cuda"
echo ""

echo "Step 2: Building V4 CUDA quick benchmark..."
~/.cargo/bin/cargo build --release --features v2,v2-gpu-cuda,v4 --example bench_v4_cuda_geometric_quick
echo "  ✅ Build complete"
echo ""

echo "Step 3: Running V4 CUDA geometric product quick test..."
echo "  (N=1024, ~5-10 minutes total)"
echo ""

~/.cargo/bin/cargo run --release --features v2,v2-gpu-cuda,v4 --example bench_v4_cuda_geometric_quick

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                      TEST COMPLETE                            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
