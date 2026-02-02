#!/bin/bash
# Find CUDA libraries and suggest environment variable fixes
#
# This script helps diagnose CUDA library path issues when running
# CUDA-enabled examples and tests.
#
# Usage:
#   bash scripts/find_cuda.sh

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           CUDA Library Path Diagnostic Tool                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

echo -e "\n1. Searching for libnvrtc.so:"
NVRTC_PATHS=$(find /usr -name "libnvrtc.so*" 2>/dev/null)
if [ -n "$NVRTC_PATHS" ]; then
    echo "$NVRTC_PATHS" | head -5
else
    echo "  ❌ libnvrtc.so not found in /usr"
fi

echo -e "\n2. Common CUDA installation paths:"
if ls -ld /usr/local/cuda* 2>/dev/null; then
    :
else
    echo "  ❌ Not found in /usr/local/cuda*"
fi

if ls -ld /opt/cuda* 2>/dev/null; then
    :
else
    echo "  ℹ️  Not found in /opt/cuda*"
fi

echo -e "\n3. Checking /usr/local/cuda symlink:"
if [ -L /usr/local/cuda ]; then
    CUDA_TARGET=$(readlink /usr/local/cuda)
    echo "  ✅ /usr/local/cuda -> $CUDA_TARGET"
    if [ -d /usr/local/cuda/lib64 ]; then
        echo "  Libraries in /usr/local/cuda/lib64:"
        ls -lh /usr/local/cuda/lib64/libnvrtc* 2>/dev/null | head -3 || echo "    ❌ No libnvrtc* found"
    fi
else
    echo "  ❌ /usr/local/cuda is not a symlink or doesn't exist"
fi

echo -e "\n4. Current environment variables:"
echo "  CUDA_HOME: ${CUDA_HOME:-❌ not set}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-❌ not set}"

echo -e "\n5. CUDA driver version (from nvidia-smi):"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi | grep "CUDA Version" || echo "  ❌ Could not detect CUDA version"
else
    echo "  ❌ nvidia-smi not found"
fi

echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "RECOMMENDED FIX:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

NVRTC_PATH=$(find /usr -name "libnvrtc.so*" 2>/dev/null | head -1)
if [ -n "$NVRTC_PATH" ]; then
    CUDA_LIB_DIR=$(dirname "$NVRTC_PATH")
    CUDA_HOME_DIR=$(dirname "$CUDA_LIB_DIR")

    echo ""
    echo "Add these lines to your ~/.bashrc or run them before cargo:"
    echo ""
    echo "  export LD_LIBRARY_PATH=$CUDA_LIB_DIR:\$LD_LIBRARY_PATH"
    echo "  export CUDA_HOME=$CUDA_HOME_DIR"
    echo ""
    echo "Then run:"
    echo "  source ~/.bashrc"
    echo "  # OR just export them in current shell"
    echo ""
    echo "To test immediately (without modifying ~/.bashrc):"
    echo ""
    echo "  LD_LIBRARY_PATH=$CUDA_LIB_DIR:\$LD_LIBRARY_PATH \\"
    echo "  CUDA_HOME=$CUDA_HOME_DIR \\"
    echo "  cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap"
    echo ""
else
    echo ""
    echo "  ❌ ERROR: libnvrtc.so not found on system"
    echo ""
    echo "  CUDA runtime libraries are required but not found."
    echo "  Please install CUDA toolkit or check your CUDA installation."
    echo ""
    echo "  On Ubuntu/Debian:"
    echo "    sudo apt-get install nvidia-cuda-toolkit"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
