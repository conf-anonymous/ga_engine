#!/bin/bash
# scripts/run_cuda_benchmarks.sh
#
# Run comprehensive CUDA benchmarks for Clifford FHE
# Designed for RunPod RTX 4090 environment
#
# Usage:
#   chmod +x scripts/run_cuda_benchmarks.sh
#   ./scripts/run_cuda_benchmarks.sh [quick|full]
#
# Options:
#   quick - Run only essential benchmarks (~5-10 min)
#   full  - Run all benchmarks including stress tests (~30-60 min)

set -e

MODE=${1:-quick}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/cuda_benchmarks_$TIMESTAMP"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Clifford FHE CUDA Benchmark Suite                          ║"
echo "║   Mode: $MODE                                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Create results directory
mkdir -p $RESULTS_DIR
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Collect system information
echo "[1/7] Collecting system information... (steps 8-11 are full mode only)"
cat > $RESULTS_DIR/system_info.txt << EOF
================================================================================
SYSTEM INFORMATION
================================================================================
Date: $(date)
Hostname: $(hostname)

--- GPU Information ---
$(nvidia-smi)

--- CUDA Version ---
$(nvcc --version 2>/dev/null || echo "nvcc not in PATH")

--- Rust Version ---
$(rustc --version)
$(cargo --version)

--- CPU Information ---
$(lscpu | grep -E "Model name|CPU\(s\)|Thread|Core" || cat /proc/cpuinfo | grep -E "model name|cpu cores" | head -4)

--- Memory ---
$(free -h)

--- OS ---
$(cat /etc/os-release | head -5)
================================================================================
EOF
cat $RESULTS_DIR/system_info.txt
echo ""

# Build with CUDA features
# Note: Use --no-default-features to avoid lattice-reduction which requires Fortran compiler
echo "[2/7] Building with CUDA features..."
CUDA_FEATURES="--no-default-features --features f64,nd,v2,v2-gpu-cuda,v4"
echo "Running: cargo build --release $CUDA_FEATURES"
cargo build --release $CUDA_FEATURES 2>&1 | tee $RESULTS_DIR/build.log
echo "Build complete!"
echo ""

# Verify CUDA integration
echo "[3/7] Verifying CUDA integration..."
cargo run --release $CUDA_FEATURES --example test_v4_cuda_basic 2>&1 | tee $RESULTS_DIR/cuda_verify.log
if [ $? -eq 0 ]; then
    echo "✓ CUDA integration verified!"
else
    echo "✗ CUDA verification failed. Check cuda_verify.log"
    exit 1
fi
echo ""

# Run comprehensive homomorphic operations benchmark
echo "[4/8] Running Comprehensive Homomorphic Ops Benchmark..."
cargo run --release $CUDA_FEATURES --example bench_cuda_all_ops 2>&1 | tee $RESULTS_DIR/all_ops.log
echo ""

# Run V4 CUDA geometric product benchmark (quick version)
echo "[5/9] Running V4 CUDA Geometric Product Benchmark (Quick)..."
cargo run --release $CUDA_FEATURES --example bench_v4_cuda_geometric_quick 2>&1 | tee $RESULTS_DIR/v4_geometric_quick.log
echo ""

# ============================================================================
# CliffordPointNet Benchmarks (Privacy-Preserving 3D Point Cloud Classification)
# ============================================================================

# Run CliffordPointNet encrypted inference demo (KEY BENCHMARK)
echo "[6/9] Running CliffordPointNet Encrypted Inference..."
echo "      This is the key benchmark for privacy-preserving 3D classification!"
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda \
    --example clifford_pointnet_encrypted 2>&1 | tee $RESULTS_DIR/clifford_pointnet_encrypted.log
echo ""

# Run plaintext experiment for accuracy baseline
echo "[7/9] Running CliffordPointNet Plaintext Baseline (10 classes, quick)..."
CLASSES=10 POINTS=256 SAMPLES=50 HIDDEN=64 EPOCHS=30 LR=0.005 \
    cargo run --release --no-default-features --features f64,nd \
    --example experiment_plaintext 2>&1 | tee $RESULTS_DIR/plaintext_baseline.log
echo ""

if [ "$MODE" = "full" ]; then
    # Run full V4 CUDA geometric product benchmark
    echo "[8/11] Running V4 CUDA Geometric Product Benchmark (Full)..."
    cargo run --release $CUDA_FEATURES --example bench_v4_cuda_geometric 2>&1 | tee $RESULTS_DIR/v4_geometric_full.log
    echo ""

    # Run V4 CUDA packing benchmark
    echo "[9/11] Running V4 CUDA Packing Benchmark..."
    cargo run --release $CUDA_FEATURES --example bench_v4_cuda_packing 2>&1 | tee $RESULTS_DIR/v4_packing.log
    echo ""

    # Run division benchmark
    echo "[10/11] Running CUDA Division Benchmark..."
    cargo run --release $CUDA_FEATURES --example bench_division_cuda_gpu 2>&1 | tee $RESULTS_DIR/division_cuda.log
    echo ""

    # Run bootstrap benchmark (requires v3 feature)
    echo "[11/11] Running CUDA Bootstrap Benchmark..."
    cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 --example bench_cuda_bootstrap 2>&1 | tee $RESULTS_DIR/bootstrap_cuda.log
    echo ""
else
    echo "[8/7] Skipping full geometric benchmark (quick mode)"
    echo "[9/7] Skipping packing benchmark (quick mode)"
    echo "[10/7] Skipping division/bootstrap benchmarks (quick mode)"
    echo ""
fi

# Generate summary
echo "Generating summary..."
cat > $RESULTS_DIR/SUMMARY.md << EOF
# CUDA Benchmark Results: $TIMESTAMP

## Environment
- **GPU**: $(nvidia-smi --query-gpu=name --format=csv,noheader)
- **VRAM**: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
- **CUDA**: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo "N/A")
- **Mode**: $MODE

## Benchmarks Run
EOF

if [ "$MODE" = "full" ]; then
    cat >> $RESULTS_DIR/SUMMARY.md << EOF
1. ✓ CUDA Integration Verification
2. ✓ All Homomorphic Operations (Encode/Encrypt/Add/Mult/Rotate/etc.)
3. ✓ V4 Geometric Product (Quick)
4. ✓ **CliffordPointNet Encrypted Inference** (KEY RESULT)
5. ✓ CliffordPointNet Plaintext Baseline
6. ✓ V4 Geometric Product (Full)
7. ✓ V4 Packing
8. ✓ Division
9. ✓ Bootstrap (V3 CKKS)

## Key Results - CliffordPointNet

Extract CliffordPointNet encrypted inference timing:
\`\`\`bash
grep -E "Geometric Product|Encryption|Decryption|Total inference|ms" $RESULTS_DIR/clifford_pointnet_encrypted.log
\`\`\`

Extract plaintext accuracy:
\`\`\`bash
grep -E "Test Accuracy|accuracy" $RESULTS_DIR/plaintext_baseline.log
\`\`\`

## Comparison with M3 Max CPU Results
| Operation | M3 Max CPU | NVIDIA GPU | Speedup |
|-----------|------------|------------|---------|
| Geometric Product | 959ms | TBD | TBD |
| Encryption (4 pts) | 93ms | TBD | TBD |
| Decryption | 7ms | TBD | TBD |

**Target: Geometric Product < 100ms = Practical encrypted 3D inference**
EOF
else
    cat >> $RESULTS_DIR/SUMMARY.md << EOF
1. ✓ CUDA Integration Verification
2. ✓ All Homomorphic Operations (Encode/Encrypt/Add/Mult/Rotate/etc.)
3. ✓ V4 Geometric Product (Quick)
4. ✓ **CliffordPointNet Encrypted Inference** (KEY RESULT)
5. ✓ CliffordPointNet Plaintext Baseline
6. ○ V4 Geometric Product (Full) - skipped
7. ○ V4 Packing - skipped
8. ○ Division - skipped
9. ○ Bootstrap - skipped

## Key Results - CliffordPointNet

Extract CliffordPointNet encrypted inference timing:
\`\`\`bash
grep -E "Geometric Product|Encryption|Decryption|Total inference|ms" $RESULTS_DIR/clifford_pointnet_encrypted.log
\`\`\`

## Comparison with M3 Max CPU Results
| Operation | M3 Max CPU | NVIDIA GPU | Speedup |
|-----------|------------|------------|---------|
| Geometric Product | 959ms | TBD | TBD |

**Target: Geometric Product < 100ms = Practical encrypted 3D inference**

To run full benchmarks:
\`\`\`
./scripts/run_cuda_benchmarks.sh full
\`\`\`
EOF
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Benchmarks Complete!                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Files:"
ls -la $RESULTS_DIR/
echo ""
echo "To extract key metrics:"
echo "  grep -E 'ms|µs|speedup|Geometric Product' $RESULTS_DIR/*.log"
echo ""
echo "To download results:"
echo "  # From your local machine:"
echo "  # scp -r root@<pod-ip>:/workspace/ga_engine/$RESULTS_DIR ."
echo ""
