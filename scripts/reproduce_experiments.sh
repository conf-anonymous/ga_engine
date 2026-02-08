#!/bin/bash
# scripts/reproduce_experiments.sh
#
# Reproduces all experimental results from the paper
#
# Usage:
#   chmod +x scripts/reproduce_experiments.sh
#   ./scripts/reproduce_experiments.sh [cpu|cuda|all]
#
# Options:
#   cpu   - Run CPU-only experiments
#   cuda  - Run CUDA GPU experiments (requires NVIDIA GPU)
#   all   - Run all experiments (default)

set -e

MODE=${1:-all}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/paper_$TIMESTAMP"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Clifford FHE Paper Experiment Reproduction                 ║"
echo "║   Mode: $MODE                                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p $RESULTS_DIR
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# System information
echo "[0/7] Collecting system information..."
cat > $RESULTS_DIR/system_info.txt << EOF
================================================================================
SYSTEM INFORMATION
================================================================================
Date: $(date)
Hostname: $(hostname)
Mode: $MODE

--- Rust Version ---
$(rustc --version)
$(cargo --version)

--- OS ---
$(uname -a)
EOF

if command -v nvidia-smi &> /dev/null && [ "$MODE" != "cpu" ]; then
    echo "--- GPU Information ---" >> $RESULTS_DIR/system_info.txt
    nvidia-smi >> $RESULTS_DIR/system_info.txt
    echo "" >> $RESULTS_DIR/system_info.txt
    echo "--- CUDA Version ---" >> $RESULTS_DIR/system_info.txt
    nvcc --version 2>/dev/null >> $RESULTS_DIR/system_info.txt || echo "nvcc not in PATH" >> $RESULTS_DIR/system_info.txt
fi
cat $RESULTS_DIR/system_info.txt
echo ""

# ============================================================================
# EXPERIMENT 1: Plaintext Training Accuracy (Baseline)
# ============================================================================
echo "[1/7] Running Plaintext Training (Baseline)..."
echo "      This establishes the maximum achievable accuracy without encryption."
cargo run --release --no-default-features --features f64,nd \
    --example experiment_plaintext \
    2>&1 | tee $RESULTS_DIR/01_plaintext_training.log
echo ""

# ============================================================================
# EXPERIMENT 2: Encrypted Accuracy Validation
# ============================================================================
echo "[2/7] Running Encrypted Accuracy Validation (40 samples)..."
echo "      This verifies encryption introduces zero accuracy degradation."
cargo run --release --no-default-features --features f64,nd,v2,v3 \
    --example encrypted_accuracy_benchmark \
    2>&1 | tee $RESULTS_DIR/02_encrypted_accuracy.log
echo ""

# ============================================================================
# EXPERIMENT 3: V3 CPU Encrypted Inference
# ============================================================================
echo "[3/7] Running V3 CPU Encrypted Inference..."
echo "      End-to-end encrypted inference on CPU."
cargo run --release --no-default-features --features f64,nd,v2,v3 \
    --example clifford_pointnet_v3_encrypted \
    2>&1 | tee $RESULTS_DIR/03_v3_cpu_inference.log
echo ""

if [ "$MODE" != "cpu" ]; then
    # ============================================================================
    # EXPERIMENT 4: V3 CUDA Encrypted Inference
    # ============================================================================
    echo "[4/7] Running V3 CUDA Encrypted Inference..."
    echo "      End-to-end encrypted inference on NVIDIA GPU."
    cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda \
        --example clifford_pointnet_v3_encrypted_cuda \
        2>&1 | tee $RESULTS_DIR/04_v3_cuda_inference.log
    echo ""

    # ============================================================================
    # EXPERIMENT 5: V3 Batched GPU Benchmark
    # ============================================================================
    echo "[5/7] Running V3 Batched CUDA Benchmark..."
    echo "      Measures per-geometric-product timing at N=1024 and N=8192."
    cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda \
        --example bench_v3_cuda_geometric \
        2>&1 | tee $RESULTS_DIR/05_v3_cuda_benchmark.log
    echo ""

    # ============================================================================
    # EXPERIMENT 6: All CKKS Operations Benchmark
    # ============================================================================
    echo "[6/7] Running CKKS Operations Benchmark..."
    echo "      Measures all core homomorphic operations on GPU."
    cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda \
        --example bench_cuda_all_ops \
        2>&1 | tee $RESULTS_DIR/06_ckks_operations.log
    echo ""

    # ============================================================================
    # EXPERIMENT 7: Bootstrap Benchmark
    # ============================================================================
    echo "[7/7] Running Bootstrap Benchmark..."
    echo "      Measures CKKS bootstrapping time on GPU."
    cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda \
        --example bench_cuda_bootstrap \
        2>&1 | tee $RESULTS_DIR/07_bootstrap.log
    echo ""
else
    echo "[4/7] Skipping V3 CUDA Inference (cpu mode)"
    echo "[5/7] Skipping V3 CUDA Benchmark (cpu mode)"
    echo "[6/7] Skipping CKKS Operations Benchmark (cpu mode)"
    echo "[7/7] Skipping Bootstrap Benchmark (cpu mode)"
    echo ""
fi

# ============================================================================
# Generate Summary
# ============================================================================
echo "Generating summary..."
cat > $RESULTS_DIR/SUMMARY.md << EOF
# Experiment Reproduction Results

**Date:** $(date)
**Mode:** $MODE

## Key Metrics

### Accuracy (Experiment 2)
EOF

grep -E "Encrypted accuracy|Plaintext accuracy|Prediction agreement|Accuracy gap|Max CKKS error" \
    $RESULTS_DIR/02_encrypted_accuracy.log >> $RESULTS_DIR/SUMMARY.md 2>/dev/null || echo "Not available" >> $RESULTS_DIR/SUMMARY.md

if [ "$MODE" != "cpu" ]; then
    cat >> $RESULTS_DIR/SUMMARY.md << EOF

### Performance (Experiment 5)
EOF
    grep -E "Per product|Total time|Speedup|N=" \
        $RESULTS_DIR/05_v3_cuda_benchmark.log >> $RESULTS_DIR/SUMMARY.md 2>/dev/null || echo "Not available" >> $RESULTS_DIR/SUMMARY.md

    cat >> $RESULTS_DIR/SUMMARY.md << EOF

### Bootstrap (Experiment 7)
EOF
    grep -E "Total bootstrap|EvalMod|CoeffToSlot|SlotToCoeff" \
        $RESULTS_DIR/07_bootstrap.log >> $RESULTS_DIR/SUMMARY.md 2>/dev/null || echo "Not available" >> $RESULTS_DIR/SUMMARY.md
fi

cat >> $RESULTS_DIR/SUMMARY.md << EOF

## Files Generated

$(ls -la $RESULTS_DIR/*.log 2>/dev/null | awk '{print "- " $NF}')

## Reproduction Command

\`\`\`bash
./scripts/reproduce_experiments.sh $MODE
\`\`\`
EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Experiments Complete!                                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Files:"
ls -la $RESULTS_DIR/
echo ""
echo "Summary:"
cat $RESULTS_DIR/SUMMARY.md
echo ""
