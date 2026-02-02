#!/bin/bash
#
# Comprehensive Benchmark Suite Runner
#
# This script runs all benchmarks needed for the homomorphic division paper
# and collects results in a structured format.
#
# Usage:
#   ./scripts/run_all_benchmarks.sh
#
# Output:
#   Results are saved to results/run_YYYYMMDD_HHMMSS/
#

set -e  # Exit on error

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/run_${TIMESTAMP}"
LOG_FILE="${RESULTS_DIR}/benchmark.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1"
    echo "[$(date +%H:%M:%S)] $1" >> "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[$(date +%H:%M:%S)] ✓ $1${NC}"
    echo "[$(date +%H:%M:%S)] SUCCESS: $1" >> "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +%H:%M:%S)] ✗ $1${NC}"
    echo "[$(date +%H:%M:%S)] ERROR: $1" >> "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)] ! $1${NC}"
    echo "[$(date +%H:%M:%S)] WARNING: $1" >> "$LOG_FILE"
}

# Create results directory
mkdir -p "$RESULTS_DIR"
touch "$LOG_FILE"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║              HOMOMORPHIC DIVISION BENCHMARK SUITE                        ║"
echo "║                         RunPod Edition                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# ============================================================================
# STEP 1: Collect System Information
# ============================================================================

log "Collecting system information..."

echo "# System Information" > "$RESULTS_DIR/system_info.txt"
echo "Generated: $(date)" >> "$RESULTS_DIR/system_info.txt"
echo "" >> "$RESULTS_DIR/system_info.txt"

# GPU info
if command -v nvidia-smi &> /dev/null; then
    echo "## GPU Information" >> "$RESULTS_DIR/system_info.txt"
    nvidia-smi >> "$RESULTS_DIR/system_info.txt"
    nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv >> "$RESULTS_DIR/gpu_summary.txt"
    success "GPU info collected"
else
    warn "nvidia-smi not found, skipping GPU info"
fi

# CUDA version
if command -v nvcc &> /dev/null; then
    echo "" >> "$RESULTS_DIR/system_info.txt"
    echo "## CUDA Version" >> "$RESULTS_DIR/system_info.txt"
    nvcc --version >> "$RESULTS_DIR/system_info.txt"
    success "CUDA version collected"
else
    warn "nvcc not found, skipping CUDA version"
fi

# Rust version
echo "" >> "$RESULTS_DIR/system_info.txt"
echo "## Rust Version" >> "$RESULTS_DIR/system_info.txt"
rustc --version >> "$RESULTS_DIR/system_info.txt"
cargo --version >> "$RESULTS_DIR/system_info.txt"
success "Rust version collected"

# CPU info (limited)
echo "" >> "$RESULTS_DIR/system_info.txt"
echo "## CPU Information" >> "$RESULTS_DIR/system_info.txt"
if [[ -f /proc/cpuinfo ]]; then
    grep "model name" /proc/cpuinfo | head -1 >> "$RESULTS_DIR/system_info.txt"
    grep "cpu cores" /proc/cpuinfo | head -1 >> "$RESULTS_DIR/system_info.txt"
fi

# Memory info
echo "" >> "$RESULTS_DIR/system_info.txt"
echo "## Memory Information" >> "$RESULTS_DIR/system_info.txt"
free -h >> "$RESULTS_DIR/system_info.txt" 2>/dev/null || echo "Memory info not available" >> "$RESULTS_DIR/system_info.txt"

success "System information collected"
echo ""

# ============================================================================
# STEP 2: Build with Required Features
# ============================================================================

log "Building with CUDA support..."

cargo build --release --features v2,v2-gpu-cuda 2>&1 | tee "$RESULTS_DIR/build_v2_cuda.log"
if [ $? -eq 0 ]; then
    success "V2 CUDA build completed"
else
    error "V2 CUDA build failed"
    exit 1
fi

log "Building with V3 bootstrap support..."
cargo build --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 2>&1 | tee "$RESULTS_DIR/build_v3_cuda.log"
if [ $? -eq 0 ]; then
    success "V3 CUDA build completed"
else
    warn "V3 CUDA build failed (some benchmarks will be skipped)"
fi

echo ""

# ============================================================================
# STEP 3: Run Validation Tests
# ============================================================================

log "Running validation tests..."

cargo run --release --features v2,v2-gpu-cuda --example test_cuda_division_runpod 2>&1 | tee "$RESULTS_DIR/validation_test.txt"
if [ $? -eq 0 ]; then
    success "Validation tests passed"
else
    error "Validation tests failed"
    exit 1
fi

echo ""

# ============================================================================
# STEP 4: Run Core Division Benchmarks
# ============================================================================

log "Running comprehensive division benchmark (N=4096)..."
cargo run --release --features v2,v2-gpu-cuda --example bench_division_comprehensive 2>&1 | tee "$RESULTS_DIR/division_n4096.txt"
success "Division benchmark (N=4096) completed"

log "Running comprehensive division benchmark (N=1024)..."
RING_DIM=1024 cargo run --release --features v2,v2-gpu-cuda --example bench_division_comprehensive 2>&1 | tee "$RESULTS_DIR/division_n1024.txt" || warn "N=1024 benchmark failed"

log "Running comprehensive division benchmark (N=8192)..."
RING_DIM=8192 cargo run --release --features v2,v2-gpu-cuda --example bench_division_comprehensive 2>&1 | tee "$RESULTS_DIR/division_n8192.txt" || warn "N=8192 benchmark failed"

echo ""

# ============================================================================
# STEP 5: Run Basic Division Benchmark
# ============================================================================

log "Running basic CUDA division benchmark..."
cargo run --release --features v2,v2-gpu-cuda --example bench_division_cuda_gpu 2>&1 | tee "$RESULTS_DIR/division_basic_cuda.txt"
success "Basic division benchmark completed"

echo ""

# ============================================================================
# STEP 6: Run All Operations Benchmark
# ============================================================================

log "Running all FHE operations benchmark..."
cargo run --release --features v2,v2-gpu-cuda --example bench_cuda_all_ops 2>&1 | tee "$RESULTS_DIR/all_ops_cuda.txt" || warn "All ops benchmark failed"

echo ""

# ============================================================================
# STEP 7: Run Bootstrap Benchmark (if V3 available)
# ============================================================================

log "Running bootstrap benchmark..."
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 --example bench_cuda_bootstrap 2>&1 | tee "$RESULTS_DIR/bootstrap_cuda.txt" || warn "Bootstrap benchmark failed (V3 may not be available)"

echo ""

# ============================================================================
# STEP 8: Run Division with Bootstrap
# ============================================================================

log "Running division with bootstrap benchmark..."
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 --example bench_division_with_bootstrap 2>&1 | tee "$RESULTS_DIR/division_bootstrap.txt" || warn "Division+bootstrap benchmark failed"

echo ""

# ============================================================================
# STEP 9: Run Vector Normalization Pipeline
# ============================================================================

log "Running vector normalization pipeline benchmark..."
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 --example bench_vector_normalization_pipeline 2>&1 | tee "$RESULTS_DIR/vector_norm_pipeline.txt" || warn "Vector normalization benchmark failed"

echo ""

# ============================================================================
# STEP 10: Create Summary
# ============================================================================

log "Creating benchmark summary..."

cat > "$RESULTS_DIR/SUMMARY.md" << 'SUMMARY_EOF'
# Benchmark Results Summary

Generated: TIMESTAMP_PLACEHOLDER

## Files in this Directory

| File | Description |
|------|-------------|
| `system_info.txt` | Hardware and software configuration |
| `gpu_summary.txt` | GPU specifications |
| `validation_test.txt` | Initial validation test results |
| `division_n1024.txt` | Division benchmark with N=1024 |
| `division_n4096.txt` | Division benchmark with N=4096 |
| `division_n8192.txt` | Division benchmark with N=8192 |
| `division_basic_cuda.txt` | Basic CUDA division benchmark |
| `all_ops_cuda.txt` | All FHE operations benchmark |
| `bootstrap_cuda.txt` | V3 bootstrap performance |
| `division_bootstrap.txt` | Division with bootstrap chains |
| `vector_norm_pipeline.txt` | Vector normalization application |
| `benchmark.log` | Execution log |

## Quick Results

(Fill in after reviewing output files)

### Division Performance (N=4096, 2 iterations)

| Test | Time (ms) | Rel. Error |
|------|-----------|------------|
| 100/7 | | |
| 1000/13 | | |

### Bootstrap Performance

| Phase | Time (s) |
|-------|----------|
| Total | |

### Vector Normalization Pipeline

| Total Time (ms) | Max Error |
|-----------------|-----------|
| | |

SUMMARY_EOF

# Replace timestamp placeholder
sed -i "s/TIMESTAMP_PLACEHOLDER/$(date)/" "$RESULTS_DIR/SUMMARY.md" 2>/dev/null || \
    sed "s/TIMESTAMP_PLACEHOLDER/$(date)/" "$RESULTS_DIR/SUMMARY.md" > "$RESULTS_DIR/SUMMARY.md.tmp" && mv "$RESULTS_DIR/SUMMARY.md.tmp" "$RESULTS_DIR/SUMMARY.md"

success "Summary created"

# ============================================================================
# DONE
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                    BENCHMARK SUITE COMPLETE                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To download results:"
echo "  tar -czvf benchmark_results_${TIMESTAMP}.tar.gz $RESULTS_DIR"
echo "  # Then SCP to your local machine"
echo ""
echo "Next steps:"
echo "  1. Review $RESULTS_DIR/SUMMARY.md"
echo "  2. Fill in the quick results table"
echo "  3. Extract key numbers for paper tables"
echo ""
