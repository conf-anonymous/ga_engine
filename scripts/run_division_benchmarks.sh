#!/bin/bash
#
# Homomorphic Division Benchmark Suite
#
# This script runs all division-related benchmarks and saves results
# to results/division/ with proper timestamping and documentation.
#
# Usage:
#   ./scripts/run_division_benchmarks.sh
#
# Output:
#   Results are saved to results/division/run_YYYYMMDD_HHMMSS/
#

set -e  # Exit on error

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/division/run_${TIMESTAMP}"
LOG_FILE="${RESULTS_DIR}/benchmark.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
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

section() {
    echo ""
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "=== $1 ===" >> "$LOG_FILE"
}

# Create results directory
mkdir -p "$RESULTS_DIR"
touch "$LOG_FILE"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║           HOMOMORPHIC DIVISION BENCHMARK SUITE                           ║"
echo "║                    RTX 4090 / CUDA Edition                               ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Timestamp:    $TIMESTAMP"
echo "  Results dir:  $RESULTS_DIR"
echo ""

# ============================================================================
section "STEP 1: System Information"
# ============================================================================

log "Collecting system information..."

cat > "$RESULTS_DIR/system_info.txt" << EOF
# System Information for Homomorphic Division Benchmarks
# Generated: $(date)
# Run ID: $TIMESTAMP

EOF

# GPU info
if command -v nvidia-smi &> /dev/null; then
    echo "## GPU Information" >> "$RESULTS_DIR/system_info.txt"
    echo "" >> "$RESULTS_DIR/system_info.txt"
    nvidia-smi >> "$RESULTS_DIR/system_info.txt"
    echo "" >> "$RESULTS_DIR/system_info.txt"

    # Create compact GPU summary
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$RESULTS_DIR/gpu_summary.txt"
    success "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
    warn "nvidia-smi not found"
fi

# CUDA version
if command -v nvcc &> /dev/null; then
    echo "## CUDA Toolkit" >> "$RESULTS_DIR/system_info.txt"
    nvcc --version >> "$RESULTS_DIR/system_info.txt"
    echo "" >> "$RESULTS_DIR/system_info.txt"
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    success "CUDA: $CUDA_VER"
else
    warn "nvcc not found"
fi

# Rust version
echo "## Rust Toolchain" >> "$RESULTS_DIR/system_info.txt"
rustc --version >> "$RESULTS_DIR/system_info.txt"
cargo --version >> "$RESULTS_DIR/system_info.txt"
echo "" >> "$RESULTS_DIR/system_info.txt"
success "Rust: $(rustc --version | cut -d' ' -f2)"

# CPU info
echo "## CPU Information" >> "$RESULTS_DIR/system_info.txt"
if [[ -f /proc/cpuinfo ]]; then
    grep "model name" /proc/cpuinfo | head -1 >> "$RESULTS_DIR/system_info.txt"
    echo "CPU Cores: $(nproc)" >> "$RESULTS_DIR/system_info.txt"
fi
echo "" >> "$RESULTS_DIR/system_info.txt"

# Memory info
echo "## Memory" >> "$RESULTS_DIR/system_info.txt"
free -h >> "$RESULTS_DIR/system_info.txt" 2>/dev/null || echo "N/A" >> "$RESULTS_DIR/system_info.txt"

# ============================================================================
section "STEP 2: Build Verification"
# ============================================================================

log "Building with V2 CUDA features..."
cargo build --release --no-default-features --features f64,nd,v2,v2-gpu-cuda 2>&1 | tee "$RESULTS_DIR/build_v2.log" | tail -3
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    success "V2 CUDA build completed"
else
    error "V2 CUDA build failed - check $RESULTS_DIR/build_v2.log"
    exit 1
fi

log "Building with V3 bootstrap features..."
cargo build --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 2>&1 | tee "$RESULTS_DIR/build_v3.log" | tail -3
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    success "V3 bootstrap build completed"
    V3_AVAILABLE=true
else
    warn "V3 build failed - bootstrap benchmarks will be skipped"
    V3_AVAILABLE=false
fi

# ============================================================================
section "STEP 3: Validation Test"
# ============================================================================

log "Running CUDA division validation..."
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda \
    --example test_cuda_division_runpod 2>&1 | tee "$RESULTS_DIR/validation.txt"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    success "Validation passed"
else
    error "Validation failed - stopping"
    exit 1
fi

# ============================================================================
section "STEP 4: Basic Division Benchmark"
# ============================================================================

log "Running basic CUDA division benchmark..."
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda \
    --example bench_division_cuda_gpu 2>&1 | tee "$RESULTS_DIR/division_basic.txt"
success "Basic division benchmark completed"

# ============================================================================
section "STEP 5: Comprehensive Division (Multiple Ring Dimensions)"
# ============================================================================

# N=4096 (default, balanced)
log "Running comprehensive division benchmark (N=4096)..."
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda \
    --example bench_division_comprehensive 2>&1 | tee "$RESULTS_DIR/division_comprehensive_n4096.txt"
success "N=4096 benchmark completed"

# N=1024 (fast, lower security)
log "Running comprehensive division benchmark (N=1024)..."
RING_DIM=1024 cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda \
    --example bench_division_comprehensive 2>&1 | tee "$RESULTS_DIR/division_comprehensive_n1024.txt" \
    || warn "N=1024 benchmark failed or not supported"

# N=8192 (production security)
log "Running comprehensive division benchmark (N=8192)..."
RING_DIM=8192 cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda \
    --example bench_division_comprehensive 2>&1 | tee "$RESULTS_DIR/division_comprehensive_n8192.txt" \
    || warn "N=8192 benchmark failed (may require more GPU memory)"

# ============================================================================
section "STEP 6: Division with Bootstrap"
# ============================================================================

if [ "$V3_AVAILABLE" = true ]; then
    log "Running division with bootstrap benchmark..."
    cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 \
        --example bench_division_with_bootstrap 2>&1 | tee "$RESULTS_DIR/division_bootstrap.txt" \
        || warn "Division+bootstrap benchmark failed"
    success "Division+bootstrap benchmark completed"
else
    warn "Skipping bootstrap benchmarks (V3 not available)"
    echo "V3 bootstrap features not available" > "$RESULTS_DIR/division_bootstrap.txt"
fi

# ============================================================================
section "STEP 7: Vector Normalization Pipeline"
# ============================================================================

if [ "$V3_AVAILABLE" = true ]; then
    log "Running vector normalization pipeline..."
    cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 \
        --example bench_vector_normalization_pipeline 2>&1 | tee "$RESULTS_DIR/vector_normalization.txt" \
        || warn "Vector normalization benchmark failed"
    success "Vector normalization benchmark completed"
else
    warn "Skipping vector normalization (V3 not available)"
    echo "V3 bootstrap features not available" > "$RESULTS_DIR/vector_normalization.txt"
fi

# ============================================================================
section "STEP 8: Generate Summary"
# ============================================================================

log "Generating results summary..."

cat > "$RESULTS_DIR/SUMMARY.md" << EOF
# Division Benchmark Results

**Run ID:** $TIMESTAMP
**Date:** $(date)
**GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")
**CUDA:** $(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' || echo "Unknown")

## Files

| File | Description | Status |
|------|-------------|--------|
| \`validation.txt\` | CUDA division validation | $([ -s "$RESULTS_DIR/validation.txt" ] && echo "✓" || echo "✗") |
| \`division_basic.txt\` | Basic division timing | $([ -s "$RESULTS_DIR/division_basic.txt" ] && echo "✓" || echo "✗") |
| \`division_comprehensive_n1024.txt\` | N=1024 comprehensive | $([ -s "$RESULTS_DIR/division_comprehensive_n1024.txt" ] && echo "✓" || echo "—") |
| \`division_comprehensive_n4096.txt\` | N=4096 comprehensive | $([ -s "$RESULTS_DIR/division_comprehensive_n4096.txt" ] && echo "✓" || echo "✗") |
| \`division_comprehensive_n8192.txt\` | N=8192 comprehensive | $([ -s "$RESULTS_DIR/division_comprehensive_n8192.txt" ] && echo "✓" || echo "—") |
| \`division_bootstrap.txt\` | Division + bootstrap chains | $([ -s "$RESULTS_DIR/division_bootstrap.txt" ] && echo "✓" || echo "—") |
| \`vector_normalization.txt\` | Vector normalization pipeline | $([ -s "$RESULTS_DIR/vector_normalization.txt" ] && echo "✓" || echo "—") |

## Key Results

*(Extract from benchmark outputs and fill in below)*

### Division Performance by Ring Dimension

| N | Primes | 2 iter (ms) | 3 iter (ms) | Rel. Error |
|---|--------|-------------|-------------|------------|
| 1024 | 5 | | | |
| 4096 | 7 | | | |
| 8192 | 9 | | | |

### Division + Bootstrap

| Chain Length | Bootstraps | Total Time (s) | Final Error |
|--------------|------------|----------------|-------------|
| | | | |

### Vector Normalization

| Metric | Value |
|--------|-------|
| Total Pipeline Time | |
| Max Relative Error | |
| Throughput (vectors/s) | |

## Notes

EOF

success "Summary generated"

# ============================================================================
# COMPLETE
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║              DIVISION BENCHMARK SUITE COMPLETE                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Results saved to: $RESULTS_DIR"
echo ""
echo "  Files generated:"
ls -1 "$RESULTS_DIR"/*.txt 2>/dev/null | sed 's/.*\//    /'
echo ""
echo "  Next steps:"
echo "    1. Review $RESULTS_DIR/SUMMARY.md"
echo "    2. Extract key metrics from benchmark outputs"
echo "    3. Update results/division/RESULTS_SUMMARY.md with findings"
echo ""
echo "  To archive:"
echo "    tar -czvf division_results_${TIMESTAMP}.tar.gz $RESULTS_DIR"
echo ""
