#!/bin/bash
# scripts/collect_benchmark_data.sh
#
# Comprehensive benchmark data collection
# Runs all benchmarks and collects results in JSON format

set -e

echo "======================================"
echo "Lattice Reduction Benchmark Collection"
echo "======================================"
echo ""

# Create results directory
mkdir -p results/benchmarks
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/benchmarks/$TIMESTAMP"
mkdir -p $RESULTS_DIR

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# System information
echo "[1/6] Collecting system information..."
cat > $RESULTS_DIR/system_info.txt << EOF
Hardware: $(sysctl -n machdep.cpu.brand_string)
CPU Cores: $(sysctl -n hw.ncpu)
Memory: $(sysctl -n hw.memsize | awk '{print $0/1024/1024/1024 " GB"}')
OS: $(sw_vers -productName) $(sw_vers -productVersion)
Rust: $(rustc --version)
Date: $(date)
EOF

# Run baseline benchmarks
echo "[2/6] Running LLL baseline benchmarks..."
cargo bench --bench lattice_battle 2>&1 | tee $RESULTS_DIR/lll_baseline.log

# Copy Criterion HTML reports
if [ -d "target/criterion" ]; then
    echo "[3/6] Copying Criterion HTML reports..."
    cp -r target/criterion $RESULTS_DIR/
fi

# Run correctness tests
echo "[4/6] Running correctness tests..."
cargo test --lib lattice_reduction 2>&1 | tee $RESULTS_DIR/correctness_tests.log

# Generate summary statistics
echo "[5/6] Generating summary statistics..."
python3 scripts/analyze_benchmarks.py $RESULTS_DIR/criterion || echo "Python analysis skipped (script not yet created)"

# Create README
echo "[6/6] Creating results README..."
cat > $RESULTS_DIR/README.md << EOF
# Benchmark Results: $TIMESTAMP

## System Configuration
- Hardware: Apple M3 Max
- See system_info.txt for details

## Files
- \`lll_baseline.log\` - Raw benchmark output
- \`correctness_tests.log\` - Test results
- \`criterion/\` - Detailed Criterion HTML reports

## Viewing Results
Open \`criterion/report/index.html\` in a browser to see detailed benchmark results.

## Next Steps
1. Implement GA-LLL variants
2. Run comparative benchmarks
3. Analyze speedup ratios
EOF

echo ""
echo "======================================"
echo "Data collection complete!"
echo "Results saved to: $RESULTS_DIR"
echo "======================================"
echo ""
echo "Next: View results with:"
echo "  open $RESULTS_DIR/criterion/report/index.html"
