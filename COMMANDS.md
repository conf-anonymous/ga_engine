# Command Reference

Complete reference of all build, test, example, and benchmark commands for GA Engine.

## Important: Feature Flags

GA Engine uses feature flags to control which components are compiled:

**Default features** (local development):
```bash
cargo build --release  # Includes: v1, lattice-reduction
```

**Recommended for development** (without lattice-reduction):
```bash
cargo build --release --features f64,nd,v1,v2 --no-default-features
```

**GPU backends**:
```bash
# Metal GPU (Apple Silicon)
cargo build --release --features v2,v2-gpu-metal,v3

# CUDA GPU (NVIDIA)
cargo build --release --features v2,v2-gpu-cuda,v3
```

**Key points**:
- See [FEATURE_FLAGS.md](FEATURE_FLAGS.md) for detailed feature flag reference
- `v3` automatically includes `v2` as a dependency
- GPU backends work with both V2 and V3
- Lattice reduction is CPU-only security analysis, not needed for FHE operations

## Table of Contents

- [Important: Feature Flags](#important-feature-flags)
- [Installation](#installation)
- [V1: Baseline Reference](#v1-baseline-reference)
- [V2: CPU Optimized](#v2-cpu-optimized)
- [V2: Metal GPU](#v2-metal-gpu)
- [V2: CUDA GPU](#v2-cuda-gpu)
- [V3: Bootstrapping](#v3-bootstrapping)
- [V4: Packed Multivector Layout](#v4-packed-multivector-layout)
- [V5: Privacy Analysis](#v5-privacy-analysis)
- [V6: parallel_lift GPU Acceleration](#v6-parallel_lift-gpu-acceleration)
- [Lattice Reduction](#lattice-reduction)
- [All Versions Combined](#all-versions-combined)
- [Quick Reference Tables](#quick-reference-tables)
- [Troubleshooting](#troubleshooting)

## Installation

### Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version  # Verify version 1.75+
```

### Clone Repository
```bash
git clone https://github.com/conf-anonymous/ga_engine.git
cd ga_engine
```

## V1: Baseline Reference

### Build V1
```bash
# Development build (without lattice-reduction, RECOMMENDED)
cargo build --features f64,nd,v1 --no-default-features

# Release build (optimized, RECOMMENDED)
cargo build --release --features f64,nd,v1 --no-default-features
```

### Test V1
```bash
# Run all V1 unit tests (31 tests)
cargo test --lib --features f64,nd,v1 --no-default-features

# Run V1 integration tests
cargo test --test clifford_fhe_integration_tests --features f64,nd,v1 --no-default-features -- --nocapture

# Run comprehensive geometric operations test suite (all 7 operations)
cargo test --test test_geometric_operations --features f64,nd,v1 --no-default-features -- --nocapture

# Run all V1 tests
cargo test --features f64,nd,v1 --no-default-features
```

### Examples V1
```bash
# Encrypted 3D classification demo (main application)
cargo run --release --features f64,nd,v1 --no-default-features --example encrypted_3d_classification

# Basic FHE encryption/decryption
cargo run --release --features f64,nd,v1 --no-default-features --example clifford_fhe_basic
```

### Performance V1
```bash
# Actual performance (Apple M3 Max, 14-core):
# - Geometric product: 11.42 seconds
# - Full network inference: 308.3 seconds (27 operations)
# - Error: <1.30e-10
```

## V2: CPU Optimized

### Build V2 CPU
```bash
# Development build (without lattice-reduction, RECOMMENDED)
cargo build --features f64,nd,v2 --no-default-features

# Release build (optimized, RECOMMENDED)
cargo build --release --features f64,nd,v2 --no-default-features
```

### Test V2 CPU
```bash
# Run all V2 unit tests (127 tests, <1 second)
cargo test --lib --features f64,nd,v2 --no-default-features

# Run specific module tests
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::ntt --features f64,nd,v2 --no-default-features -- --nocapture
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::ckks --features f64,nd,v2 --no-default-features -- --nocapture

# Run V2 geometric operations integration test
cargo test --test test_geometric_operations_v2 --features f64,nd,v2 --no-default-features -- --nocapture
```

### Examples V2 CPU
```bash
# Encrypted 3D classification (38x faster than V1)
cargo run --release --features f64,nd,v2 --no-default-features --example encrypted_3d_classification
```

### Performance V2 CPU
```bash
# Actual performance (Apple M3 Max, 14-core):
# - Geometric product: 0.30s (38x faster than V1)
# - Keygen: 0.01s
# - Error: <9.67e-8 (all operations)
```

## V2: Metal GPU

### Build V2 Metal
```bash
# Install Xcode Command Line Tools (macOS only)
xcode-select --install

# Build with Metal support (RECOMMENDED)
cargo build --release --features v2,v2-gpu-metal
```

### Test V2 Metal
```bash
# Run Metal GPU tests
cargo test --release --features v2,v2-gpu-metal
```

### Performance V2 Metal
```bash
# Actual performance (Apple M3 Max GPU):
# - Geometric product: 33ms mean (30ms min, 37ms max)
# - Speedup: 346× vs V1 (11.42s → 0.033s)
# - Speedup: 9.1× vs V2 CPU (0.30s → 0.033s)
# - Throughput: 30.3 operations/second
```

## V2: CUDA GPU

### Build V2 CUDA
```bash
# Verify CUDA installation
nvcc --version

# Set CUDA path if needed
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Build with CUDA support
cargo build --release --features v2,v2-gpu-cuda
```

### Test V2 CUDA
```bash
# Run CUDA GPU tests
cargo test --release --features v2,v2-gpu-cuda
```

### Performance V2 CUDA
```bash
# Actual performance (NVIDIA RTX 5090):
# - Geometric product: 5.7ms mean
# - Speedup: 2,002× vs V1 (11.42s → 5.7ms)
# - Speedup: 77× vs V2 CPU (441ms → 5.7ms)
# - 6× faster than Metal GPU (33ms → 5.7ms)
# - Throughput: 174.8 operations/second
```

## V3: Bootstrapping

### Build V3

```bash
# Build V3 with CPU backend
cargo build --release --features v2,v3

# Build V3 with Metal GPU backend (RECOMMENDED for Apple Silicon)
cargo build --release --features v2,v2-gpu-metal,v3

# Build V3 with CUDA GPU backend (RECOMMENDED for NVIDIA GPUs)
cargo build --release --features v2,v2-gpu-cuda,v3
```

### Test V3

```bash
# Run all V3 unit tests (52 tests, 100% passing)
cargo test --lib --features v2,v3 clifford_fhe_v3

# Run full test suite (V1 + V2 + V3 = ~200 tests without lattice-reduction)
cargo test --lib --features v2,v3

# Run V3 bootstrapping tests
cargo test --lib clifford_fhe_v3::bootstrapping --features v2,v3 -- --nocapture
```

### Examples V3 - Metal GPU (Apple Silicon)

```bash
# ==== RECOMMENDED: Metal GPU Bootstrap (PRODUCTION READY) ====

# V2 Native Bootstrap (100% GPU) - Stable
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native

# V3 CPU Bootstrap Reference (correct implementation)
cargo run --release --features v2,v2-gpu-metal,v3 --example test_v3_metal_bootstrap_correct

# ==== Validation Tests ====

# GPU rescaling golden compare (bit-exact validation)
cargo run --release --features v2,v2-gpu-metal,v3 --example test_rescale_golden_compare

# Layout conversion test
cargo run --release --features v2,v2-gpu-metal,v3 --example test_multiply_rescale_layout
```

### Examples V3 - CUDA GPU (NVIDIA)

```bash
# ==== CUDA GPU Bootstrap (PRODUCTION READY) ====

# V3 CUDA GPU Bootstrap (100% GPU with relinearization) - Stable
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap

# Results:
# - Total bootstrap: ~11.95s
# - EvalMod: ~11.76s (98% of time)
# - CoeffToSlot: ~0.15s
# - SlotToCoeff: ~0.04s
# - Error: ~1e-3 (excellent accuracy)
# - 100% GPU execution with relinearization
```

### Examples V3 - CPU Reference

```bash
# Full Bootstrap Pipeline (CPU-only)
cargo run --release --features v2,v3 --example test_v3_full_bootstrap

# Fast CPU Demo Bootstrap (N=512, completes in seconds)
cargo run --release --features v2,v3 --example test_v3_cpu_demo

# V3 Parameters Test (verify dynamic prime generation)
cargo run --release --features v2,v3 --example test_v3_parameters
```

### Performance V3

**CUDA GPU Bootstrap (NVIDIA GPU)**
```bash
# V3 CUDA GPU (100% GPU with relinearization):
# - Total bootstrap: 11.95s
# - EvalMod (9 levels): 11.76s (98% of total)
# - CoeffToSlot (9 levels): ~0.15s
# - SlotToCoeff (9 levels): ~0.04s
# - Accuracy: error = ~1e-3
# - Parameters: N=1024, 30 primes (1× 60-bit, 29× 45-bit)
# - Status: PRODUCTION STABLE
# - Key Achievement: Full bootstrap with relinearization
```

**Metal GPU Bootstrap (Apple M3 Max)**
```bash
# V2 Native (100% GPU):
# - Total bootstrap: ~60s
# - CoeffToSlot (9 levels): ~50s
# - SlotToCoeff (9 levels): ~12s
# - Accuracy: error = 3.6e-3
# - Status: PRODUCTION STABLE
# - Key Achievement: GPU rescaling with Russian peasant mul_mod_128

# Validation Tests:
# - GPU rescaling golden compare: 0 mismatches (bit-exact)
# - Hardware: Apple M3 Max GPU
```

**CPU Bootstrap (Reference)**
```bash
# CPU-only performance (N=1024, 30 primes):
# - Bootstrap operation: ~70s
# - Accuracy: error = 3.6e-3
# - Hardware: Apple M3 Max, 14-core CPU

# Fast CPU demo (N=512, 7 primes):
# - Bootstrap operation: <10 seconds
# - Use case: Quick validation and testing
```

## V4: Packed Multivector Layout

V4 implements **slot-interleaved packing** to compress 8 component ciphertexts into a single packed ciphertext, achieving 8× memory reduction. The geometric product operates on packed multivectors using Metal GPU acceleration.

### Build V4
```bash
# V4 with Metal GPU backend
cargo build --release --features v4,v2-gpu-metal
```

### Test V4
```bash
# Run V4 integration tests (matches V2 style)
cargo test --test test_geometric_operations_v4 --features v4,v2-gpu-metal --no-default-features -- --nocapture

# Clean output (filter debug messages)
cargo test --test test_geometric_operations_v4 --features v4,v2-gpu-metal --no-default-features -- --nocapture 2>&1 | \
  grep -v "ROTATION DEBUG|GALOIS DEBUG|NTT|Metal Device|Metal Max"

# Run individual tests
cargo test --test test_geometric_operations_v4 test_geometric_product_simple \
  --features v4,v2-gpu-metal --no-default-features -- --nocapture

cargo test --test test_geometric_operations_v4 test_packing_unpacking \
  --features v4,v2-gpu-metal --no-default-features -- --nocapture

cargo test --test test_geometric_operations_v4 test_geometric_product_exists \
  --features v4,v2-gpu-metal --no-default-features -- --nocapture
```

### Performance V4
```bash
# Geometric product performance (Apple M3 Max GPU):
# - Operation time: ~5.0s per packed geometric product
# - Memory usage: 8× reduction vs V3 (1 packed ciphertext instead of 8)
# - GPU utilization: ~95%+
# - Test: (1 + 2e₁) ⊗ (3e₂) = 3e₂ + 6e₁₂

# What's tested:
# 1. test_geometric_product_simple: Full geometric product on Metal GPU
# 2. test_packing_unpacking: Pack 8→1, unpack 1→8, verify values
# 3. test_geometric_product_exists: Quick smoke test
```

### V4 File Location
```bash
# Integration test file
tests/test_geometric_operations_v4.rs

# Matches V2/V3 style:
# - tests/test_geometric_operations_v2.rs ← V2
# - tests/test_geometric_operations_v3.rs ← V3 (if exists)
# - tests/test_geometric_operations_v4.rs ← V4
```

## V5: Privacy Analysis

V5 provides **execution-trace collection and privacy analysis** for researching side-channel leakage in FHE computations.

### Build V5
```bash
# Build V5 (standalone, no GPU required)
cargo build --release --features v5

# Build V5 with Metal GPU tracing
cargo build --release --features v5,v2-gpu-metal

# Build V5 with CUDA GPU tracing
cargo build --release --features v5,v2-gpu-cuda
```

### Examples V5

```bash
# ==== Comprehensive Attack Suite ====

# Run all 6 privacy attacks (dimension, task, sparsity, linkability, operation count, trace length)
cargo run --release --features v5 --example v5_privacy_attacks

# Run with verbose output
cargo run --release --features v5 --example v5_privacy_attacks -- --verbose

# Export results to JSON
cargo run --release --features v5 --example v5_privacy_attacks -- --json

# ==== Individual Attacks ====

# Dimension inference attack only
cargo run --release --features v5 --example v5_dimension_attack

# ==== Trace Collection ====

# Collect traces with CPU backend
cargo run --release --features v5 --example v5_trace_collector

# Collect traces with Metal GPU backend
cargo run --release --features v5,v2-gpu-metal --example v5_trace_collector -- --metal
```

### V5 Attack Results

The comprehensive attack suite demonstrates that CliffordFHE provides stronger execution-trace privacy than CKKS:

| Attack | CKKS Accuracy | CliffordFHE Accuracy | Winner |
|--------|---------------|----------------------|--------|
| Dimension Inference | 100.0% | 16.7% (random) | **CliffordFHE** |
| Task Identification | 100.0% | 16.7% (random) | **CliffordFHE** |
| Operation Count | 100.0% | 16.7% (random) | **CliffordFHE** |
| Trace Length | 100.0% | 16.7% (random) | **CliffordFHE** |
| Sparsity Inference | ~20% | ~20% | Tie |
| Tenant Linkability | ~50% | ~50% | Tie |

**Key Finding**: CliffordFHE wins 4 attacks, ties 2. CKKS leaks 2.585 bits of dimension entropy; CliffordFHE leaks 0 bits.

### V5 File Locations
```bash
# V5 module
src/clifford_fhe_v5/

# Example files
examples/v5_privacy_attacks.rs   # Comprehensive attack suite
examples/v5_dimension_attack.rs  # Dimension inference attack
examples/v5_trace_collector.rs   # Trace collection utility
```

## V6: parallel_lift GPU Acceleration

V6 provides **external GPU acceleration** via the parallel_lift library for 25-552× speedup on CRT operations.

**Default Behavior**: Uses **stub crates** that compile without the external repository but return errors at runtime. This allows the project to build without additional setup.

### Setup V6 (Optional)

```bash
# Step 1: Clone the parallel_lift repository
cd ..  # Go to parent directory of ga_engine
git clone <parallel_lift_url> parallel_lift
cd ga_engine

# Step 2: Enable real parallel_lift
./scripts/enable_parallel_lift.sh

# Step 3: Build with V6 CUDA support
cargo build --release --features v6-cuda
```

### Revert to Stubs

```bash
# Revert to stub crates (no external dependency)
./scripts/disable_parallel_lift.sh
```

### Build V6

```bash
# Build with V6 CUDA (uses stubs by default)
cargo build --release --features v6-cuda

# Build with V6 full (V6 + V3 bootstrapping)
cargo build --release --features v6-full
```

### Test V6 (with stubs)

```bash
# Build succeeds with stubs, but FheGpuContext::new() returns error at runtime
cargo build --release --features v6-cuda
```

### Test V6 (with real parallel_lift)

```bash
# After running ./scripts/enable_parallel_lift.sh
cargo test --release --features v6-cuda

# Run V6 speedup benchmark
cargo bench --bench v6_speedup_benchmark --features v6-cuda
```

### V6 File Locations
```bash
# Stub crates (default, no external dependency)
crates/parallel_lift_stubs/parallel_lift_core/
crates/parallel_lift_stubs/parallel_lift_cuda/

# Setup scripts
scripts/enable_parallel_lift.sh   # Enable real parallel_lift
scripts/disable_parallel_lift.sh  # Revert to stubs

# Configuration example
.cargo/config.toml.example        # Setup instructions
```

## Lattice Reduction

Lattice reduction is used for **security analysis** (cryptanalysis) of the FHE scheme. It is **not required** for FHE operations.

### Build Lattice Reduction
```bash
# With lattice reduction (default)
cargo build --release --features lattice-reduction

# Or use default features
cargo build --release
```

### Test Lattice Reduction
```bash
# Run all lattice reduction tests
cargo test --lib lattice_reduction

# Run specific module tests
cargo test --lib lattice_reduction::stable_gso --features lattice-reduction
cargo test --lib lattice_reduction::bkz_stable --features lattice-reduction
```

### Examples Lattice Reduction
```bash
# Lattice reduction examples
cargo run --release --features lattice-reduction --example test_stable_bkz
cargo run --release --features lattice-reduction --example test_lll
```

## All Versions Combined

### Build All
```bash
# Build all versions (V1-V5) without lattice reduction (RECOMMENDED)
cargo build --release --features f64,nd,v1,v2,v3,v4,v5 --no-default-features

# Build with Metal GPU support (Apple Silicon)
cargo build --release --features v2,v2-gpu-metal,v3,v4,v5

# Build with CUDA GPU support (NVIDIA)
cargo build --release --features v2,v2-gpu-cuda,v3,v4,v5
```

### Test All
```bash
# Run all tests across all versions (without lattice reduction, RECOMMENDED)
cargo test --features f64,nd,v1,v2,v3,v5 --no-default-features

# Run all tests with default features (includes lattice-reduction)
cargo test --features v1,v2,v3,v5
```

### Benchmarks
```bash
# V1 vs V2 comparison benchmark
cargo bench --bench v1_vs_v2_benchmark --features v1,v2

# Run with specific backend
cargo bench --features v2,v2-gpu-cuda
```

### Documentation
```bash
# Generate and open documentation
cargo doc --open --features v2,v3

# Generate documentation for specific version
cargo doc --open --features v1
cargo doc --open --features v2
cargo doc --open --features v2,v3
```

## Quick Reference Tables

### Feature Flags

| Feature | Description | Required For |
|---------|-------------|--------------|
| `v1` | V1 baseline reference implementation | V1 examples and tests |
| `v2` | V2 CPU-optimized backend | V2 CPU examples and tests |
| `v2-gpu-metal` | V2/V3/V4 Metal GPU backend (Apple Silicon) | Metal GPU operations (V2/V3/V4) |
| `v2-gpu-cuda` | V2/V3/V4 CUDA GPU backend (NVIDIA) | CUDA GPU operations (V2/V3/V4) |
| `v3` | V3 bootstrapping (requires `v2`) | V3 bootstrap examples and tests |
| `v4` | V4 packed multivector layout (requires `v2`) | V4 packing and geometric product |
| `v5` | V5 privacy analysis (standalone) | V5 attack suite and trace collection |
| `v6` | V6 parallel_lift base (requires `v2`) | V6 GPU acceleration |
| `v6-cuda` | V6 with CUDA (uses stubs by default) | V6 CUDA operations |
| `v6-full` | V6 + V3 bootstrapping | V6 with bootstrap support |
| `lattice-reduction` | Lattice reduction for security analysis | Lattice reduction tests |

**Important**: `v3`, `v4`, and `v6` automatically include `v2` as a dependency. `v5` is standalone (does not require `v2`). GPU backends (`v2-gpu-metal`, `v2-gpu-cuda`) work with V2, V3, V4, and V5. **V6 uses stub crates by default**—run `./scripts/enable_parallel_lift.sh` to enable real parallel_lift.

### Test Counts

| Component | Test Count | Command |
|-----------|------------|---------|
| V1 Unit Tests | 31 | `cargo test --lib --features v1` |
| V2 Unit Tests | 127 | `cargo test --lib --features v2` |
| V3 Unit Tests | 52 | `cargo test --lib --features v2,v3 clifford_fhe_v3` |
| V4 Integration Tests | 3 | `cargo test --test test_geometric_operations_v4 --features v4,v2-gpu-metal` |
| Lattice Reduction | ~60 | `cargo test --lib lattice_reduction` |
| **Total (no lattice)** | **~213** | `cargo test --lib --features v1,v2,v3` |
| **Total (with lattice)** | **~273** | `cargo test --lib --features v1,v2,v3,lattice-reduction` |

### Performance Summary

#### Geometric Product (Single Operation)

| Backend | Hardware | Time | Speedup vs V1 | Memory |
|---------|----------|------|---------------|--------|
| V1 CPU | Apple M3 Max (14-core) | 11.42s | 1× | 8 ciphertexts |
| V2 CPU | Apple M3 Max (14-core) | 0.30s | 38× | 8 ciphertexts |
| V2 Metal | Apple M3 Max GPU | 33ms | 346× | 8 ciphertexts |
| V2 CUDA | NVIDIA RTX 5090 | 5.7ms | 2,002× | 8 ciphertexts |
| **V4 Metal** | **Apple M3 Max GPU** | **~5.0s** | **2.3×** | **1 packed** (8× reduction) |

#### Bootstrap (Full Operation)

| Backend | Hardware | Total Time | Speedup vs CPU |
|---------|----------|------------|----------------|
| V3 CPU | Apple M3 Max | ~70s | 1× |
| V3 Metal GPU | Apple M3 Max | ~60s | 1.17× |
| V3 CUDA GPU | NVIDIA GPU | **11.95s** | **5.86×** |

**Key Insight**: CUDA bootstrap is ~5× faster than Metal, primarily due to different GPU architectures and optimizations.

## Troubleshooting

### Compilation Issues

**Problem**: Feature flag not recognized
```bash
# Solution: Ensure correct feature syntax
cargo build --features v2,v3  # Correct
cargo build --features v2 v3  # Incorrect
```

**Problem**: Metal not found
```bash
# Solution: Install Xcode Command Line Tools
xcode-select --install
```

**Problem**: CUDA not found
```bash
# Solution: Install CUDA Toolkit and set environment variables
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
nvcc --version  # Verify installation
```

**Problem**: CMake errors with netlib-src
```bash
# Error: "Compatibility with CMake < 3.5 has been removed from CMake"

# Solution: Automatic fix via .cargo/config.toml
cargo clean
cargo build --release --features v2,v3

# If issues persist, verify .cargo/config.toml contains:
# [env]
# CMAKE_POLICY_VERSION_MINIMUM = "3.5"
```

### Test Failures

**Problem**: Tests timing out
```bash
# Solution: Use release mode for performance-intensive tests
cargo test --release --features v1,v2,v3
```

**Problem**: Specific test failing
```bash
# Solution: Run test in isolation with nocapture
cargo test --lib specific_test_name --features v2 -- --nocapture
```

**Problem**: GPU tests failing
```bash
# Solution: Verify GPU is available and drivers are installed

# Metal (macOS):
system_profiler SPDisplaysDataType | grep Metal

# CUDA (Linux/Windows):
nvidia-smi
nvcc --version
```

### Performance Issues

**Problem**: Slower than expected performance
```bash
# Solution: Always use --release flag for benchmarking
cargo run --release --features v2 --example encrypted_3d_classification

# Solution: Verify optimization level in Cargo.toml (already configured)
# [profile.release]
# opt-level = 3
# lto = true
```

**Problem**: CUDA GPU not being utilized
```bash
# Solution: Verify CUDA is properly installed and LD_LIBRARY_PATH is set
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Check GPU usage during execution
nvidia-smi -l 1  # Monitor GPU utilization in real-time
```

### V6 parallel_lift Issues

**Problem**: "parallel_lift is not available" error at runtime
```bash
# This is expected with stub crates. To enable real parallel_lift:

# Step 1: Clone parallel_lift repository
cd ..
git clone <parallel_lift_url> parallel_lift
cd ga_engine

# Step 2: Enable real parallel_lift
./scripts/enable_parallel_lift.sh

# Step 3: Rebuild
cargo build --release --features v6-cuda
```

**Problem**: Cannot find parallel_lift repository
```bash
# The enable script checks for ../parallel_lift/rust/crates/parallel_lift_core
# Make sure parallel_lift is cloned to the correct location:
ls ../parallel_lift/rust/crates/parallel_lift_core  # Should exist
```

**Problem**: Want to revert to stubs
```bash
# Revert Cargo.toml to use stub crates
./scripts/disable_parallel_lift.sh
cargo build --release --features v6-cuda  # Now uses stubs
```

## Additional Resources

### Essential Documentation
- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and component overview
- **Installation**: See [INSTALLATION.md](INSTALLATION.md) - Setup and dependencies
- **Testing**: See [TESTING_GUIDE.md](TESTING_GUIDE.md) - Comprehensive testing procedures
- **Benchmarks**: See [BENCHMARKS.md](BENCHMARKS.md) - Performance measurements
- **Feature Flags**: See [FEATURE_FLAGS.md](FEATURE_FLAGS.md) - Complete feature flag reference
- **README**: See [README.md](README.md) - Project overview and quick start

## Support

*Contact information withheld for anonymous review.*
