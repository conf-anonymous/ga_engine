# Testing Guide

Complete guide for running all tests in GA Engine.

## Quick Start

### Run All Tests (CPU)
```bash
cargo test --lib --features v1,v2,v3,v5
```

**Expected Result**: `ok. ~223 passed; 0 failed; 0 ignored; 0 measured`

**Time**: ~75 seconds

### Run GPU Tests

**Metal GPU (Apple Silicon):**
```bash
cargo test --release --features v2,v2-gpu-metal,v3
```

**CUDA GPU (NVIDIA):**
```bash
cargo test --release --features v2,v2-gpu-cuda,v3
```

## Test Breakdown

### 1. V3 Bootstrap Tests (52 tests)
```bash
cargo test --lib --features v2,v3 clifford_fhe_v3
```

**What's Tested**:
- Rotation infrastructure (Galois automorphisms)
- CoeffToSlot/SlotToCoeff transforms
- Diagonal matrix multiplication
- EvalMod (homomorphic modular reduction)
- ModRaise (modulus chain extension)
- Bootstrap pipeline integration

**Result**: 52/52 passing (100%)

### 2. V2 Backend Tests (127 tests)
```bash
cargo test --lib --features v2 clifford_fhe_v2
```

**What's Tested**:
- NTT (Number Theoretic Transform)
- RNS (Residue Number System)
- CKKS encryption scheme
- Key generation and management
- Polynomial multiplication
- Geometric operations

**Result**: 127/127 passing (100%)

### 3. V1 Baseline Tests (31 tests)
```bash
cargo test --lib --features v1 clifford_fhe_v1
```

**What's Tested**:
- Reference CKKS implementation
- Canonical embedding
- Slot encoding
- Automorphisms
- Geometric neural network
- Rotation keys

**Result**: 31/31 passing (100%)

### 4. V5 Privacy Analysis Tests (~10 tests)
```bash
cargo test --lib --features v5 clifford_fhe_v5
```

**What's Tested**:
- Execution trace collection
- Privacy attack implementations
- Dimension inference attacks
- Trace metadata analysis

**Result**: All passing

### 5. Lattice Reduction Tests (~60 tests)
```bash
cargo test --lib lattice_reduction --features lattice-reduction
```

**What's Tested**:
- Gram-Schmidt orthogonalization
- LLL reduction
- BKZ reduction
- Enumeration algorithms
- GA-accelerated lattice operations

**Note**: Optional feature - not required for FHE operations

**Result**: All passing

## GPU Bootstrap Testing

### Metal GPU (Apple Silicon)

#### Build V3 with Metal Backend
```bash
cargo build --release --features v2,v2-gpu-metal,v3
```

#### Run Metal Bootstrap Tests
```bash
# Unit tests
cargo test --release --features v2,v2-gpu-metal,v3

# Full bootstrap example (100% GPU)
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native

# Expected results:
# - Total bootstrap: ~60s
# - CoeffToSlot: ~50s
# - SlotToCoeff: ~12s
# - Error: ~3.6e-3
# - Status: Production Stable
```

#### Metal Validation Tests
```bash
# GPU rescaling golden compare (bit-exact verification)
cargo run --release --features v2,v2-gpu-metal,v3 --example test_rescale_golden_compare

# Expected: 0 mismatches (bit-exact with CPU)

# Layout conversion test
cargo run --release --features v2,v2-gpu-metal,v3 --example test_multiply_rescale_layout
```

### CUDA GPU (NVIDIA)

#### Build V3 with CUDA Backend
```bash
# Set CUDA environment
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Build
cargo build --release --features v2,v2-gpu-cuda,v3
```

#### Run CUDA Bootstrap Tests
```bash
# Unit tests
cargo test --release --features v2,v2-gpu-cuda,v3

# Full bootstrap example (100% GPU with relinearization)
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap

# Monitor GPU usage in separate terminal:
nvidia-smi -l 1

# Expected results:
# - Total bootstrap: ~11.95s
# - EvalMod: ~11.76s (98% of time)
# - CoeffToSlot: ~0.15s
# - SlotToCoeff: ~0.04s
# - Error: ~1e-3
# - GPU utilization: >90% during EvalMod
# - Status: Production Stable
```

#### CUDA Performance Verification
```bash
# Verify GPU is being utilized
nvidia-smi -l 1  # Run in separate terminal during test

# Expected during bootstrap:
# - GPU Utilization: >80%
# - Memory Usage: 2-4GB
# - Temperature: Varies by hardware
```

## Specialized Test Commands

### Bootstrap-Specific Tests

#### Rotation Tests
```bash
cargo test --lib clifford_fhe_v3::bootstrapping::rotation --features v2,v3 -- --nocapture
```

#### CoeffToSlot/SlotToCoeff Tests
```bash
cargo test --lib clifford_fhe_v3::bootstrapping::coeff_to_slot --features v2,v3 -- --nocapture
cargo test --lib clifford_fhe_v3::bootstrapping::slot_to_coeff --features v2,v3 -- --nocapture
```

#### EvalMod Tests
```bash
cargo test --lib clifford_fhe_v3::bootstrapping::eval_mod --features v2,v3 -- --nocapture
```

#### Bootstrap Context Tests
```bash
cargo test --lib clifford_fhe_v3::bootstrapping::bootstrap_context --features v2,v3 -- --nocapture
```

## Build Verification

### Compile All Examples
```bash
cargo build --release --features v2,v3 --examples
```

**Time**: ~60 seconds (CPU only)

**Expected**: `Finished release [optimized] target(s)`

### Compile GPU Examples
```bash
# Metal
cargo build --release --features v2,v2-gpu-metal,v3 --examples

# CUDA
cargo build --release --features v2,v2-gpu-cuda,v3 --examples
```

## Performance Testing

### CPU Backend Performance

#### V2 CPU Example
```bash
cargo run --release --features v2 --example encrypted_3d_classification

# Expected: ~300ms per geometric product (38× faster than V1)
```

#### V3 CPU Bootstrap
```bash
cargo run --release --features v2,v3 --example test_v3_full_bootstrap

# Expected: ~70s total bootstrap
```

### GPU Backend Performance

#### Metal GPU Bootstrap
```bash
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native

# Expected: ~60s total bootstrap
# Speedup: 1.17× vs CPU
```

#### CUDA GPU Bootstrap
```bash
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap

# Expected: ~11.95s total bootstrap
# Speedup: 5.86× vs CPU, ~5× vs Metal
```

## Test Output Interpretation

### Successful Test Run
```
running 210 tests
test clifford_fhe_v1::... ok
test clifford_fhe_v2::... ok
test clifford_fhe_v3::... ok
...

test result: ok. 210 passed; 0 failed; 0 ignored; 0 measured; finished in 70.22s
```

### Successful GPU Bootstrap
```
╔═══════════════════════════════════════════════════════════════╗
║           V3 CUDA GPU Bootstrap Test                         ║
╚═══════════════════════════════════════════════════════════════╝

Step 1: Initializing parameters
- Generated 30 NTT-friendly primes

Step 2: Initializing CUDA contexts
- CUDA device initialized

Step 3: Generating keys
- Keys generated

Step 4: Encrypting input
- Input encrypted

Step 5: Running bootstrap
  CoeffToSlot completed in 0.15s
  EvalMod completed in 11.76s
  SlotToCoeff completed in 0.04s

═══════════════════════════════════════════════════════════════
Bootstrap completed in 11.95s
═══════════════════════════════════════════════════════════════

V3 CUDA GPU BOOTSTRAP COMPLETE
   Full implementation with relinearization.
```

## Clean Build Testing

### Complete Clean Rebuild
```bash
cargo clean
cargo test --lib --features v1,v2,v3
```

**Purpose**: Verify everything builds from scratch (useful after pulling updates)

**Time**: ~3-5 minutes (first build) + ~70 seconds (tests)

### Clean GPU Rebuild
```bash
cargo clean

# Metal
cargo build --release --features v2,v2-gpu-metal,v3

# CUDA
cargo build --release --features v2,v2-gpu-cuda,v3
```

**Time**: ~3-5 minutes (includes shader/kernel compilation)

## Troubleshooting

### GPU Tests Not Running

**Metal GPU:**
```bash
# Verify Metal support
system_profiler SPDisplaysDataType | grep Metal

# Should show: Metal: Supported

# Install Xcode Command Line Tools if needed
xcode-select --install
```

**CUDA GPU:**
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Check environment variables
echo $CUDA_PATH
echo $LD_LIBRARY_PATH

# Set if needed
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### Test Timeouts

**Symptom**: Tests hang or timeout

**Solution**: Always use `--release` for performance-intensive tests:
```bash
cargo test --release --lib --features v2,v3
```

### Specific Test Failures

**Symptom**: One test fails, want to debug

**Solution**: Run in isolation with output:
```bash
cargo test --lib test_name --features v2,v3 -- --nocapture
```

### CUDA GPU Not Being Utilized

**Symptom**: Bootstrap runs but GPU shows 0% usage

**Solution**:
```bash
# Rebuild CUDA kernels from scratch
cargo clean
cargo build --release --features v2,v2-gpu-cuda,v3

# Verify CUDA device is accessible
nvidia-smi

# Run with GPU monitoring
nvidia-smi -l 1  # In separate terminal
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
```

### Lattice Reduction Build Issues

**Symptom**: Build stuck at netlib-src compilation

**Solution**: Lattice reduction is optional - skip it:
```bash
cargo clean
cargo build --release --features v2,v3

# Or install system BLAS (Linux)
sudo apt-get install libblas-dev liblapack-dev
```

## Continuous Integration

### Pre-Commit Checklist
```bash
# 1. Run full test suite
cargo test --lib --features v1,v2,v3,v5

# 2. Build all examples
cargo build --release --features v2,v3,v5 --examples

# 3. Format code
cargo fmt

# 4. Run clippy
cargo clippy --features v2,v3,v5 -- -D warnings
```

**All should pass before committing.**

### GPU-Specific Pre-Commit
```bash
# Metal (on macOS)
cargo test --release --features v2,v2-gpu-metal,v3
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native

# CUDA (on Linux/Windows with NVIDIA GPU)
cargo test --release --features v2,v2-gpu-cuda,v3
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
```

## Test Statistics

### CPU Tests

| Category | Tests | Pass Rate | Time |
|----------|-------|-----------|------|
| V1 Baseline | 31 | 100% | ~5s |
| V2 Optimized | 127 | 100% | ~15s |
| V3 Bootstrap | 52 | 100% | ~40s |
| V5 Privacy | ~10 | 100% | ~5s |
| Lattice Reduction | ~60 | 100% | ~5s |
| **Total (no lattice)** | **~223** | **100%** | **~75s** |
| **Total (with lattice)** | **~283** | **100%** | **~80s** |

### GPU Bootstrap Performance

| Backend | Hardware | Total Time | Speedup vs CPU | Status |
|---------|----------|------------|----------------|--------|
| V3 CPU | Apple M3 Max | ~70s | 1× | Reference |
| V3 Metal GPU | Apple M3 Max | ~60s | 1.17× | Production Stable |
| V3 CUDA GPU | NVIDIA GPU | ~11.95s | 5.86× | Production Stable |

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture (V3 uses V2 backend)
- [COMMANDS.md](COMMANDS.md) - Complete command reference
- [FEATURE_FLAGS.md](FEATURE_FLAGS.md) - Feature flag reference
- [INSTALLATION.md](INSTALLATION.md) - Installation and setup
- [BENCHMARKS.md](BENCHMARKS.md) - Performance measurements
- [README.md](README.md) - Project overview

## Support

For test failures or questions:
- Check [COMMANDS.md](COMMANDS.md) troubleshooting section
- Review test output carefully (use `--nocapture` for details)
- Verify GPU drivers and CUDA/Metal are properly installed
- *Issue tracker withheld for anonymous review*
