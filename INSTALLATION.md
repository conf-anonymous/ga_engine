# Installation Guide

Complete setup instructions for GA Engine across all platforms and backends.

## System Requirements

### Minimum Requirements
- **CPU**: Multi-core x86_64 or ARM64 processor
- **RAM**: 4GB
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows
- **Rust**: 1.75 or newer

### Recommended Configuration
- **CPU**: 8+ cores (Apple M3/M4, AMD Ryzen 9, Intel Core i9)
- **RAM**: 16GB+ (32GB for large datasets)
- **OS**: macOS 14+ (for Metal) or Linux (for CUDA)

### GPU Backend Requirements

**Metal GPU** (Apple Silicon):
- macOS 13.0+
- Apple M1/M2/M3/M4 chip
- Xcode Command Line Tools

**CUDA GPU** (NVIDIA):
- NVIDIA GPU with Compute Capability 7.0+ (RTX 20xx series or newer)
- CUDA Toolkit 11.0+ (12.0+ recommended)
- Linux (Ubuntu 22.04+ recommended) or Windows
- NVIDIA drivers 450.80.02+ (Linux) or 452.39+ (Windows)

## Installation Steps

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version  # Verify ≥1.75
```

### 2. Clone Repository

```bash
git clone https://github.com/conf-anonymous/ga_engine.git
cd ga_engine
```

### 3. Choose Backend

#### Option A: CPU Backend (Recommended for Local Development)

```bash
# Build V2 CPU backend
cargo build --release --features v2

# Run tests
cargo test --lib --features v2

# Run example
cargo run --release --features v2 --example encrypted_3d_classification
```

**Performance**: ~300ms geometric product (38× faster than V1)

#### Option B: Metal GPU (Apple Silicon)

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify installation
xcode-select -p

# Build V2 with Metal support
cargo build --release --features v2,v2-gpu-metal

# Run Metal GPU tests
cargo test --release --features v2,v2-gpu-metal

# Run Metal GPU example
cargo run --release --features v2,v2-gpu-metal --example test_metal_gpu_bootstrap_native
```

**Performance**:
- Geometric product: ~33ms (346× faster than V1)
- V3 Bootstrap: ~60s (100% GPU)

#### Option C: CUDA GPU (NVIDIA)

```bash
# 1. Install CUDA Toolkit
# Visit: https://developer.nvidia.com/cuda-downloads
# Or use package manager (Ubuntu):
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# 2. Verify CUDA installation
nvcc --version
nvidia-smi

# 3. Set CUDA environment variables
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_PATH/bin:$PATH

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export CUDA_PATH=/usr/local/cuda' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=$CUDA_PATH/bin:$PATH' >> ~/.bashrc

# 4. Build with CUDA support
cargo build --release --features v2,v2-gpu-cuda

# 5. Run CUDA GPU tests
cargo test --release --features v2,v2-gpu-cuda

# 6. Verify GPU is being utilized
nvidia-smi -l 1  # Monitor GPU usage in real-time (in separate terminal)
```

**Performance**:
- Geometric product: ~5.7ms (2,002× faster than V1, 6× faster than Metal)
- V3 Bootstrap: ~11.95s (100% GPU with relinearization, 5× faster than Metal)

#### Option D: V3 Bootstrap - CUDA GPU (RECOMMENDED for NVIDIA GPUs)

```bash
# Build V3 with CUDA backend
cargo build --release --features v2,v2-gpu-cuda,v3

# Run CUDA GPU bootstrap
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap

# Expected results:
# - Total bootstrap: ~11.95s
# - EvalMod: ~11.76s (98% of time)
# - CoeffToSlot: ~0.15s
# - SlotToCoeff: ~0.04s
# - Error: ~1e-3 (excellent accuracy)
# - 100% GPU execution with relinearization
```

#### Option E: V3 Bootstrap - Metal GPU (Apple Silicon)

```bash
# Build V3 with Metal backend
cargo build --release --features v2,v2-gpu-metal,v3

# Run Metal GPU bootstrap native (100% GPU) - PRODUCTION READY
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native

# Expected results:
# - Total bootstrap: ~60s
# - CoeffToSlot: ~50s
# - SlotToCoeff: ~12s
# - Error: ~3.6e-3 (excellent accuracy)
# - 100% GPU execution

# Validation tests
cargo run --release --features v2,v2-gpu-metal,v3 --example test_rescale_golden_compare
```

#### Option F: V3 Bootstrap - CPU Reference

```bash
# Build V3 with CPU backend
cargo build --release --features v2,v3

# Run CPU bootstrap (reference implementation)
cargo run --release --features v2,v3 --example test_v3_full_bootstrap

# Expected results:
# - Total bootstrap: ~70s
# - Error: ~3.6e-3
```

#### Option G: V5 Privacy Analysis (Standalone)

V5 provides **execution-trace collection and privacy analysis** for researching side-channel leakage in FHE computations.

```bash
# Build V5 (standalone, no GPU required)
cargo build --release --features v5

# Run comprehensive attack suite
cargo run --release --features v5 --example v5_privacy_attacks

# Run dimension inference attack only
cargo run --release --features v5 --example v5_dimension_attack

# Expected results:
# - CliffordFHE: 0% information leakage (dimension inference)
# - CKKS baseline: 100% information leakage
# - CliffordFHE wins 4 attacks, ties 2
```

**V5 with GPU tracing:**
```bash
# V5 with Metal GPU tracing (Apple Silicon)
cargo build --release --features v5,v2-gpu-metal
cargo run --release --features v5,v2-gpu-metal --example v5_trace_collector -- --metal

# V5 with CUDA GPU tracing (NVIDIA)
cargo build --release --features v5,v2-gpu-cuda
cargo run --release --features v5,v2-gpu-cuda --example v5_trace_collector -- --cuda
```

**Note**: V5 is standalone and does NOT require V2 (unlike V3/V4). GPU tracing is optional for collecting execution traces on GPU backends.

## Verification

### Quick Test

```bash
# V2 CPU (38× speedup over V1)
cargo run --release --features v2 --example encrypted_3d_classification

# Expected output: Geometric product completes in ~300ms
```

### GPU Bootstrap Test

**CUDA GPU:**
```bash
# Run CUDA bootstrap test
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap

# Monitor GPU usage in separate terminal:
nvidia-smi -l 1

# Expected: GPU utilization >90% during bootstrap
```

**Metal GPU:**
```bash
# Run Metal bootstrap test
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native

# Monitor GPU usage:
sudo powermetrics --samplers gpu_power -i 1000 -n 1
```

### Full Test Suite

```bash
# V1 Baseline (31 tests)
cargo test --lib --features v1

# V2 Optimized (127 tests)
cargo test --lib --features v2

# V3 Bootstrapping (52 tests - 100% passing)
cargo test --lib --features v2,v3 clifford_fhe_v3

# V5 Privacy Analysis (~10 tests)
cargo test --lib --features v5 clifford_fhe_v5

# All versions (V1 + V2 + V3 + V5 = ~223 tests without lattice-reduction)
cargo test --lib --features v1,v2,v3,v5
```

## Troubleshooting

### Metal GPU Issues

**Problem**: "Metal not found" error

**Solution**:
```bash
xcode-select --install
# Restart terminal
xcode-select -p  # Verify installation
```

**Problem**: Metal tests fail with "GPU not found"

**Solution**:
```bash
# Verify Metal support
system_profiler SPDisplaysDataType | grep Metal

# Should show: Metal: Supported, feature set macOS GPUFamily2 v1
```

### CUDA Issues

**Problem**: "CUDA not found" or linking errors

**Solution**:
```bash
# 1. Verify CUDA installation
nvcc --version
nvidia-smi

# 2. Check CUDA libraries exist
ls $CUDA_PATH/lib64/libcudart.so
ls $CUDA_PATH/lib64/libcuda.so

# 3. Ensure environment variables are set
echo $CUDA_PATH
echo $LD_LIBRARY_PATH

# 4. Add to shell profile if not persistent
echo 'export CUDA_PATH=/usr/local/cuda' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 5. Rebuild from clean
cargo clean
cargo build --release --features v2,v2-gpu-cuda,v3
```

**Problem**: CUDA build succeeds but runtime fails with "libcuda.so.1: cannot open shared object file"

**Solution**:
```bash
# Install NVIDIA drivers if not already installed
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

# Or manually install specific version
sudo apt-get install nvidia-driver-535

# Reboot
sudo reboot

# Verify after reboot
nvidia-smi
```

**Problem**: CUDA GPU not being utilized during bootstrap

**Solution**:
```bash
# Monitor GPU usage in real-time
nvidia-smi -l 1

# Check CUDA device is accessible
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap

# Expected: GPU utilization should be >80% during EvalMod phase
# If GPU shows 0% usage, there may be a CUDA initialization issue

# Try rebuilding CUDA kernels
cargo clean
cargo build --release --features v2,v2-gpu-cuda,v3
```

### Build Performance

For faster builds:
```bash
# Use more cores (adjust based on your CPU)
export CARGO_BUILD_JOBS=8

# Build specific examples only
cargo build --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap

# Use Rust nightly for faster compile times (optional)
rustup default nightly
cargo build --release --features v2,v2-gpu-cuda,v3
rustup default stable  # Switch back if needed
```

### Lattice Reduction Issues

**Problem**: Build stuck at netlib-src compilation

**Solution**:
```bash
# Lattice reduction is optional - skip it for GPU work
cargo clean
cargo build --release --features v2,v2-gpu-cuda,v3

# Note: Default features include lattice-reduction
# Use --no-default-features or omit lattice-reduction to avoid
```

**Problem**: Want to include lattice reduction but build is slow

**Solution** (Linux):
```bash
# Install system BLAS to avoid compiling from source
sudo apt-get install libblas-dev liblapack-dev
cargo clean
cargo build --release --features v2,v2-gpu-cuda,v3,lattice-reduction
```

**Solution** (macOS):
```bash
# macOS includes Accelerate framework - should be fast
xcode-select --install
cargo clean
cargo build --release --features v2,v2-gpu-metal,v3,lattice-reduction
```

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

```bash
# Install build essentials
sudo apt-get update
sudo apt-get install build-essential pkg-config libssl-dev

# For CUDA support
sudo apt-get install nvidia-driver-535
sudo apt-get install cuda-toolkit-12-3

# Verify CUDA
nvidia-smi
nvcc --version
```

### macOS

```bash
# Install Xcode Command Line Tools (required for all builds)
xcode-select --install

# For Metal GPU support (no additional steps - built-in)
cargo build --release --features v2,v2-gpu-metal,v3
```

### Windows

```bash
# Install Rust for Windows
# Download from: https://rustup.rs/

# Install CUDA Toolkit for Windows
# Download from: https://developer.nvidia.com/cuda-downloads

# Set environment variables in PowerShell:
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
$env:PATH += ";$env:CUDA_PATH\bin"

# Build with CUDA
cargo build --release --features v2,v2-gpu-cuda,v3
```

## Performance Verification

After installation, verify performance matches expected benchmarks:

### V2 CPU (Apple M3 Max)
```bash
cargo run --release --features v2 --example encrypted_3d_classification
# Expected: ~300ms geometric product
```

### V2 Metal GPU (Apple M3 Max)
```bash
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native
# Expected: ~60s bootstrap
```

### V2 CUDA GPU (NVIDIA RTX 5090)
```bash
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
# Expected: ~11.95s bootstrap
```

If performance is significantly slower, check:
1. Using `--release` flag (debug builds are 10-100× slower)
2. GPU is actually being utilized (`nvidia-smi` for CUDA)
3. No thermal throttling (check temperatures)
4. Sufficient RAM/VRAM available

## Next Steps

### Documentation
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system architecture overview
- See [COMMANDS.md](COMMANDS.md) for complete command reference
- See [FEATURE_FLAGS.md](FEATURE_FLAGS.md) for feature flag reference
- See [BENCHMARKS.md](BENCHMARKS.md) for performance measurements
- See [TESTING_GUIDE.md](TESTING_GUIDE.md) for testing procedures
- See [README.md](README.md) for project overview and quick start

### Getting Started

1. **Start with V2 CPU** to verify basic functionality
2. **Add GPU backend** (Metal or CUDA) for performance
3. **Run V3 bootstrap** to test full implementation
4. **Run V5 privacy analysis** to verify privacy guarantees
5. **Check benchmarks** to ensure performance matches expectations

## Support

*Contact information withheld for anonymous review.*
