# CliffordPointNet: Privacy-Preserving 3D Point Cloud Classification

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [API Reference](#api-reference)
6. [Benchmark Reproduction](#benchmark-reproduction)
7. [Shell Scripts](#shell-scripts)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

---

## Overview

**GA Engine** is a Rust framework for privacy-preserving computation combining:

- **CKKS Fully Homomorphic Encryption (FHE)**: Enables computation on encrypted data
- **Clifford Geometric Algebra Cl(3,0)**: Native support for 3D geometric operations
- **GPU Acceleration**: CUDA (NVIDIA) and Metal (Apple Silicon) backends

### Key Results

| Metric | Value |
|--------|-------|
| Geometric Product (N=1024, CUDA) | **17.68ms** |
| Geometric Product (N=8192, CUDA) | **45.41ms** |
| Encrypted vs Plaintext Agreement | **100%** |
| Security Level (N=8192) | 128-bit post-quantum |

---

## Installation

### Prerequisites

- **Rust 1.75+**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **CUDA 12.0+** (optional, for NVIDIA GPU acceleration)
- **macOS** (for Metal GPU acceleration)

### Build Commands

```bash
# Clone repository
git clone <repository-url>
cd ga_engine_anonymous

# CPU-only build
cargo build --release --features v2

# CUDA GPU build (NVIDIA)
cargo build --release --no-default-features --features f64,nd,v2,v2-gpu-cuda

# Metal GPU build (Apple Silicon)
cargo build --release --features v2,v2-gpu-metal

# Full build with V3 bootstrapping
cargo build --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda

# All features (CPU + all GPUs + V3 + V4)
cargo build --release --features v2,v3,v4,v2-gpu-metal
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `v2` | Core V2 optimized CKKS implementation |
| `v3` | CKKS bootstrapping for unlimited depth |
| `v4` | Packed multivector layout (8x memory reduction) |
| `v2-gpu-cuda` | NVIDIA CUDA GPU acceleration |
| `v2-gpu-metal` | Apple Metal GPU acceleration |
| `f64` | 64-bit floating point (recommended) |
| `nd` | N-dimensional array support |

---

## Quick Start

### Basic Encrypted Geometric Product

```rust
use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CkksContext;
use ga_engine::clifford_fhe_v3::batched::{encode_batch, decode_batch, geometric_product_batched};

// 1. Create CKKS context
let params = CkksParams::new(1024, 3, 40.0); // N=1024, 3 primes, scale=2^40
let ctx = CkksContext::new(&params)?;

// 2. Generate keys
let (sk, pk) = ctx.generate_keys();
let relin_keys = ctx.generate_relin_keys(&sk);
let rot_keys = ctx.generate_rotation_keys(&sk);

// 3. Encode multivectors (8 components each)
let multivectors: Vec<[f64; 8]> = points.iter()
    .map(|p| [1.0, p.x, p.y, p.z, 0.0, 0.0, 0.0, 0.0])
    .collect();

// 4. Batch encode and encrypt
let plaintext = encode_batch(&multivectors, ctx.scale());
let ciphertext = ctx.encrypt(&plaintext, &pk);

// 5. Compute encrypted geometric product
let result_ct = geometric_product_batched(&ciphertext, &ciphertext, &ctx, &relin_keys, &rot_keys)?;

// 6. Decrypt and decode
let result_pt = ctx.decrypt(&result_ct, &sk);
let result_mvs = decode_batch(&result_pt);
```

### CliffordPointNet Encrypted Inference

```rust
use ga_engine::clifford_pointnet::{GPFeatureClassifier, encode_point_augmented, train_gp_classifier};

// 1. Train classifier on plaintext data
let classifier = train_gp_classifier(&train_data, num_classes, hidden_dim, epochs)?;

// 2. Encode point cloud as multivectors
let points: Vec<[f64; 3]> = load_point_cloud();
let multivectors: Vec<[f64; 8]> = points.iter()
    .map(|p| encode_point_augmented(p[0], p[1], p[2]))
    .collect();

// 3. Encrypt and compute GP features on server
let encrypted_features = server_compute_gp_features(&multivectors, &ctx, &keys)?;

// 4. Client decrypts and classifies
let plaintext_features = client_decrypt(&encrypted_features, &sk);
let class = classifier.predict(&plaintext_features);
```

---

## Architecture

### Version Hierarchy

```
V1 (Baseline)
 └── Reference implementation, O(n^2) algorithms
     ├── 31 unit tests
     └── ~11s per geometric product

V2 (Production Optimized)
 ├── CPU Backend (Rayon parallelization)
 ├── Metal GPU Backend (Apple Silicon)
 └── CUDA GPU Backend (NVIDIA)
     ├── Harvey butterfly NTT
     ├── Barrett reduction
     └── RNS modular arithmetic

V3 (Unlimited Depth)
 └── CKKS Bootstrapping
     ├── CoeffToSlot / SlotToCoeff
     ├── EvalMod (polynomial sine)
     └── SIMD batched encoding
         ├── 64 MVs batched (N=1024)
         └── 512 MVs batched (N=8192)

V4 (Packed Layout)
 └── 8 components in 1 ciphertext
     ├── Butterfly packing network
     └── 8x memory reduction
```

### Module Organization

```
src/
├── clifford_fhe_v1/          # Baseline reference
├── clifford_fhe_v2/          # Optimized core
│   ├── backends/
│   │   ├── cpu_optimized/    # Rayon parallelization
│   │   ├── gpu_metal/        # Apple Metal shaders
│   │   └── gpu_cuda/         # CUDA kernels
│   │       ├── ckks.rs       # CKKS operations
│   │       ├── ntt.rs        # NTT transforms
│   │       ├── rotation_keys.rs
│   │       └── relin_keys.rs
│   ├── core.rs               # Trait definitions
│   └── params.rs             # Security parameters
├── clifford_fhe_v3/          # Bootstrapping
│   ├── bootstrapping/
│   │   ├── coeff_to_slot.rs
│   │   ├── slot_to_coeff.rs
│   │   └── eval_mod.rs
│   └── batched/
│       ├── encoding.rs       # SIMD slot packing
│       ├── extraction.rs     # Component extraction
│       └── cuda_batched.rs   # GPU-accelerated batched GP
├── clifford_fhe_v4/          # Packed multivectors
├── clifford_fhe_v5/          # Privacy trace analysis
└── clifford_pointnet/        # ML models
    ├── gp_classifier.rs      # Production classifier
    ├── simple_model.rs       # Training utilities
    └── serialization.rs      # Weight save/load
```

---

## API Reference

### Core Types

#### `CkksContext`

Central context for all CKKS operations.

```rust
impl CkksContext {
    /// Create new context with parameters
    pub fn new(params: &CkksParams) -> Result<Self>;

    /// Generate secret/public key pair
    pub fn generate_keys(&self) -> (SecretKey, PublicKey);

    /// Generate relinearization keys (for multiplication)
    pub fn generate_relin_keys(&self, sk: &SecretKey) -> RelinKeys;

    /// Generate rotation keys (for SIMD operations)
    pub fn generate_rotation_keys(&self, sk: &SecretKey) -> RotationKeys;

    /// Encrypt a plaintext
    pub fn encrypt(&self, pt: &Plaintext, pk: &PublicKey) -> Ciphertext;

    /// Decrypt a ciphertext
    pub fn decrypt(&self, ct: &Ciphertext, sk: &SecretKey) -> Plaintext;

    /// Homomorphic addition
    pub fn add(&self, ct1: &Ciphertext, ct2: &Ciphertext) -> Ciphertext;

    /// Homomorphic multiplication (requires relinearization)
    pub fn multiply(&self, ct1: &Ciphertext, ct2: &Ciphertext, rk: &RelinKeys) -> Ciphertext;

    /// Homomorphic rotation (SIMD slot shift)
    pub fn rotate(&self, ct: &Ciphertext, steps: i32, rot_keys: &RotationKeys) -> Ciphertext;
}
```

#### `Multivector`

8-component Clifford algebra Cl(3,0) element.

```rust
pub struct Multivector {
    /// Components: [scalar, e1, e2, e3, e12, e13, e23, e123]
    pub components: [f64; 8],
}

impl Multivector {
    /// Create from scalar
    pub fn scalar(s: f64) -> Self;

    /// Create from vector (e1, e2, e3 components)
    pub fn vector(x: f64, y: f64, z: f64) -> Self;

    /// Geometric product: self * other
    pub fn geometric_product(&self, other: &Self) -> Self;

    /// Reverse (conjugate): flip sign of grade 2,3 components
    pub fn reverse(&self) -> Self;

    /// Norm squared: self * self.reverse()
    pub fn norm_squared(&self) -> f64;
}
```

### V3 Batched API

#### `encode_batch` / `decode_batch`

Pack multiple multivectors into SIMD slots.

```rust
/// Encode N multivectors into a single plaintext
/// Layout: slot[i*8 + j] = mv[i].components[j]
pub fn encode_batch(multivectors: &[[f64; 8]], scale: f64) -> Plaintext;

/// Decode plaintext back to multivectors
pub fn decode_batch(plaintext: &Plaintext) -> Vec<[f64; 8]>;
```

#### `geometric_product_batched`

Encrypted geometric product on batched multivectors.

```rust
/// Compute GP(a, b) for all multivectors in batch
/// Performs 64 multiplications + rotations + additions
pub fn geometric_product_batched(
    ct_a: &Ciphertext,
    ct_b: &Ciphertext,
    ctx: &CkksContext,
    relin_keys: &RelinKeys,
    rot_keys: &RotationKeys,
) -> Result<Ciphertext>;

/// CUDA-accelerated version
pub fn geometric_product_batched_cuda(
    ct_a: &Ciphertext,
    ct_b: &Ciphertext,
    ctx: &CudaCkksContext,
    relin_keys: &CudaRelinKeys,
    rot_keys: &CudaRotationKeys,
) -> Result<Ciphertext>;
```

### CliffordPointNet API

#### `GPFeatureClassifier`

Production classifier operating on GP features.

```rust
impl GPFeatureClassifier {
    /// Create new classifier
    pub fn new(input_dim: usize, hidden_dim: usize, num_classes: usize) -> Self;

    /// Forward pass: features -> logits
    pub fn forward(&self, features: &[f64; 8]) -> Vec<f64>;

    /// Predict class (argmax of logits)
    pub fn predict(&self, features: &[f64; 8]) -> usize;

    /// Training step with gradient descent
    pub fn train_step(&mut self, features: &[f64; 8], target: usize, lr: f64);

    /// Save weights to JSON file
    pub fn save_weights(&self, path: &str) -> Result<()>;

    /// Load weights from JSON file
    pub fn load_weights(path: &str) -> Result<Self>;
}

/// Train classifier from scratch
pub fn train_gp_classifier(
    train_data: &[(Vec<[f64; 3]>, usize)],  // (point_cloud, class)
    num_classes: usize,
    hidden_dim: usize,
    epochs: usize,
) -> GPFeatureClassifier;
```

#### `encode_point_augmented`

Convert 3D point to augmented multivector.

```rust
/// Encode (x, y, z) as [1, x, y, z, 0, 0, 0, 0]
/// The scalar=1 ensures non-trivial GP self-product
pub fn encode_point_augmented(x: f64, y: f64, z: f64) -> [f64; 8];
```

---

## Benchmark Reproduction

### Quick Verification (5-10 minutes)

```bash
# CUDA environment (RTX 4090/5090)
./scripts/run_cuda_benchmarks.sh quick

# Expected output:
# - V3 CUDA N=1024: ~17-20ms per GP
# - V3 CUDA N=8192: ~45-50ms per GP
```

### Full Benchmark Suite (30-60 minutes)

```bash
./scripts/run_cuda_benchmarks.sh full
```

### Individual Benchmarks

```bash
# V3 Batched CPU vs CUDA
cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda \
    --example bench_v3_cuda_geometric

# All GP implementations comparison
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3,v4 \
    --example bench_all_geometric_products

# Encrypted accuracy validation (40 samples)
cargo run --release --no-default-features --features f64,nd,v2,v3 \
    --example encrypted_accuracy_benchmark

# Plaintext training (10 classes)
CLASSES=10 POINTS=256 SAMPLES=100 EPOCHS=50 \
    cargo run --release --example experiment_plaintext

# CKKS bootstrap benchmark
cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda \
    --example bench_cuda_bootstrap
```

### Expected Results

| Benchmark | Configuration | Expected Time |
|-----------|--------------|---------------|
| V3 CUDA GP (N=1024) | 64 MVs batched | 17-20ms/product |
| V3 CUDA GP (N=8192) | 512 MVs batched | 45-50ms/product |
| V3 CPU GP (N=1024) | 64 MVs batched | 50-55ms/product |
| Bootstrap (CUDA) | N=1024, 20 primes | 11-12s |
| Encrypt (CUDA) | N=4096 | 7-8ms |
| Decrypt (CUDA) | N=4096 | 3-4ms |

---

## Shell Scripts

### `scripts/run_cuda_benchmarks.sh`

Comprehensive CUDA benchmark automation.

```bash
#!/bin/bash
# Usage: ./scripts/run_cuda_benchmarks.sh [quick|full]
#
# Options:
#   quick - Essential benchmarks (~5-10 min)
#   full  - All benchmarks including stress tests (~30-60 min)

set -e

MODE=${1:-quick}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/cuda_benchmarks_$TIMESTAMP"

echo "Clifford FHE CUDA Benchmark Suite"
echo "Mode: $MODE"

mkdir -p $RESULTS_DIR

# Collect system information
nvidia-smi > $RESULTS_DIR/system_info.txt

# Build with CUDA features
CUDA_FEATURES="--no-default-features --features f64,nd,v2,v2-gpu-cuda,v4"
cargo build --release $CUDA_FEATURES

# Verify CUDA integration
cargo run --release $CUDA_FEATURES --example test_v4_cuda_basic

# Run benchmarks
cargo run --release $CUDA_FEATURES --example bench_cuda_all_ops \
    2>&1 | tee $RESULTS_DIR/all_ops.log

cargo run --release $CUDA_FEATURES --example bench_v4_cuda_geometric_quick \
    2>&1 | tee $RESULTS_DIR/v4_geometric_quick.log

# CliffordPointNet encrypted inference
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda \
    --example clifford_pointnet_encrypted \
    2>&1 | tee $RESULTS_DIR/clifford_pointnet_encrypted.log

if [ "$MODE" = "full" ]; then
    # Full geometric product benchmark
    cargo run --release $CUDA_FEATURES --example bench_v4_cuda_geometric \
        2>&1 | tee $RESULTS_DIR/v4_geometric_full.log

    # Bootstrap benchmark
    cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda \
        --example bench_cuda_bootstrap \
        2>&1 | tee $RESULTS_DIR/bootstrap_cuda.log
fi

echo "Results saved to: $RESULTS_DIR"
```

### `scripts/setup_runpod_cuda.sh`

Environment setup for RunPod cloud GPU instances.

```bash
#!/bin/bash
# Usage: ./scripts/setup_runpod_cuda.sh
#
# Run ONCE after spinning up a new RunPod pod

set -e

echo "RunPod CUDA Environment Setup"

# Check NVIDIA GPU
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Check CUDA version
nvcc --version | grep "release"

# Install Rust if not present
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Update Rust
rustup default stable
rustup update

# Install build dependencies
apt-get update && apt-get install -y \
    build-essential pkg-config libssl-dev cmake git

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Add to bashrc
cat >> ~/.bashrc << 'EOF'
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

echo "Setup Complete!"
echo "Next: cd /workspace/ga_engine && ./scripts/run_cuda_benchmarks.sh"
```

### `scripts/reproduce_experiments.sh`

Reproduce all paper results.

```bash
#!/bin/bash
# Usage: ./scripts/reproduce_experiments.sh
#
# Reproduces all experimental results from the paper

set -e

echo "Reproducing Clifford FHE Experiments"
RESULTS_DIR="results/paper_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# 1. Plaintext training accuracy (baseline)
echo "[1/5] Plaintext training..."
cargo run --release --no-default-features --features f64,nd,v2,v3 \
    --example experiment_plaintext \
    2>&1 | tee $RESULTS_DIR/plaintext_training.log

# 2. V3 CPU vs CUDA geometric product benchmark
echo "[2/5] V3 Batched GP benchmark..."
cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda \
    --example bench_v3_batched_cuda \
    2>&1 | tee $RESULTS_DIR/v3_batched_benchmark.log

# 3. End-to-end encrypted inference (CPU)
echo "[3/5] Encrypted inference (CPU)..."
cargo run --release --no-default-features --features f64,nd,v2,v3 \
    --example clifford_pointnet_v3_encrypted \
    2>&1 | tee $RESULTS_DIR/encrypted_inference_cpu.log

# 4. End-to-end encrypted inference (CUDA)
echo "[4/5] Encrypted inference (CUDA)..."
cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda \
    --example clifford_pointnet_v3_encrypted_cuda \
    2>&1 | tee $RESULTS_DIR/encrypted_inference_cuda.log

# 5. Accuracy validation (40 samples)
echo "[5/5] Accuracy validation..."
cargo run --release --no-default-features --features f64,nd,v2,v3 \
    --example encrypted_accuracy_benchmark \
    2>&1 | tee $RESULTS_DIR/accuracy_validation.log

echo "All experiments complete!"
echo "Results: $RESULTS_DIR"

# Generate summary
echo "=== SUMMARY ===" > $RESULTS_DIR/SUMMARY.txt
grep -E "accuracy|agreement|error|ms|speedup" $RESULTS_DIR/*.log >> $RESULTS_DIR/SUMMARY.txt
cat $RESULTS_DIR/SUMMARY.txt
```

---

## Examples

### Available Examples

| Example | Description | Command |
|---------|-------------|---------|
| `experiment_plaintext` | Plaintext training baseline | `cargo run --release --example experiment_plaintext` |
| `clifford_pointnet_v3_encrypted` | CPU encrypted inference | `cargo run --release --features v2,v3 --example clifford_pointnet_v3_encrypted` |
| `clifford_pointnet_v3_encrypted_cuda` | CUDA encrypted inference | `cargo run --release --features v2,v3,v2-gpu-cuda --example clifford_pointnet_v3_encrypted_cuda` |
| `encrypted_accuracy_benchmark` | Accuracy validation | `cargo run --release --features v2,v3 --example encrypted_accuracy_benchmark` |
| `privacy_preserving_detection` | Autonomous vehicle demo | `cargo run --release --features v2,v3 --example privacy_preserving_detection` |
| `medical_scan_classification` | Medical imaging demo | `cargo run --release --features v2,v3 --example medical_scan_classification` |
| `cloud_inference_service` | Multi-client service demo | `cargo run --release --features v2,v3 --example cloud_inference_service` |
| `bench_v3_cuda_geometric` | V3 batched GP benchmark | `cargo run --release --features v2,v3,v2-gpu-cuda --example bench_v3_cuda_geometric` |
| `bench_cuda_all_ops` | All CKKS operations | `cargo run --release --features v2,v2-gpu-cuda --example bench_cuda_all_ops` |
| `bench_cuda_bootstrap` | Bootstrap benchmark | `cargo run --release --features v2,v3,v2-gpu-cuda --example bench_cuda_bootstrap` |

### Example: Privacy-Preserving Object Detection

```rust
// examples/privacy_preserving_detection.rs
//
// Scenario: Autonomous vehicle sends encrypted LiDAR scan to cloud server
// Server classifies objects without seeing raw sensor data

fn main() {
    // Client: Vehicle sensors capture objects
    let objects = vec![
        Object { class: "car", points: scan_object(0) },
        Object { class: "pedestrian", points: scan_object(1) },
        Object { class: "cyclist", points: scan_object(2) },
    ];

    // Client: Encrypt each object
    let encrypted_objects: Vec<_> = objects.iter()
        .map(|obj| encrypt_point_cloud(&obj.points, &pk))
        .collect();

    // Server: Compute encrypted GP features (never sees plaintext)
    let encrypted_features: Vec<_> = encrypted_objects.iter()
        .map(|ct| geometric_product_batched(ct, ct, &ctx, &rk, &rot))
        .collect();

    // Client: Decrypt and classify locally
    for (obj, enc_feat) in objects.iter().zip(encrypted_features.iter()) {
        let features = decrypt_features(enc_feat, &sk);
        let predicted_class = classifier.predict(&features);

        println!("Object: {} -> Predicted: {}", obj.class, predicted_class);

        // Safety analysis
        if is_vulnerable_road_user(predicted_class) {
            println!("  WARNING: Vulnerable road user detected!");
        }
    }
}
```

---

## Troubleshooting

### CUDA Not Found

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Out of GPU Memory

```bash
# Check GPU memory
nvidia-smi

# Use smaller ring dimension
# N=1024: ~2GB VRAM
# N=4096: ~8GB VRAM
# N=8192: ~16GB VRAM
```

### Build Errors with BLAS/LAPACK

```bash
# Disable lattice-reduction feature
cargo build --release --no-default-features --features f64,nd,v2,v2-gpu-cuda
```

### cudarc Version Mismatch

```bash
# Check CUDA version
nvcc --version

# Project uses CUDA 12.3 by default
# Modify Cargo.toml if needed:
# cudarc = { version = "0.11", features = ["cuda-12000"] }  # For CUDA 12.0
```

### Metal Shader Compilation Errors (macOS)

```bash
# Ensure Xcode command line tools are installed
xcode-select --install

# Build with Metal
cargo build --release --features v2,v2-gpu-metal
```

---

## Security Considerations

### Post-Quantum Security

| Ring Size | Security Level | Recommended Use |
|-----------|---------------|-----------------|
| N=1024 | 80-bit | Development/testing |
| N=4096 | 110-bit | Production (classical) |
| N=8192 | 128-bit | Production (post-quantum) |

### Important Notes

1. **Research Prototype**: This implementation is for research purposes. It requires a full security audit before production deployment.

2. **Not Constant-Time**: Side-channel attacks are not mitigated. Do not use in adversarial environments without hardening.

3. **CKKS Precision**: Approximate arithmetic means results have bounded error (~10^-4 to 10^-8). Verify error bounds for your application.

4. **Key Management**: Securely store and transmit encryption keys. The examples use in-memory key generation for simplicity.

---

## Citation

*Citation information withheld for anonymous review.*

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) file
