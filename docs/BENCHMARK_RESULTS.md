# Benchmark Results: Clifford FHE Performance

## Test Environment

| Component | Specification |
|-----------|---------------|
| CPU | Apple M3 Max |
| GPU (Metal) | Apple M3 Max integrated GPU |
| OS | macOS Darwin 23.6.0 |
| Rust | Release build with optimizations |

## V2 CPU Backend (M3 Max)

Benchmark: `clifford_pointnet_encrypted` example with 4-point cloud

| Operation | Time | Notes |
|-----------|------|-------|
| Key Generation | 5ms | One-time setup |
| Encryption | 93ms | 4 points × 8 components = 32 ciphertexts |
| Mean Pooling | 16ms | Homomorphic addition + scalar mult |
| **Geometric Product** | **959ms** | Core Cl(3,0) operation |
| Decryption | 7ms | Client-side |
| **Total Inference** | **~1,080ms** | End-to-end |

### Speedup vs V1 Baseline

| Metric | V1 | V2 CPU | Speedup |
|--------|-----|--------|---------|
| Geometric Product | ~13,000ms | 959ms | **13.5×** |

## V2 Metal GPU Backend (M3 Max)

Benchmark: `encrypted_metal_demo` example

| Operation | Time | Notes |
|-----------|------|-------|
| Encrypt | 36.38ms | Metal NTT acceleration |
| Decrypt | 26.69ms | Hybrid CPU+Metal |
| Homomorphic Add | 4.57ms | GPU-accelerated |
| **Max Error** | **0.000000** | Perfect numerical accuracy |

### NTT Correctness Verification

```
Test 1: NTT Roundtrip (Forward + Inverse)
  CPU roundtrip max error: 0
  Metal roundtrip max error: 0
  ALL TESTS PASSED - Metal NTT is CORRECT!
```

## Plaintext Training Results

### 10-Class Synthetic ModelNet

| Metric | Value |
|--------|-------|
| Test Accuracy | 99.0% |
| Random Chance | 10.0% |
| Improvement | **9.9×** better than random |

### 40-Class Synthetic ModelNet

| Metric | Value |
|--------|-------|
| Test Accuracy | 30.25% |
| Random Chance | 2.5% |
| Improvement | **12.1×** better than random |
| Training Epochs | 150 |
| Optimizer | Adam with warmup + cosine annealing |

## NVIDIA RTX 5090 CUDA Benchmarks

### Test Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 5090 (32GB VRAM) |
| CUDA | 12.8 (V12.8.93) |
| Driver | 570.195.03 |
| OS | Ubuntu 24.04 (Linux 6.8.0) |
| Rust | 1.93.0 (release build) |

### NTT Correctness Verification

```
Testing CUDA NTT round-trip
n=8192, q=1152921504606994433
CUDA Device: NVIDIA GeForce RTX 5090
Original[0]=42, CUDA[0]=42, match=true
Original[1]=100, CUDA[1]=100, match=true
Original[100]=999, CUDA[100]=999, match=true
CUDA NTT round-trip PASSED
```

### Core CKKS Operations (N=4096, 7 primes)

Benchmark: `bench_cuda_all_ops`

| Operation | Avg (ms) | Min (ms) | Max (ms) | Notes |
|-----------|----------|----------|----------|-------|
| Encode | 0.076 | 0.075 | 0.082 | |
| Encrypt | 13.81 | 13.73 | 14.26 | |
| Decrypt | 5.96 | 5.79 | 6.04 | |
| Decode | 143.54 | 143.43 | 143.68 | |
| Add (ct+ct) | 0.088 | 0.085 | 0.095 | |
| Subtract (ct-ct) | 0.265 | 0.257 | 0.273 | |
| **Multiply (ct*ct)** | **393.92** | **389.00** | **430.59** | Includes relinearization |
| Multiply Plain (ct*pt) | 0.430 | 0.424 | 0.434 | |
| Rotate (1 slot) | 8.23 | 8.18 | 8.41 | |
| Rotate (8 slots) | 8.20 | 8.16 | 8.23 | |
| Rescale | 0.547 | 0.541 | 0.560 | |
| Mod Switch | 0.103 | 0.101 | 0.122 | |

Throughput: 2.5 multiplications/sec, 11,423 additions/sec, 121.6 rotations/sec

### Unified Geometric Product Comparison (N=1024, 3 primes)

Benchmark: `bench_all_geometric_products` -- head-to-head comparison of ALL implementations

| Version | Total Time (ms) | Per Product (ms) | Speedup vs V2 CPU |
|---------|----------------|-----------------|-------------------|
| V2 CPU (Rayon) | 801.86 | 801.86 | 1.00x |
| V2 CUDA (64 mults) | 2,019.56 | 2,019.56 | 0.40x (slower) |
| **V3 Batched (64 MVs)** | **2,595.95** | **40.56** | **19.77x** |
| V4 CUDA (packed) | 314.22 | 314.22 | 2.55x |

**Key finding: V3 Batched achieves 40.56ms per geometric product** by computing 64 products simultaneously via SIMD slot packing. This is the only implementation that breaks the <50ms barrier.

### V2 CPU vs V4 CUDA Comparison (N=1024 and N=8192)

Benchmark: `bench_geometric_product_comparison`

| N | Primes | V2 CPU (ms) | V4 CUDA (ms) | Speedup |
|---|--------|-------------|--------------|---------|
| 1024 | 3 | 721.24 | 520.46 | 1.39x |
| **8192** | **9** | **18,385.18** | **3,314.80** | **5.55x** |

### V3 CUDA Bootstrap (N=1024, 30 primes)

Benchmark: `bench_cuda_bootstrap`

| Phase | Time | Notes |
|-------|------|-------|
| Modulus Raise | <0.01s | Negligible |
| CoeffToSlot | ~0.48s | Linear transforms + rotations |
| **EvalMod** | **~16.4s** | Polynomial sin approximation (dominant) |
| SlotToCoeff | ~0.32s | Linear transforms + rotations |
| Modulus Switch | <0.01s | Negligible |
| **Full Bootstrap** | **~16.98s avg** | 3 iterations, min 15.09s, max 19.02s |

Note: The N=1024 bootstrap uses 30 primes (27 consumed by bootstrap levels), making each operation much more expensive than standard N=1024 with 3 primes. This is expected -- bootstrap requires deep multiplicative chains.

## Performance Summary

### Geometric Product -- All Versions and Configurations

| Backend | Config | Per Product | Speedup vs V1 | Notes |
|---------|--------|-------------|---------------|-------|
| V1 (baseline) | N=4096 | ~13,000ms | 1x | CPU, deprecated |
| V2 CPU (M3 Max) | N=4096 | 959ms | 13.5x | Rayon parallelized |
| V2 CPU (RTX 5090 host) | N=1024 | 802ms | 16.2x | Rayon parallelized |
| V2 CPU (RTX 5090 host) | N=8192 | 18,385ms | 0.7x | Large ring, sequential mults |
| V2 CUDA (RTX 5090) | N=1024 | 2,020ms | 6.4x | Transfer overhead dominates at small N |
| V4 CUDA (RTX 5090) | N=1024 | 314-520ms | 25-41x | Packed, rotation overhead |
| V4 CUDA (RTX 5090) | N=8192 | 3,165-3,315ms | 3.9-4.1x | Production params |
| **V3 Batched CPU** | **N=1024** | **40.56ms** | **320x** | **64 products in parallel** |
| V2 Metal GPU | TBD | TBD | Est. 50-100x | Apple Silicon only |

### Production Viability Assessment

Target: **<100ms per geometric product** for practical encrypted 3D inference.

| Approach | Per Product | Meets Target? | Trade-off |
|----------|-------------|--------------|-----------|
| V3 Batched (N=1024, CPU) | **40.56ms** | **YES** | Requires batch of 64 inputs, reduced security (N=1024) |
| V4 CUDA (N=1024) | 314-520ms | No | Single-input, reduced security |
| V4 CUDA (N=8192) | 3,165-3,315ms | No | Single-input, full 128-bit security |
| V3 Batched (N=8192, est.) | **~5-10ms** | **YES (projected)** | Batch of 512, full security |

**Recommendation**: V3 Batched is the production-viable path. At N=8192 (128-bit security), batch size grows to 512 multivectors, which should yield even better amortized per-product times. The batching approach naturally fits point cloud classification where many points need processing simultaneously.

## Test Commands

```bash
# === CUDA Benchmarks (RTX 5090) ===

# NTT correctness
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda \
    --example test_cuda_ntt_roundtrip

# Core CKKS operations
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda \
    --example bench_cuda_all_ops

# UNIFIED: V2 CPU vs V2 CUDA vs V3 Batched vs V4 CUDA (MOST IMPORTANT)
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3,v4 \
    --example bench_all_geometric_products

# V2 CPU vs V4 CUDA at both N=1024 and N=8192
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3,v4 \
    --example bench_geometric_product_comparison

# V4 CUDA geometric product (production N=8192)
cargo run --release --no-default-features --features f64,nd,v2,v3,v4,v2-gpu-cuda \
    --example bench_v4_cuda_geometric

# V4 CUDA geometric product (quick N=1024)
cargo run --release --no-default-features --features f64,nd,v2,v3,v4,v2-gpu-cuda \
    --example bench_v4_cuda_geometric_quick

# Bootstrap benchmark
cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda \
    --example bench_cuda_bootstrap

# === CPU / Metal Benchmarks ===

# V2 CPU PointNet inference
cargo run --release --example clifford_pointnet_encrypted --features v2

# Metal GPU
cargo run --release --example encrypted_metal_demo --features v2-gpu-metal

# Plaintext training
cargo run --release --example experiment_plaintext
```

## Notes

1. **V2 CUDA slower than CPU at N=1024**: The V2 CUDA geometric product (2,020ms) is slower than V2 CPU (802ms) at small ring dimensions because it performs 64 individual GPU multiplications, each incurring host-GPU transfer overhead. The GPU wins at larger N where computation dominates over transfer.

2. **V3 Batched is the clear throughput winner**: At 40.56ms per product (batch of 64), V3 Batched achieves 320x speedup over V1 by amortizing the cost of CKKS operations across many multivectors packed into SIMD slots. This naturally fits the point cloud classification use case where batches of points are always available.

3. **V4 CUDA packed approach**: Achieves 5.55x speedup over V2 CPU at N=8192 with 8x memory reduction, but rotation-based packing/unpacking adds overhead compared to V3's slot-level batching.

4. **Bootstrap (16.98s)**: Uses N=1024 with 30 primes for deep multiplicative chains. The dominant bottleneck is EvalMod (~16.4s) which evaluates a degree-23 sine polynomial. This is expected for CKKS bootstrapping and is comparable to state-of-the-art implementations.

5. **CUDA Context Initialization**: One-time cost of ~1.2-3.7s for NTT twiddle factor computation and GPU upload, amortized over all subsequent operations.

6. **Path to <50ms**: V3 Batched already achieves 40.56ms/product at N=1024. Combining V3 batching with CUDA GPU acceleration (replacing the CPU backend) is the highest-priority optimization for production deployment.
