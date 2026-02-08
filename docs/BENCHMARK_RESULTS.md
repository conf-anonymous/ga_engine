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

## Performance Summary

| Backend | Geometric Product | Speedup vs V1 |
|---------|------------------|---------------|
| V1 (baseline) | ~13,000ms | 1× |
| V2 CPU (M3 Max) | 959ms | 13.5× |
| V2 Metal GPU | TBD (geometric product) | Est. 50-100× |
| V2 CUDA GPU | Pending | Est. 100-200× |

## Test Commands

```bash
# V2 CPU benchmark
cargo run --release --example clifford_pointnet_encrypted --features v2

# Metal GPU benchmark
cargo run --release --example encrypted_metal_demo --features v2-gpu-metal

# Metal NTT correctness
cargo run --release --example test_metal_ntt_correctness --features v2-gpu-metal

# Plaintext training
cargo run --release --example experiment_plaintext
```

## Notes

1. **Decoding Issue**: V2 CPU decryption shows scale/encoding errors (values off by large factors). The timing benchmarks are accurate; correctness fix is in progress.

2. **Metal Status**: NTT kernels fully integrated and verified. Full geometric product on Metal pending.

3. **CUDA**: Benchmarks pending - requires NVIDIA GPU environment.
