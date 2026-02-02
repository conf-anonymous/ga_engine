# Performance Benchmarks

This document contains performance benchmarks for the GA Engine Clifford FHE implementation across V1-V5 versions.

## Table of Contents

1. [V1 vs V2 Core Operations](#v1-vs-v2-core-operations)
2. [V2 Metal GPU Operations](#v2-metal-gpu-operations)
3. [V2 CUDA GPU Operations](#v2-cuda-gpu-operations)
4. [GPU Bootstrap Performance](#gpu-bootstrap-performance)
5. [V4 Packed Operations](#v4-packed-operations)
6. [V5 Privacy Analysis](#v5-privacy-analysis)
7. [Detailed Analysis](#detailed-analysis)

## V1 vs V2 Core Operations

### Benchmark Setup

- **Hardware**: Apple M3 Max (Apple Silicon)
- **Compiler**: Rust 1.86.0+ with `--release` profile
- **Optimization**: LTO enabled, opt-level 3, single codegen unit
- **Parameters**:
  - Ring dimension: N = 1024
  - RNS moduli: 4 primes (~60 bits each)
  - Security level: ~128 bits
- **Benchmark Framework**: Criterion 0.4
- **Sample Size**: 100 samples for core operations, 50 for geometric operations
- **Last Updated**: January 2026

### Command

```bash
cargo bench --bench v1_vs_v2_benchmark --features v1,v2
```

### Results Summary

#### Core FHE Operations

| Operation | V1 Time | V2 Time | **Speedup** |
|-----------|---------|---------|-------------|
| **Key Generation** | 50.2 ms | 13.4 ms | **3.7×** |
| **Single Encryption** | 10.9 ms | 2.3 ms | **4.7×** |
| **Single Decryption** | 5.4 ms | 1.1 ms | **4.9×** |
| **Ciphertext Multiplication** | 123.5 ms | 34.0 ms | **3.6×** |

#### Geometric Operations (CPU Backend)

Both V1 and V2 support geometric operations on CPU:

| Operation | V1 Time | V2 Time | **Speedup** |
|-----------|---------|---------|-------------|
| **Reverse** | 697 µs | 671 µs | **~1×** |
| **Geometric Product** | 13.0 s | 12.98 s | **~1×** |
| **Wedge Product** | TBD | 4.15 s | TBD |
| **Inner Product** | TBD | 4.11 s | TBD |

**Note**: The CPU geometric product times are similar for V1 and V2 because both use the same underlying FHE multiplication. For significant speedups on geometric operations, use GPU backends (Metal or CUDA)—see sections below.

## V2 Metal GPU Operations

### Basic Operations (Apple M3 Max)

**Command:**
```bash
cargo run --release --features v2,v2-gpu-metal --example encrypted_metal_demo
```

**Parameters**: N=1024, 3 primes

| Operation | Time | Notes |
|-----------|------|-------|
| **Encryption** | 37.84ms | Full GPU pipeline |
| **Decryption** | 26.90ms | Full GPU pipeline |
| **Ciphertext Add** | 4.93ms | Element-wise |
| **Max Error** | 0.000000 | Round-trip accuracy |

## V2 CUDA GPU Operations

### Homomorphic Operations (NVIDIA RTX 4090)

**Command:**
```bash
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v4 --example bench_cuda_all_ops
```

**Parameters**: N=4096, 4 primes

| Operation | Time | Notes |
|-----------|------|-------|
| **Encode** | 0.029ms | Plaintext encoding |
| **Encrypt** | 7.89ms | Full GPU pipeline |
| **Decrypt** | 3.47ms | Full GPU pipeline |
| **Ciphertext Add** | 0.079ms | Element-wise |
| **Ciphertext Multiply** | 286.35ms | With relinearization |
| **Multiply Plain** | 0.41ms | Ciphertext × plaintext |
| **Rotate** | 6.5-6.8ms | Galois automorphism |
| **Rescale** | 0.54ms | Level reduction |

## GPU Bootstrap Performance

### V3 Bootstrap

V3 implements full CKKS bootstrap with three backends: CPU, Metal GPU, and CUDA GPU.

#### Parameters

- **Ring Dimension**: N = 1024
- **RNS Moduli**: 20 primes (1× 60-bit, 19× 45-bit)
- **Bootstrap Levels**: 18 levels total
  - CoeffToSlot: 9 levels
  - SlotToCoeff: 9 levels
- **Security Level**: ~128 bits
- **Last Updated**: January 2026

### Metal GPU Bootstrap (Apple M3 Max)

**Command:**
```bash
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native
```

#### Results

| Operation | Time | Backend | Notes |
|-----------|------|---------|-------|
| **Key Generation** | 697.19s | CPU | Rotation keys + NTT pre-computation |
| **Encryption** | 203.14ms | GPU | Single ciphertext |
| **CoeffToSlot** (9 levels) | 59.13s | GPU | Linear transforms + rotations |
| **SlotToCoeff** (9 levels) | 12.24s | GPU | Linear transforms + rotations |
| **Decryption** | 10.34ms | GPU | Single ciphertext |
| **Total Bootstrap** | **71.37s** | **GPU** | **Entirely GPU execution** |

**Error**: 3.61e-3

**Key Features:**
- Entirely GPU execution (no CPU fallback)
- Uses Metal shaders for all operations
- GPU-resident ciphertexts (minimal PCIe transfers)
- Exact rescaling with Russian peasant `mul_mod_128`

### CUDA GPU Bootstrap (NVIDIA RTX 4090)

**Command:**
```bash
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 --example bench_cuda_bootstrap
```

#### Results

| Operation | Time | Backend | Notes |
|-----------|------|---------|-------|
| **CoeffToSlot** | ~0.4s | GPU | Linear transforms + rotations |
| **EvalMod** | **11.3s** | GPU | Modular reduction with BSGS |
| **SlotToCoeff** | ~0.3s | GPU | Linear transforms + rotations |
| **Total Bootstrap** | **11.69s** | **GPU** | **With relinearization** |

**Error**: ~1e-3

**Key Features:**
- Entirely GPU execution with CUDA kernels
- Full relinearization support
- GPU-resident ciphertexts throughout
- Optimized RNS operations (add, sub, pointwise multiply)
- Exact rescaling with Russian peasant `mul_mod_128`

### Bootstrap Performance Comparison

| Backend | Hardware | Total Time | Speedup vs CPU | Notes |
|---------|----------|------------|----------------|-------|
| **V3 CPU** | Apple M3 Max | ~70s | 1.0× | Reference |
| **V3 Metal GPU** | Apple M3 Max | 71.37s | ~1× | Entirely GPU |
| **V3 CUDA GPU** | NVIDIA RTX 4090 | **11.69s** | **6.0×** | **Entirely GPU + Relin** |

**Key Insight**: CUDA implementation is ~6× faster than Metal on this workload, primarily due to:
- Different GPU architectures (NVIDIA vs Apple Silicon)
- Optimized CUDA kernels for FHE operations
- Efficient relinearization implementation
- Hardware-specific optimizations

## V4 Packed Operations

V4 introduces slot-interleaved packing, storing all 8 multivector components in a single ciphertext.

### Parameters

- **Ring Dimension**: N = 1024 (quick) / N = 8192 (production)
- **RNS Moduli**: 3-15 primes (~60 bits each)
- **Batch Size**: N/8 multivectors per ciphertext (128 for N=1024, 1024 for N=8192)
- **Last Updated**: January 2026

### Commands

```bash
# Metal GPU (Apple Silicon)
cargo run --release --features v4,v2-gpu-metal --example test_v4_geometric_product

# CUDA GPU (NVIDIA)
cargo run --release --features v4,v2-gpu-cuda --example bench_v4_cuda_geometric_quick
```

### V4 Performance Results

| Operation | Backend | Time | Notes |
|-----------|---------|------|-------|
| **Key Generation** | Metal (N=1024) | 4.45s | Rotation keys for steps 1-8 |
| **Packing (8→1)** | Metal (N=1024) | ~0.5s | 8 ciphertexts → 1 |
| **Geometric Product** | Metal (N=1024) | **3.89s** | On packed data |
| **Packing (8→1)** | CUDA (N=1024) | 0.734s | 8 ciphertexts → 1 |
| **Unpacking (1→8)** | CUDA (N=1024) | 0.067s | 1 ciphertext → 8 |
| **Geometric Product** | CUDA (N=1024) | **0.588s** | Quick test, 3 primes |
| **Geometric Product** | CUDA (N=8192) | **3.40s** | Full test, 9 primes |
| **Per-MV Cost** | CUDA (N=8192 batched) | ~3.3ms | 1024 MVs in parallel |

**V4 GPU vs V2 CPU Geometric Product (N=1024):**
- V2 CPU: 12.98s
- V4 Metal (M3 Max): 3.89s → **3.3× speedup**
- V4 CUDA (RTX 4090): 0.588s → **22.1× speedup**, 6.6× faster than Metal

### Memory Comparison

| Configuration | V2/V3 Memory | V4 Memory | Savings |
|---------------|--------------|-----------|---------|
| N=1024, 4 primes | 262 KB/MV | 33 KB/MV | 8× |
| N=8192, 15 primes | 15.7 MB/MV | 1.97 MB/MV | 8× |
| N=32768, 30 primes | 125.8 MB/MV | 15.7 MB/MV | 8× |

### V4 vs V2 Trade-offs

| Metric | V2 (Separate) | V4 (Packed) | Winner |
|--------|---------------|-------------|--------|
| Memory per MV | 8 ciphertexts | 1 ciphertext | **V4** |
| Single-MV latency | 33ms (GPU) | 36s (includes pack/unpack) | **V2** |
| Batch throughput | 1× | 64-1024× | **V4** |
| Implementation complexity | Lower | Higher | V2 |

**Use V4 when**: Processing batches of multivectors, memory-constrained environments, SIMD-style operations.

**Use V2 when**: Single multivector operations, low-latency requirements, interactive applications.

### Division Operations (CUDA)

**Command:**
```bash
cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v4 --example bench_division_cuda_gpu
```

| Operation | Backend | Time | Notes |
|-----------|---------|------|-------|
| **Division** | CUDA (RTX 4090) | **1.37s** | Goldschmidt iteration |

## V5 Privacy Analysis

V5 provides execution-trace collection and privacy analysis for security research.

### Commands

```bash
# Run comprehensive attack suite
cargo run --release --features v5 --example v5_privacy_attacks

# Run dimension inference attack only
cargo run --release --features v5 --example v5_dimension_attack

# Collect traces with Metal GPU
cargo run --release --features v5,v2-gpu-metal --example v5_trace_collector -- --metal
```

### Privacy Attack Results

Results from the comprehensive attack suite (6 attack vectors):

| Attack | CKKS Accuracy | CliffordFHE Accuracy | Random Baseline | Winner |
|--------|---------------|----------------------|-----------------|--------|
| **Dimension Inference** | 100.0% | 16.7% | 16.7% | **CliffordFHE** |
| **Task Identification** | 100.0% | 16.7% | 16.7% | **CliffordFHE** |
| **Operation Count** | 100.0% | 16.7% | 16.7% | **CliffordFHE** |
| **Trace Length** | 100.0% | 16.7% | 16.7% | **CliffordFHE** |
| **Sparsity Inference** | ~20% | ~20% | 20% | Tie |
| **Tenant Linkability** | ~50% | ~50% | 50% | Tie |

### Information Leakage

| Metric | CKKS | CliffordFHE |
|--------|------|-------------|
| Dimension entropy (6 classes) | 2.585 bits | 2.585 bits |
| Bits leaked | 2.585 bits (100%) | 0.0 bits (0%) |
| Conditional entropy H(D\|T) | 0.0 bits | 2.585 bits |
| Mutual information I(D;T) | 2.585 bits | 0.0 bits |

### Key Finding

CliffordFHE achieves **information-theoretic privacy** against dimension inference attacks:
- CKKS: Rotation count directly reveals input dimension
- CliffordFHE: Fixed 64-multiplication structure regardless of input

**Score**: CliffordFHE wins 4 attacks, ties 2.

## Detailed Analysis

### V2 Optimizations (CPU)

**Important:** Both V1 and V2 use O(N log N) NTT for polynomial multiplication and RNS representation. The speedups come from implementation-level optimizations, not algorithmic changes.

The V2 implementation achieves 3-5× speedups through:

1. **Harvey Butterfly NTT** (1.5-2× speedup over V1's Cooley-Tukey NTT)
   - More cache-efficient butterfly operations
   - Better memory access patterns
   - Optimized modular arithmetic with Barrett reduction
   - Lazy reduction techniques (fewer modular reductions)

2. **RNS Operation Optimizations** (1.2-1.5× speedup)
   - Both versions use RNS, but V2 has faster per-prime operations
   - Better vectorization opportunities
   - Reduced overhead in CRT reconstruction
   - More efficient modulus switching

3. **Memory Layout and Data Structures** (1.3-1.8× speedup)
   - Improved cache locality for ciphertext operations
   - Reduced allocations and copying
   - Better memory alignment for potential SIMD
   - Streamlined ciphertext representation

**Combined effect:** These multiplicative improvements result in the observed 3.2-4.8× overall speedup.

### GPU Backend Optimizations

Both Metal and CUDA GPU backends achieve significant speedups through:

1. **Parallel NTT** - All N coefficients × num_primes limbs computed in parallel
2. **GPU-Resident Data** - Ciphertexts stay on GPU between operations
3. **Batched Operations** - Multiple primes processed simultaneously
4. **Exact Rescaling on GPU** - Russian peasant `mul_mod_128` avoids overflow
5. **Optimized Memory Access** - Coalesced reads/writes, minimized transfers

**Metal-Specific** (Apple Silicon):
- Metal shaders optimized for Apple GPU architecture
- Unified memory reduces CPU↔GPU transfer overhead
- Threadgroup memory for NTT twiddle factors

**CUDA-Specific** (NVIDIA):
- CUDA kernels optimized for NVIDIA architecture
- Shared memory for fast data exchange
- Coalesced global memory access patterns
- Strided layout kernels avoid expensive conversions

### Bootstrap Operation Breakdown

The bootstrap consists of three main phases (CoeffToSlot + EvalMod + SlotToCoeff):

1. **CoeffToSlot** - Convert coefficient encoding to slot encoding
   - Linear transformations (matrix multiplications)
   - Rotations via Galois automorphisms
   - CUDA: ~0.4s, Metal: 59.13s

2. **EvalMod** - Modular reduction (the main bottleneck)
   - Polynomial approximation of modular reduction
   - Baby-step giant-step (BSGS) optimization
   - CUDA: ~15.7s (97% of total), Metal: included in CoeffToSlot

3. **SlotToCoeff** - Convert slot encoding back to coefficients
   - Inverse of CoeffToSlot
   - Linear transformations + rotations
   - CUDA: ~0.3s, Metal: 12.24s

**Performance Note**: The Metal backend shows different performance characteristics, with CoeffToSlot taking significantly longer. This is due to:
- Different GPU memory architectures (discrete vs unified)
- Metal shader execution model differences
- Apple Silicon's GPU compute architecture
- NTT-transformed key pre-computation overhead (612s on Metal)

### Geometric Operations Performance

The geometric operations are computationally expensive because they involve multiple homomorphic operations:

- **Reverse**: Simple coefficient reordering (very fast)
- **Geometric Product**: 8×8 = 64 homomorphic multiplications + additions
- **Wedge Product**: Geometric product + subtraction + scalar division by 2
- **Inner Product**: Geometric product + addition + scalar division by 2

Each homomorphic multiplication requires:
1. Tensor product of ciphertexts (polynomial multiplication in NTT domain)
2. Relinearization (reduce ciphertext size using evaluation key)
3. Rescaling (manage noise growth)

**Geometric Product Performance (N=1024, 3 primes):**

| Backend | Time | Speedup | Notes |
|---------|------|---------|-------|
| V1 CPU | 13.0s | 1.0× | Baseline |
| V2 CPU | 12.98s | ~1× | Same algorithm as V1 |
| V2 CUDA GPU (RTX 4090) | 1.395s | 9.3× | Single product |
| V3 Batched CUDA (RTX 4090) | 1.519s total | 12.48× per | 64 products in 23.73ms each |
| V4 Metal GPU (M3 Max) | **3.89s** | **3.3×** | Apple Silicon unified memory |
| V4 CUDA GPU (RTX 4090) | **0.588s** | **22.1×** | Best performance, 6.6× faster than Metal |

**Geometric Product Performance (N=8192, 9 primes):**

| Backend | Time | Notes |
|---------|------|-------|
| V4 CUDA GPU (RTX 4090) | **3.40s** | Production parameters, 1024 MVs batched |

**V3 Batched Geometric Product Breakdown (RTX 4090):**
- Total time for 64 products: 1.519s
- Per-product time: 23.73ms
- Effective speedup: **12.48× per product** vs sequential
- This demonstrates the efficiency of batched operations in FHE

## Accuracy Verification

All implementations maintain high accuracy:

### V2 CPU Operations
- Key Generation: Exact
- Encryption/Decryption: < 1e-6 error
- Multiplication: < 1e-6 error
- Reverse: < 2e-10 error
- Geometric Product: < 8e-10 error
- Wedge Product: < 2e-10 error
- Inner Product: < 1e-10 error
- Projection: < 2e-10 error
- Rejection: < 1e-7 error

### V3 Bootstrap Operations
- **Metal GPU Bootstrap**: 3.6e-3 error
- **CUDA GPU Bootstrap**: ~1e-3 error
- **CPU Reference**: 3.6e-3 error

All errors are within acceptable bounds for FHE applications.

## Future Optimization Opportunities

Based on feature flags and current development:

1. **SIMD Vectorization** (`v2-simd-batched`)
   - Slot packing for batch operations
   - Estimated 8-16× throughput improvement
   - Status: Experimental

2. **GPU Pipeline Optimization**
   - Persistent GPU buffers (eliminate redundant transfers)
   - Kernel fusion for multi-step operations
   - Async compute for overlapping operations
   - Estimated 20-30% additional speedup

3. **EvalMod Optimization** (CUDA)
   - Current bottleneck: 11.3s / 11.69s = 97% of bootstrap time
   - Potential: Optimize BSGS polynomial evaluation
   - Potential: Better rotation key caching
   - Target: 30-50% reduction in EvalMod time

4. **Multi-GPU Support**
   - Distribute bootstrap across multiple GPUs
   - Parallel ciphertext processing
   - Estimated 2-4× additional speedup

## Benchmark Reproducibility

### V1 vs V2 CPU Benchmarks

To reproduce these benchmarks:

1. Clone the repository
2. Ensure you have Rust 1.86.0 or later
3. Run: `cargo bench --bench v1_vs_v2_benchmark --features v1,v2`

### GPU Bootstrap Benchmarks

**Metal GPU (Apple Silicon required):**
```bash
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native
```

**CUDA GPU (NVIDIA GPU required):**
```bash
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
```

**Note**: GPU benchmarks require specific hardware and drivers:
- Metal: Apple Silicon Mac (M1/M2/M3)
- CUDA: NVIDIA GPU with CUDA Toolkit 11.0+

Results may vary based on:
- GPU architecture and compute capability
- CPU architecture and clock speed
- Available RAM and VRAM
- PCIe bandwidth (for discrete GPUs)
- System load and thermal throttling
- Compiler version and optimizations

For consistent results:
- Close other applications
- Ensure adequate cooling
- Run multiple times and average results
- Use the same compiler and CUDA/Metal SDK versions
