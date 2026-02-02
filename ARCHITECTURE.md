# GA Engine Architecture

## Overview

The GA Engine implements Clifford Homomorphic Encryption with **five parallel implementations**:

- **V1:** Research prototype (deprecated, stable baseline)
- **V2:** GPU-accelerated backend (CPU + Metal + CUDA)
- **V3:** Full bootstrap implementation (unlimited depth)
- **V4:** Packed slot-interleaved layout (8× memory reduction)
- **V5:** Privacy-trace collection and analysis framework

## Directory Structure

```
src/
├── clifford_fhe/              # V1 - STABLE (Deprecated)
│   ├── ckks_rns.rs            # Basic RNS-CKKS
│   ├── geometric_product_rns.rs
│   └── ...
│
├── clifford_fhe_v2/           # V2 - GPU Backend Foundation
│   ├── backends/
│   │   ├── cpu_optimized/     # CPU with NTT optimization
│   │   │   ├── ckks.rs        # Core CKKS operations
│   │   │   ├── ntt.rs         # Harvey butterfly NTT
│   │   │   └── keys.rs        # Key generation
│   │   │
│   │   ├── gpu_metal/         # Apple Metal GPU backend
│   │   │   ├── ckks.rs        # Core CKKS operations
│   │   │   ├── ntt.rs         # GPU NTT transforms
│   │   │   ├── keys.rs        # Key generation (CPU)
│   │   │   ├── rotation.rs    # Rotation operations
│   │   │   ├── rotation_keys.rs   # Rotation key switching
│   │   │   ├── relin_keys.rs  # Relinearization keys
│   │   │   ├── device.rs      # Metal device management
│   │   │   ├── ciphertext_ops.rs  # Ciphertext operations
│   │   │   └── shaders/
│   │   │       ├── ntt.metal         # NTT kernels
│   │   │       ├── pointwise.metal   # Element-wise ops
│   │   │       ├── galois.metal      # Galois automorphism
│   │   │       ├── gadget.metal      # Key switching
│   │   │       └── rns_fixed.metal   # GPU rescaling
│   │   │
│   │   ├── gpu_cuda/          # NVIDIA CUDA GPU backend
│   │   │   ├── ckks.rs        # Core CKKS operations
│   │   │   ├── ntt.rs         # GPU NTT transforms
│   │   │   ├── rotation.rs    # Rotation operations
│   │   │   ├── rotation_keys.rs   # Rotation key switching
│   │   │   ├── relin_keys.rs  # Relinearization keys
│   │   │   ├── device.rs      # CUDA device management
│   │   │   ├── geometric.rs   # Geometric product
│   │   │   ├── ciphertext_ops.rs  # Ciphertext operations
│   │   │   └── kernels/
│   │   │       ├── ntt.cu            # NTT kernels
│   │   │       ├── galois.cu         # Galois automorphism
│   │   │       ├── gadget.cu         # Gadget decomposition
│   │   │       └── rns.cu            # RNS operations
│   │   │
│   │   └── simd_batched/      # SIMD slot packing (experimental)
│   │
│   └── params.rs              # Parameter sets
│
├── clifford_fhe_v3/           # V3 - Full Bootstrap
│   ├── bootstrapping/         # Bootstrap operations
│   │   ├── bootstrap_context.rs   # Bootstrap orchestration
│   │   │
│   │   ├── coeff_to_slot.rs   # CPU CoeffToSlot transform
│   │   ├── slot_to_coeff.rs   # CPU SlotToCoeff transform
│   │   ├── eval_mod.rs        # CPU modular reduction
│   │   │
│   │   ├── cuda_bootstrap.rs      # CUDA bootstrap orchestration
│   │   ├── cuda_coeff_to_slot.rs  # CUDA CoeffToSlot transform
│   │   ├── cuda_slot_to_coeff.rs  # CUDA SlotToCoeff transform
│   │   ├── cuda_eval_mod.rs       # CUDA modular reduction
│   │   │
│   │   ├── keys.rs            # Rotation key generation
│   │   ├── rotation.rs        # Rotation helpers
│   │   ├── diagonal_mult.rs   # Diagonal matrix multiplication
│   │   ├── mod_raise.rs       # Modulus raising
│   │   └── sin_approx.rs      # Sine approximation for EvalMod
│   │
│   ├── params.rs              # V3-optimized parameters
│   └── prime_gen.rs           # Dynamic NTT-friendly prime generation
│
├── clifford_fhe_v4/           # V4 - Packed Slot-Interleaved
│   ├── mod.rs                 # Module exports with feature gating
│   ├── multivector.rs         # PackedMultivector type
│   ├── packing.rs             # Metal/CPU packing (1-param encode)
│   ├── packing_cuda.rs        # CUDA packing (3-param encode)
│   ├── packing_butterfly.rs   # Shared butterfly algorithm
│   └── geometric_ops.rs       # Packed geometric operations
│
└── clifford_fhe_v5/           # V5 - Privacy-Trace Collection
    ├── mod.rs                 # Module exports
    ├── trace_collector.rs     # Execution trace instrumentation
    ├── trace_types.rs         # TraceEvent, TraceMetadata types
    ├── attacks/               # Privacy attack implementations
    │   ├── branching.rs       # Branching pattern analysis
    │   ├── rotation.rs        # Rotation pattern analysis
    │   ├── depth.rs           # Depth pattern analysis
    │   └── timing.rs          # Timing side-channel analysis
    └── mitigations/           # Countermeasure implementations
        ├── oblivious.rs       # Oblivious operation patterns
        └── padding.rs         # Trace padding strategies
```

## Feature Flags

Defined in `Cargo.toml`:

```toml
[features]
default = ["f64", "nd", "v1", "lattice-reduction"]

# Version selection
v1 = []                        # Deprecated research implementation
v2 = []                        # GPU backend (CPU + Metal + CUDA)
v3 = ["v2"]                    # Full bootstrap (uses V2 backend)
v4 = ["v2"]                    # Packed layout (uses V2 backend)
v5 = []                        # Privacy-trace collection (standalone)

# V2 backends (requires v2)
v2-cpu-optimized = ["v2"]           # CPU with NTT
v2-gpu-metal = ["v2", "metal"]      # Apple Metal GPU
v2-gpu-cuda = ["v2", "cudarc"]      # NVIDIA CUDA GPU
v2-simd-batched = ["v2"]            # SIMD slot packing
```

## Architecture Principles

### Backend Hierarchy

All versions build on V2's backend infrastructure:

```
V1 (Standalone)
  └─ Basic CKKS, no GPU support

V2 (Foundation)
  ├─ CPU Backend (cpu_optimized/)
  ├─ Metal GPU Backend (gpu_metal/)
  └─ CUDA GPU Backend (gpu_cuda/)

V3 (Uses V2 Backend)
  ├─ Bootstrap algorithms
  ├─ CoeffToSlot, SlotToCoeff, EvalMod
  └─ Delegates low-level ops to V2

V4 (Uses V2 Backend)
  ├─ Packing/unpacking algorithms
  ├─ Butterfly network transforms
  └─ Delegates low-level ops to V2
```

**Key Principle:** V3 and V4 are **not backend-agnostic**. They directly use V2's GPU infrastructure for all low-level operations.

### V2 Backend Operations

V2 provides these operations used by V3 and V4:

**Core CKKS Operations:**
- `encode()` - Encode plaintext values
- `encrypt()` / `decrypt()` - Basic encryption/decryption
- `add()` / `subtract()` - Ciphertext arithmetic
- `multiply_plain()` - Plaintext multiplication (with rescaling)
- `exact_rescale_gpu()` - Exact rescaling with centered rounding

**NTT Operations:**
- `ntt_forward_batched_gpu()` - Forward NTT transform
- `ntt_inverse_batched_gpu()` - Inverse NTT transform

**Layout Conversions:**
- `strided_to_flat()` - Convert strided → flat layout
- `flat_to_strided()` - Convert flat → strided layout

**Polynomial Operations:**
- `add_polynomials_gpu()` - Polynomial addition
- `subtract_polynomials_gpu()` - Polynomial subtraction
- `pointwise_multiply_polynomials_gpu_strided()` - Pointwise multiplication

**Rotation & Key Switching:**
- `apply_galois_automorphism_gpu()` - Galois automorphism
- `key_switch_hoisted_gpu()` - Key switching for rotation
- `rotate_by_steps()` - High-level rotation API (V4)
- `rotate_batch_with_hoisting()` - Batched rotation (V4)

All these operations are entirely GPU-resident (no CPU fallback in hot path).

## Version Details

### V1: Research Prototype (Deprecated)

**Purpose:** Proof of concept, correctness verification

**Status:** Stable, frozen

**Characteristics:**
- Straightforward implementation
- O(n²) schoolbook multiplication
- CPU-only
- 8 separate ciphertexts per multivector
- Serves as baseline for benchmarking

**Performance:** ~11.4s per geometric product (M3 Max CPU)

**Use Case:** Historical reference, correctness validation

### V2: GPU-Accelerated Backend (Foundation)

**Purpose:** Production-quality CKKS implementation, multi-platform GPU support

**Status:** Production Candidate

**Backends:**
1. **CPU (cpu_optimized/)**: SIMD, Rayon parallelization, Harvey butterfly NTT
2. **Metal GPU (gpu_metal/)**: Apple Silicon, unified memory, Metal shaders
3. **CUDA GPU (gpu_cuda/)**: NVIDIA GPUs, strided layout, CUDA kernels

**Performance (Geometric Product):**
- CPU (M3 Max, 14-core): 300ms
- Metal (M3 Max GPU): 33ms
- CUDA (RTX 5090): 5.7ms

**Key Optimizations:**
- Harvey butterfly NTT (O(n log n))
- RNS representation (multi-prime modular arithmetic)
- Barrett reduction (fast modular reduction)
- GPU memory coalescing (CUDA strided layout)
- Unified memory architecture (Metal)

**Advanced Operations:**
- Homomorphic division via Newton-Raphson iteration (6 iterations, depth 7)

### V3: Full Bootstrap Implementation

**Purpose:** Unlimited circuit depth via noise refresh

**Status:** Production Candidate

**Architecture:** Uses V2 backend infrastructure

**Components:**
1. **ModRaise:** Lift ciphertext to higher modulus
2. **CoeffToSlot:** Transform coefficient → slot encoding
3. **EvalMod:** Homomorphic modular reduction
4. **SlotToCoeff:** Transform slot → coefficient encoding
5. **ModDown:** Reduce to original modulus

**Performance (Full Bootstrap):**
- CPU: ~70s (reference)
- Metal GPU (M3 Max): ~60s
- CUDA GPU (RTX 5090): **11.95s**

**Ciphertext Representation:** 8 separate ciphertexts per multivector (8× memory cost)

**Use Case:** Deep circuits, complex computations, unlimited multiplication depth

### V4: Packed Slot-Interleaved Layout

**Purpose:** Memory-efficient geometric operations, batch processing

**Status:** Production Candidate (Metal + CUDA)

**Key Innovation:** Pack all 8 multivector components into **1 ciphertext**

**Slot Layout (N=8192):**
```
Slots: [c₀ c₁ c₂ c₃ c₄ c₅ c₆ c₇ | c₀ c₁ c₂ c₃ c₄ c₅ c₆ c₇ | ...]
        └──── batch 0 ─────┘     └──── batch 1 ─────┘

Batch size: N/8 = 1024 multivectors per ciphertext
```

**Operations:**

1. **Packing (8 → 1):** Butterfly network
   - Stage 1: Combine pairs (rotation by 1)
   - Stage 2: Combine quads (rotation by 2)
   - Stage 3: Combine octets (rotation by 4)
   - Complexity: O(log k) rotations for k components

2. **Unpacking (1 → 8):** Reverse butterfly with masking
   - Extract each component via rotations and masks
   - Complexity: O(log k) rotations + O(k) multiplications

3. **Geometric Product (packed):** Unpack → Compute → Repack
   - Unpack into 8 components
   - For each RNS prime (parallel on GPU):
     - Extract coefficients for that prime
     - Compute geometric product using structure constants
     - Insert results back into RNS representation
   - Pack result back into single ciphertext

**Backend Support:**
- **Metal:** Full implementation (1-parameter `encode()`)
- **CUDA:** Full implementation (3-parameter `encode()`)

**Critical Implementation Detail (CUDA):**
After rescaling operations (`multiply_plain`, `add`), both `level` and `num_primes` fields **must** be updated together:

```rust
// After rescaling drops one prime:
let new_level = self.level.saturating_sub(1);
let new_num_primes = new_level + 1;  // CRITICAL!

CudaCiphertext {
    c0: rescaled_c0,
    c1: rescaled_c1,
    num_primes: new_num_primes,  // Must match array size
    level: new_level,
    scale: new_scale,
}
```

**Performance:**
- Metal GPU: ~5.0s per packed geometric product
- CUDA GPU (N=1024): 36.84s per packed geometric product
- **Memory:** 8× reduction vs V3

**Batch Throughput:**
- Single operation: Higher latency than V2
- Batch of 1024 MVs: **1024× throughput** improvement

**Use Case:** Batch processing, memory-constrained environments, SIMD operations

### V5: Privacy-Trace Collection and Analysis

**Purpose:** Research framework for execution-trace privacy analysis

**Status:** Research / Paper Support

**Architecture:** Standalone instrumentation layer (works with V2-V4)

**Components:**
1. **Trace Collector:** Instruments FHE operations to capture execution patterns
2. **Attack Implementations:**
   - Branching pattern analysis (data-dependent control flow)
   - Rotation pattern analysis (slot access patterns)
   - Depth pattern analysis (multiplicative depth fingerprinting)
   - Timing side-channel analysis
3. **Mitigation Strategies:**
   - Oblivious operation patterns (constant-time execution)
   - Trace padding (dummy operations for uniformity)

**Key Insight:** Even with encrypted data, execution traces can leak information about the computation structure, enabling inference attacks on the underlying algorithm or data characteristics.

**Use Case:** Security research, privacy analysis, academic paper support

## Key Components

### RNS (Residue Number System)

Multi-prime representation for efficiency:

```
Ciphertext mod q = q₀ · q₁ · ... · qₗ
Represented as: (c mod q₀, c mod q₁, ..., c mod qₗ)
```

**Advantages:**
- Parallel computation across primes
- No large integer arithmetic
- Efficient modulus switching (drop primes)

**Prime Selection:**
- V1/V2: Fixed prime sets
- V3: Dynamic NTT-friendly prime generation
- V4: Uses V2 prime management

**Example (V3 Bootstrap):** N=1024, 30 primes
- 1× ~60-bit prime
- 29× ~45-bit primes
- Total modulus: ~900 bits

### NTT (Number Theoretic Transform)

Fast polynomial multiplication via FFT in finite fields:

**Implementations:**
- **V1:** Schoolbook multiplication O(n²)
- **V2 CPU:** Harvey butterfly NTT O(n log n)
- **V2 Metal:** Metal shader with optimized memory access
- **V2 CUDA:** CUDA kernel with coalesced memory
- **V3/V4:** Use V2's NTT implementations

**Metal NTT Shader:**
```metal
kernel void ntt_forward(
    device ulong* data [[buffer(0)]],
    constant ulong* twiddles [[buffer(1)]],
    constant ulong& modulus [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Cooley-Tukey butterfly algorithm
    // Optimized for Metal GPU architecture
}
```

**CUDA NTT Kernel:**
```cuda
__global__ void batched_ntt_kernel(
    uint64_t* polys,      // Strided layout
    const uint64_t* twiddles,
    const uint64_t* moduli,
    int n,
    int num_primes
) {
    // Process all RNS primes in parallel
    // Coalesced memory access pattern
}
```

### GPU Memory Layouts

**Metal (Flat Layout):**
```
RNS representation for polynomial of degree n:
[coeff₀_q₀, coeff₁_q₀, ..., coeff_{n-1}_q₀,
 coeff₀_q₁, coeff₁_q₁, ..., coeff_{n-1}_q₁,
 ...]
└────────── prime q₀ ────────────┘└── prime q₁ ──...
```

**CUDA (Strided Layout):**
```
RNS representation for polynomial of degree n:
[coeff₀_q₀, coeff₀_q₁, ..., coeff₀_qₗ,
 coeff₁_q₀, coeff₁_q₁, ..., coeff₁_qₗ,
 ...]
└───── coeff 0 ─────┘└───── coeff 1 ─────┘
```

**Advantage of Strided (CUDA):** Better memory coalescing when processing all primes for one coefficient.

### GPU Rescaling

Exact rescaling requires 128-bit multiplication without overflow.

**Russian Peasant Multiplication (Metal `rns_fixed.metal` and CUDA `rns.cu`):**

```c
// Computes (a * b) mod q without 128-bit overflow
ulong mul_mod_128(ulong a, ulong b, ulong q) {
    ulong result = 0;
    a = a % q;

    while (b > 0) {
        if (b & 1) {
            result = add_mod_lazy(result, a, q);
            if (result >= q) result -= q;
        }
        a = add_mod_lazy(a, a, q);
        if (a >= q) a -= q;
        b >>= 1;
    }

    return result;
}
```

**Validation:**
- Golden compare: Bit-exact with CPU reference
- Bootstrap accuracy: Error < 10⁻³

## Testing

### Unit Tests

```bash
# V1 tests (31 tests)
cargo test --lib --features v1

# V2 CPU tests (127 tests)
cargo test --lib --features v2,v2-cpu-optimized

# V2 Metal GPU tests
cargo test --lib --features v2,v2-gpu-metal

# V2 CUDA GPU tests
cargo test --lib --features v2,v2-gpu-cuda

# V3 tests (52 tests)
cargo test --lib --features v2,v3

# V4 tests
cargo test --lib --features v2,v4
```

### Integration Tests

**V3 Bootstrap:**
```bash
# Metal GPU bootstrap (~60s)
cargo run --release --features v2,v2-gpu-metal,v3 \
  --example test_metal_gpu_bootstrap_native

# CUDA GPU bootstrap (~11.95s)
cargo run --release --features v2,v2-gpu-cuda,v3 \
  --example test_cuda_bootstrap
```

**V4 Packed Operations:**
```bash
# Metal GPU packed geometric product
cargo test --release --features v4,v2-gpu-metal \
  --test test_geometric_operations_v4 -- --nocapture

# CUDA GPU packed geometric product (quick test, N=1024)
cargo run --release --features v4,v2-gpu-cuda \
  --example bench_v4_cuda_geometric_quick
```

### Validation Tests

```bash
# GPU rescaling golden compare (Metal)
cargo run --release --features v2,v2-gpu-metal,v3 \
  --example test_rescale_golden_compare

# V4 CUDA basic pack/unpack
cargo run --release --features v4,v2-gpu-cuda \
  --example test_v4_cuda_basic
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing instructions.

## Performance Summary

### V2 Geometric Product (Single Operation)

| Backend | Hardware | Time | Speedup |
|---------|----------|------|---------|
| V1 Baseline | M3 Max CPU | 11,420 ms | 1× |
| V2 CPU | M3 Max (14-core) | 300 ms | 38× |
| V2 Metal | M3 Max GPU | 33 ms | 346× |
| V2 CUDA | RTX 5090 | 5.7 ms | 2,002× |

### V3 Bootstrap (Full Pipeline)

| Backend | Hardware | Time | Speedup |
|---------|----------|------|---------|
| V3 CPU | M3 Max | ~70s | 1× |
| V3 Metal | M3 Max GPU | ~60s | 1.17× |
| V3 CUDA | RTX 5090 | ~11.95s | 5.86× |

**CUDA Bootstrap Breakdown:**
- EvalMod: 11.76s (98% of time)
- CoeffToSlot: 0.15s
- SlotToCoeff: 0.04s
- Error: ~10⁻³ (excellent accuracy)

### V4 Packed Operations

| Operation | Backend | Time | Notes |
|-----------|---------|------|-------|
| Packing (8→1) | CUDA (N=1024) | 31.38s | One-time per batch |
| Geometric Product | CUDA (N=1024) | 36.84s | On packed data |
| Per-MV Cost (batched) | CUDA | ~36ms | 1024 MVs in parallel |

**Memory Comparison (N=8192, 15 primes):**
- V3: 15.7 MB per multivector
- V4: 1.97 MB per multivector
- **Savings: 8× reduction**

See [BENCHMARKS.md](BENCHMARKS.md) for detailed performance analysis.

## Development Guidelines

### V1 - Historical Baseline
- Frozen, no new development
- Critical bug fixes only
- Maintained for reproducibility

### V2 - Active Development
- Performance optimization
- New GPU kernel development
- Backend improvements
- Foundation for V3/V4

### V3 - Bootstrap Research
- Algorithm improvements
- Parameter optimization
- Note: Ensure V2 backend has needed GPU operations first

### V4 - Packing & Batching
- Packing algorithm improvements
- Batch operation optimization
- Note: Ensure V2 backend has needed rotation operations first
- Note: Maintain synchronization of `level` and `num_primes` fields

### V5 - Privacy Research
- Execution trace analysis
- Attack implementation and validation
- Mitigation strategy development
- Paper-supporting experiments

## Documentation

### Core Documentation

- **[README.md](README.md)** - Project overview, quick start
- **[CLIFFORD_FHE_VERSIONS.md](CLIFFORD_FHE_VERSIONS.md)** - Complete V1-V5 technical history
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - This file (system architecture)
- **[INSTALLATION.md](INSTALLATION.md)** - Setup instructions
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Testing procedures
- **[BENCHMARKS.md](BENCHMARKS.md)** - Performance measurements
- **[COMMANDS.md](COMMANDS.md)** - Command reference
- **[FEATURE_FLAGS.md](FEATURE_FLAGS.md)** - Feature flag reference

## Key Achievements

### V1 (Deprecated)
- First working Clifford FHE implementation
- All 7 geometric operations
- Correctness validated

### V2 (Production Candidate)
- Three backends: CPU, Metal GPU, CUDA GPU
- NTT optimization (10-100× speedup)
- Rotation operations
- Relinearization keys
- Homomorphic division (Newton-Raphson)
- Foundation for V3/V4

### V3 (Production Candidate)
- Full bootstrap implementation
- Dynamic prime generation
- CPU reference implementation
- Metal GPU implementation (~60s)
- CUDA GPU implementation (~11.95s)
- Unlimited computation depth

### V4 (Production Candidate)
- Packed slot-interleaved layout
- 8× memory reduction
- Butterfly network packing
- Metal backend complete
- CUDA backend complete
- Batch processing (1024× throughput)

### V5 (Research)
- Execution-trace privacy analysis framework
- Four attack categories implemented
- Mitigation strategies documented
- Research paper experiments

## Future Work

1. **V4 Bootstrap:** Adapt V3 bootstrap to packed layout
2. **Fused Kernels:** GPU kernels operating directly on packed data
3. **Hoisting for V4:** Apply rotation hoisting to butterfly operations
4. **Multi-GPU:** Distribute batches across multiple GPUs
5. **Vulkan Backend:** Cross-platform GPU support

## License

See [LICENSE](LICENSE) file in the repository root.
