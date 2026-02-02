# Clifford FHE: Version History and Technical Overview

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Version 1 (V1): Proof of Concept](#version-1-v1-proof-of-concept)
3. [Version 2 (V2): Production CKKS Backend](#version-2-v2-production-ckks-backend)
4. [Version 3 (V3): Full Bootstrap](#version-3-v3-full-bootstrap)
5. [Version 4 (V4): Packed Slot-Interleaved](#version-4-v4-packed-slot-interleaved)
6. [Version 5 (V5): Privacy-Trace Collection](#version-5-v5-privacy-trace-collection)
7. [Version Comparison Matrix](#version-comparison-matrix)
8. [Packing Strategies Comparison](#packing-strategies-comparison)
9. [Build and Run](#build-and-run)
10. [Cryptographic Parameters](#cryptographic-parameters)
11. [References](#references)

---

## Executive Summary

**Clifford FHE** is a fully homomorphic encryption system that combines CKKS lattice-based encryption with Clifford algebra (geometric algebra) for efficient encrypted geometric computation. The project has evolved through five major versions:

| Version | Focus | Key Innovation | Status |
|---------|-------|----------------|--------|
| **V1** | Proof of Concept | First FHE for geometric algebra | Deprecated |
| **V2** | Production Backend | Multi-GPU (Metal, CUDA) + NTT optimization | Production |
| **V3** | Unlimited Depth | CKKS bootstrapping with sine approximation | Production |
| **V4** | Memory Efficiency | Slot-interleaved packing (8× memory reduction) | Production |
| **V5** | Privacy Research | Execution-trace collection and privacy analysis | Production |

### Key Achievement

The system demonstrates that **representation choice affects execution-trace privacy**:
- **CKKS**: Leaks input dimensions via rotation counts (100% attack accuracy)
- **CliffordFHE**: Fixed 64-multiplication structure eliminates dimension leakage (random-guessing accuracy)

## Version 1 (V1): Proof of Concept

### Overview

V1 was the initial proof-of-concept demonstrating that Clifford algebra operations could be performed homomorphically using CKKS encryption.

### File Structure

```
src/clifford_fhe_v1/
├── mod.rs                      # Module exports
├── params.rs                   # Parameter sets (128/192/256-bit security)
├── ckks_rns.rs                # RNS-CKKS encryption/decryption, NTT
├── rns.rs                      # Residue Number System core operations
├── keys_rns.rs                # Key generation (public/secret/evaluation)
├── geometric_product_rns.rs    # Homomorphic geometric products (2D/3D)
├── slot_encoding.rs            # SIMD slot encoding using FFT
├── canonical_embedding.rs      # CKKS canonical embedding
├── automorphisms.rs            # Galois automorphisms
├── rotation_keys.rs            # Rotation key generation
└── geometric_nn.rs             # Encrypted geometric neural networks
```

### Technical Details

**Multivector Representation (Cl(3,0))**:
```
M = a₀·1 + a₁·e₁ + a₂·e₂ + a₃·e₃ + a₄·e₁₂ + a₅·e₂₃ + a₆·e₃₁ + a₇·e₁₂₃
```

**Ciphertext Layout**: Component-separate (8 independent ciphertexts per multivector)

**Key Data Structures**:
```rust
pub struct RnsCiphertext {
    pub c0: RnsPolynomial,       // First component
    pub c1: RnsPolynomial,       // Second component
    pub level: usize,            // Current level in modulus chain
    pub scale: f64,              // Current scaling factor
    pub n: usize,                // Ring dimension
}

pub struct RnsPolynomial {
    pub rns_coeffs: Vec<Vec<i64>>,  // [N coefficients][L residues]
    pub n: usize,
    pub level: usize,
    pub domain: Domain,          // Coef or NTT
}
```

**Geometric Product Implementation**:
- Uses structure constants for Cl(2,0) and Cl(3,0)
- 2D: 8 homomorphic multiplications
- 3D: 32 homomorphic multiplications
- Each multiplication: O(N log N) via NTT

### Supported Operations

| Operation | 2D Components | 3D Components | Multiplications |
|-----------|---------------|---------------|-----------------|
| Geometric Product | 4 | 8 | 8 (2D) / 32 (3D) |
| Wedge Product | 4 | 8 | 16 (2D) / 64 (3D) |
| Inner Product | 4 | 8 | 16 (2D) / 64 (3D) |
| Rotation (R·v·R̃) | 4 | 8 | 2× geometric product |

### Limitations

- **No GPU acceleration** (CPU-only)
- **No ciphertext packing** (inefficient slot utilization)
- **No bootstrapping** (limited circuit depth: 8-12 levels)
- **Naive polynomial multiplication** before NTT optimization

### Status

**Deprecated** - Preserved for historical reference. All functionality superseded by V2.

## Version 2 (V2): Production CKKS Backend

### Overview

V2 is a complete rewrite providing production-quality CKKS implementation with multi-platform GPU support and significant performance improvements (10-100× over V1).

### File Structure

```
src/clifford_fhe_v2/
├── mod.rs                               # Module organization
├── params.rs                            # FHE parameter management
├── inversion.rs                         # Homomorphic division (Newton-Raphson)
├── core/
│   ├── traits.rs                        # CliffordFHE trait abstraction
│   └── types.rs                         # Backend enum, SecurityLevel
├── backends/
│   ├── cpu_optimized/
│   │   ├── ckks.rs                     # CKKS encode/encrypt/decrypt
│   │   ├── keys.rs                     # KeyContext, key generation
│   │   ├── ntt.rs                      # Harvey butterfly NTT
│   │   ├── rns.rs                      # Barrett reduction
│   │   ├── multiplication.rs           # Ciphertext multiplication
│   │   ├── geometric.rs                # Cl(3,0) geometric operations
│   │   └── simd/                       # AVX2/NEON/Scalar backends
│   ├── gpu_cuda/
│   │   ├── device.rs                   # CUDA context
│   │   ├── ckks.rs                     # GPU CKKS operations
│   │   ├── ntt.rs                      # CUDA NTT kernels
│   │   ├── geometric.rs                # GPU geometric product
│   │   ├── rotation.rs                 # GPU-accelerated rotations
│   │   └── rotation_keys.rs            # GPU rotation key generation
│   └── gpu_metal/
│       ├── device.rs                   # Metal device context
│       ├── ckks.rs                     # Metal CKKS operations
│       ├── ntt.rs                      # Metal compute shader NTT
│       ├── geometric.rs                # Metal geometric product
│       ├── bootstrap.rs                # Bootstrap support
│       └── hoisting.rs                 # Key hoisting optimization
```

### Key Innovations

**1. Harvey Butterfly NTT**
- O(N log N) polynomial multiplication (vs O(N²) naive)
- Negacyclic convolution for ring Z[x]/(x^N + 1)
- Twisted NTT with precomputed twiddle factors
- In-place computation

**2. Barrett Reduction**
- ~2× faster than native modulo operator
- Formula: `x mod q ≈ x - q * floor(x * mu / 2^64)`
- Eliminates expensive division operations

**3. Multi-Backend Architecture**
```rust
pub trait CliffordFHE {
    type Ciphertext: Clone;
    type Plaintext: Clone;

    fn keygen(params) -> (PublicKey, SecretKey, EvaluationKey);
    fn encrypt(pk, pt, params) -> Ciphertext;
    fn decrypt(sk, ct, params) -> Plaintext;
    fn geometric_product_3d(a, b, evk, params) -> [Ciphertext; 8];
    // ... additional operations
}
```

**4. SIMD Optimization**
- Runtime CPU feature detection (AVX2, NEON, Scalar)
- Vectorized butterfly operations
- 2-4× additional speedup on supported hardware

### Performance Comparison (V1 → V2)

| Operation | V1 (CPU) | V2 (CPU) | V2 (CUDA) | V2 (Metal) |
|-----------|----------|----------|-----------|------------|
| Geometric Product | 13.0s | 220ms | 20-25ms | 30-50ms |
| Key Generation | 52ms | 16ms | 10ms | 12ms |
| NTT (per prime) | ~500ms | 2-3ms | <1ms | <1ms |
| **Speedup vs V1** | 1× | 59× | 520-650× | 260-430× |

### Ciphertext Packing: Component-Separate Layout

**Layout**: 8 independent ciphertexts per multivector
```
Multivector M = [c₀, c₁, c₂, c₃, c₄, c₅, c₆, c₇]
                 ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
              Enc(s) Enc(e₁) Enc(e₂) Enc(e₃) Enc(e₁₂) Enc(e₂₃) Enc(e₃₁) Enc(e₁₂₃)
```

**Advantages**:
- Direct component access (no rotations needed)
- Simple implementation
- Optimal for single-multivector operations

**Disadvantages**:
- 8× memory overhead
- 8× ciphertext management overhead

### Homomorphic Division: Newton-Raphson Inversion

V2 introduces a significant cryptographic innovation: **homomorphic division** via Newton-Raphson iterative inversion. This addresses a fundamental limitation in FHE: most schemes cannot perform division on encrypted data, and those that do rely on expensive binary circuit decomposition.

#### The Problem with FHE Division

Standard CKKS and other arithmetic FHE schemes natively support:
- Addition: `Enc(a) + Enc(b) → Enc(a + b)`
- Multiplication: `Enc(a) × Enc(b) → Enc(a × b)`

But division `Enc(a) / Enc(b)` has no direct operation. Traditional approaches:
- **Binary circuit decomposition**: Convert to O(n log n) bit-level gates—prohibitively expensive
- **Avoid division entirely**: Restructure algorithms to eliminate division—limits applicability

#### Our Solution: Newton-Raphson Inversion

We compute division by computing the multiplicative inverse of the denominator, then multiplying:

```
a / b = a × (1/b) = a × inverse(b)
```

The inverse is computed using Newton-Raphson iteration:

```
x_{n+1} = x_n × (2 - b × x_n)
```

This iteration converges quadratically to `1/b` when started with a good initial guess.

#### Algorithm Details

**File**: `src/clifford_fhe_v2/inversion.rs`

```rust
pub fn newton_raphson_inverse(
    ct: &Ciphertext,           // Encrypted denominator
    initial_guess: f64,        // Starting point (e.g., 1/mean_expected_value)
    iterations: usize,         // Convergence iterations (typically 3-5)
    evk: &EvaluationKey,
    key_ctx: &KeyContext,
    pk: &PublicKey,
) -> Ciphertext

pub fn scalar_division(
    numerator: &Ciphertext,    // Enc(a)
    denominator: &Ciphertext,  // Enc(b)
    initial_guess: f64,
    nr_iterations: usize,
    ...
) -> Ciphertext               // Returns Enc(a/b)
```

**Iteration Breakdown**:
```
Each Newton-Raphson iteration:
  1. Multiply: temp = b × x_n           (1 ciphertext mult)
  2. Subtract: temp = 2 - temp          (1 plaintext sub)
  3. Multiply: x_{n+1} = x_n × temp     (1 ciphertext mult)
  4. Rescale: maintain scale            (1 rescale)

Total per iteration: 2 multiplications + 1 rescale
```

#### Depth and Precision Analysis

| Iterations | Multiplicative Depth | Precision (\|error\|) | Use Case |
|------------|----------------------|-----------------------|----------|
| 3 | 6 levels | ~10⁻³ | Low precision, fast |
| 4 | 8 levels | ~10⁻⁴ | Standard applications |
| 5 | 10 levels | ~10⁻⁶ | High precision |

**Depth Cost**: O(k) for k iterations, vs O(n log n) for binary circuit approach

#### GPU Implementations

**CUDA Backend**: `src/clifford_fhe_v2/backends/gpu_cuda/inversion.rs`
- Kernel-fused NR iterations
- Shared memory for intermediate results
- 20-32× speedup over binary circuits

**Metal Backend**: `src/clifford_fhe_v2/backends/gpu_metal/inversion.rs`
- Metal compute shader pipeline
- Optimized for Apple Silicon
- Comparable speedup to CUDA

#### Related Operations

The Newton-Raphson machinery enables several derived operations:

```rust
// Magnitude squared: |v|² for vectors
pub fn magnitude_squared(v: &[Ciphertext; 3], ...) -> Ciphertext

// Vector inverse: 1/|v|²
pub fn vector_inverse(v: &[Ciphertext; 3], ...) -> Ciphertext

// Full vector division: a/b for multivectors
pub fn vector_division(
    numerator: &[Ciphertext; 8],
    denominator: &[Ciphertext; 8],
    ...
) -> [Ciphertext; 8]
```

#### Performance Comparison

| Approach | Depth | Time (N=8192, CPU) | Time (GPU) |
|----------|-------|-------------------|------------|
| Binary circuit | O(n log n) | ~10-20 seconds | ~1-2 seconds |
| Newton-Raphson (5 iter) | O(1) = 10 | ~200-400 ms | ~20-40 ms |
| **Speedup** | **>100×** | **20-50×** | **20-50×** |

#### Applications Enabled

Homomorphic division enables encrypted computation of:
- **Normalization**: `v / |v|` for unit vectors
- **Similarity metrics**: Cosine similarity requires division
- **Geometric algebra**: Inverse of multivectors
- **Machine learning**: Softmax, layer normalization, attention mechanisms

This capability is essential for the full expressiveness of CliffordFHE in encrypted geometric computation.

### Status

**Production Candidate** - Foundation for V3, V4, and V5. All backends (CPU, Metal, CUDA) fully functional.

## Version 3 (V3): Full Bootstrap

### Overview

V3 adds **CKKS bootstrapping** capability, enabling unlimited circuit depth by refreshing ciphertext noise without decryption.

### File Structure

```
src/clifford_fhe_v3/
├── mod.rs                           # Module definition
├── params.rs                        # Bootstrap-aware parameters
├── prime_gen.rs                     # Dynamic NTT-friendly prime generation
├── bootstrapping/
│   ├── bootstrap_context.rs         # Main API: BootstrapContext
│   ├── mod_raise.rs                 # Stage 1: Modulus raising
│   ├── coeff_to_slot.rs             # Stage 2: Coefficient→Slot transform
│   ├── eval_mod.rs                  # Stage 3: Homomorphic modular reduction
│   ├── slot_to_coeff.rs             # Stage 4: Slot→Coefficient transform
│   ├── sin_approx.rs                # Sine polynomial approximation
│   ├── keys.rs                      # Rotation key generation
│   ├── rotation.rs                  # Homomorphic rotation
│   └── cuda_*.rs                    # CUDA bootstrap implementations
└── batched/
    ├── encoding.rs                  # Batch encoding (512 multivectors)
    ├── extraction.rs                # Component extraction
    └── geometric.rs                 # Batched geometric product
```

### Bootstrap Pipeline

```
Input: Noisy ciphertext at level 0-1
       ↓
┌──────────────────┐
│ 1. ModRaise      │  Lift to higher modulus (~10ms)
└──────────────────┘
       ↓
┌──────────────────┐
│ 2. CoeffToSlot   │  FFT-like transform (~200ms)
│    (O(log N)     │  Uses log₂(N) rotations
│     rotations)   │
└──────────────────┘
       ↓
┌──────────────────┐
│ 3. EvalMod       │  Homomorphic modular reduction (~500ms)
│    sin(2πx/q)    │  Uses Taylor/Chebyshev polynomial
└──────────────────┘
       ↓
┌──────────────────┐
│ 4. SlotToCoeff   │  Inverse FFT transform (~200ms)
└──────────────────┘
       ↓
Output: Fresh ciphertext at level ≥10
```

### Key Innovation: Sine Approximation for Modular Reduction

**Mathematical Basis**:
```
x mod q ≈ x - (q/2π) · sin(2πx/q)
```

**Implementation**:
- Taylor series: `sin(x) = x - x³/6 + x⁵/120 - x⁷/5040 + ...`
- Horner's method for polynomial evaluation
- Configurable precision via polynomial degree

**Parameter Presets**:
| Preset | Sin Degree | Bootstrap Levels | Precision |
|--------|------------|------------------|-----------|
| `fast()` | 15 | 10 | 10⁻³ |
| `balanced()` | 23 | 12 | 10⁻⁴ |
| `conservative()` | 31 | 15 | 10⁻⁶ |

### V3 Batching: SIMD Multivector Packing

**Layout**: 512 multivectors in single CKKS ciphertext (N=8192)
```
Slot Layout (4096 slots, 8 components each):
[mv₀.s, mv₀.e₁, mv₀.e₂, ..., mv₀.e₁₂₃, mv₁.s, mv₁.e₁, ..., mv₅₁₁.e₁₂₃]
```

**Throughput Improvement**: 512× (amortized bootstrap cost)

### Performance

| Platform | Bootstrap Time | Notes |
|----------|----------------|-------|
| CPU (Apple M3 Max) | ~70s | Reference implementation |
| Metal GPU (Apple M3 Max) | 71.37s | 100% GPU execution |
| CUDA GPU (NVIDIA RTX 4090) | **11.69s** | **6× faster than CPU** |

See [BENCHMARKS.md](BENCHMARKS.md) for detailed breakdown.

### Ciphertext Structure

**Still uses Component-Separate layout** (8 ciphertexts per multivector)

This is intentional:
- Bootstrap operates per-ciphertext
- Batching is at the slot level (512 scalars per component ciphertext)
- Maintains compatibility with V2 geometric operations

### Limitations

- Bootstrap consumes 40 levels (12 CoeffToSlot + 16 EvalMod + 12 SlotToCoeff)
- Rotation key generation: ~90 seconds (CPU), ~30 seconds (GPU)
- Batched bootstrap not fully implemented (skeleton only)

### Status

**Production Candidate** - Full bootstrap capability demonstrated. Metal and CUDA GPU backends operational.

## Version 4 (V4): Packed Slot-Interleaved

### Overview

V4 introduces a novel **slot-interleaved packing scheme** that stores all 8 multivector components in a single CKKS ciphertext, achieving 8× memory reduction.

### File Structure

```
src/clifford_fhe_v4/
├── mod.rs                    # Module organization
├── params.rs                 # V4 parameter definitions
├── packed_multivector.rs     # Core packed data structure
├── packing.rs                # Naive rotation-based packing
├── packing_butterfly.rs      # Optimized butterfly algorithm
├── packing_cuda.rs           # CUDA-specific packing
├── cuda_adapter.rs           # CUDA API compatibility
├── cuda_context.rs           # Unified CUDA context
├── mult_table.rs             # Clifford multiplication table
├── geometric_ops.rs          # Packed geometric operations
└── bootstrapping/mod.rs      # Placeholder for Phase 3
```

### Slot-Interleaved Layout

**Memory Organization** (N=1024, 512 slots):
```
┌─ Multivector 0 ─┐  ┌─ Multivector 1 ─┐      ┌─ Multivector 63 ─┐
│ s  e₁ e₂ e₃     │  │ s  e₁ e₂ e₃     │ ···  │ s  e₁ e₂ e₃      │
│ e₁₂ e₂₃ e₃₁ I   │  │ e₁₂ e₂₃ e₃₁ I   │ ···  │ e₁₂ e₂₃ e₃₁ I    │
└─────────────────┘  └─────────────────┘      └──────────────────┘
  Slots [0-7]          Slots [8-15]             Slots [504-511]
```

**Slot Index Formula**: `slot = batch_idx × 8 + component_idx`

**Batch Capacity**:
| Ring Dimension | CKKS Slots | Batch Size |
|----------------|------------|------------|
| N=1024 | 512 | 64 multivectors |
| N=2048 | 1024 | 128 multivectors |
| N=8192 | 4096 | 512 multivectors |

### Key Innovation: Butterfly Packing Transform

**Problem**: Naive packing requires 7 rotations (one per component 1-7)

**Solution**: Butterfly network reduces to 3 rotation stages

**Pack Algorithm (3 Stages)**:
```
Stage 1: Combine pairs via rot(1)
  q0 = c0 + c1·rot(1)    // Components (0,1)
  q1 = c2 + c3·rot(1)    // Components (2,3)
  q2 = c4 + c5·rot(1)    // Components (4,5)
  q3 = c6 + c7·rot(1)    // Components (6,7)

Stage 2: Combine quads via rot(2)
  h0 = q0 + q1·rot(2)    // Components (0,1,2,3)
  h1 = q2 + q3·rot(2)    // Components (4,5,6,7)

Stage 3: Final combine via rot(4)
  packed = h0 + h1·rot(4)

Total: 3 unique rotation distances (1, 2, 4)
```

**Unpack Algorithm**: Inverse butterfly with coefficient-wise negation

**Rotation Comparison**:
| Method | Pack Rotations | Unpack Rotations | Total |
|--------|----------------|------------------|-------|
| Naive | 7 | 7 | 14 |
| Butterfly | 3 | 3 | 6 |
| **Improvement** | 2.3× | 2.3× | **2.3×** |

### PackedMultivector Data Structure

```rust
pub struct PackedMultivector {
    pub ct: Ciphertext,          // Single CKKS ciphertext
    pub batch_size: usize,        // Number of multivectors (64 for N=1024)
    pub n: usize,                 // Ring dimension
    pub num_primes: usize,        // RNS primes at current level
    pub level: usize,             // Current CKKS level
    pub scale: f64,               // Scaling factor
}
```

### Geometric Operations on Packed Data

**Architecture**: Unpack → Compute → Pack

```
Input: 2 PackedMultivectors (64 multivectors each)
       ↓
┌──────────────────────────────────┐
│ 1. Unpack (butterfly inverse)    │  6-8 rotation kernels
└──────────────────────────────────┘
       ↓
┌──────────────────────────────────┐
│ 2. Compute (per-prime GPU)       │  64 ciphertext multiplications
│    Using Metal/CUDA              │  Parallel across RNS primes
└──────────────────────────────────┘
       ↓
┌──────────────────────────────────┐
│ 3. Pack (butterfly forward)      │  3-4 rotation kernels
└──────────────────────────────────┘
       ↓
Output: Single PackedMultivector with 64 results
```

### Memory Savings

| Metric | V2/V3 (Component-Separate) | V4 (Slot-Interleaved) | Improvement |
|--------|----------------------------|-----------------------|-------------|
| Ciphertexts per MV | 8 | 1 | **8×** |
| Memory per MV (N=1024) | 512 KB | 64 KB | **8×** |
| GPU Memory (64 batches) | 32 MB | 4 MB | **8×** |

### Trade-offs

| Aspect | V2/V3 | V4 |
|--------|-------|-----|
| Per-operation latency | Baseline | 2-4× slower |
| Throughput | 1 MV/op | 64 MV/op |
| Memory | 8× | 1× |
| Rotation overhead | 0 | 6-14 per batch |
| Implementation complexity | Simple | Moderate |

### When to Use V4

**Use V4 when**:
- Processing batches of 64+ multivectors
- Memory bandwidth is the bottleneck
- GPU utilization needs maximization
- Cost per operation is critical

**Use V2/V3 when**:
- Single multivector operations
- Low latency is critical
- Heterogeneous batch sizes
- Interactive applications

### Status

**Production Candidate** - Metal and CUDA backends operational. Bootstrap integration planned.

## Version 5 (V5): Privacy-Trace Collection

### Overview

V5 is a **non-invasive instrumentation layer** for privacy research, enabling systematic analysis of execution-trace leakage in encrypted computation.

### File Structure

```
src/clifford_fhe_v5/
├── mod.rs              # Module entry point
├── trace.rs            # ExecutionTrace data structures
├── collector.rs        # Trace aggregation & management
├── workloads.rs        # Workload definitions
├── cpu.rs              # TracedCpuBackend
├── metal.rs            # TracedMetalBackend
├── cuda.rs             # TracedCudaBackend
├── analysis.rs         # Information-theoretic analysis
└── ml_classifier.rs    # Privacy attack classifiers
```

### Design Philosophy

**Non-invasive**: Wraps V2 operations without modifying cryptographic computations
**Retrocompatible**: Maintains compatibility with existing papers
**Fair comparison**: Enables identical-computation-different-representation comparison

### TracedCpuBackend Architecture

```rust
pub struct TracedCpuBackend {
    pub ckks_ctx: CkksContext,          // V2 CKKS context
    pub geom_ctx: GeometricContext,     // V2 Geometric algebra context
    pub key_ctx: KeyContext,            // V2 Key generation context
    pub params: CliffordFHEParams,
    pub pk: PublicKey,
    pub sk: SecretKey,
    pub evk: EvaluationKey,
}
```

**Pattern**: Each operation is wrapped with timing instrumentation:
1. Start `OperationTimer`
2. Call V2 backend method
3. Record trace event with metadata

### Execution Trace Structure

```rust
pub struct ExecutionTrace {
    pub workload_type: String,           // "similarity", "dot_product", etc.
    pub representation: String,          // "ckks" or "clifford"
    pub backend: String,                 // "cpu", "metal", "cuda"
    pub ring_dimension: usize,           // FHE parameter N
    pub num_primes: usize,               // RNS prime count
    pub input_metadata: InputMetadata,   // Input characteristics
    pub events: Vec<OperationEvent>,     // Sequence of operations
    pub summary: TraceSummary,           // Aggregated statistics
}

pub struct OperationEvent {
    pub op_type: OperationType,          // 35 operation types
    pub duration: Duration,              // Wall-clock time
    pub level_before: usize,
    pub level_after: usize,
    pub rotation_count: usize,
    pub rotation_amounts: Vec<i32>,      // Specific amounts (1,2,4,8,...)
    pub rescale_count: usize,
    pub relin_count: usize,
    pub bootstrap_occurred: bool,
}
```

### Privacy Attack Classifiers

V5 implements **6 attack classifiers** demonstrating execution-trace leakage:

#### 1. Dimension Inference Attack
- **Goal**: Predict input dimension from trace
- **Mechanism**: CKKS rotation count = log₂(dimension)
- **CKKS Accuracy**: ~100%
- **CliffordFHE Accuracy**: ~17% (random guessing)

#### 2. Task Identification Attack
- **Goal**: Identify computation type
- **Features**: Operation sequences, rotation patterns

#### 3. Sparsity Inference Attack
- **Goal**: Infer input sparsity from timing
- **Buckets**: [0.0, 0.25, 0.5, 0.75, 0.9]

#### 4. Multi-Tenant Linkability Attack
- **Goal**: Link traces from same user across sessions
- **Metrics**: Link accuracy, TPR, FPR

#### 5. Operation Count Attack
- **Goal**: Fingerprint via total operation count
- **CKKS**: Variable (O(log n))
- **CliffordFHE**: Fixed (64 multiplications)

#### 6. Trace Length Attack
- **Goal**: Infer dimension from event count
- **CKKS**: Multiple distinct lengths
- **CliffordFHE**: Single length for all inputs

### Information-Theoretic Analysis

```rust
pub struct LeakageAnalysis {
    pub mutual_information: f64,         // I(D; T) bits leaked
    pub dimension_entropy: f64,          // H(D) total possible
    pub conditional_entropy: f64,        // H(D|T) remaining uncertainty
    pub leakage_ratio: f64,              // I/H (0 = perfect privacy)
}
```

**Key Results** (6 dimension classes):

| Metric | CKKS | CliffordFHE |
|--------|------|-------------|
| H(Dimension) | 2.585 bits | 2.585 bits |
| H(Dimension \| Trace) | 0.000 bits | 2.585 bits |
| I(Dimension; Trace) | 2.585 bits | 0.000 bits |
| **Leakage Ratio** | **100%** | **0%** |

### Relationship to Earlier Versions

```
┌─────────────────────────────────────────────────┐
│       V5: Trace Collection Layer                │
│  (Non-invasive instrumentation)                 │
├─────────────────────────────────────────────────┤
│  V5 Backend Wrappers                            │
│  TracedCPU │ TracedMetal │ TracedCUDA           │
├─────────────────────────────────────────────────┤
│  V2 Backend (Unchanged)                         │
│  CkksContext │ GeometricContext │ KeyContext    │
└─────────────────────────────────────────────────┘
```

**V5 wraps but does NOT modify V2**.

### Status

**Production Candidate** - Complete privacy analysis framework for encrypted computation research.

## Version Comparison Matrix

### Feature Matrix

| Feature | V1 | V2 | V3 | V4 | V5 |
|---------|:--:|:--:|:--:|:--:|:--:|
| Encryption/Decryption | ✓ | ✓ | ✓ | ✓ | ✓ |
| Geometric Product | ✓ | ✓ | ✓ | ✓ | ✓ |
| Wedge/Inner Product | ✓ | ✓ | ✓ | ✓ | ✓ |
| Homomorphic Division | - | ✓ | ✓ | ✓ | ✓ |
| CPU Backend | ✓ | ✓ | ✓ | ✓ | ✓ |
| Metal GPU | - | ✓ | ✓ | ✓ | ✓ |
| CUDA GPU | - | ✓ | ✓ | ✓ | ✓ |
| NTT Optimization | - | ✓ | ✓ | ✓ | ✓ |
| SIMD (AVX2/NEON) | - | ✓ | ✓ | ✓ | ✓ |
| Bootstrap | - | - | ✓ | Planned | - |
| Slot-Interleaved | - | - | - | ✓ | - |
| Privacy Analysis | - | - | - | - | ✓ |

### Performance Evolution

| Metric | V1 | V2 | V3 | V4 |
|--------|---:|---:|---:|---:|
| Geometric Product (CPU) | 13.0s | 220ms | 220ms | N/A |
| Geometric Product (GPU) | N/A | 20-50ms | 20-50ms | 36ms (batched) |
| Memory per MV | 8 CT | 8 CT | 8 CT | 1 CT |
| Throughput | 1× | 1× | 512× (batch) | 64× (batch) |
| Max Depth | 8-12 | 8-12 | Unlimited | 8-12 |

## Packing Strategies Comparison

### Two Packing Paradigms

| Aspect | Component-Separate (V2/V3) | Slot-Interleaved (V4) |
|--------|----------------------------|----------------------|
| **Layout** | 8 ciphertexts per MV | 1 ciphertext per MV |
| **Rotations for GP** | 0 | 6-14 |
| **Memory** | 8× baseline | 1× baseline |
| **Batch capacity** | N/16 (SIMD) | N/16 (slot interleaved) |
| **Complexity** | Simple | Moderate |

### Privacy Implications

| Property | Component-Separate | Slot-Interleaved |
|----------|-------------------|------------------|
| Dimension privacy | ✓ Perfect (0 rotations) | ✓ Perfect (fixed rotations) |
| Rotation signature | 0 rotations | 3-7 fixed rotations |
| System fingerprinting | Low surface | Higher surface |
| Query count privacy | Visible | Obfuscated by batching |

**Key Insight**: Both achieve dimension privacy because rotation count is constant (0 for V2/V3, fixed for V4), but V4 introduces a distinguishing trace signature.

## Build and Run

### Feature Flags

```bash
# V2 Backends
cargo build --release --features v2,v2-cpu-optimized
cargo build --release --features v2,v2-gpu-metal
cargo build --release --features v2,v2-gpu-cuda

# V3 Bootstrap
cargo build --release --features v3,v2-gpu-cuda

# V4 Packed
cargo build --release --features v4,v2-gpu-metal
cargo build --release --features v4,v2-gpu-cuda

# V5 Privacy Analysis
cargo build --release --features v5
```

### Key Examples

```bash
# V3 Bootstrap Demo
cargo run --release --features v3,v2-gpu-cuda --example test_v3_full_bootstrap

# V4 Geometric Product
cargo run --release --features v4,v2-gpu-cuda --example bench_v4_cuda_geometric_quick

# V5 Privacy Attack Analysis
cargo run --release --features v5 --example v5_privacy_attacks
```

## Cryptographic Parameters

### Standard Parameters (N=8192)

```rust
CliffordFHEParams {
    n: 8192,                     // Ring dimension
    moduli: [q₀, q₁, ..., q₁₄], // 15 RNS primes (~60 bits each)
    scale: 2^40,                 // CKKS scaling factor
    error_std: 3.2,              // Gaussian error stddev
}
```

### Security Estimates

| Ring Dimension | Modulus Size | Security Level |
|----------------|--------------|----------------|
| N=1024 | ~180 bits | ~50 bits (testing only) |
| N=8192 | ~900 bits | ~128 bits (NIST L1) |
| N=16384 | ~900 bits | ~192 bits (NIST L3) |
| N=32768 | ~900 bits | ~256 bits (NIST L5) |

### NTT-Friendly Prime Requirement

All primes satisfy: **q ≡ 1 (mod 2N)**

This ensures primitive 2N-th roots of unity exist for efficient negacyclic NTT.

## References

### Clifford Algebra / Geometric Algebra

- **Hestenes, D. & Sobczyk, G.** "Clifford Algebra to Geometric Calculus" (1984)
- **Dorst, L., et al.** "Geometric Algebra for Computer Science" (2007)
- **Hildenbrand, D.** "Foundations of Geometric Algebra Computing" (2013)

### CKKS and FHE

- **Cheon et al.** "Homomorphic Encryption for Arithmetic of Approximate Numbers" (ASIACRYPT 2017)
- **Gentry, C.** "Fully Homomorphic Encryption Using Ideal Lattices" (STOC 2009)
- **Gentry et al.** "Homomorphic Evaluation of the AES Circuit" (CRYPTO 2012)

### This Project

*Publication information withheld for anonymous review.*

## License

Apache License 2.0 - See [LICENSE](LICENSE) file

## Contact

*Contact information withheld for anonymous review.*
