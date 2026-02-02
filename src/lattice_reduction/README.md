# Lattice Reduction Module

GA-accelerated lattice reduction for cryptanalysis.

## Overview

This module implements lattice reduction algorithms with Geometric Algebra (GA) optimizations for analyzing post-quantum cryptographic schemes.

## Structure

```
lattice_reduction/
├── mod.rs              # Module documentation and exports
├── rotor_nd.rs         # n-Dimensional rotors (Week 2)
├── lll_baseline.rs     # Standard LLL algorithm (✅ COMPLETE)
├── svp_challenge.rs    # SVP Challenge parser (✅ COMPLETE)
└── README.md           # This file
```

## Components

### 1. LLL Baseline (`lll_baseline.rs`)

**Status**: ✅ **COMPLETE & TESTED**

Standard Lenstra-Lenstra-Lovász algorithm using Gram-Schmidt orthogonalization.

**Features**:
- Modified GSO with μ coefficient matrix (O(n²) storage)
- Configurable Lovász constant δ (typically 0.99)
- Statistics tracking (swaps, size reductions, GSO updates)
- Hermite factor computation

**Performance**:
- Dimension 10: ~20 µs
- Dimension 20: ~8 µs
- Dimension 30: ~22 µs

**Usage**:
```rust
use ga_engine::lattice_reduction::lll_baseline::LLL;

let basis = vec![
    vec![12.0, 2.0],
    vec![5.0, 13.0],
];

let mut lll = LLL::new(basis, 0.99);
lll.reduce();

let reduced_basis = lll.get_basis();
let hermite_factor = lll.hermite_factor();
println!("Hermite factor: {:.6}", hermite_factor);
```

**Tests**: 3/3 passing ✅

### 2. SVP Challenge Parser (`svp_challenge.rs`)

**Status**: ✅ **COMPLETE & TESTED**

Parses lattice files from [SVP Challenge](https://www.latticechallenge.org/svp-challenge/).

**Features**:
- Multi-line bracket format parsing
- Arbitrary precision integer conversion to f64
- Shortest vector finding
- Dimension validation

**Supported Dimensions**: 40-140 (tested)
- Dimensions 40-100: Full precision
- Dimensions 110-140: f64 overflow (entries → inf)

**Usage**:
```rust
use ga_engine::lattice_reduction::svp_challenge;

let basis = svp_challenge::parse_lattice_file(
    "data/lattices/svpchallengedim40seed0.txt"
)?;

let (idx, norm) = svp_challenge::find_shortest_vector(&basis);
println!("Shortest: b{} with norm {:.6e}", idx, norm);
```

**Tests**: 4/4 passing ✅

### 3. n-Dimensional Rotors (`rotor_nd.rs`)

**Status**: ⚠️ **PARTIAL** (Week 2)

Implements rotors (oriented rotations) in n-dimensional Clifford algebra Cl(n,0).

**What Works**:
- Identity rotors ✅
- Rotor composition ✅
- Construction from two vectors ✅
- Unit rotor verification ✅

**What Needs Work**:
- `apply()` method (sandwich product R·v·R†) ⚠️
- Numerical stability for large n

**Tests**: 4/6 passing (2 tests depend on apply())

**Deferred**: Not needed for baseline LLL. Will fix for GA-accelerated variant (Week 2).

## Examples

### Parse & Display

```bash
cargo run --release --example test_svp_parse_only
```

Parses all SVP Challenge files (dim 40-140) and displays statistics.

### LLL Correctness Verification

```bash
cargo run --release --example test_lll_synthetic
```

Runs LLL on synthetic lattices (dim 10-30) to verify correctness.

### LLL on SVP Challenge

```bash
cargo run --release --example test_lll_svp40
```

⚠️ **Warning**: Takes >10 minutes. SVP Challenge instances are specifically designed to be hard for standard LLL.

## Testing

```bash
# All lattice reduction tests
cargo test --lib lattice_reduction

# Specific components
cargo test --lib lattice_reduction::lll_baseline
cargo test --lib lattice_reduction::svp_challenge
cargo test --lib lattice_reduction::rotor_nd
```

**Current Status**: 11/13 tests passing
- LLL: 3/3 ✅
- Parser: 4/4 ✅
- Rotors: 4/6 ✅ (2 deferred to Week 2)

## Known Limitations

### 1. f64 Precision

**Issue**: SVP Challenge uses arbitrary precision integers. Our f64-based implementation:
- Works well for dimensions 40-100
- Overflows for dimensions 110-140 (entries → inf)

**Impact**: Cannot accurately process very large dimensions

**Solution**: Implement arbitrary precision arithmetic (Week 3+) or work with normalized lattices

### 2. Hard Instance Performance

**Issue**: Standard LLL takes >10 minutes on SVP Challenge dim 40

**Why**:
- Lattices specifically designed to be hard
- Huge entries (~10^120) cause numerical instability
- LLL complexity: O(n^4 log B)

**Solution**: Need stronger algorithms (BKZ-2.0, Week 2-3)

### 3. Rotor Implementation

**Issue**: Sandwich product formula needs fixing

**Impact**: Cannot use GA-accelerated projections yet

**Solution**: Week 2 priority - derive correct formula or use matrix representation

## Development Roadmap

### Week 1: Foundation (✅ COMPLETE)
- [x] Standard LLL baseline
- [x] SVP Challenge parser
- [x] Validation framework
- [~] n-Dimensional rotors (partial, deferred)

### Week 2: Optimization
- [ ] Fix rotor sandwich product
- [ ] Implement BKZ-2.0
- [ ] Profile and optimize LLL
- [ ] GA-accelerated projections

### Week 3: Advanced Algorithms
- [ ] Progressive BKZ
- [ ] Enumeration optimizations
- [ ] Arbitrary precision arithmetic
- [ ] Parallel processing

### Week 4: Benchmarking
- [ ] Compare: LLL vs BKZ vs GA-accelerated
- [ ] Test on dimensions 40-100
- [ ] Performance profiling
- [ ] Quality metrics (Hermite factors)

### Week 5: Reproducibility
- [ ] Documentation
- [ ] Benchmark suite
- [ ] Result visualization
- [ ] Technical documentation

## References

1. **LLL Algorithm**:
   - Lenstra, Lenstra, Lovász (1982): "Factoring Polynomials with Rational Coefficients"
   - Nguyen, Stehlé (2009): "Low-dimensional Lattice Basis Reduction Revisited"

2. **SVP Challenge**:
   - https://www.latticechallenge.org/svp-challenge/
   - Schneider et al.: "SVP Challenge" database

3. **Geometric Algebra**:
   - Doran, Lasenby (2003): "Geometric Algebra for Physicists"
   - Dorst, Fontijne, Mann (2007): "Geometric Algebra for Computer Science"

4. **Cryptanalysis**:
   - Albrecht et al. (2017): "Estimate all the LWE, NTRU schemes!"
   - Chen, Nguyen (2011): "BKZ 2.0: Better Lattice Security Estimates"

## Contributing

This is research code for lattice reduction with geometric algebra optimizations.

## License

Apache 2.0 - See LICENSE file in repository root.
