//! Benchmark: Clifford Ring S for Cryptographic Operations
//!
//! This tests the math expert's suggestion: use the closed ring S ⊂ M₈(ℝ)
//! as an algebraic structure for crypto, where S ≅ Cl(3,0).
//!
//! Key insight: S is closed under addition and multiplication!
//! - Addition: ρ(a) + ρ(b) = ρ(a+b) ∈ S  (52 ns via GA)
//! - Multiplication: ρ(a)·ρ(b) = ρ(ab) ∈ S  (52 ns via GA geometric product!)
//!
//! Classical 8×8 matrix operations:
//! - Addition: 64 element-wise adds (~10 ns)
//! - Multiplication: 512 FMAs (~82 ns)
//!
//! So for multiplication, GA gives 1.58× speedup!
//!
//! Question: Can we build NTRU-like crypto over S instead of Z[x]/(x^N - 1)?

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::clifford_ring::{CliffordRingElement, CliffordPolynomial};

/// Benchmark: Basic ring operations
fn bench_ring_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("clifford_ring_ops");

    // Test elements
    let a = CliffordRingElement::from_multivector([1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5]);
    let b = CliffordRingElement::from_multivector([2.0, 1.0, 4.0, 3.0, 1.0, 1.0, 1.0, 1.0]);

    group.bench_function("addition", |bencher| {
        bencher.iter(|| {
            black_box(a.add(black_box(&b)))
        })
    });

    group.bench_function("multiplication", |bencher| {
        bencher.iter(|| {
            black_box(a.multiply(black_box(&b)))
        })
    });

    group.bench_function("scalar_mul", |bencher| {
        bencher.iter(|| {
            black_box(a.scalar_mul(black_box(3.14159)))
        })
    });

    group.finish();
}

/// Benchmark: Polynomial operations over Clifford ring
///
/// This is like NTRU but with coefficients in S instead of Z[x]
fn bench_polynomial_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("clifford_polynomial");

    // Create small polynomials (degree 4, like mini-NTRU)
    let f = CliffordPolynomial::new(vec![
        CliffordRingElement::from_multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]);

    let g = CliffordPolynomial::new(vec![
        CliffordRingElement::from_multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]);

    group.bench_function("poly_add_deg4", |bencher| {
        bencher.iter(|| {
            black_box(f.add(black_box(&g)))
        })
    });

    group.bench_function("poly_mult_deg4", |bencher| {
        bencher.iter(|| {
            black_box(f.multiply(black_box(&g)))
        })
    });

    // Larger polynomials (degree 8)
    let f8 = CliffordPolynomial::new(vec![
        CliffordRingElement::from_multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]); 8
    ]);

    let g8 = CliffordPolynomial::new(vec![
        CliffordRingElement::from_multivector([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]); 8
    ]);

    group.bench_function("poly_mult_deg8", |bencher| {
        bencher.iter(|| {
            black_box(f8.multiply(black_box(&g8)))
        })
    });

    group.finish();
}

/// Theoretical analysis: What dimension can we achieve?
///
/// Standard NTRU: Z[x]/(x^N - 1) with N=509, 677, 821
/// - Each coefficient: 1 integer (small, -1/0/+1 for ternary)
/// - Total space: N integers
///
/// Clifford NTRU: S[x]/(x^N - 1) where S ≅ Cl(3,0)
/// - Each coefficient: 8 floats (one multivector)
/// - Total space: 8N floats
///
/// For same total space:
/// Classical N=509 ints ≈ Clifford N=64 multivectors (509/8 ≈ 64)
///
/// Security question: Is S[x]/(x^64 - 1) as secure as Z[x]/(x^509 - 1)?
/// - S is 8-dimensional over ℝ
/// - Effective dimension: 64 × 8 = 512 ≈ 509 ✓
/// - But: working over ℝ vs Z may have different hardness!
fn bench_dimension_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension_comparison");

    // N=8: Very small polynomial
    let f_n8 = CliffordPolynomial::new(vec![
        CliffordRingElement::from_multivector([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]); 8
    ]);

    let g_n8 = CliffordPolynomial::new(vec![
        CliffordRingElement::from_multivector([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]); 8
    ]);

    group.bench_function("N=8_clifford_poly_mult", |bencher| {
        bencher.iter(|| {
            black_box(f_n8.multiply(black_box(&g_n8)))
        })
    });

    // N=16: Moderate size
    let f_n16 = CliffordPolynomial::new(vec![
        CliffordRingElement::from_multivector([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]); 16
    ]);

    let g_n16 = CliffordPolynomial::new(vec![
        CliffordRingElement::from_multivector([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]); 16
    ]);

    group.bench_function("N=16_clifford_poly_mult", |bencher| {
        bencher.iter(|| {
            black_box(f_n16.multiply(black_box(&g_n16)))
        })
    });

    // N=32: Target size (32 × 8 = 256 effective dimension)
    let f_n32 = CliffordPolynomial::new(vec![
        CliffordRingElement::from_multivector([1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]); 32
    ]);

    let g_n32 = CliffordPolynomial::new(vec![
        CliffordRingElement::from_multivector([1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0]); 32
    ]);

    group.bench_function("N=32_clifford_poly_mult", |bencher| {
        bencher.iter(|| {
            black_box(f_n32.multiply(black_box(&g_n32)))
        })
    });

    group.finish();
}

/// Comparison with classical polynomial multiplication
fn bench_vs_classical(c: &mut Criterion) {
    let mut group = c.benchmark_group("clifford_vs_classical");

    // Classical: polynomial over integers (standard NTRU)
    let classical_f = vec![1i64, 1, 0, 1, -1, 0, 1, 1];
    let classical_g = vec![1i64, 0, 1, 1, 0, -1, 1, 0];

    group.bench_function("classical_poly_mult_N8", |bencher| {
        bencher.iter(|| {
            let mut result = vec![0i64; 15]; // degree 7 + 7 = 14
            for i in 0..8 {
                for j in 0..8 {
                    result[i + j] += classical_f[i] * classical_g[j];
                }
            }
            black_box(result)
        })
    });

    // Clifford: polynomial over S
    let clifford_f = CliffordPolynomial::new(vec![
        CliffordRingElement::from_multivector([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]);

    let clifford_g = CliffordPolynomial::new(vec![
        CliffordRingElement::from_multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        CliffordRingElement::from_multivector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]);

    group.bench_function("clifford_poly_mult_N8", |bencher| {
        bencher.iter(|| {
            black_box(clifford_f.multiply(black_box(&clifford_g)))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_ring_operations,
    bench_polynomial_operations,
    bench_dimension_comparison,
    bench_vs_classical
);
criterion_main!(benches);
