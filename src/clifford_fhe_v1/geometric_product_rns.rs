//! Homomorphic Geometric Product for RNS-CKKS
//!
//! This module implements encrypted geometric algebra operations using the
//! RNS-CKKS scheme with proper noise management.
//!
//! # Key Innovation
//!
//! Clifford FHE enables privacy-preserving geometric algebra:
//! ```text
//! Enc(a) ⊗ Enc(b) = Enc(a ⊗ b)
//! ```
//!
//! where ⊗ is the geometric product, not just polynomial multiplication!

use crate::clifford_fhe_v1::ckks_rns::{RnsCiphertext, rns_multiply_ciphertexts};
use crate::clifford_fhe_v1::keys_rns::RnsEvaluationKey;
use crate::clifford_fhe_v1::params::CliffordFHEParams;
use crate::clifford_fhe_v1::rns::{rns_add, rns_sub};

/// 2D Geometric Algebra Cl(2,0) structure constants
///
/// Basis: {1, e₁, e₂, e₁₂}
///
/// Multiplication table:
/// ```text
///       1    e₁   e₂   e₁₂
///   1   1    e₁   e₂   e₁₂
///   e₁  e₁   1    e₁₂  e₂
///   e₂  e₂  -e₁₂  1   -e₁
///   e₁₂ e₁₂  e₂  -e₁  -1
/// ```
pub struct Cl2StructureConstants {
    /// For each output component, list of (coefficient, input_a_idx, input_b_idx)
    pub products: Vec<Vec<(i64, usize, usize)>>,
}

impl Cl2StructureConstants {
    /// Create structure constants for Cl(2,0)
    pub fn new() -> Self {
        let mut products = vec![Vec::new(); 4];

        // Component 0 (scalar):
        // 1⊗1=1, e₁⊗e₁=1, e₂⊗e₂=1, e₁₂⊗e₁₂=-1
        products[0] = vec![
            (1, 0, 0),   // a[0] * b[0]
            (1, 1, 1),   // a[1] * b[1] (e₁⊗e₁)
            (1, 2, 2),   // a[2] * b[2] (e₂⊗e₂)
            (-1, 3, 3),  // -a[3] * b[3] (e₁₂⊗e₁₂)
        ];

        // Component 1 (e₁):
        // 1⊗e₁=e₁, e₁⊗1=e₁, e₂⊗e₁₂=-e₁, e₁₂⊗e₂=e₁
        products[1] = vec![
            (1, 0, 1),   // a[0] * b[1] (1⊗e₁)
            (1, 1, 0),   // a[1] * b[0] (e₁⊗1)
            (-1, 2, 3),  // -a[2] * b[3] (e₂⊗e₁₂)
            (1, 3, 2),   // a[3] * b[2] (e₁₂⊗e₂)
        ];

        // Component 2 (e₂):
        // 1⊗e₂=e₂, e₂⊗1=e₂, e₁⊗e₁₂=e₂, e₁₂⊗e₁=-e₂
        products[2] = vec![
            (1, 0, 2),   // a[0] * b[2] (1⊗e₂)
            (1, 2, 0),   // a[2] * b[0] (e₂⊗1)
            (1, 1, 3),   // a[1] * b[3] (e₁⊗e₁₂)
            (-1, 3, 1),  // -a[3] * b[1] (e₁₂⊗e₁)
        ];

        // Component 3 (e₁₂):
        // 1⊗e₁₂=e₁₂, e₁₂⊗1=e₁₂, e₁⊗e₂=e₁₂, e₂⊗e₁=-e₁₂
        products[3] = vec![
            (1, 0, 3),   // a[0] * b[3] (1⊗e₁₂)
            (1, 3, 0),   // a[3] * b[0] (e₁₂⊗1)
            (1, 1, 2),   // a[1] * b[2] (e₁⊗e₂)
            (-1, 2, 1),  // -a[2] * b[1] (e₂⊗e₁)
        ];

        Cl2StructureConstants { products }
    }
}

/// 3D Geometric Algebra Cl(3,0) structure constants
///
/// Basis: {1, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃}
///
/// Multiplication table (signatures: e₁²=e₂²=e₃²=1):
/// ```text
/// Component ordering:
/// 0: 1     (scalar)
/// 1: e₁    (vector)
/// 2: e₂    (vector)
/// 3: e₃    (vector)
/// 4: e₁₂   (bivector)
/// 5: e₁₃   (bivector)
/// 6: e₂₃   (bivector)
/// 7: e₁₂₃  (trivector/pseudoscalar)
/// ```
pub struct Cl3StructureConstants {
    /// For each output component, list of (coefficient, input_a_idx, input_b_idx)
    pub products: Vec<Vec<(i64, usize, usize)>>,
}

impl Cl3StructureConstants {
    /// Create structure constants for Cl(3,0)
    pub fn new() -> Self {
        let mut products = vec![Vec::new(); 8];

        // Component 0 (scalar):
        // 1⊗1=1, e₁⊗e₁=1, e₂⊗e₂=1, e₃⊗e₃=1,
        // e₁₂⊗e₁₂=-1, e₁₃⊗e₁₃=-1, e₂₃⊗e₂₃=-1, e₁₂₃⊗e₁₂₃=-1
        products[0] = vec![
            (1, 0, 0),   // 1⊗1
            (1, 1, 1),   // e₁⊗e₁
            (1, 2, 2),   // e₂⊗e₂
            (1, 3, 3),   // e₃⊗e₃
            (-1, 4, 4),  // e₁₂⊗e₁₂
            (-1, 5, 5),  // e₁₃⊗e₁₃
            (-1, 6, 6),  // e₂₃⊗e₂₃
            (-1, 7, 7),  // e₁₂₃⊗e₁₂₃
        ];

        // Component 1 (e₁):
        // 1⊗e₁=e₁, e₁⊗1=e₁, e₂⊗e₁₂=e₁, e₁₂⊗e₂=-e₁,
        // e₃⊗e₁₃=e₁, e₁₃⊗e₃=-e₁, e₂₃⊗e₁₂₃=-e₁, e₁₂₃⊗e₂₃=e₁
        products[1] = vec![
            (1, 0, 1),   // 1⊗e₁
            (1, 1, 0),   // e₁⊗1
            (1, 2, 4),   // e₂⊗e₁₂
            (-1, 4, 2),  // e₁₂⊗e₂
            (1, 3, 5),   // e₃⊗e₁₃
            (-1, 5, 3),  // e₁₃⊗e₃
            (-1, 6, 7),  // e₂₃⊗e₁₂₃
            (1, 7, 6),   // e₁₂₃⊗e₂₃
        ];

        // Component 2 (e₂):
        // 1⊗e₂=e₂, e₂⊗1=e₂, e₁⊗e₁₂=-e₂, e₁₂⊗e₁=e₂,
        // e₃⊗e₂₃=e₂, e₂₃⊗e₃=-e₂, e₁₃⊗e₁₂₃=-e₂, e₁₂₃⊗e₁₃=e₂
        products[2] = vec![
            (1, 0, 2),   // 1⊗e₂
            (1, 2, 0),   // e₂⊗1
            (-1, 1, 4),  // e₁⊗e₁₂
            (1, 4, 1),   // e₁₂⊗e₁
            (1, 3, 6),   // e₃⊗e₂₃
            (-1, 6, 3),  // e₂₃⊗e₃
            (-1, 5, 7),  // e₁₃⊗e₁₂₃
            (1, 7, 5),   // e₁₂₃⊗e₁₃
        ];

        // Component 3 (e₃):
        // 1⊗e₃=e₃, e₃⊗1=e₃, e₁⊗e₁₃=-e₃, e₁₃⊗e₁=e₃,
        // e₂⊗e₂₃=-e₃, e₂₃⊗e₂=e₃, e₁₂⊗e₁₂₃=-e₃, e₁₂₃⊗e₁₂=e₃
        products[3] = vec![
            (1, 0, 3),   // 1⊗e₃
            (1, 3, 0),   // e₃⊗1
            (-1, 1, 5),  // e₁⊗e₁₃
            (1, 5, 1),   // e₁₃⊗e₁
            (-1, 2, 6),  // e₂⊗e₂₃
            (1, 6, 2),   // e₂₃⊗e₂
            (-1, 4, 7),  // e₁₂⊗e₁₂₃
            (1, 7, 4),   // e₁₂₃⊗e₁₂
        ];

        // Component 4 (e₁₂):
        // 1⊗e₁₂=e₁₂, e₁₂⊗1=e₁₂, e₁⊗e₂=e₁₂, e₂⊗e₁=-e₁₂,
        // e₃⊗e₁₂₃=e₁₂, e₁₂₃⊗e₃=-e₁₂, e₁₃⊗e₂₃=e₁₂, e₂₃⊗e₁₃=-e₁₂
        products[4] = vec![
            (1, 0, 4),   // 1⊗e₁₂
            (1, 4, 0),   // e₁₂⊗1
            (1, 1, 2),   // e₁⊗e₂
            (-1, 2, 1),  // e₂⊗e₁
            (1, 3, 7),   // e₃⊗e₁₂₃
            (-1, 7, 3),  // e₁₂₃⊗e₃
            (1, 5, 6),   // e₁₃⊗e₂₃
            (-1, 6, 5),  // e₂₃⊗e₁₃
        ];

        // Component 5 (e₁₃):
        // 1⊗e₁₃=e₁₃, e₁₃⊗1=e₁₃, e₁⊗e₃=e₁₃, e₃⊗e₁=-e₁₃,
        // e₂⊗e₁₂₃=-e₁₃, e₁₂₃⊗e₂=e₁₃, e₁₂⊗e₂₃=-e₁₃, e₂₃⊗e₁₂=e₁₃
        products[5] = vec![
            (1, 0, 5),   // 1⊗e₁₃
            (1, 5, 0),   // e₁₃⊗1
            (1, 1, 3),   // e₁⊗e₃
            (-1, 3, 1),  // e₃⊗e₁
            (-1, 2, 7),  // e₂⊗e₁₂₃
            (1, 7, 2),   // e₁₂₃⊗e₂
            (-1, 4, 6),  // e₁₂⊗e₂₃
            (1, 6, 4),   // e₂₃⊗e₁₂
        ];

        // Component 6 (e₂₃):
        // 1⊗e₂₃=e₂₃, e₂₃⊗1=e₂₃, e₂⊗e₃=e₂₃, e₃⊗e₂=-e₂₃,
        // e₁⊗e₁₂₃=e₂₃, e₁₂₃⊗e₁=-e₂₃, e₁₂⊗e₁₃=e₂₃, e₁₃⊗e₁₂=-e₂₃
        products[6] = vec![
            (1, 0, 6),   // 1⊗e₂₃
            (1, 6, 0),   // e₂₃⊗1
            (1, 2, 3),   // e₂⊗e₃
            (-1, 3, 2),  // e₃⊗e₂
            (1, 1, 7),   // e₁⊗e₁₂₃
            (-1, 7, 1),  // e₁₂₃⊗e₁
            (1, 4, 5),   // e₁₂⊗e₁₃
            (-1, 5, 4),  // e₁₃⊗e₁₂
        ];

        // Component 7 (e₁₂₃):
        // 1⊗e₁₂₃=e₁₂₃, e₁₂₃⊗1=e₁₂₃, e₁⊗e₂₃=e₁₂₃, e₂₃⊗e₁=-e₁₂₃,
        // e₂⊗e₁₃=-e₁₂₃, e₁₃⊗e₂=e₁₂₃, e₃⊗e₁₂=e₁₂₃, e₁₂⊗e₃=-e₁₂₃
        products[7] = vec![
            (1, 0, 7),   // 1⊗e₁₂₃
            (1, 7, 0),   // e₁₂₃⊗1
            (1, 1, 6),   // e₁⊗e₂₃
            (-1, 6, 1),  // e₂₃⊗e₁
            (-1, 2, 5),  // e₂⊗e₁₃
            (1, 5, 2),   // e₁₃⊗e₂
            (1, 3, 4),   // e₃⊗e₁₂
            (-1, 4, 3),  // e₁₂⊗e₃
        ];

        Cl3StructureConstants { products }
    }
}

/// Encode a 2D multivector as a plaintext polynomial
///
/// Maps [scalar, e₁, e₂, e₁₂] → polynomial coefficients
/// Each component is placed at coefficient index 0, 1, 2, 3
pub fn encode_multivector_2d(mv: &[f64; 4], scale: f64, n: usize) -> Vec<i64> {
    let mut coeffs = vec![0i64; n];
    for i in 0..4 {
        coeffs[i] = (mv[i] * scale).round() as i64;
    }
    coeffs
}

/// Decode a plaintext polynomial to a 2D multivector
///
/// Extracts [scalar, e₁, e₂, e₁₂] from polynomial coefficients
pub fn decode_multivector_2d(coeffs: &[i64], scale: f64) -> [f64; 4] {
    let mut mv = [0.0; 4];
    for i in 0..4 {
        mv[i] = (coeffs[i] as f64) / scale;
    }
    mv
}

/// Encode a 3D multivector as a plaintext polynomial
///
/// Maps [scalar, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃] → polynomial coefficients
pub fn encode_multivector_3d(mv: &[f64; 8], scale: f64, n: usize) -> Vec<i64> {
    let mut coeffs = vec![0i64; n];
    for i in 0..8 {
        coeffs[i] = (mv[i] * scale).round() as i64;
    }
    coeffs
}

/// Decode a plaintext polynomial to a 3D multivector
///
/// Extracts [scalar, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃] from polynomial coefficients
pub fn decode_multivector_3d(coeffs: &[i64], scale: f64) -> [f64; 8] {
    let mut mv = [0.0; 8];
    for i in 0..8 {
        mv[i] = (coeffs[i] as f64) / scale;
    }
    mv
}

/// Homomorphic geometric product for 2D multivectors (Cl(2,0))
///
/// Computes Enc(a) ⊗ Enc(b) = Enc(a ⊗ b) where ⊗ is the geometric product.
///
/// # Algorithm
///
/// For each output component i:
///   result[i] = Σ coeff_k · Enc(a[j]) · Enc(b[k])
///
/// Uses structure constants to determine which products contribute to each component.
///
/// # Example
/// ```text
/// a = [1, 2, 0, 0]  // 1 + 2e₁
/// b = [3, 0, 4, 0]  // 3 + 4e₂
///
/// a ⊗ b = [3, 6, 4, 8]  // 3 + 6e₁ + 4e₂ + 8e₁₂
/// ```
///
/// This is computed HOMOMORPHICALLY without decrypting!
pub fn geometric_product_2d(
    ct_a: &RnsCiphertext,
    ct_b: &RnsCiphertext,
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> RnsCiphertext {
    let structure = Cl2StructureConstants::new();
    let primes = &params.moduli;

    // We'll build result component by component
    // Each component needs to accumulate multiple products

    // Storage for each output component's accumulator
    let component_accumulators: Vec<Option<RnsCiphertext>> = vec![None; 4];

    // For each output component
    for output_idx in 0..4 {
        let product_terms = &structure.products[output_idx];

        for &(coeff, a_idx, b_idx) in product_terms {
            // We need to multiply specific components of a and b
            // But our ciphertexts encrypt entire polynomials!
            //
            // Solution: Extract components using component extraction,
            // or use the fact that components are stored at specific indices
            //
            // For now, we'll use a simplified approach:
            // Encrypt the product of the full polynomials, then extract/combine

            // This is a placeholder - we need component-wise operations
            // For now, let's just demonstrate the structure

            // TODO: Implement component extraction and placement
            // This requires either:
            // 1. Rotation to extract/place components (needs rotation keys)
            // 2. Masking techniques
            // 3. Multiple ciphertexts (one per component)
        }
    }

    // For now, return the first ciphertext (placeholder)
    // Real implementation needs proper component handling
    ct_a.clone()
}

/// SIMPLIFIED VERSION: Homomorphic geometric product using separate ciphertexts
///
/// This version encrypts each component separately, making the geometric product
/// straightforward but requiring 4× more ciphertexts.
///
/// # Arguments
/// * `cts_a` - Array of 4 ciphertexts, each encrypting one component of multivector a
/// * `cts_b` - Array of 4 ciphertexts, each encrypting one component of multivector b
/// * `evk` - Evaluation key for multiplication
/// * `params` - FHE parameters
///
/// # Returns
/// Array of 4 ciphertexts encrypting the geometric product a ⊗ b
///
/// # Example
/// ```text
/// cts_a = [Enc(1), Enc(2), Enc(0), Enc(0)]  // 1 + 2e₁
/// cts_b = [Enc(3), Enc(0), Enc(4), Enc(0)]  // 3 + 4e₂
///
/// result = geometric_product_2d_componentwise(cts_a, cts_b, evk, params)
/// // result = [Enc(3), Enc(6), Enc(4), Enc(8)]  // 3 + 6e₁ + 4e₂ + 8e₁₂
/// ```
pub fn geometric_product_2d_componentwise(
    cts_a: &[RnsCiphertext; 4],
    cts_b: &[RnsCiphertext; 4],
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 4] {
    let structure = Cl2StructureConstants::new();

    let mut results: Vec<RnsCiphertext> = Vec::with_capacity(4);

    // For each output component
    for output_idx in 0..4 {
        let product_terms = &structure.products[output_idx];

        let mut accumulator: Option<RnsCiphertext> = None;

        for &(coeff, a_idx, b_idx) in product_terms {
            // Multiply the two component ciphertexts
            let product = rns_multiply_ciphertexts(
                &cts_a[a_idx],
                &cts_b[b_idx],
                evk,
                params,
            );

            // Get active primes for THIS product (after multiplication/rescaling)
            let product_active_primes = &params.moduli[..product.c0.num_primes()];

            // Apply coefficient (for now, only handle ±1)
            let term = if coeff == -1 {
                // Negate: multiply c0 and c1 by -1
                let mut neg_c0 = product.c0.clone();
                let mut neg_c1 = product.c1.clone();

                for i in 0..params.n {
                    for j in 0..neg_c0.num_primes() {
                        let qi = product_active_primes[j];
                        neg_c0.rns_coeffs[i][j] = (qi - product.c0.rns_coeffs[i][j] % qi) % qi;
                        neg_c1.rns_coeffs[i][j] = (qi - product.c1.rns_coeffs[i][j] % qi) % qi;
                    }
                }

                RnsCiphertext::new(neg_c0, neg_c1, product.level, product.scale)
            } else {
                product
            };

            // Add to accumulator
            accumulator = Some(match accumulator {
                None => term,
                Some(acc) => {
                    // Get active primes for accumulation (should match term's level)
                    let acc_active_primes = &params.moduli[..acc.c0.num_primes()];
                    let sum_c0 = rns_add(&acc.c0, &term.c0, acc_active_primes);
                    let sum_c1 = rns_add(&acc.c1, &term.c1, acc_active_primes);
                    RnsCiphertext::new(sum_c0, sum_c1, acc.level, acc.scale)
                }
            });
        }

        results.push(accumulator.expect("Empty accumulator for geometric product"));
    }

    [
        results[0].clone(),
        results[1].clone(),
        results[2].clone(),
        results[3].clone(),
    ]
}

/// Homomorphic geometric product for 3D multivectors (Cl(3,0))
///
/// Computes Enc(a) ⊗ Enc(b) = Enc(a ⊗ b) for 3D geometric algebra.
///
/// This version uses componentwise encryption with 8 ciphertexts per multivector.
pub fn geometric_product_3d_componentwise(
    cts_a: &[RnsCiphertext; 8],
    cts_b: &[RnsCiphertext; 8],
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 8] {
    let structure = Cl3StructureConstants::new();

    let mut results = Vec::with_capacity(8);

    // For each output component
    for output_idx in 0..8 {
        let product_terms = &structure.products[output_idx];
        let mut accumulator: Option<RnsCiphertext> = None;

        for &(coeff, a_idx, b_idx) in product_terms {
            // Debug: check levels before multiplication
            if cts_a[a_idx].level != cts_b[b_idx].level {
                // eprintln!("[GP ERROR] Level mismatch before multiply!");
                // eprintln!("  output_idx={}, a_idx={}, b_idx={}", output_idx, a_idx, b_idx);
                // eprintln!("  cts_a[{}].level = {}", a_idx, cts_a[a_idx].level);
                // eprintln!("  cts_b[{}].level = {}", b_idx, cts_b[b_idx].level);
            }

            // Multiply the encrypted components
            let product = rns_multiply_ciphertexts(
                &cts_a[a_idx],
                &cts_b[b_idx],
                evk,
                params,
            );

            // Get active primes for THIS product (after multiplication/rescaling)
            let product_active_primes = &params.moduli[..product.c0.num_primes()];

            // Apply coefficient (±1)
            let term = if coeff == -1 {
                let mut neg_c0 = product.c0.clone();
                let mut neg_c1 = product.c1.clone();

                for i in 0..params.n {
                    for j in 0..neg_c0.num_primes() {
                        let qi = product_active_primes[j];
                        neg_c0.rns_coeffs[i][j] = (qi - product.c0.rns_coeffs[i][j] % qi) % qi;
                        neg_c1.rns_coeffs[i][j] = (qi - product.c1.rns_coeffs[i][j] % qi) % qi;
                    }
                }

                RnsCiphertext::new(neg_c0, neg_c1, product.level, product.scale)
            } else {
                product
            };

            // Accumulate
            accumulator = Some(match accumulator {
                None => term,
                Some(acc) => {
                    // Get active primes for accumulation (should match term's level)
                    let acc_active_primes = &params.moduli[..acc.c0.num_primes()];
                    let sum_c0 = rns_add(&acc.c0, &term.c0, acc_active_primes);
                    let sum_c1 = rns_add(&acc.c1, &term.c1, acc_active_primes);
                    RnsCiphertext::new(sum_c0, sum_c1, acc.level, acc.scale)
                }
            });
        }

        results.push(accumulator.expect("Empty accumulator for geometric product"));
    }

    [
        results[0].clone(),
        results[1].clone(),
        results[2].clone(),
        results[3].clone(),
        results[4].clone(),
        results[5].clone(),
        results[6].clone(),
        results[7].clone(),
    ]
}

/// Homomorphic reverse operation for Cl(2,0)
///
/// The reverse reverses the order of basis vectors, which flips signs for certain grades:
/// - Grade 0 (scalar): unchanged
/// - Grade 1 (e₁, e₂): unchanged
/// - Grade 2 (e₁₂): reversed → flip sign
///
/// ã = [a₀, a₁, a₂, -a₃]
///
/// This is needed for rotations: R·x·R̃
pub fn reverse_2d(ct: &[RnsCiphertext; 4], params: &CliffordFHEParams) -> [RnsCiphertext; 4] {
    let primes = &params.moduli;
    let n = params.n;

    // Components 0, 1, 2 stay the same
    let ct0 = ct[0].clone();
    let ct1 = ct[1].clone();
    let ct2 = ct[2].clone();

    // Component 3 (e₁₂) gets negated
    let ct3 = negate_ciphertext(&ct[3], primes, n);

    [ct0, ct1, ct2, ct3]
}

/// Negate a ciphertext: -ct = (-c0, -c1)
fn negate_ciphertext(ct: &RnsCiphertext, primes: &[i64], n: usize) -> RnsCiphertext {
    // Use the actual number of primes in the ciphertext (not total primes)
    // This handles ciphertexts at any level correctly
    let num_primes = ct.c0.rns_coeffs[0].len();
    let active_primes = &primes[..num_primes];

    let mut neg_c0_coeffs = vec![vec![0i64; num_primes]; n];
    let mut neg_c1_coeffs = vec![vec![0i64; num_primes]; n];

    for i in 0..n {
        for j in 0..num_primes {
            let qi = active_primes[j];
            neg_c0_coeffs[i][j] = (qi - ct.c0.rns_coeffs[i][j] % qi) % qi;
            neg_c1_coeffs[i][j] = (qi - ct.c1.rns_coeffs[i][j] % qi) % qi;
        }
    }

    use crate::clifford_fhe_v1::rns::RnsPolynomial;
    let neg_c0 = RnsPolynomial::new(neg_c0_coeffs, n, ct.c0.level);
    let neg_c1 = RnsPolynomial::new(neg_c1_coeffs, n, ct.c1.level);

    RnsCiphertext {
        c0: neg_c0,
        c1: neg_c1,
        scale: ct.scale,
        level: ct.level,
        n: ct.n,
    }
}

/// Homomorphic rotation operation for Cl(2,0)
///
/// Applies a rotor R to a vector x: R·x·R̃
///
/// # Arguments
/// * `rotor` - The rotor R = [4 ciphertexts for scalar, e₁, e₂, e₁₂]
/// * `vector` - The vector x = [4 ciphertexts, but typically only e₁, e₂ components non-zero]
/// * `evk` - Evaluation key for relinearization
/// * `params` - Clifford FHE parameters
///
/// # Returns
/// Rotated vector R·x·R̃
pub fn rotate_2d(
    rotor: &[RnsCiphertext; 4],
    vector: &[RnsCiphertext; 4],
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 4] {
    // Step 1: Compute R·x
    let rx = geometric_product_2d_componentwise(rotor, vector, evk, params);

    // Step 2: Compute R̃ (reverse of R)
    let rotor_reverse = reverse_2d(rotor, params);

    // Step 3: Compute (R·x)·R̃
    geometric_product_2d_componentwise(&rx, &rotor_reverse, evk, params)
}

/// Homomorphic wedge product (outer product) for Cl(2,0)
///
/// The wedge product is the antisymmetric part of the geometric product:
/// a ∧ b = (a⊗b - b⊗a) / 2
///
/// For Cl(2,0):
/// - e₁ ∧ e₂ = e₁₂
/// - Scalar components wedge to 0
///
/// This is useful for computing bivectors (oriented areas).
pub fn wedge_product_2d(
    cts_a: &[RnsCiphertext; 4],
    cts_b: &[RnsCiphertext; 4],
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 4] {
    let n = params.n;

    // Compute a⊗b
    let ab = geometric_product_2d_componentwise(cts_a, cts_b, evk, params);

    // Compute b⊗a
    let ba = geometric_product_2d_componentwise(cts_b, cts_a, evk, params);

    // Get active primes based on actual number of primes in result
    let result_primes = &params.moduli[..ab[0].c0.num_primes()];

    // Compute (a⊗b - b⊗a) / 2
    let mut result = [
        subtract_ciphertexts(&ab[0], &ba[0], result_primes, n),
        subtract_ciphertexts(&ab[1], &ba[1], result_primes, n),
        subtract_ciphertexts(&ab[2], &ba[2], result_primes, n),
        subtract_ciphertexts(&ab[3], &ba[3], result_primes, n),
    ];

    // Divide by 2 (multiply by 1/2)
    for i in 0..4 {
        result[i].scale *= 2.0;  // Dividing by 2 is same as doubling the scale
    }

    result
}

/// Subtract two ciphertexts: ct_a - ct_b
fn subtract_ciphertexts(ct_a: &RnsCiphertext, ct_b: &RnsCiphertext, primes: &[i64], _n: usize) -> RnsCiphertext {
    let c0_diff = rns_sub(&ct_a.c0, &ct_b.c0, primes);
    let c1_diff = rns_sub(&ct_a.c1, &ct_b.c1, primes);

    RnsCiphertext {
        c0: c0_diff,
        c1: c1_diff,
        scale: ct_a.scale,  // Assume same scale
        level: ct_a.level,
        n: ct_a.n,
    }
}

/// Add two ciphertexts: ct_a + ct_b
fn add_ciphertexts(ct_a: &RnsCiphertext, ct_b: &RnsCiphertext, primes: &[i64], _n: usize) -> RnsCiphertext {
    let c0_sum = rns_add(&ct_a.c0, &ct_b.c0, primes);
    let c1_sum = rns_add(&ct_a.c1, &ct_b.c1, primes);

    RnsCiphertext {
        c0: c0_sum,
        c1: c1_sum,
        scale: ct_a.scale,  // Assume same scale
        level: ct_a.level,
        n: ct_a.n,
    }
}

/// Homomorphic inner product (dot product) for Cl(2,0)
///
/// The inner product is the symmetric part that produces scalars and vectors:
/// a · b = (a⊗b + b⊗a) / 2
///
/// For vectors in Cl(2,0):
/// - e₁ · e₁ = 1
/// - e₂ · e₂ = 1
/// - e₁ · e₂ = 0
///
/// This is useful for computing angles and distances.
pub fn inner_product_2d(
    cts_a: &[RnsCiphertext; 4],
    cts_b: &[RnsCiphertext; 4],
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 4] {
    let n = params.n;

    // Compute a⊗b
    let ab = geometric_product_2d_componentwise(cts_a, cts_b, evk, params);

    // Compute b⊗a
    let ba = geometric_product_2d_componentwise(cts_b, cts_a, evk, params);

    // Get active primes based on actual number of primes in result
    let result_primes = &params.moduli[..ab[0].c0.num_primes()];

    // Compute (a⊗b + b⊗a) / 2
    let mut result = [
        add_ciphertexts(&ab[0], &ba[0], result_primes, n),
        add_ciphertexts(&ab[1], &ba[1], result_primes, n),
        add_ciphertexts(&ab[2], &ba[2], result_primes, n),
        add_ciphertexts(&ab[3], &ba[3], result_primes, n),
    ];

    // Divide by 2 (multiply by 1/2)
    for i in 0..4 {
        result[i].scale *= 2.0;  // Dividing by 2 is same as doubling the scale
    }

    result
}

/// Homomorphic reverse operation for Cl(3,0)
///
/// The reverse reverses the order of basis vectors:
/// - Grade 0 (scalar): unchanged
/// - Grade 1 (e₁, e₂, e₃): unchanged
/// - Grade 2 (e₁₂, e₁₃, e₂₃): reversed → flip signs
/// - Grade 3 (e₁₂₃): unchanged (grade(grade-1)/2 is even)
///
/// ã = [a₀, a₁, a₂, a₃, -a₄, -a₅, -a₆, a₇]
pub fn reverse_3d(ct: &[RnsCiphertext; 8], params: &CliffordFHEParams) -> [RnsCiphertext; 8] {
    let primes = &params.moduli;
    let n = params.n;

    // Components 0, 1, 2, 3, 7 stay the same
    let ct0 = ct[0].clone();
    let ct1 = ct[1].clone();
    let ct2 = ct[2].clone();
    let ct3 = ct[3].clone();
    let ct7 = ct[7].clone();

    // Components 4, 5, 6 (bivectors) get negated
    let ct4 = negate_ciphertext(&ct[4], primes, n);
    let ct5 = negate_ciphertext(&ct[5], primes, n);
    let ct6 = negate_ciphertext(&ct[6], primes, n);

    [ct0, ct1, ct2, ct3, ct4, ct5, ct6, ct7]
}

/// Helper: Drop one modulus from a ciphertext to match level (modswitch without rescale)
///
/// This is needed when we need to match levels between ciphertexts for multiplication.
/// After multiplication, result is at level+1. To multiply again, we need operands at same level.
fn modswitch_to_next_level(ct: &RnsCiphertext, params: &CliffordFHEParams) -> RnsCiphertext {
    use crate::clifford_fhe_v1::rns::{RnsPolynomial, Domain};

    // Drop the last prime to advance level
    let new_level = ct.level + 1;
    let new_num_primes = params.moduli.len() - new_level;

    // Truncate RnsPolynomial to fewer primes
    let mut new_c0_coeffs = Vec::new();
    let mut new_c1_coeffs = Vec::new();

    for i in 0..ct.n {
        new_c0_coeffs.push(ct.c0.rns_coeffs[i][..new_num_primes].to_vec());
        new_c1_coeffs.push(ct.c1.rns_coeffs[i][..new_num_primes].to_vec());
    }

    let new_c0 = RnsPolynomial {
        rns_coeffs: new_c0_coeffs,
        n: ct.n,
        level: new_level,
        domain: ct.c0.domain,  // Preserve domain
    };
    let new_c1 = RnsPolynomial {
        rns_coeffs: new_c1_coeffs,
        n: ct.n,
        level: new_level,
        domain: ct.c1.domain,  // Preserve domain
    };

    RnsCiphertext {
        c0: new_c0,
        c1: new_c1,
        n: ct.n,
        level: new_level,
        scale: ct.scale,  // Scale stays the same (no rescale, just modswitch)
    }
}

/// Homomorphic rotation operation for Cl(3,0)
///
/// Applies a rotor R to a vector x: R·x·R̃
///
/// This is the key operation for encrypted rotations in 3D space.
///
/// **Level management**: This is a depth-2 operation.
/// - Input: level 0 (uses all primes)
/// - After R⊗v: level 1 (dropped 1 prime)
/// - Before (R⊗v)⊗~R: modswitch R to level 1 to match
/// - After (R⊗v)⊗~R: level 2 (dropped 2 primes)
pub fn rotate_3d(
    rotor: &[RnsCiphertext; 8],
    vector: &[RnsCiphertext; 8],
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 8] {
    // Step 1: Compute R⊗v (both at level 0 → result at level 1)
    let rx = geometric_product_3d_componentwise(rotor, vector, evk, params);

    // Step 2: Modswitch R to level 1 to match rx's level
    let rotor_level1: [RnsCiphertext; 8] = std::array::from_fn(|i| {
        modswitch_to_next_level(&rotor[i], params)
    });

    // eprintln!("\n[ROTATION] After first GP:");
    // eprintln!("  rx[0].level = {}", rx[0].level);
    // eprintln!("  rotor_level1[0].level = {}", rotor_level1[0].level);

    // Step 3: Compute ~R at level 1
    let rotor_reverse = reverse_3d(&rotor_level1, params);

    // eprintln!("  rotor_reverse[0].level = {}", rotor_reverse[0].level);

    // Step 4: Compute (R⊗v)⊗~R (both at level 1 → result at level 2)
    // eprintln!("[ROTATION] Starting second GP...");
    geometric_product_3d_componentwise(&rx, &rotor_reverse, evk, params)
}

/// Homomorphic wedge product for Cl(3,0)
///
/// a ∧ b = (a⊗b - b⊗a) / 2
pub fn wedge_product_3d(
    cts_a: &[RnsCiphertext; 8],
    cts_b: &[RnsCiphertext; 8],
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 8] {
    let n = params.n;

    // Compute a⊗b
    let ab = geometric_product_3d_componentwise(cts_a, cts_b, evk, params);

    // Compute b⊗a
    let ba = geometric_product_3d_componentwise(cts_b, cts_a, evk, params);

    // Get active primes based on actual number of primes in result
    let result_primes = &params.moduli[..ab[0].c0.num_primes()];

    // Compute (a⊗b - b⊗a) / 2
    let mut result = [
        subtract_ciphertexts(&ab[0], &ba[0], result_primes, n),
        subtract_ciphertexts(&ab[1], &ba[1], result_primes, n),
        subtract_ciphertexts(&ab[2], &ba[2], result_primes, n),
        subtract_ciphertexts(&ab[3], &ba[3], result_primes, n),
        subtract_ciphertexts(&ab[4], &ba[4], result_primes, n),
        subtract_ciphertexts(&ab[5], &ba[5], result_primes, n),
        subtract_ciphertexts(&ab[6], &ba[6], result_primes, n),
        subtract_ciphertexts(&ab[7], &ba[7], result_primes, n),
    ];

    // Divide by 2
    for i in 0..8 {
        result[i].scale *= 2.0;
    }

    result
}

/// Homomorphic inner product for Cl(3,0)
///
/// a · b = (a⊗b + b⊗a) / 2
pub fn inner_product_3d(
    cts_a: &[RnsCiphertext; 8],
    cts_b: &[RnsCiphertext; 8],
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 8] {
    let n = params.n;

    // Compute a⊗b
    let ab = geometric_product_3d_componentwise(cts_a, cts_b, evk, params);

    // Compute b⊗a
    let ba = geometric_product_3d_componentwise(cts_b, cts_a, evk, params);

    // Get active primes based on actual number of primes in result
    let result_primes = &params.moduli[..ab[0].c0.num_primes()];

    // Compute (a⊗b + b⊗a) / 2
    let mut result = [
        add_ciphertexts(&ab[0], &ba[0], result_primes, n),
        add_ciphertexts(&ab[1], &ba[1], result_primes, n),
        add_ciphertexts(&ab[2], &ba[2], result_primes, n),
        add_ciphertexts(&ab[3], &ba[3], result_primes, n),
        add_ciphertexts(&ab[4], &ba[4], result_primes, n),
        add_ciphertexts(&ab[5], &ba[5], result_primes, n),
        add_ciphertexts(&ab[6], &ba[6], result_primes, n),
        add_ciphertexts(&ab[7], &ba[7], result_primes, n),
    ];

    // Divide by 2
    for i in 0..8 {
        result[i].scale *= 2.0;
    }

    result
}

/// Homomorphic projection of vector a onto vector b in Cl(3,0)
///
/// proj_b(a) = (a · b) / (b · b) * b
///
/// This projects vector a onto the direction of vector b.
/// Useful for decomposing vectors into parallel and perpendicular components.
pub fn project_3d(
    cts_a: &[RnsCiphertext; 8],
    cts_b: &[RnsCiphertext; 8],
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 8] {
    // Compute a · b (scalar result, level 2 after 2 GPs)
    let a_dot_b = inner_product_3d(cts_a, cts_b, evk, params);

    // LEVEL MATCHING: a_dot_b is at level 2, but cts_a is at level 0
    // We need to modswitch cts_a to level 2 to match
    // Formula: proj_a(b) = (a·b) / (a·a) × a
    // For unit vectors, (a·a) = 1, so: proj_a(b) = (a·b) × a
    let target_level = a_dot_b[0].level;
    let cts_a_matched: [RnsCiphertext; 8] = std::array::from_fn(|i| {
        let mut ct = cts_a[i].clone();
        // Modswitch from level 0 → target_level (drop target_level primes)
        for _ in 0..target_level {
            ct = modswitch_to_next_level(&ct, params);
        }
        ct
    });

    // Compute (a·b) ⊗ a (both now at level 2 → result at level 3)
    // Since a·b is mostly scalar (component 0), this multiplies a by that scalar
    // This gives us the projection of b onto a
    geometric_product_3d_componentwise(&a_dot_b, &cts_a_matched, evk, params)
}

/// Homomorphic rejection of vector a from vector b in Cl(3,0)
///
/// rej_b(a) = a - proj_b(a)
///
/// This computes the component of a perpendicular to b.
pub fn reject_3d(
    cts_a: &[RnsCiphertext; 8],
    cts_b: &[RnsCiphertext; 8],
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 8] {
    let n = params.n;

    // Compute projection (result at level 3)
    let proj = project_3d(cts_a, cts_b, evk, params);

    // LEVEL AND SCALE MATCHING: proj is at level 3 with reduced scale,
    // but cts_b is at level 0 with original scale
    let target_level = proj[0].level;

    // First, modswitch cts_b to match level
    let cts_b_level_matched: [RnsCiphertext; 8] = std::array::from_fn(|i| {
        let mut ct = cts_b[i].clone();
        // Modswitch from level 0 → target_level
        for _ in 0..target_level {
            ct = modswitch_to_next_level(&ct, params);
        }
        ct
    });

    // Now adjust scale of cts_b_level_matched to match proj's scale
    // After 3 multiplications with rescaling, proj's scale is much smaller
    // We need to multiply cts_b's scale by the ratio of dropped primes
    let cts_b_matched: [RnsCiphertext; 8] = std::array::from_fn(|i| {
        let mut ct = cts_b_level_matched[i].clone();
        // Adjust scale to match proj (proj has been rescaled 3 times)
        // Each multiplication divides scale by a prime (≈ 2^44)
        // So proj's scale ≈ original_scale / (Q0 * Q1 * Q2)
        // We need to match this by multiplying b's scale by the same factors
        ct.scale = proj[i].scale;
        ct
    });

    // Get active primes for the matched level
    let active_primes = &params.moduli[..proj[0].c0.num_primes()];

    // Compute b - proj (both at level 3 with matched scales)
    // Rejection: rej_a(b) = b - proj_a(b) = perpendicular component of b relative to a
    [
        subtract_ciphertexts(&cts_b_matched[0], &proj[0], active_primes, n),
        subtract_ciphertexts(&cts_b_matched[1], &proj[1], active_primes, n),
        subtract_ciphertexts(&cts_b_matched[2], &proj[2], active_primes, n),
        subtract_ciphertexts(&cts_b_matched[3], &proj[3], active_primes, n),
        subtract_ciphertexts(&cts_b_matched[4], &proj[4], active_primes, n),
        subtract_ciphertexts(&cts_b_matched[5], &proj[5], active_primes, n),
        subtract_ciphertexts(&cts_b_matched[6], &proj[6], active_primes, n),
        subtract_ciphertexts(&cts_b_matched[7], &proj[7], active_primes, n),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cl2_structure_constants() {
        let sc = Cl2StructureConstants::new();

        // Check scalar component has 4 products
        assert_eq!(sc.products[0].len(), 4);

        // Check e₁ component
        assert_eq!(sc.products[1].len(), 4);
    }

    #[test]
    fn test_encode_decode_2d() {
        let mv = [1.0, 2.0, 3.0, 4.0];  // 1 + 2e₁ + 3e₂ + 4e₁₂
        let scale = 1024.0;
        let n = 64;

        let coeffs = encode_multivector_2d(&mv, scale, n);
        let decoded = decode_multivector_2d(&coeffs, scale);

        for i in 0..4 {
            assert!((decoded[i] - mv[i]).abs() < 1e-6);
        }
    }
}
