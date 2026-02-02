//! Batch Geometric Product Operations
//!
//! Homomorphic geometric product on batched multivectors.
//! Processes multiple geometric products in parallel using SIMD slots.
//!
//! # Performance
//!
//! - Single geometric product (V2 CPU): ~327ms at N=1024
//! - Batched (64× at N=1024): Expected ~30ms total = ~0.47ms per product (70× speedup)

use super::BatchedMultivector;
use super::extraction::{extract_all_components, reassemble_components};
use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{CkksContext, Ciphertext};
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::EvaluationKey;
use crate::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use crate::clifford_fhe_v3::bootstrapping::RotationKeys;
use rayon::prelude::*;

/// Cl(3,0) structure constants for geometric product
///
/// For each output component, stores list of (coefficient, input_a_idx, input_b_idx)
struct Cl3StructureConstants {
    products: Vec<Vec<(i64, usize, usize)>>,
}

impl Cl3StructureConstants {
    fn new() -> Self {
        let mut products = vec![Vec::new(); 8];

        // Component 0 (scalar)
        products[0] = vec![
            (1, 0, 0), (1, 1, 1), (1, 2, 2), (1, 3, 3),
            (-1, 4, 4), (-1, 5, 5), (-1, 6, 6), (-1, 7, 7),
        ];

        // Component 1 (e₁)
        products[1] = vec![
            (1, 0, 1), (1, 1, 0), (1, 2, 4), (-1, 4, 2),
            (1, 3, 5), (-1, 5, 3), (-1, 6, 7), (1, 7, 6),
        ];

        // Component 2 (e₂)
        products[2] = vec![
            (1, 0, 2), (1, 2, 0), (-1, 1, 4), (1, 4, 1),
            (1, 3, 6), (-1, 6, 3), (-1, 5, 7), (1, 7, 5),
        ];

        // Component 3 (e₃)
        products[3] = vec![
            (1, 0, 3), (1, 3, 0), (-1, 1, 5), (1, 5, 1),
            (-1, 2, 6), (1, 6, 2), (-1, 4, 7), (1, 7, 4),
        ];

        // Component 4 (e₁₂)
        products[4] = vec![
            (1, 0, 4), (1, 4, 0), (1, 1, 2), (-1, 2, 1),
            (1, 3, 7), (-1, 7, 3), (1, 5, 6), (-1, 6, 5),
        ];

        // Component 5 (e₁₃)
        products[5] = vec![
            (1, 0, 5), (1, 5, 0), (1, 1, 3), (-1, 3, 1),
            (-1, 2, 7), (1, 7, 2), (-1, 4, 6), (1, 6, 4),
        ];

        // Component 6 (e₂₃)
        products[6] = vec![
            (1, 0, 6), (1, 6, 0), (1, 2, 3), (-1, 3, 2),
            (1, 1, 7), (-1, 7, 1), (1, 4, 5), (-1, 5, 4),
        ];

        // Component 7 (e₁₂₃)
        products[7] = vec![
            (1, 0, 7), (1, 7, 0), (1, 1, 6), (-1, 6, 1),
            (-1, 2, 5), (1, 5, 2), (1, 3, 4), (-1, 4, 3),
        ];

        Cl3StructureConstants { products }
    }
}

/// Negate a ciphertext
fn negate_ciphertext(ct: &Ciphertext, ckks_ctx: &CkksContext) -> Ciphertext {
    let moduli: Vec<u64> = ckks_ctx.params.moduli[..=ct.level].to_vec();

    let neg_c0: Vec<_> = ct.c0.iter().map(|rns| {
        let negated_values: Vec<u64> = rns.values.iter()
            .zip(&moduli)
            .map(|(&val, &q)| if val == 0 { 0 } else { q - val })
            .collect();
        crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation::new(
            negated_values,
            moduli.clone(),
        )
    }).collect();

    let neg_c1: Vec<_> = ct.c1.iter().map(|rns| {
        let negated_values: Vec<u64> = rns.values.iter()
            .zip(&moduli)
            .map(|(&val, &q)| if val == 0 { 0 } else { q - val })
            .collect();
        crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation::new(
            negated_values,
            moduli.clone(),
        )
    }).collect();

    Ciphertext::new(neg_c0, neg_c1, ct.level, ct.scale)
}

/// Batch geometric product: a ⊗ b for all pairs in batches
///
/// Computes geometric product on all multivector pairs simultaneously.
/// The slot-level parallelism means the same ciphertext operations
/// compute the product for all multivector pairs in the batch.
///
/// # Algorithm
///
/// 1. Extract all 8 components from both batches (16 ciphertexts total)
/// 2. For each output component, compute 8 cross-terms:
///    - Multiply extracted component ciphertexts (operates on all batch items)
///    - Apply sign and accumulate
/// 3. Reassemble 8 result components into batched multivector
///
/// # Arguments
///
/// * `a_batch` - First batch of multivectors
/// * `b_batch` - Second batch of multivectors
/// * `rotation_keys` - Rotation keys for component extraction
/// * `evk` - Evaluation key for relinearization
/// * `ckks_ctx` - CKKS context
///
/// # Returns
///
/// Batch of result multivectors (a[i] ⊗ b[i] for all i)
///
/// # Performance
///
/// - Single geometric product: ~327ms (V2 CPU)
/// - Batched (64× at N=1024): ~X ms total for 64 products
pub fn geometric_product_batched(
    a_batch: &BatchedMultivector,
    b_batch: &BatchedMultivector,
    rotation_keys: &RotationKeys,
    evk: &EvaluationKey,
    ckks_ctx: &CkksContext,
) -> Result<BatchedMultivector, String> {
    assert_eq!(
        a_batch.batch_size, b_batch.batch_size,
        "Batch sizes must match"
    );

    let key_ctx = KeyContext::new(ckks_ctx.params.clone());
    let constants = Cl3StructureConstants::new();

    // Step 1: Extract all 8 components from each batch
    let a_components = extract_all_components(a_batch, rotation_keys, ckks_ctx)?;
    let b_components = extract_all_components(b_batch, rotation_keys, ckks_ctx)?;

    // Step 2: Compute 8 output components
    // Each output component is the sum of 8 cross-terms
    let mut result_components: Vec<Ciphertext> = Vec::with_capacity(8);

    for out_idx in 0..8 {
        let product_terms = &constants.products[out_idx];

        // Compute first term
        let (first_coeff, first_a_idx, first_b_idx) = product_terms[0];
        let mut accumulated = multiply_ciphertexts(
            &a_components[first_a_idx],
            &b_components[first_b_idx],
            evk,
            &key_ctx,
        );

        // Apply sign if negative
        if first_coeff < 0 {
            accumulated = negate_ciphertext(&accumulated, ckks_ctx);
        }

        // Accumulate remaining 7 terms
        for &(coeff, a_idx, b_idx) in &product_terms[1..] {
            let mut ct_product = multiply_ciphertexts(
                &a_components[a_idx],
                &b_components[b_idx],
                evk,
                &key_ctx,
            );

            // Apply sign
            if coeff < 0 {
                ct_product = negate_ciphertext(&ct_product, ckks_ctx);
            }

            // Accumulate
            accumulated = accumulated.add(&ct_product);
        }

        result_components.push(accumulated);
    }

    // Step 3: Reassemble into batched multivector
    let result_array: [Ciphertext; 8] = result_components.try_into()
        .map_err(|_| "Failed to convert to array".to_string())?;

    reassemble_components(
        &result_array,
        rotation_keys,
        ckks_ctx,
        a_batch.batch_size,
        a_batch.n,
    )
}

/// Parallel version of batch geometric product
///
/// Uses Rayon to parallelize computation of the 8 output components.
pub fn geometric_product_batched_parallel(
    a_batch: &BatchedMultivector,
    b_batch: &BatchedMultivector,
    rotation_keys: &RotationKeys,
    evk: &EvaluationKey,
    ckks_ctx: &CkksContext,
) -> Result<BatchedMultivector, String> {
    assert_eq!(
        a_batch.batch_size, b_batch.batch_size,
        "Batch sizes must match"
    );

    let key_ctx = KeyContext::new(ckks_ctx.params.clone());
    let constants = Cl3StructureConstants::new();

    // Step 1: Extract all 8 components from each batch
    let a_components = extract_all_components(a_batch, rotation_keys, ckks_ctx)?;
    let b_components = extract_all_components(b_batch, rotation_keys, ckks_ctx)?;

    // Step 2: Compute 8 output components in parallel
    let results: Vec<Result<Ciphertext, String>> = (0..8)
        .into_par_iter()
        .map(|out_idx| {
            let product_terms = &constants.products[out_idx];

            // Compute first term
            let (first_coeff, first_a_idx, first_b_idx) = product_terms[0];
            let mut accumulated = multiply_ciphertexts(
                &a_components[first_a_idx],
                &b_components[first_b_idx],
                evk,
                &key_ctx,
            );

            if first_coeff < 0 {
                accumulated = negate_ciphertext(&accumulated, ckks_ctx);
            }

            // Accumulate remaining terms
            for &(coeff, a_idx, b_idx) in &product_terms[1..] {
                let mut ct_product = multiply_ciphertexts(
                    &a_components[a_idx],
                    &b_components[b_idx],
                    evk,
                    &key_ctx,
                );

                if coeff < 0 {
                    ct_product = negate_ciphertext(&ct_product, ckks_ctx);
                }

                accumulated = accumulated.add(&ct_product);
            }

            Ok(accumulated)
        })
        .collect();

    // Collect results
    let mut result_components: Vec<Ciphertext> = Vec::with_capacity(8);
    for result in results {
        result_components.push(result?);
    }

    // Step 3: Reassemble into batched multivector
    let result_array: [Ciphertext; 8] = result_components.try_into()
        .map_err(|_| "Failed to convert to array".to_string())?;

    reassemble_components(
        &result_array,
        rotation_keys,
        ckks_ctx,
        a_batch.batch_size,
        a_batch.n,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_constants() {
        let constants = Cl3StructureConstants::new();
        assert_eq!(constants.products.len(), 8);
        for terms in &constants.products {
            assert_eq!(terms.len(), 8);
        }
    }
}
