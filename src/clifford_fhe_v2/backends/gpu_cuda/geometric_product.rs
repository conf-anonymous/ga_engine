//! V2 CUDA Geometric Product for Full CKKS Ciphertexts
//!
//! Implements homomorphic geometric product for Cl(3,0) Clifford algebra
//! using CUDA GPU acceleration on full CKKS ciphertexts.
//!
//! This is the CUDA equivalent of V2 CPU's `GeometricContext::geometric_product()`.
//!
//! ## Performance
//!
//! - V2 CPU (Rayon): ~327ms at N=1024, ~16s at N=8192
//! - V2 CUDA (this): Expected ~50ms at N=1024, ~2s at N=8192 (based on GPU multiplication speedup)
//!
//! ## Algorithm
//!
//! 1. For each of 8 output components (in parallel on GPU)
//! 2. Compute 8 cross-terms according to Cl(3,0) structure constants
//! 3. Each term is a ciphertext multiplication (with relinearization + rescale)
//! 4. Accumulate terms with appropriate signs
//!
//! Total: 64 ciphertext multiplications per geometric product

use super::ckks::{CudaCkksContext, CudaCiphertext};
use super::relin_keys::CudaRelinKeys;
use super::inversion::multiply_ciphertexts_gpu;
use rayon::prelude::*;

/// Multivector ciphertext in Cl(3,0) - 8 encrypted components
///
/// Components represent:
/// - [0]: scalar (grade 0)
/// - [1,2,3]: vectors e₁, e₂, e₃ (grade 1)
/// - [4,5,6]: bivectors e₁₂, e₁₃, e₂₃ (grade 2)
/// - [7]: trivector e₁₂₃ (grade 3)
pub type CudaMultivectorCiphertext = [CudaCiphertext; 8];

/// Cl(3,0) structure constants for geometric product
///
/// For each output component, stores list of (coefficient, input_a_idx, input_b_idx)
/// This encodes the Clifford algebra multiplication table
pub struct Cl3StructureConstants {
    pub products: Vec<Vec<(i64, usize, usize)>>,
}

impl Cl3StructureConstants {
    /// Create structure constants for Cl(3,0)
    ///
    /// Basis: {1, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃}
    /// Signature: e₁²=e₂²=e₃²=1
    pub fn new() -> Self {
        let mut products = vec![Vec::new(); 8];

        // Component 0 (scalar): vectors square to +1, bivectors to -1
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

        // Component 1 (e₁)
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

        // Component 2 (e₂)
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

        // Component 3 (e₃)
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

        // Component 4 (e₁₂)
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

        // Component 5 (e₁₃)
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

        // Component 6 (e₂₃)
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

        // Component 7 (e₁₂₃)
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

/// CUDA Geometric Product context for Cl(3,0)
pub struct CudaGeometricProductContext {
    /// Structure constants for multiplication table
    constants: Cl3StructureConstants,
}

impl CudaGeometricProductContext {
    /// Create a new CUDA geometric product context
    pub fn new() -> Self {
        Self {
            constants: Cl3StructureConstants::new(),
        }
    }

    /// Negate a CUDA ciphertext: compute -ct (mod q for each prime)
    fn negate_ciphertext(&self, ct: &CudaCiphertext, ctx: &CudaCkksContext) -> CudaCiphertext {
        let params = ctx.params();
        let n = ct.n;
        let num_primes = ct.num_primes;

        let mut neg_c0 = vec![0u64; ct.c0.len()];
        let mut neg_c1 = vec![0u64; ct.c1.len()];

        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let idx = coeff_idx * num_primes + prime_idx;
                let q = params.moduli[prime_idx];

                // Negate: -x mod q = q - x (if x != 0)
                neg_c0[idx] = if ct.c0[idx] == 0 { 0 } else { q - ct.c0[idx] };
                neg_c1[idx] = if ct.c1[idx] == 0 { 0 } else { q - ct.c1[idx] };
            }
        }

        CudaCiphertext {
            c0: neg_c0,
            c1: neg_c1,
            n: ct.n,
            num_primes: ct.num_primes,
            level: ct.level,
            scale: ct.scale,
        }
    }

    /// Add two CUDA ciphertexts
    fn add_ciphertexts(
        &self,
        a: &CudaCiphertext,
        b: &CudaCiphertext,
        ctx: &CudaCkksContext,
    ) -> Result<CudaCiphertext, String> {
        ctx.add(a, b)
    }

    /// Geometric product: a ⊗ b (CUDA GPU accelerated)
    ///
    /// Computes the full Clifford algebra geometric product using CUDA GPU
    /// for all 64 ciphertext multiplications.
    ///
    /// # Arguments
    /// * `a` - First multivector ciphertext (8 components)
    /// * `b` - Second multivector ciphertext (8 components)
    /// * `relin_keys` - Relinearization keys for multiplication
    /// * `ctx` - CUDA CKKS context
    ///
    /// # Returns
    /// Result multivector encrypting geometric product a ⊗ b
    pub fn geometric_product(
        &self,
        a: &CudaMultivectorCiphertext,
        b: &CudaMultivectorCiphertext,
        relin_keys: &CudaRelinKeys,
        ctx: &CudaCkksContext,
    ) -> Result<CudaMultivectorCiphertext, String> {
        // Process all 8 output components
        // Note: We use sequential processing here because each GPU multiplication
        // already uses all GPU cores. Parallel CPU threads would just contend for GPU.
        let mut results: Vec<CudaCiphertext> = Vec::with_capacity(8);

        for out_idx in 0..8 {
            let product_terms = &self.constants.products[out_idx];

            // Compute first term
            let (first_coeff, first_a_idx, first_b_idx) = product_terms[0];
            let mut accumulated = multiply_ciphertexts_gpu(
                &a[first_a_idx],
                &b[first_b_idx],
                relin_keys,
                ctx,
            )?;

            // Apply sign if negative
            if first_coeff < 0 {
                accumulated = self.negate_ciphertext(&accumulated, ctx);
            }

            // Accumulate remaining 7 terms
            for &(coeff, a_idx, b_idx) in &product_terms[1..] {
                // Multiply a[a_idx] * b[b_idx]
                let mut ct_product = multiply_ciphertexts_gpu(
                    &a[a_idx],
                    &b[b_idx],
                    relin_keys,
                    ctx,
                )?;

                // Apply sign
                if coeff < 0 {
                    ct_product = self.negate_ciphertext(&ct_product, ctx);
                }

                // Accumulate (handle level mismatch from rescaling)
                // After multiplication+rescale, products are at level-1
                // All products in this component are at the same level, so no mod_switch needed
                accumulated = self.add_ciphertexts(&accumulated, &ct_product, ctx)?;
            }

            results.push(accumulated);
        }

        // Convert Vec to array
        Ok([
            results[0].clone(),
            results[1].clone(),
            results[2].clone(),
            results[3].clone(),
            results[4].clone(),
            results[5].clone(),
            results[6].clone(),
            results[7].clone(),
        ])
    }

    /// Geometric product with parallel output component processing
    ///
    /// Uses Rayon to parallelize across the 8 output components.
    /// This may or may not be faster than sequential, depending on GPU contention.
    pub fn geometric_product_parallel(
        &self,
        a: &CudaMultivectorCiphertext,
        b: &CudaMultivectorCiphertext,
        relin_keys: &CudaRelinKeys,
        ctx: &CudaCkksContext,
    ) -> Result<CudaMultivectorCiphertext, String> {
        // Clone what we need for parallel processing
        let constants = Cl3StructureConstants::new();

        // Parallel computation: process all 8 output components in parallel
        let results: Vec<Result<CudaCiphertext, String>> = (0..8)
            .into_par_iter()
            .map(|out_idx| {
                let product_terms = &constants.products[out_idx];

                // Compute first term
                let (first_coeff, first_a_idx, first_b_idx) = product_terms[0];
                let mut accumulated = multiply_ciphertexts_gpu(
                    &a[first_a_idx],
                    &b[first_b_idx],
                    relin_keys,
                    ctx,
                )?;

                // Apply sign if negative
                if first_coeff < 0 {
                    accumulated = self.negate_ciphertext(&accumulated, ctx);
                }

                // Accumulate remaining 7 terms
                for &(coeff, a_idx, b_idx) in &product_terms[1..] {
                    let mut ct_product = multiply_ciphertexts_gpu(
                        &a[a_idx],
                        &b[b_idx],
                        relin_keys,
                        ctx,
                    )?;

                    if coeff < 0 {
                        ct_product = self.negate_ciphertext(&ct_product, ctx);
                    }

                    accumulated = self.add_ciphertexts(&accumulated, &ct_product, ctx)?;
                }

                Ok(accumulated)
            })
            .collect();

        // Collect results, propagating any errors
        let mut final_results: Vec<CudaCiphertext> = Vec::with_capacity(8);
        for result in results {
            final_results.push(result?);
        }

        Ok([
            final_results[0].clone(),
            final_results[1].clone(),
            final_results[2].clone(),
            final_results[3].clone(),
            final_results[4].clone(),
            final_results[5].clone(),
            final_results[6].clone(),
            final_results[7].clone(),
        ])
    }
}

impl Default for CudaGeometricProductContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_constants() {
        let constants = Cl3StructureConstants::new();

        // Check that we have 8 components
        assert_eq!(constants.products.len(), 8);

        // Each component should have exactly 8 terms
        for (idx, terms) in constants.products.iter().enumerate() {
            assert_eq!(
                terms.len(),
                8,
                "Component {} should have 8 product terms",
                idx
            );
        }

        // Verify scalar component: 1⊗1=1, e₁⊗e₁=1, e₁₂⊗e₁₂=-1
        let scalar_terms = &constants.products[0];
        assert_eq!(scalar_terms[0], (1, 0, 0)); // 1⊗1
        assert_eq!(scalar_terms[1], (1, 1, 1)); // e₁⊗e₁
        assert_eq!(scalar_terms[4], (-1, 4, 4)); // e₁₂⊗e₁₂
    }
}
