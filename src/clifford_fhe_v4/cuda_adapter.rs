///! CUDA Adapter for V4 Packed Operations
///!
///! Provides V4-compatible methods for CUDA ciphertexts to match the CPU/Metal API.
///! This adapter bridges the gap between V4's assumptions and CUDA's actual API.

use crate::clifford_fhe_v2::backends::gpu_cuda::{
    ckks::{CudaCiphertext, CudaCkksContext, CudaPlaintext},
    rotation::CudaRotationContext,
    rotation_keys::CudaRotationKeys,
};
use std::sync::Arc;

/// Extension trait for CudaCiphertext to provide V4-compatible methods
pub trait CudaCiphertextExt {
    /// Add two ciphertexts (wrapper for CudaCkksContext::add)
    fn add(
        &self,
        other: &CudaCiphertext,
        ckks_ctx: &CudaCkksContext,
    ) -> Result<CudaCiphertext, String>;

    /// Rotate ciphertext by given number of slots
    fn rotate_by_steps(
        &self,
        steps: i32,
        rot_keys: &CudaRotationKeys,
        rot_ctx: &CudaRotationContext,
        ckks_ctx: &CudaCkksContext,
    ) -> Result<CudaCiphertext, String>;

    /// Multiply by plaintext
    fn multiply_plain(
        &self,
        plaintext: &CudaPlaintext,
        ckks_ctx: &CudaCkksContext,
    ) -> Result<CudaCiphertext, String>;
}

impl CudaCiphertextExt for CudaCiphertext {
    fn add(
        &self,
        other: &CudaCiphertext,
        ckks_ctx: &CudaCkksContext,
    ) -> Result<CudaCiphertext, String> {
        // Use the existing add method on CudaCkksContext
        ckks_ctx.add(self, other)
    }

    fn rotate_by_steps(
        &self,
        steps: i32,
        rot_keys: &CudaRotationKeys,
        rot_ctx: &CudaRotationContext,
        ckks_ctx: &CudaCkksContext,
    ) -> Result<CudaCiphertext, String> {
        // DEPRECATED: This file is redundant with ciphertext_ops.rs
        // Just delegate to the extension method
        self.rotate_by_steps(steps, rot_keys, ckks_ctx)
    }

    fn multiply_plain(
        &self,
        plaintext: &CudaPlaintext,
        ckks_ctx: &CudaCkksContext,
    ) -> Result<CudaCiphertext, String> {
        // Verify level compatibility
        let num_primes_at_level = self.level + 1;

        if plaintext.num_primes < num_primes_at_level {
            return Err(format!(
                "Plaintext has {} primes but ciphertext is at level {} (needs {} primes)",
                plaintext.num_primes, self.level, num_primes_at_level
            ));
        }

        // Multiply c0 and c1 by plaintext
        let c0_mult = ckks_ctx.pointwise_multiply_polynomials_gpu_strided(
            &self.c0,
            &plaintext.poly,
            self.num_primes,
            self.num_primes,
        )?;

        let c1_mult = ckks_ctx.pointwise_multiply_polynomials_gpu_strided(
            &self.c1,
            &plaintext.poly,
            self.num_primes,
            self.num_primes,
        )?;

        // Rescale to maintain scale
        let c0_rescaled = ckks_ctx.exact_rescale_gpu_strided(&c0_mult, self.level)?;
        let c1_rescaled = ckks_ctx.exact_rescale_gpu_strided(&c1_mult, self.level)?;

        Ok(CudaCiphertext {
            c0: c0_rescaled,
            c1: c1_rescaled,
            n: self.n,
            num_primes: self.num_primes,
            level: self.level.saturating_sub(1), // Consumed one level
            scale: self.scale * plaintext.scale / ckks_ctx.params().scale,
        })
    }
}

/// Helper function to create properly-scaled plaintexts for CUDA
pub fn encode_for_v4(
    ckks_ctx: &CudaCkksContext,
    values: &[f64],
    scale: f64,
    level: usize,
) -> Result<CudaPlaintext, String> {
    ckks_ctx.encode(values, scale, level)
}

/// Compute Galois element for rotation by k slots
/// For cyclotomic ring Z[X]/(X^N + 1), galois_elt = 5^k mod 2N
fn compute_galois_element(rotation_steps: i32, n: usize) -> Result<usize, String> {
    let two_n = 2 * n;
    let k = if rotation_steps >= 0 {
        rotation_steps as usize % (n / 2)
    } else {
        let abs_steps = (-rotation_steps) as usize % (n / 2);
        (n / 2) - abs_steps
    };

    // Compute 5^k mod 2N using modular exponentiation
    let mut result = 1usize;
    let mut base = 5usize;
    let mut exp = k;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % two_n;
        }
        base = (base * base) % two_n;
        exp >>= 1;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_galois_element_computation() {
        // For N=1024, test common rotations
        assert_eq!(compute_galois_element(1, 1024).unwrap(), 5);
        assert_eq!(compute_galois_element(2, 1024).unwrap(), 25);
        assert_eq!(compute_galois_element(0, 1024).unwrap(), 1);
    }
}
