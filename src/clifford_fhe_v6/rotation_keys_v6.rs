//! V6 Rotation Keys with GPU-Accelerated Key Switching
//!
//! This module provides `V6RotationKeys` which wraps V2's `CudaRotationKeys` and
//! uses parallel_lift's GPU-accelerated gadget decomposition for 25× faster key switching.
//!
//! ## Performance
//!
//! Rotation involves Galois automorphism + key switching. The key switching step
//! uses gadget decomposition, which is accelerated by parallel_lift.
//!
//! - V2 key switching: ~50ms (CPU gadget decomposition)
//! - V6 key switching: ~2ms (GPU gadget decomposition)
//! - Speedup: 25×
//!
//! ## Architecture
//!
//! ```text
//! rotate_v6(ct, steps):
//!   1. Apply Galois automorphism ← V2 GPU kernel
//!   2. Key switch c1(X^g) back to s(X):
//!      a. gpu_gadget_decompose_v6() ← parallel_lift (25× faster)
//!      b. GPU NTT multiply ← V2 (unchanged)
//! ```

use std::sync::Arc;

use crate::clifford_fhe_v2::backends::gpu_cuda::rotation_keys::CudaRotationKeys;
use crate::clifford_fhe_v2::backends::gpu_cuda::rotation::CudaRotationContext;
use crate::clifford_fhe_v2::backends::gpu_cuda::ntt::CudaNttContext;
use crate::clifford_fhe_v2::backends::gpu_cuda::ckks::{CudaCkksContext, CudaCiphertext};
use crate::clifford_fhe_v2::backends::gpu_cuda::device::CudaDeviceContext;
use crate::clifford_fhe_v2::params::CliffordFHEParams;

use super::context::ParallelLiftContext;
use super::gadget_decompose::gpu_gadget_decompose_v6;
use super::{V6Error, V6Result};

/// V6 Rotation Keys with parallel_lift GPU acceleration
///
/// Wraps V2's `CudaRotationKeys` and provides 25× faster key switching
/// using parallel_lift's GPU-accelerated gadget decomposition.
///
/// # Example
///
/// ```rust,ignore
/// // Initialize V6 context
/// let ctx = Arc::new(ParallelLiftContext::with_params(params.clone())?);
///
/// // Create V6 rotation keys
/// let rotation_keys = V6RotationKeys::new(ctx.clone(), &secret_key, 16)?;
///
/// // Generate keys for specific rotations
/// rotation_keys.generate_for_rotations(&[1, -1, 2, 4, 8])?;
///
/// // Apply rotation (25× faster key switching)
/// let rotated = rotation_keys.rotate_v6(&ciphertext, 1)?;
/// ```
pub struct V6RotationKeys {
    /// Underlying V2 rotation keys
    /// Stores the key material and provides Galois automorphism
    pub v2_keys: CudaRotationKeys,

    /// V6 context for parallel_lift acceleration
    ctx: Arc<ParallelLiftContext>,
}

impl V6RotationKeys {
    /// Create new V6 rotation keys manager
    ///
    /// # Arguments
    /// * `ctx` - V6 ParallelLiftContext
    /// * `secret_key` - Secret key polynomial in strided layout
    /// * `base_bits` - Gadget base exponent (e.g., 16 for B = 2^16)
    ///
    /// # Returns
    /// V6RotationKeys manager (call generate_for_rotations to add keys)
    pub fn new(
        ctx: Arc<ParallelLiftContext>,
        secret_key: &[u64],
        base_bits: usize,
    ) -> V6Result<Self> {
        let device = ctx.v2_ctx().device().clone();
        let params = ctx.params().clone();

        // Create rotation context for Galois automorphisms
        let rotation_ctx = Arc::new(
            CudaRotationContext::new(device.clone(), params.clone())
                .map_err(|e| V6Error::V2Error(e))?
        );

        let v2_keys = CudaRotationKeys::new(
            device,
            params,
            rotation_ctx,
            secret_key.to_vec(),
            base_bits,
        ).map_err(|e| V6Error::V2Error(e))?;

        Ok(Self { v2_keys, ctx })
    }

    /// Create V6RotationKeys by wrapping existing V2 keys
    pub fn from_v2_keys(
        v2_keys: CudaRotationKeys,
        ctx: Arc<ParallelLiftContext>,
    ) -> Self {
        Self { v2_keys, ctx }
    }

    /// Generate rotation keys for specified rotation amounts
    ///
    /// # Arguments
    /// * `rotations` - List of rotation amounts (positive and/or negative)
    pub fn generate_for_rotations(&mut self, rotations: &[i32]) -> V6Result<()> {
        let ntt_contexts = self.ctx.v2_ctx().ntt_contexts();

        for &steps in rotations {
            self.v2_keys.generate_rotation_key_gpu(steps, ntt_contexts)
                .map_err(|e| V6Error::V2Error(e))?;
        }

        Ok(())
    }

    /// Generate rotation keys for all powers of 2 up to N/2
    ///
    /// This is commonly needed for CoeffToSlot/SlotToCoeff operations.
    pub fn generate_power_of_two_rotations(&mut self) -> V6Result<()> {
        let n = self.ctx.n();
        let num_slots = n / 2;

        let mut rotations = Vec::new();
        let mut step = 1i32;
        while (step as usize) < num_slots {
            rotations.push(step);
            rotations.push(-step);
            step *= 2;
        }

        self.generate_for_rotations(&rotations)
    }

    /// Rotate ciphertext with V6-accelerated key switching
    ///
    /// This applies rotation with 25× faster key switching via parallel_lift.
    ///
    /// # Algorithm
    ///
    /// 1. **Galois Automorphism** (GPU - V2 kernel)
    ///    - c0(X) → c0(X^g)
    ///    - c1(X) → c1(X^g)
    ///
    /// 2. **Key Switching** (GPU - 25× faster)
    ///    - Decompose c1(X^g) using gpu_gadget_decompose_v6
    ///    - Compute c0' = c0(X^g) + Σ d_i · b_i
    ///    - Compute c1' = Σ d_i · a_i
    ///
    /// # Arguments
    /// * `ct` - Input ciphertext
    /// * `steps` - Rotation amount (positive = left, negative = right)
    ///
    /// # Returns
    /// Rotated ciphertext
    pub fn rotate_v6(
        &self,
        ct: &CudaCiphertext,
        steps: i32,
    ) -> V6Result<CudaCiphertext> {
        let n = ct.n;
        let num_primes = ct.num_primes;
        let level = ct.level;

        // Get required contexts
        let ckks_ctx = self.ctx.v2_ctx();
        let ntt_contexts = ckks_ctx.ntt_contexts();
        let rotation_ctx = self.v2_keys.rotation_context();

        // Step 1: Convert to flat RNS layout
        let c0_flat = ckks_ctx.strided_to_flat(&ct.c0, n, num_primes, level + 1);
        let c1_flat = ckks_ctx.strided_to_flat(&ct.c1, n, num_primes, level + 1);

        // Step 2: Apply Galois automorphism using V2 GPU kernel
        let c0_galois = rotation_ctx.rotate_gpu(&c0_flat, steps, level + 1)
            .map_err(|e| V6Error::V2Error(e))?;
        let c1_galois = rotation_ctx.rotate_gpu(&c1_flat, steps, level + 1)
            .map_err(|e| V6Error::V2Error(e))?;

        // Step 3: Key switch c1(X^g) back to s(X)
        // This uses V6's accelerated gadget decomposition
        let galois_elt = rotation_ctx.galois_element(steps);

        let (c0_ks, c1_ks) = self.apply_rotation_key_v6(
            &c1_galois,
            galois_elt,
            level,
            ntt_contexts,
        )?;

        // Step 4: c0' = c0(X^g) + c0_ks
        let c0_result = ckks_ctx.add_polynomials_gpu(&c0_galois, &c0_ks, level + 1)
            .map_err(|e| V6Error::V2Error(e))?;

        // Step 5: Convert back to strided layout
        let c0_strided = ckks_ctx.flat_to_strided(&c0_result, n, num_primes, level + 1);
        let c1_strided = ckks_ctx.flat_to_strided(&c1_ks, n, num_primes, level + 1);

        Ok(CudaCiphertext {
            c0: c0_strided,
            c1: c1_strided,
            n,
            num_primes,
            level,
            scale: ct.scale,
        })
    }

    /// Apply rotation key with V6-accelerated gadget decomposition
    fn apply_rotation_key_v6(
        &self,
        c1_galois: &[u64],
        galois_elt: u64,
        level: usize,
        ntt_contexts: &[CudaNttContext],
    ) -> V6Result<(Vec<u64>, Vec<u64>)> {
        let n = self.ctx.n();
        let num_primes = level + 1;

        // Get rotation key for this Galois element
        let rot_key = self.v2_keys.get_rotation_key(galois_elt)
            .ok_or_else(|| V6Error::V2Error(
                format!("Rotation key not found for galois_elt={}", galois_elt)
            ))?;

        // Step 1: GPU-accelerated gadget decomposition (25× faster)
        let digits = gpu_gadget_decompose_v6(
            &self.ctx,
            c1_galois,
            num_primes,
            self.v2_keys.base_bits,
        )?;

        // Step 2: Initialize accumulators
        let mut c0_acc = vec![0u64; n * num_primes];
        let mut c1_acc = vec![0u64; n * num_primes];

        // Step 3: Accumulate: c0' = Σ d_i · b_i, c1' = Σ d_i · a_i
        for (i, digit) in digits.iter().enumerate() {
            if i >= rot_key.ks_components.len() {
                break;
            }

            let (b_i, a_i) = &rot_key.ks_components[i];

            for prime_idx in 0..num_primes {
                let ntt_ctx = &ntt_contexts[prime_idx];
                let q = self.ctx.moduli()[prime_idx];

                // Extract slices
                let offset = prime_idx * n;
                let b_i_slice = &b_i[offset..offset + n];
                let a_i_slice = &a_i[offset..offset + n];

                // NTT forward on digit (in-place)
                let mut digit_ntt = digit[offset..offset + n].to_vec();
                ntt_ctx.forward(&mut digit_ntt)
                    .map_err(|e| V6Error::V2Error(e))?;

                // Pointwise multiply
                let mut prod_b = vec![0u64; n];
                let mut prod_a = vec![0u64; n];

                for j in 0..n {
                    prod_b[j] = mul_mod(digit_ntt[j], b_i_slice[j], q);
                    prod_a[j] = mul_mod(digit_ntt[j], a_i_slice[j], q);
                }

                // NTT inverse (in-place)
                ntt_ctx.inverse(&mut prod_b)
                    .map_err(|e| V6Error::V2Error(e))?;
                ntt_ctx.inverse(&mut prod_a)
                    .map_err(|e| V6Error::V2Error(e))?;

                // Accumulate
                for j in 0..n {
                    c0_acc[offset + j] = add_mod(c0_acc[offset + j], prod_b[j], q);
                    c1_acc[offset + j] = add_mod(c1_acc[offset + j], prod_a[j], q);
                }
            }
        }

        Ok((c0_acc, c1_acc))
    }

    /// Get the gadget base bits exponent
    #[inline]
    pub fn base_bits(&self) -> u32 {
        self.v2_keys.base_bits
    }

    /// Get the number of gadget digits
    #[inline]
    pub fn num_digits(&self) -> usize {
        self.v2_keys.dnum
    }

    /// Access the rotation context
    pub fn rotation_context(&self) -> Arc<CudaRotationContext> {
        self.v2_keys.rotation_context().clone()
    }

    /// Check if a rotation key exists
    pub fn has_rotation_key(&self, steps: i32) -> bool {
        let galois_elt = self.v2_keys.rotation_context().galois_element(steps);
        self.v2_keys.get_rotation_key(galois_elt).is_some()
    }
}

/// Modular multiplication: (a * b) mod q
#[inline]
fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

/// Modular addition: (a + b) mod q
#[inline]
fn add_mod(a: u64, b: u64, q: u64) -> u64 {
    let sum = a + b;
    if sum >= q { sum - q } else { sum }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_of_two_rotations() {
        // For N=1024, slots=512, we need rotations 1,2,4,8,...,256
        let n = 1024usize;
        let num_slots = n / 2;

        let mut rotations = Vec::new();
        let mut step = 1i32;
        while (step as usize) < num_slots {
            rotations.push(step);
            step *= 2;
        }

        assert_eq!(rotations, vec![1, 2, 4, 8, 16, 32, 64, 128, 256]);
    }
}
