//! V4 CUDA Bootstrap Implementation
//!
//! Provides bootstrap for PackedMultivector using V3's CUDA bootstrap infrastructure.
//!
//! ## How It Works
//!
//! V4's PackedMultivector stores 8 Clifford components in interleaved slots:
//! [s₀, e1₀, e2₀, e3₀, e12₀, e23₀, e31₀, I₀, s₁, e1₁, ...]
//!
//! Bootstrap operates uniformly on all slots, so:
//! 1. Extract the underlying CudaCiphertext from PackedMultivector
//! 2. Apply V3 CUDA bootstrap (CoeffToSlot → EvalMod → SlotToCoeff)
//! 3. Wrap the refreshed ciphertext back into PackedMultivector
//!
//! The slot layout is preserved because all bootstrap operations are slot-wise.

use crate::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
use crate::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext as V2CudaCiphertext;
use crate::clifford_fhe_v2::backends::gpu_cuda::rotation::CudaRotationContext;
use crate::clifford_fhe_v2::backends::gpu_cuda::rotation_keys::CudaRotationKeys;
use crate::clifford_fhe_v2::backends::gpu_cuda::relin_keys::CudaRelinKeys;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v3::bootstrapping::BootstrapParams;
use crate::clifford_fhe_v3::bootstrapping::cuda_bootstrap::CudaCiphertext as V3CudaCiphertext;
use crate::clifford_fhe_v3::bootstrapping::cuda_coeff_to_slot::cuda_coeff_to_slot;
use crate::clifford_fhe_v3::bootstrapping::cuda_slot_to_coeff::cuda_slot_to_coeff;
use crate::clifford_fhe_v3::bootstrapping::cuda_eval_mod::cuda_eval_mod;
use crate::clifford_fhe_v4::PackedMultivector;
use std::sync::Arc;

/// Convert V2 CudaCiphertext to V3 CudaCiphertext
fn v2_to_v3_ciphertext(ct: &V2CudaCiphertext) -> V3CudaCiphertext {
    V3CudaCiphertext {
        c0: ct.c0.clone(),
        c1: ct.c1.clone(),
        n: ct.n,
        num_primes: ct.num_primes,
        level: ct.level,
        scale: ct.scale,
    }
}

/// Convert V3 CudaCiphertext to V2 CudaCiphertext
fn v3_to_v2_ciphertext(ct: &V3CudaCiphertext) -> V2CudaCiphertext {
    V2CudaCiphertext {
        c0: ct.c0.clone(),
        c1: ct.c1.clone(),
        n: ct.n,
        num_primes: ct.num_primes,
        level: ct.level,
        scale: ct.scale,
    }
}

/// V4 Bootstrap Context for Packed Multivectors
///
/// Wraps V3 CUDA bootstrap infrastructure for use with PackedMultivector.
pub struct V4BootstrapContext {
    /// CKKS context for basic operations
    ckks_ctx: Arc<CudaCkksContext>,

    /// Rotation context for Galois automorphisms
    #[allow(dead_code)]
    rotation_ctx: Arc<CudaRotationContext>,

    /// Rotation keys for key switching
    rotation_keys: Arc<CudaRotationKeys>,

    /// Relinearization keys for ciphertext multiplication
    relin_keys: Arc<CudaRelinKeys>,

    /// Bootstrap parameters
    bootstrap_params: BootstrapParams,

    /// Base FHE parameters
    params: CliffordFHEParams,
}

impl V4BootstrapContext {
    /// Create new V4 bootstrap context
    pub fn new(
        ckks_ctx: Arc<CudaCkksContext>,
        rotation_ctx: Arc<CudaRotationContext>,
        rotation_keys: Arc<CudaRotationKeys>,
        relin_keys: Arc<CudaRelinKeys>,
        bootstrap_params: BootstrapParams,
        params: CliffordFHEParams,
    ) -> Result<Self, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║      V4 CUDA GPU Bootstrap Context Initialized               ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        Ok(Self {
            ckks_ctx,
            rotation_ctx,
            rotation_keys,
            relin_keys,
            bootstrap_params,
            params,
        })
    }

    /// Bootstrap a packed multivector
    ///
    /// Refreshes the ciphertext noise while preserving the packed slot layout.
    ///
    /// Input: Noisy PackedMultivector at low level
    /// Output: Refreshed PackedMultivector with reduced noise
    pub fn bootstrap(
        &self,
        mv_in: &PackedMultivector,
    ) -> Result<PackedMultivector, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║          V4 CUDA GPU Bootstrap Pipeline                      ║");
        println!("║  (Packed Multivector - 8 components bootstrapped together)  ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let total_start = std::time::Instant::now();

        // Convert V2 ciphertext to V3 format
        let ct_v3 = v2_to_v3_ciphertext(&mv_in.ct);

        // Step 1: Modulus raise
        println!("Step 1: Modulus raise");
        let step1_start = std::time::Instant::now();
        let ct_raised = self.modulus_raise(&ct_v3)?;
        println!("  Modulus raised in {:.2}s\n", step1_start.elapsed().as_secs_f64());

        // Step 2: CoeffToSlot transformation
        println!("Step 2: CoeffToSlot transformation");
        let step2_start = std::time::Instant::now();
        let ct_slots = cuda_coeff_to_slot(&ct_raised, &self.rotation_keys, &self.ckks_ctx)?;
        println!("  CoeffToSlot completed in {:.2}s\n", step2_start.elapsed().as_secs_f64());

        // Step 3: EvalMod (sine evaluation for modular reduction)
        println!("Step 3: EvalMod (modular reduction)");
        let step3_start = std::time::Instant::now();
        let q = self.params.moduli[ct_slots.level];
        let ct_evalmod = cuda_eval_mod(
            &ct_slots,
            q,
            self.bootstrap_params.sin_degree,
            &self.ckks_ctx,
            Some(&self.relin_keys),
        )?;
        println!("  EvalMod completed in {:.2}s\n", step3_start.elapsed().as_secs_f64());

        // Step 4: SlotToCoeff transformation
        println!("Step 4: SlotToCoeff transformation");
        let step4_start = std::time::Instant::now();
        let ct_coeffs = cuda_slot_to_coeff(&ct_evalmod, &self.rotation_keys, &self.ckks_ctx)?;
        println!("  SlotToCoeff completed in {:.2}s\n", step4_start.elapsed().as_secs_f64());

        // Step 5: Modulus switch
        println!("Step 5: Modulus switch");
        let step5_start = std::time::Instant::now();
        let ct_out_v3 = self.modulus_switch(&ct_coeffs, ct_v3.level)?;
        println!("  Modulus switched in {:.2}s\n", step5_start.elapsed().as_secs_f64());

        let total_time = total_start.elapsed().as_secs_f64();
        println!("═══════════════════════════════════════════════════════════════");
        println!("V4 Bootstrap completed in {:.2}s", total_time);
        println!("  (8 Clifford components refreshed simultaneously)");
        println!("═══════════════════════════════════════════════════════════════\n");

        // Convert back to V2 ciphertext and wrap in PackedMultivector
        let ct_out_v2 = v3_to_v2_ciphertext(&ct_out_v3);

        Ok(PackedMultivector::new(
            ct_out_v2,
            mv_in.batch_size,
            mv_in.n,
            mv_in.num_primes,
            mv_in.level,
            mv_in.scale,
        ))
    }

    /// Step 1: Modulus raise - extend ciphertext to higher modulus
    fn modulus_raise(&self, ct: &V3CudaCiphertext) -> Result<V3CudaCiphertext, String> {
        let target_level = self.params.moduli.len() - 1;

        if ct.level >= target_level {
            return Ok(ct.clone());
        }

        let n = ct.n;
        let mut c0_raised = vec![0u64; n * (target_level + 1)];
        let mut c1_raised = vec![0u64; n * (target_level + 1)];

        // Copy existing coefficients (strided layout)
        for coeff_idx in 0..n {
            for prime_idx in 0..=ct.level {
                c0_raised[coeff_idx * (target_level + 1) + prime_idx] =
                    ct.c0[coeff_idx * (ct.level + 1) + prime_idx];
                c1_raised[coeff_idx * (target_level + 1) + prime_idx] =
                    ct.c1[coeff_idx * (ct.level + 1) + prime_idx];
            }
        }

        Ok(V3CudaCiphertext {
            c0: c0_raised,
            c1: c1_raised,
            n: ct.n,
            num_primes: target_level + 1,
            level: target_level,
            scale: ct.scale,
        })
    }

    /// Step 5: Modulus switch - reduce ciphertext to target level
    fn modulus_switch(&self, ct: &V3CudaCiphertext, target_level: usize) -> Result<V3CudaCiphertext, String> {
        if ct.level <= target_level {
            return Ok(ct.clone());
        }

        let mut current_ct = ct.clone();

        for _ in target_level..ct.level {
            let c0_rescaled = self.ckks_ctx.exact_rescale_gpu(&current_ct.c0, current_ct.level)?;
            let c1_rescaled = self.ckks_ctx.exact_rescale_gpu(&current_ct.c1, current_ct.level)?;

            let new_level = current_ct.level - 1;
            let q_dropped = self.params.moduli[current_ct.level] as f64;
            let new_scale = current_ct.scale / q_dropped;

            current_ct = V3CudaCiphertext {
                c0: c0_rescaled,
                c1: c1_rescaled,
                n: current_ct.n,
                num_primes: new_level + 1,
                level: new_level,
                scale: new_scale,
            };
        }

        Ok(current_ct)
    }
}

/// Convenience function for bootstrapping a packed multivector
///
/// Creates a temporary context and bootstraps the input.
/// For multiple bootstraps, prefer creating a V4BootstrapContext once.
pub fn bootstrap_packed_multivector(
    mv: &PackedMultivector,
    ckks_ctx: &Arc<CudaCkksContext>,
    rotation_ctx: &Arc<CudaRotationContext>,
    rotation_keys: &Arc<CudaRotationKeys>,
    relin_keys: &Arc<CudaRelinKeys>,
    bootstrap_params: &BootstrapParams,
    params: &CliffordFHEParams,
) -> Result<PackedMultivector, String> {
    let ctx = V4BootstrapContext::new(
        ckks_ctx.clone(),
        rotation_ctx.clone(),
        rotation_keys.clone(),
        relin_keys.clone(),
        bootstrap_params.clone(),
        params.clone(),
    )?;

    ctx.bootstrap(mv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ciphertext_conversion() {
        // Test V2 to V3 conversion
        let v2_ct = V2CudaCiphertext {
            c0: vec![1, 2, 3],
            c1: vec![4, 5, 6],
            n: 1024,
            num_primes: 3,
            level: 2,
            scale: 1e10,
        };

        let v3_ct = v2_to_v3_ciphertext(&v2_ct);
        assert_eq!(v3_ct.c0, v2_ct.c0);
        assert_eq!(v3_ct.n, v2_ct.n);

        // Test V3 to V2 conversion
        let v2_ct_back = v3_to_v2_ciphertext(&v3_ct);
        assert_eq!(v2_ct_back.c0, v2_ct.c0);
        assert_eq!(v2_ct_back.n, v2_ct.n);
    }
}
