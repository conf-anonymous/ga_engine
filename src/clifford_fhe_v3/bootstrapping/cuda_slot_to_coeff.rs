//! CUDA GPU SlotToCoeff Transformation
//!
//! GPU-accelerated slot-to-coefficient transformation for CKKS bootstrapping.
//! This is the inverse of CoeffToSlot.
//!
//! **Algorithm**: Inverse FFT-like butterfly structure with O(log N) rotations
//!
//! **GPU Optimizations**:
//! - Rotation operations use GPU Galois kernel
//! - Diagonal matrix multiplication (plaintext multiply)
//! - Rescaling uses GPU RNS kernel
//!
//! **Performance Target**: ~2-3s on RTX 5090 (vs ~6s on Metal M3 Max)

use crate::clifford_fhe_v2::backends::gpu_cuda::rotation_keys::CudaRotationKeys;
use crate::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
use crate::clifford_fhe_v3::bootstrapping::cuda_bootstrap::CudaCiphertext;
use crate::clifford_fhe_v3::bootstrapping::cuda_coeff_to_slot::{
    cuda_add_ciphertexts, cuda_multiply_plain, cuda_rotate_ciphertext
};
use std::f64::consts::PI;
use std::sync::Arc;

/// CUDA GPU SlotToCoeff transformation
///
/// Transforms ciphertext from slot (evaluation) representation back to coefficient representation.
/// This is the inverse of CoeffToSlot.
///
/// # Arguments
/// * `ct` - Input ciphertext in slot representation
/// * `rotation_keys` - Rotation keys for all required rotations
/// * `ckks_ctx` - CUDA CKKS context for multiply/rescale operations
///
/// # Returns
/// Ciphertext in coefficient representation
///
/// # Algorithm
/// ```text
/// // Reverse order compared to CoeffToSlot
/// for level_idx in (0..log(N/2)).rev():
///     rotation = 2^level_idx  (N/4, ..., 4, 2, 1)
///
///     ct_rotated = rotate(ct, rotation)
///
///     // Inverse DFT twiddle factors (conjugate)
///     diag1[j] = (1 + cos(2πk/N)) / 2
///     diag2[j] = (1 - cos(2πk/N)) / 2
///
///     // Inverse butterfly operation
///     ct = diag1 * ct + diag2 * ct_rotated
/// ```
pub fn cuda_slot_to_coeff(
    ct: &CudaCiphertext,
    rotation_keys: &Arc<CudaRotationKeys>,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<CudaCiphertext, String> {
    let n = ct.n;
    let num_slots = n / 2;
    let num_levels = (num_slots as f64).log2() as usize;

    println!("  [CUDA SlotToCoeff] N={}, slots={}, inverse FFT levels={}", n, num_slots, num_levels);

    let mut current = ct.clone();
    let initial_level = current.level;

    // Apply inverse FFT-like butterfly structure (reverse order)
    for level_idx in (0..num_levels).rev() {
        let rotation_amount = 1 << level_idx;  // N/4, ..., 4, 2, 1

        println!("    Level {}/{}: rotation by ±{}, current level={}",
            num_levels - level_idx, num_levels, rotation_amount, current.level);

        // Step 1: Rotate by +rotation_amount using GPU
        let ct_rotated = cuda_rotate_ciphertext(&current, rotation_amount, rotation_keys, ckks_ctx)?;

        // Step 2: Compute inverse DFT twiddle factors (same as forward for real CKKS)
        let (diag1, diag2) = compute_inverse_dft_twiddle_factors(n, level_idx);

        // Step 3: Encode diagonal matrices as plaintexts
        let scale_for_diag = ckks_ctx.params().moduli[current.level] as f64;
        let pt_diag1 = cuda_encode_diagonal(&diag1, scale_for_diag, current.level, ckks_ctx)?;
        let pt_diag2 = cuda_encode_diagonal(&diag2, scale_for_diag, current.level, ckks_ctx)?;

        // Step 4: Multiply by diagonal matrices
        let ct_mul1 = cuda_multiply_plain(&current, &pt_diag1, ckks_ctx, scale_for_diag)?;
        let ct_mul2 = cuda_multiply_plain(&ct_rotated, &pt_diag2, ckks_ctx, scale_for_diag)?;

        // Step 5: Add the two results (inverse butterfly operation)
        current = cuda_add_ciphertexts(&ct_mul1, &ct_mul2, ckks_ctx)?;

        println!("      → After inverse butterfly: level={}, scale={:.2e}",
            current.level, current.scale);
    }

    let levels_consumed = initial_level - current.level;
    println!("  [CUDA SlotToCoeff] Complete: consumed {} levels", levels_consumed);

    Ok(current)
}

/// Compute inverse DFT twiddle factors for a given FFT level
///
/// For real-valued CKKS, the inverse DFT uses the same twiddle factors as forward DFT
/// (since we're working with real FFT, not complex FFT)
///
/// Returns (diag1, diag2) where:
/// - diag1[j] = (1 + cos(2πk/N)) / 2
/// - diag2[j] = (1 - cos(2πk/N)) / 2
fn compute_inverse_dft_twiddle_factors(n: usize, level_idx: usize) -> (Vec<f64>, Vec<f64>) {
    let num_slots = n / 2;
    let stride = 1 << level_idx;

    let mut diag1 = Vec::with_capacity(num_slots);
    let mut diag2 = Vec::with_capacity(num_slots);

    for j in 0..num_slots {
        let k = (j / stride) * stride;
        let theta = 2.0 * PI * (k as f64) / (n as f64);
        let cos_theta = theta.cos();

        // Same as forward DFT for real CKKS
        diag1.push((1.0 + cos_theta) / 2.0);
        diag2.push((1.0 - cos_theta) / 2.0);
    }

    (diag1, diag2)
}

/// Encode diagonal matrix as plaintext for CUDA operations
fn cuda_encode_diagonal(
    values: &[f64],
    scale: f64,
    level: usize,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<Vec<u64>, String> {
    // Encode using CUDA CKKS context
    let pt = ckks_ctx.encode(values, scale, level)?;

    // Return the polynomial coefficients
    Ok(pt.poly)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse_dft_twiddle_factors() {
        let n = 1024;
        let level_idx = 0;

        let (diag1, diag2) = compute_inverse_dft_twiddle_factors(n, level_idx);

        assert_eq!(diag1.len(), n / 2);
        assert_eq!(diag2.len(), n / 2);

        // Check that diag1 + diag2 = 1
        for i in 0..n/2 {
            let sum = diag1[i] + diag2[i];
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_c2s_s2c_symmetry() {
        // Verify that CoeffToSlot and SlotToCoeff use compatible twiddle factors
        let n = 1024;

        for level_idx in 0..5 {
            let (fwd_d1, fwd_d2) = super::super::cuda_coeff_to_slot::compute_dft_twiddle_factors(n, level_idx);
            let (inv_d1, inv_d2) = compute_inverse_dft_twiddle_factors(n, level_idx);

            // For real CKKS, forward and inverse use same factors
            assert_eq!(fwd_d1.len(), inv_d1.len());
            assert_eq!(fwd_d2.len(), inv_d2.len());

            for i in 0..fwd_d1.len() {
                assert!((fwd_d1[i] - inv_d1[i]).abs() < 1e-10);
                assert!((fwd_d2[i] - inv_d2[i]).abs() < 1e-10);
            }
        }
    }
}
