//! CUDA GPU CoeffToSlot Transformation
//!
//! GPU-accelerated coefficient-to-slot transformation for CKKS bootstrapping.
//!
//! **Algorithm**: FFT-like butterfly structure with O(log N) rotations
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
use std::f64::consts::PI;
use std::sync::Arc;

/// CUDA GPU CoeffToSlot transformation
///
/// Transforms ciphertext from coefficient representation to slot (evaluation) representation.
///
/// # Arguments
/// * `ct` - Input ciphertext in coefficient representation
/// * `rotation_keys` - Rotation keys for all required rotations
/// * `ckks_ctx` - CUDA CKKS context for multiply/rescale operations
///
/// # Returns
/// Ciphertext in slot representation
///
/// # Algorithm
/// ```text
/// for level_idx in 0..log(N/2):
///     rotation = 2^level_idx  (1, 2, 4, 8, ..., N/4)
///
///     ct_rotated = rotate(ct, rotation)
///
///     // Compute DFT twiddle factors
///     diag1[j] = (1 + cos(2πk/N)) / 2
///     diag2[j] = (1 - cos(2πk/N)) / 2
///
///     // Butterfly operation
///     ct = diag1 * ct + diag2 * ct_rotated
/// ```
pub fn cuda_coeff_to_slot(
    ct: &CudaCiphertext,
    rotation_keys: &Arc<CudaRotationKeys>,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<CudaCiphertext, String> {
    let n = ct.n;
    let num_slots = n / 2;
    let num_levels = (num_slots as f64).log2() as usize;

    println!("  [CUDA CoeffToSlot] N={}, slots={}, FFT levels={}", n, num_slots, num_levels);

    let mut current = ct.clone();
    let initial_level = current.level;

    // Apply FFT-like butterfly structure
    for level_idx in 0..num_levels {
        let rotation_amount = 1 << level_idx;  // 1, 2, 4, 8, ..., N/4

        println!("    Level {}/{}: rotation by ±{}, current level={}",
            level_idx + 1, num_levels, rotation_amount, current.level);

        // Step 1: Rotate by +rotation_amount using GPU
        let ct_rotated = cuda_rotate_ciphertext(&current, rotation_amount, rotation_keys, ckks_ctx)?;

        // Step 2: Compute DFT twiddle factors
        let (diag1, diag2) = compute_dft_twiddle_factors(n, level_idx);

        // Step 3: Encode diagonal matrices as plaintexts
        let scale_for_diag = ckks_ctx.params().moduli[current.level] as f64;
        let pt_diag1 = cuda_encode_diagonal(&diag1, scale_for_diag, current.level, ckks_ctx)?;
        let pt_diag2 = cuda_encode_diagonal(&diag2, scale_for_diag, current.level, ckks_ctx)?;

        // Step 4: Multiply by diagonal matrices
        let ct_mul1 = cuda_multiply_plain(&current, &pt_diag1, ckks_ctx, scale_for_diag)?;
        let ct_mul2 = cuda_multiply_plain(&ct_rotated, &pt_diag2, ckks_ctx, scale_for_diag)?;

        // Step 5: Add the two results (butterfly operation)
        current = cuda_add_ciphertexts(&ct_mul1, &ct_mul2, ckks_ctx)?;

        println!("      → After butterfly: level={}, scale={:.2e}",
            current.level, current.scale);
    }

    let levels_consumed = initial_level - current.level;
    println!("  [CUDA CoeffToSlot] Complete: consumed {} levels", levels_consumed);

    Ok(current)
}

/// Compute DFT twiddle factors for a given FFT level
///
/// Returns (diag1, diag2) where:
/// - diag1[j] = (1 + cos(2πk/N)) / 2
/// - diag2[j] = (1 - cos(2πk/N)) / 2
pub(crate) fn compute_dft_twiddle_factors(n: usize, level_idx: usize) -> (Vec<f64>, Vec<f64>) {
    let num_slots = n / 2;
    let stride = 1 << level_idx;

    let mut diag1 = Vec::with_capacity(num_slots);
    let mut diag2 = Vec::with_capacity(num_slots);

    for j in 0..num_slots {
        let k = (j / stride) * stride;
        let theta = 2.0 * PI * (k as f64) / (n as f64);
        let cos_theta = theta.cos();

        diag1.push((1.0 + cos_theta) / 2.0);
        diag2.push((1.0 - cos_theta) / 2.0);
    }

    (diag1, diag2)
}

/// Rotate ciphertext using GPU rotation operations
///
/// Full rotation with key switching:
/// 1. Apply Galois automorphism to c0 and c1: (c0(X), c1(X)) → (c0(X^g), c1(X^g))
/// 2. Apply rotation key to c1(X^g) to get (c0_ks, c1_ks)
/// 3. Final ciphertext: (c0(X^g) + c0_ks, c1_ks)
pub fn cuda_rotate_ciphertext(
    ct: &CudaCiphertext,
    rotation_steps: usize,
    rotation_keys: &Arc<CudaRotationKeys>,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<CudaCiphertext, String> {
    let n = ct.n;
    let num_primes = ct.num_primes;
    let level = ct.level;

    // Convert to flat RNS layout using V2 GPU function
    let c0_flat = ckks_ctx.strided_to_flat(&ct.c0, n, ct.num_primes, num_primes);
    let c1_flat = ckks_ctx.strided_to_flat(&ct.c1, n, ct.num_primes, num_primes);

    // Step 1: Apply Galois automorphism to c0 and c1 using GPU
    let rotation_ctx = rotation_keys.rotation_context();
    let c0_galois = rotation_ctx.rotate_gpu(&c0_flat, rotation_steps as i32, num_primes)?;
    let c1_galois = rotation_ctx.rotate_gpu(&c1_flat, rotation_steps as i32, num_primes)?;

    // Step 2: Compute Galois element for this rotation
    let galois_elt = rotation_ctx.galois_element(rotation_steps as i32);

    // Step 3: Apply rotation key to c1(X^g) using GPU NTT
    let (c0_ks, c1_ks) = rotation_keys.apply_rotation_key_gpu(
        &c1_galois,
        galois_elt,
        level,
        ckks_ctx.ntt_contexts(),
    )?;

    // Step 4: Add c0(X^g) + c0_ks using V2 GPU function
    let c0_result = ckks_ctx.add_polynomials_gpu(&c0_galois, &c0_ks, num_primes)?;

    // Convert back from flat to strided layout using V2 GPU function
    let c0_strided = ckks_ctx.flat_to_strided(&c0_result, n, num_primes, num_primes);
    let c1_strided = ckks_ctx.flat_to_strided(&c1_ks, n, num_primes, num_primes);

    Ok(CudaCiphertext {
        c0: c0_strided,
        c1: c1_strided,
        n,
        num_primes,
        level,
        scale: ct.scale,
    })
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

/// Multiply ciphertext by plaintext using CUDA
///
/// Uses V2's GPU-accelerated pointwise_multiply_polynomials_gpu_strided() for parallel computation.
pub fn cuda_multiply_plain(
    ct: &CudaCiphertext,
    pt: &[u64],
    ckks_ctx: &Arc<CudaCkksContext>,
    scale_for_diag: f64,
) -> Result<CudaCiphertext, String> {
    let n = ct.n;
    let num_primes = ct.num_primes;

    // Use V2's GPU pointwise multiplication with strided layout
    // Both ct and pt are in strided layout: [coeff_idx * num_primes + prime_idx]
    let c0_result = ckks_ctx.pointwise_multiply_polynomials_gpu_strided(&ct.c0, pt, num_primes, num_primes)?;
    let c1_result = ckks_ctx.pointwise_multiply_polynomials_gpu_strided(&ct.c1, pt, num_primes, num_primes)?;

    // After multiplication, we need to rescale to maintain scale
    // The scale becomes: ct.scale * scale(pt)
    // Then we rescale by dropping the top modulus
    let new_scale = ct.scale * scale_for_diag;

    // Rescale using GPU
    let c0_rescaled = ckks_ctx.exact_rescale_gpu(&c0_result, num_primes - 1)?;
    let c1_rescaled = ckks_ctx.exact_rescale_gpu(&c1_result, num_primes - 1)?;

    Ok(CudaCiphertext {
        c0: c0_rescaled,
        c1: c1_rescaled,
        n,
        num_primes: num_primes - 1,
        level: ct.level - 1,
        scale: new_scale / ckks_ctx.params().moduli[num_primes - 1] as f64,
    })
}

/// Add two ciphertexts (assumes levels and scales match)
///
/// Uses V2's GPU-accelerated add_polynomials_gpu() for parallel computation.
pub fn cuda_add_ciphertexts(
    ct1: &CudaCiphertext,
    ct2: &CudaCiphertext,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<CudaCiphertext, String> {
    // Verify levels match
    if ct1.level != ct2.level {
        return Err(format!("Ciphertexts must be at same level: {} vs {}", ct1.level, ct2.level));
    }

    let n = ct1.n;
    let num_active_primes = ct1.level + 1;
    let num_primes = ct1.num_primes;

    // Convert to flat layout
    let c0_1_flat = ckks_ctx.strided_to_flat(&ct1.c0, n, num_primes, num_active_primes);
    let c1_1_flat = ckks_ctx.strided_to_flat(&ct1.c1, n, num_primes, num_active_primes);
    let c0_2_flat = ckks_ctx.strided_to_flat(&ct2.c0, n, num_primes, num_active_primes);
    let c1_2_flat = ckks_ctx.strided_to_flat(&ct2.c1, n, num_primes, num_active_primes);

    // Add on GPU
    let c0_flat_result = ckks_ctx.add_polynomials_gpu(&c0_1_flat, &c0_2_flat, num_active_primes)?;
    let c1_flat_result = ckks_ctx.add_polynomials_gpu(&c1_1_flat, &c1_2_flat, num_active_primes)?;

    // Convert back to strided
    let c0 = ckks_ctx.flat_to_strided(&c0_flat_result, n, num_primes, num_active_primes);
    let c1 = ckks_ctx.flat_to_strided(&c1_flat_result, n, num_primes, num_active_primes);

    Ok(CudaCiphertext {
        c0,
        c1,
        n,
        num_primes,
        level: ct1.level,
        scale: ct1.scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dft_twiddle_factors() {
        let n = 1024;
        let level_idx = 0;  // First level, stride = 1

        let (diag1, diag2) = compute_dft_twiddle_factors(n, level_idx);

        assert_eq!(diag1.len(), n / 2);
        assert_eq!(diag2.len(), n / 2);

        // Check that diag1 + diag2 = 1 (since (1+cos)/2 + (1-cos)/2 = 1)
        for i in 0..n/2 {
            let sum = diag1[i] + diag2[i];
            assert!((sum - 1.0).abs() < 1e-10, "diag1[{}] + diag2[{}] = {} != 1.0", i, i, sum);
        }
    }

    #[test]
    fn test_twiddle_symmetry() {
        let n = 1024;

        for level_idx in 0..5 {
            let (diag1, diag2) = compute_dft_twiddle_factors(n, level_idx);

            // Verify range [0, 1]
            for &val in diag1.iter().chain(diag2.iter()) {
                assert!(val >= 0.0 && val <= 1.0, "Twiddle factor {} out of range", val);
            }
        }
    }
}
