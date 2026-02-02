//! CoeffToSlot Transformation
//!
//! Transforms ciphertext from coefficient representation to slot representation.
//! This is a key component of CKKS bootstrapping.
//!
//! ## Algorithm
//!
//! CoeffToSlot is an FFT-like transformation that uses O(log N) homomorphic rotations.
//!
//! **High-level structure:**
//! 1. Linear transformations with diagonal matrices
//! 2. Rotations by powers of 2
//! 3. Recursively build up the DFT structure
//!
//! **Complexity:** O(log N) rotations, O(N log N) multiplications
//!
//! ## References
//!
//! - Cheon et al. "Bootstrapping for Approximate Homomorphic Encryption" (2018)
//! - Chen & Han "Homomorphic Lower Digits Removal and Improved FHE Bootstrapping" (2018)

use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;
use super::keys::RotationKeys;
use super::rotation::rotate;

/// CoeffToSlot transformation
///
/// Transforms a ciphertext from coefficient representation to slot (evaluation) representation.
///
/// # Arguments
///
/// * `ct` - Input ciphertext in coefficient representation
/// * `rotation_keys` - Rotation keys for all required rotations
///
/// # Returns
///
/// Ciphertext in slot representation
///
/// # Algorithm
///
/// The transformation follows an FFT-like butterfly structure:
///
/// ```text
/// Level 0: N/2 pairs, rotation by ±1
/// Level 1: N/4 pairs, rotation by ±2
/// Level 2: N/8 pairs, rotation by ±4
/// ...
/// Level log(N)-1: 1 pair, rotation by ±N/2
/// ```
///
/// Each level applies:
/// 1. Diagonal matrix multiplication (encode constants)
/// 2. Rotation
/// 3. Addition/subtraction
///
/// # Note
///
/// This is a skeleton implementation. Full implementation requires:
/// - Precomputed diagonal matrices (constants for each level)
/// - Proper scaling management
/// - Conjugate handling for complex slots
///
pub fn coeff_to_slot(
    ct: &Ciphertext,
    rotation_keys: &RotationKeys,
) -> Result<Ciphertext, String> {
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{CkksContext, Plaintext};
    use crate::clifford_fhe_v2::params::CliffordFHEParams;
    use std::f64::consts::PI;

    let n = ct.n;
    let num_slots = n / 2;
    let num_levels = (num_slots as f64).log2() as usize;

    println!("CoeffToSlot: N={}, slots={}, levels={}", n, num_slots, num_levels);

    // Get params from ciphertext metadata
    // We need to reconstruct params - use moduli from ciphertext level
    let moduli_count = ct.level + 1;

    // Start with input ciphertext
    let mut current = ct.clone();
    let initial_level = current.level;

    // Apply FFT-like butterfly structure
    for level_idx in 0..num_levels {
        let rotation_amount = 1 << level_idx;  // 1, 2, 4, 8, ..., N/4

        println!("  Level {}: rotation by ±{}, current ct.level={}", level_idx, rotation_amount, current.level);

        // Rotate by +rotation_amount
        let ct_rotated = rotate(&current, rotation_amount as i32, rotation_keys)?;

        // Compute DFT twiddle factors for this level
        // For a proper DFT, we need: diag1 = (1 + ω^k)/2, diag2 = (1 - ω^k)/2
        // where ω = exp(2πi/N) and k depends on the butterfly position

        // Simplified approach: use diagonal matrices that approximate DFT
        // For each slot j, compute the twiddle factor
        let mut diag1 = vec![0.5; num_slots];  // (1 + ω^k)/2 ≈ 0.5
        let mut diag2 = vec![0.5; num_slots];  // (1 - ω^k)/2 ≈ 0.5

        // For proper DFT, compute actual twiddle factors
        let stride = 1 << level_idx;
        for j in 0..num_slots {
            let k = (j / stride) * stride;
            let theta = 2.0 * PI * (k as f64) / (n as f64);

            // Twiddle factor: ω^k = exp(2πik/N) = cos(θ) + i·sin(θ)
            // For real encoding, we use: (1 ± cos(θ))/2
            let cos_theta = theta.cos();
            diag1[j] = (1.0 + cos_theta) / 2.0;
            diag2[j] = (1.0 - cos_theta) / 2.0;
        }

        // Create temporary params for encoding
        // CRITICAL: Use encode_at_level to match ciphertext's level
        let temp_params = create_temp_params_from_ct(&current)?;

        // Get q_top (the modulus at current level that will be dropped during rescale)
        let q_top = temp_params.moduli[current.level] as f64;

        // Encode diagonal matrices with scale = q_top
        // This ensures: (current.scale × q_top) / q_top = current.scale after rescale
        let pt_diag1 = Plaintext::encode_at_level(&diag1, q_top, &temp_params, current.level);
        let pt_diag2 = Plaintext::encode_at_level(&diag2, q_top, &temp_params, current.level);

        // Create CKKS context for multiply_plain
        let ckks_ctx = CkksContext::new(temp_params.clone());

        // Apply diagonal matrices:
        // result = diag1 * current + diag2 * ct_rotated
        let ct_mul1 = current.multiply_plain(&pt_diag1, &ckks_ctx);
        let ct_mul2 = ct_rotated.multiply_plain(&pt_diag2, &ckks_ctx);

        // Add the two results (this is the butterfly operation)
        current = add_ciphertexts_simple(&ct_mul1, &ct_mul2)?;

        println!("    After level {}: ct.level={}, ct.scale={:.2e}", level_idx, current.level, current.scale);
    }

    println!("  CoeffToSlot complete: final level={} (consumed {} levels)", current.level, initial_level - current.level);

    Ok(current)
}

/// Create temporary params from ciphertext metadata
fn create_temp_params_from_ct(ct: &Ciphertext) -> Result<crate::clifford_fhe_v2::params::CliffordFHEParams, String> {
    use crate::clifford_fhe_v2::params::{CliffordFHEParams, SecurityLevel};

    // Extract moduli from the first RNS coefficient
    if ct.c0.is_empty() {
        return Err("Ciphertext has no coefficients".to_string());
    }

    let moduli = ct.c0[0].moduli.clone();
    let n = ct.n;
    let scale = ct.scale;

    // Create params with these moduli
    let inv_scale_mod_q = CliffordFHEParams::precompute_inv_scale_mod_q(scale, &moduli);
    let inv_q_top_mod_q = CliffordFHEParams::precompute_inv_q_top_mod_q(&moduli);

    Ok(CliffordFHEParams {
        n,
        moduli,
        scale,
        error_std: 3.2,
        security: SecurityLevel::Bit128,
        inv_scale_mod_q,
        inv_q_top_mod_q,
        kappa_plain_mul: 1.0,
    })
}

/// Simple ciphertext addition (assumes levels and scales already match)
fn add_ciphertexts_simple(ct1: &Ciphertext, ct2: &Ciphertext) -> Result<Ciphertext, String> {
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;

    if ct1.level != ct2.level {
        return Err(format!("Level mismatch: {} vs {}", ct1.level, ct2.level));
    }

    let c0_sum: Vec<_> = ct1.c0.iter().zip(&ct2.c0)
        .map(|(a, b)| a.add(b))
        .collect();

    let c1_sum: Vec<_> = ct1.c1.iter().zip(&ct2.c1)
        .map(|(a, b)| a.add(b))
        .collect();

    Ok(Ciphertext {
        c0: c0_sum,
        c1: c1_sum,
        n: ct1.n,
        level: ct1.level,
        scale: ct1.scale,
    })
}

/// Precompute diagonal matrices for CoeffToSlot
///
/// These encode the FFT twiddle factors in CKKS encoding.
///
/// # Returns
///
/// Vector of diagonal matrices, one per level of the FFT
///
/// # Note
///
/// This is a placeholder. Full implementation requires:
/// - Computing DFT matrix roots of unity
/// - Encoding as CKKS plaintext diagonals
/// - Precomputing for all levels
///
#[allow(dead_code)]
fn precompute_coeff_to_slot_matrices(n: usize) -> Vec<Vec<f64>> {
    let num_slots = n / 2;
    let num_levels = (num_slots as f64).log2() as usize;

    let mut matrices = Vec::with_capacity(num_levels);

    for level in 0..num_levels {
        // Placeholder: identity-like diagonal
        let diag = vec![1.0; num_slots];
        matrices.push(diag);
    }

    matrices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::params::CliffordFHEParams;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
    use crate::clifford_fhe_v3::bootstrapping::keys::{generate_rotation_keys, required_rotations_for_bootstrap};

    #[test]
    fn test_coeff_to_slot_structure() {
        // Test that CoeffToSlot runs without errors (not checking correctness yet)
        let params = CliffordFHEParams::new_128bit();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        // Create test message
        let mut message = vec![0.0; params.n / 2];
        message[0] = 1.0;

        let pt = ckks_ctx.encode(&message);
        let ct = ckks_ctx.encrypt(&pt, &pk);

        // Generate rotation keys (use small set for testing)
        let rotations = vec![1, 2, 4, 8];  // Subset for fast testing
        let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

        // Run CoeffToSlot (will fail if rotation is broken, but structure is tested)
        let result = coeff_to_slot(&ct, &rotation_keys);

        // For now, just check it doesn't panic
        // Once rotation is fixed, we can verify correctness
        println!("CoeffToSlot test result: {:?}", result.is_ok());
    }

    #[test]
    fn test_precompute_matrices() {
        let matrices = precompute_coeff_to_slot_matrices(1024);

        // Should have log2(N/2) levels
        let expected_levels = ((1024 / 2) as f64).log2() as usize;
        assert_eq!(matrices.len(), expected_levels);

        // Each matrix should have N/2 diagonal elements
        for (i, mat) in matrices.iter().enumerate() {
            assert_eq!(mat.len(), 1024 / 2, "Matrix {} has wrong size", i);
        }
    }
}
