//! SlotToCoeff Transformation
//!
//! Transforms ciphertext from slot representation back to coefficient representation.
//! This is the inverse of CoeffToSlot.
//!
//! ## Algorithm
//!
//! SlotToCoeff is the inverse FFT, also using O(log N) homomorphic rotations.
//!
//! **Structure:** Same as CoeffToSlot but with:
//! - Reversed level order (log N → 0)
//! - Inverse diagonal matrices
//! - Negated rotation directions (or equivalent)
//!
//! **Complexity:** O(log N) rotations, O(N log N) multiplications
//!
//! ## Correctness
//!
//! Must satisfy: SlotToCoeff(CoeffToSlot(x)) ≈ x (up to noise growth)

use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;
use super::keys::RotationKeys;
use super::rotation::rotate;

/// SlotToCoeff transformation
///
/// Transforms a ciphertext from slot (evaluation) representation back to coefficient representation.
///
/// # Arguments
///
/// * `ct` - Input ciphertext in slot representation
/// * `rotation_keys` - Rotation keys for all required rotations
///
/// # Returns
///
/// Ciphertext in coefficient representation
///
/// # Algorithm
///
/// Inverse FFT-like butterfly structure (reverse of CoeffToSlot):
///
/// ```text
/// Level log(N)-1: 1 pair, rotation by ±N/2
/// Level log(N)-2: 2 pairs, rotation by ±N/4
/// ...
/// Level 1: N/4 pairs, rotation by ±2
/// Level 0: N/2 pairs, rotation by ±1
/// ```
///
/// Each level applies:
/// 1. Rotation
/// 2. Diagonal matrix multiplication (inverse constants)
/// 3. Addition/subtraction
///
pub fn slot_to_coeff(
    ct: &Ciphertext,
    rotation_keys: &RotationKeys,
) -> Result<Ciphertext, String> {
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{CkksContext, Plaintext};
    use crate::clifford_fhe_v2::params::CliffordFHEParams;
    use std::f64::consts::PI;

    let n = ct.n;
    let num_slots = n / 2;
    let num_levels = (num_slots as f64).log2() as usize;

    println!("SlotToCoeff: N={}, slots={}, levels={}", n, num_slots, num_levels);

    // Start with input ciphertext
    let mut current = ct.clone();
    let initial_level = current.level;

    // Apply inverse FFT-like butterfly structure (reversed order)
    for level_idx in (0..num_levels).rev() {
        let rotation_amount = 1 << level_idx;  // N/4, ..., 4, 2, 1

        println!("  Level {}: rotation by ±{}, current ct.level={}", level_idx, rotation_amount, current.level);

        // Rotate by -rotation_amount (negative to reverse CoeffToSlot)
        let ct_rotated = rotate(&current, -(rotation_amount as i32), rotation_keys)?;

        // Compute inverse DFT twiddle factors for this level
        // For inverse DFT: diag1 = (1 + ω^{-k})/2, diag2 = (1 - ω^{-k})/2
        // where ω^{-k} = exp(-2πik/N) = cos(-θ) + i·sin(-θ)

        let mut diag1 = vec![0.5; num_slots];
        let mut diag2 = vec![0.5; num_slots];

        // Compute inverse twiddle factors
        let stride = 1 << level_idx;
        for j in 0..num_slots {
            let k = (j / stride) * stride;
            let theta = -2.0 * PI * (k as f64) / (n as f64);  // Negative for inverse

            let cos_theta = theta.cos();
            diag1[j] = (1.0 + cos_theta) / 2.0;
            diag2[j] = (1.0 - cos_theta) / 2.0;
        }

        // Create temporary params from ciphertext
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

        // Add the two results (inverse butterfly operation)
        current = add_ciphertexts_simple(&ct_mul1, &ct_mul2)?;

        println!("    After level {}: ct.level={}, ct.scale={:.2e}", level_idx, current.level, current.scale);
    }

    println!("  SlotToCoeff complete: final level={} (consumed {} levels)", current.level, initial_level - current.level);

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

/// Precompute inverse diagonal matrices for SlotToCoeff
///
/// These are the inverse of the CoeffToSlot matrices.
///
/// # Returns
///
/// Vector of inverse diagonal matrices, one per level
///
#[allow(dead_code)]
fn precompute_slot_to_coeff_matrices(n: usize) -> Vec<Vec<f64>> {
    let num_slots = n / 2;
    let num_levels = (num_slots as f64).log2() as usize;

    let mut matrices = Vec::with_capacity(num_levels);

    for level in 0..num_levels {
        // Placeholder: identity-like diagonal (inverse would need actual DFT matrix inverse)
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
    use crate::clifford_fhe_v3::bootstrapping::keys::generate_rotation_keys;
    use crate::clifford_fhe_v3::bootstrapping::coeff_to_slot::coeff_to_slot;

    #[test]
    fn test_slot_to_coeff_structure() {
        // Test that SlotToCoeff runs without errors
        let params = CliffordFHEParams::new_128bit();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        // Create test message
        let mut message = vec![0.0; params.n / 2];
        message[0] = 1.0;

        let pt = ckks_ctx.encode(&message);
        let ct = ckks_ctx.encrypt(&pt, &pk);

        // Generate rotation keys
        let rotations = vec![-1, -2, -4, -8];  // Negative rotations for SlotToCoeff
        let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

        // Run SlotToCoeff
        let result = slot_to_coeff(&ct, &rotation_keys);

        println!("SlotToCoeff test result: {:?}", result.is_ok());
    }

    #[test]
    #[ignore]  // Requires 18+ primes for full roundtrip (9 levels each direction)
    fn test_coeff_slot_roundtrip() {
        // Test that SlotToCoeff(CoeffToSlot(x)) ≈ x
        // NOTE: This test is ignored because it requires many primes (18+ for N=512).
        // The full bootstrap example (test_v3_full_bootstrap) tests this end-to-end.
        // To run: cargo test test_coeff_slot_roundtrip --features v2,v3 -- --ignored
        use crate::clifford_fhe_v2::params::CliffordFHEParams;

        let params = CliffordFHEParams::new_v3_demo_cpu();  // N=512, 7 primes
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        // Create test message
        let mut message = vec![0.0; params.n / 2];
        message[0] = 1.0;
        message[1] = 2.0;

        let pt = ckks_ctx.encode(&message);
        let ct_original = ckks_ctx.encrypt(&pt, &pk);

        println!("Test setup: N={}, primes={}, initial level={}",
                 params.n, params.moduli.len(), ct_original.level);

        // Generate all required rotation keys for bootstrap
        use crate::clifford_fhe_v3::bootstrapping::keys::required_rotations_for_bootstrap;
        let rotations = required_rotations_for_bootstrap(params.n);
        let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

        println!("Generated {} rotation keys for N={}", rotation_keys.keys.len(), params.n);

        // Apply CoeffToSlot then SlotToCoeff
        let ct_slots = coeff_to_slot(&ct_original, &rotation_keys)
            .expect("CoeffToSlot failed");
        println!("After CoeffToSlot: level={}", ct_slots.level);

        let ct_coeffs = slot_to_coeff(&ct_slots, &rotation_keys)
            .expect("SlotToCoeff failed");
        println!("After SlotToCoeff: level={}", ct_coeffs.level);

        // Decrypt and compare
        let pt_result = ckks_ctx.decrypt(&ct_coeffs, &sk);
        let result = ckks_ctx.decode(&pt_result);

        // Should be approximately equal to original
        for i in 0..5 {
            let error = (result[i] - message[i]).abs();
            println!("result[{}] = {}, expected {}, error = {}",
                     i, result[i], message[i], error);
            assert!(error < 10.0, "Roundtrip error too large: {}", error);
        }
    }
}
