//! Diagonal Matrix Multiplication for CKKS
//!
//! Implements homomorphic diagonal matrix-vector multiplication.
//! This is a key primitive for bootstrapping operations like CoeffToSlot/SlotToCoeff.
//!
//! ## Theory
//!
//! A diagonal matrix D multiplied by a vector v encoded in slots:
//! ```text
//! [d₀ 0  0  ...] [v₀]   [d₀·v₀]
//! [0  d₁ 0  ...] [v₁] = [d₁·v₁]
//! [0  0  d₂ ...] [v₂]   [d₂·v₂]
//! [... ... ...]  [...]  [...]
//! ```
//!
//! In CKKS slot encoding, this becomes element-wise multiplication:
//! `result[i] = diag[i] * ct[i]` for each slot i.
//!
//! ## Implementation
//!
//! We use plaintext-ciphertext multiplication since the diagonal is known:
//! 1. Encode diagonal values as plaintext
//! 2. Multiply: ct_result = ct_input * pt_diagonal
//! 3. Result has diagonal applied to all slots

use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{Ciphertext, Plaintext};
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

/// Multiply ciphertext by a diagonal matrix
///
/// Applies element-wise multiplication of ciphertext slots by diagonal values.
///
/// # Arguments
///
/// * `ct` - Input ciphertext (slots encode vector)
/// * `diagonal` - Diagonal values (length must match number of slots)
/// * `params` - FHE parameters
/// * `key_ctx` - Key context (for relinearization after multiplication)
///
/// # Returns
///
/// Ciphertext with diagonal applied: result[i] = ct[i] * diagonal[i]
///
/// # Errors
///
/// Returns error if:
/// - Diagonal length doesn't match slot count
/// - Multiplication/relinearization fails
///
/// # Example
///
/// ```rust,ignore
/// // Multiply ciphertext by diagonal [2.0, 3.0, 4.0, ...]
/// let diagonal = vec![2.0, 3.0, 4.0, ...];
/// let ct_result = diagonal_mult(&ct_input, &diagonal, &params, &key_ctx)?;
/// // Now ct_result[0] = 2.0 * ct_input[0], ct_result[1] = 3.0 * ct_input[1], ...
/// ```
pub fn diagonal_mult(
    ct: &Ciphertext,
    diagonal: &[f64],
    params: &CliffordFHEParams,
    key_ctx: &KeyContext,
) -> Result<Ciphertext, String> {
    // Verify diagonal length matches slot count
    let num_slots = params.n / 2;  // CKKS uses N/2 complex slots
    if diagonal.len() != num_slots {
        return Err(format!(
            "Diagonal length {} doesn't match slot count {}",
            diagonal.len(),
            num_slots
        ));
    }

    // Encode diagonal as plaintext using V2 CKKS encoding
    // Use params.scale to match ciphertext scale (rescale is now implemented correctly)
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{Plaintext, CkksContext};

    let mut pt_diagonal = Plaintext::encode(diagonal, params.scale, params);

    eprintln!("DEBUG: ct.level = {}, pt_diagonal.level = {}, params.moduli.len() = {}", ct.level, pt_diagonal.level, params.moduli.len());
    eprintln!("DEBUG: ct num moduli = {}, pt num moduli = {}", ct.c0[0].moduli.len(), pt_diagonal.coeffs[0].moduli.len());

    // Adjust plaintext level to match ciphertext level if needed
    if ct.level < params.moduli.len() - 1 {
        use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

        let moduli = &params.moduli[..=ct.level];
        eprintln!("DEBUG: Truncating pt from {} to {} moduli", pt_diagonal.coeffs[0].moduli.len(), moduli.len());
        for coeff in &mut pt_diagonal.coeffs {
            coeff.values.truncate(moduli.len());
            coeff.moduli = moduli.to_vec();
        }
        pt_diagonal.level = ct.level;
    }

    // Debug: Check what the diagonal plaintext encodes
    let diagonal_decoded = pt_diagonal.decode(params);
    eprintln!("DEBUG: Diagonal plaintext encodes: {:?}", &diagonal_decoded[..8]);

    // Use V2's built-in plaintext multiplication
    let ckks_ctx = CkksContext::new(params.clone());
    Ok(ct.multiply_plain(&pt_diagonal, &ckks_ctx))
}

/// Encode diagonal values as CKKS plaintext
///
/// Converts real-valued diagonal into CKKS plaintext encoding.
///
/// # Arguments
///
/// * `diagonal` - Real diagonal values
/// * `params` - FHE parameters (for scaling)
/// * `level` - Level of the plaintext (determines which moduli to use)
///
/// # Returns
///
/// Plaintext encoding of diagonal
fn encode_diagonal_as_plaintext(
    diagonal: &[f64],
    params: &CliffordFHEParams,
    level: usize,
) -> Result<Plaintext, String> {
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;

    // Use proper V2 CKKS encoding
    let mut pt = Plaintext::encode(diagonal, params.scale, params);

    // Adjust the plaintext to match the ciphertext level
    // If the ciphertext is at a lower level, we need to drop higher-level moduli
    if level < params.moduli.len() - 1 {
        use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

        // Truncate RNS representation to match the level
        let moduli = &params.moduli[..=level];
        for coeff in &mut pt.coeffs {
            coeff.values.truncate(moduli.len());
            coeff.moduli = moduli.to_vec();
        }
        pt.level = level;
    }

    Ok(pt)
}

/// Multiply ciphertext by plaintext (element-wise in slots)
///
/// Performs plaintext-ciphertext multiplication without relinearization.
///
/// # Arguments
///
/// * `ct` - Input ciphertext
/// * `pt` - Plaintext (diagonal encoded)
/// * `params` - FHE parameters
///
/// # Returns
///
/// Result ciphertext: ct_result = ct * pt (slot-wise)
pub fn multiply_by_plaintext(
    ct: &Ciphertext,
    pt: &Plaintext,
    params: &CliffordFHEParams,
) -> Result<Ciphertext, String> {
    // Plaintext-ciphertext multiplication is simpler than ciphertext-ciphertext
    // For ct = (c0, c1), result = (c0 * pt, c1 * pt)
    // No relinearization needed since degree doesn't increase

    use crate::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;
    use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

    // Ensure plaintext level matches ciphertext level
    if pt.level != ct.level {
        return Err(format!(
            "Plaintext level {} doesn't match ciphertext level {}. Adjust plaintext level before multiplication.",
            pt.level, ct.level
        ));
    }

    // Get active moduli for this level
    let moduli: Vec<u64> = params.moduli[..=ct.level].to_vec();
    let n = ct.n;

    // Helper to multiply two RNS polynomials
    let multiply_rns_polys = |a: &[RnsRepresentation], b: &[RnsRepresentation]| -> Vec<RnsRepresentation> {
        let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.clone()); n];

        for (prime_idx, &q) in moduli.iter().enumerate() {
            let ntt_ctx = NttContext::new(n, q);

            let a_mod_q: Vec<u64> = a.iter().map(|rns| rns.values[prime_idx]).collect();
            let b_mod_q: Vec<u64> = b.iter().map(|rns| rns.values[prime_idx]).collect();

            let product_mod_q = ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q);

            for i in 0..n {
                result[i].values[prime_idx] = product_mod_q[i];
            }
        }
        result
    };

    // Multiply c0 by pt
    let c0_result = multiply_rns_polys(&ct.c0, &pt.coeffs);

    // Multiply c1 by pt
    let c1_result = multiply_rns_polys(&ct.c1, &pt.coeffs);

    // New scale is product of scales
    let new_scale = ct.scale * pt.scale;

    Ok(Ciphertext {
        c0: c0_result,
        c1: c1_result,
        n: ct.n,
        level: ct.level,
        scale: new_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{Plaintext, CkksContext};

    #[test]
    fn test_diagonal_mult_simple() {
        // Create test parameters
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, evk) = key_ctx.keygen();

        // Create test vector
        let vec = vec![1.0, 2.0, 3.0, 4.0];

        // Create CKKS context for encoding/encryption
        let ckks_ctx = CkksContext::new(params.clone());

        // Encode and encrypt
        let pt = Plaintext::encode(&vec, params.scale, &params);
        let ct = ckks_ctx.encrypt(&pt, &pk);

        // Verify encryption works by decrypting before diagonal_mult
        let pt_check = ckks_ctx.decrypt(&ct, &sk);
        let check_values = pt_check.decode(&params);
        println!("Input ciphertext decrypts to: {:?}", &check_values[..4]);

        // Create diagonal [2.0, 3.0, 4.0, 5.0, ...]
        let num_slots = params.n / 2;
        let mut diagonal = vec![1.0; num_slots];
        diagonal[0] = 2.0;
        diagonal[1] = 3.0;
        diagonal[2] = 4.0;
        diagonal[3] = 5.0;

        // Apply diagonal multiplication
        let ct_result = diagonal_mult(&ct, &diagonal, &params, &key_ctx).unwrap();

        println!("Before diagonal_mult: scale = {}", ct.scale);
        println!("After diagonal_mult: scale = {}", ct_result.scale);

        // Decrypt and decode
        let pt_result = ckks_ctx.decrypt(&ct_result, &sk);
        println!("Plaintext scale after decryption: {}", pt_result.scale);
        let result = pt_result.decode(&params);
        println!("Result after diagonal_mult: {:?}", &result[..4]);

        // Expected: [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
        assert!((result[0] - 2.0).abs() < 1.0, "slot 0: expected 2, got {}", result[0]);
        assert!((result[1] - 6.0).abs() < 1.0, "slot 1: expected 6, got {}", result[1]);
        assert!((result[2] - 12.0).abs() < 1.0, "slot 2: expected 12, got {}", result[2]);
        assert!((result[3] - 20.0).abs() < 1.0, "slot 3: expected 20, got {}", result[3]);
    }

    #[test]
    fn test_diagonal_mult_wrong_size() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, _, _) = key_ctx.keygen();

        let ckks_ctx = CkksContext::new(params.clone());
        let vec = vec![1.0, 2.0, 3.0];
        let pt = Plaintext::encode(&vec, params.scale, &params);
        let ct = ckks_ctx.encrypt(&pt, &pk);

        // Wrong diagonal size
        let diagonal = vec![1.0, 2.0, 3.0];  // Should be N/2 = 512 elements

        let result = diagonal_mult(&ct, &diagonal, &params, &key_ctx);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("doesn't match"));
    }
}
