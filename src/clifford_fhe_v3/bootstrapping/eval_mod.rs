//! EvalMod - Homomorphic Modular Reduction
//!
//! Implements the core bootstrapping operation: homomorphic evaluation of
//! modular reduction to remove noise from ciphertexts.
//!
//! ## Theory
//!
//! For CKKS, decryption is: `m = (c0 + c1·s) mod q`
//!
//! The challenge: `mod q` is not a polynomial operation!
//!
//! **Solution**: Use sine approximation:
//! ```text
//! x mod q ≈ x - (q/2π) · sin(2πx/q)
//! ```
//!
//! ## Why This Works
//!
//! The sine function is periodic with period 2π, and when properly scaled,
//! approximates the "sawtooth" shape of modular reduction.
//!
//! For x in [0, q]:
//! - sin(2πx/q) oscillates from 0 to 0
//! - Scaled by q/2π, it subtracts the excess over each modulus
//! - Result: x mod q (approximately)
//!
//! ## Implementation Steps
//!
//! 1. **Scale input**: x' = 2πx/q (map to sine period)
//! 2. **Evaluate sine**: sin(x') using polynomial approximation
//! 3. **Scale and subtract**: result = x - (q/2π)·sin(x')
//!
//! The sine polynomial is evaluated homomorphically using Baby-Step Giant-Step
//! to minimize multiplication depth.

use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{Ciphertext, Plaintext};
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{EvaluationKey, KeyContext};
use super::sin_approx::eval_polynomial;
use super::diagonal_mult::multiply_by_plaintext;
use std::f64::consts::PI;

/// Homomorphically evaluate modular reduction
///
/// Implements: x mod q ≈ x - (q/2π) · sin(2πx/q)
///
/// # Arguments
///
/// * `ct` - Input ciphertext (in slot representation)
/// * `q` - Modulus to reduce by
/// * `sin_coeffs` - Polynomial coefficients for sine approximation
/// * `evk` - Evaluation key (for relinearization)
/// * `params` - FHE parameters
/// * `key_ctx` - Key context (for NTT operations)
///
/// # Returns
///
/// Ciphertext with modular reduction applied
///
/// # Performance
///
/// For degree-23 sine polynomial:
/// - ~12 ciphertext multiplications
/// - ~500ms on CPU
/// - ~100ms on GPU (future)
///
/// # Example
///
/// ```rust,ignore
/// // Reduce ciphertext modulo q
/// let q = params.moduli[0];
/// let sin_coeffs = taylor_sin_coeffs(23);
/// let ct_reduced = eval_mod(&ct, q, &sin_coeffs, &params, &key_ctx)?;
/// ```
pub fn eval_mod(
    ct: &Ciphertext,
    q: u64,
    sin_coeffs: &[f64],
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
    key_ctx: &KeyContext,
) -> Result<Ciphertext, String> {
    println!("    EvalMod: Starting modular reduction");
    println!("      Modulus: {}", q);
    println!("      Sine degree: {}", sin_coeffs.len() - 1);

    // Step 1: Scale input: x' = (2π/q) · x
    println!("      [1/3] Scaling input by 2π/q...");
    let scale_factor = 2.0 * PI / (q as f64);
    let ct_scaled = multiply_by_constant(ct, scale_factor, params)?;

    // Step 2: Evaluate sine polynomial: sin(x')
    println!("      [2/3] Evaluating sine polynomial...");
    let ct_sin = eval_sine_polynomial(&ct_scaled, sin_coeffs, evk, params, key_ctx)?;

    // Step 3: Compute result: x - (q/2π)·sin(x')
    println!("      [3/3] Computing final result...");
    let rescale_factor = (q as f64) / (2.0 * PI);
    let ct_sin_scaled = multiply_by_constant(&ct_sin, rescale_factor, params)?;

    // Subtract: ct_result = ct - ct_sin_scaled
    let ct_result = subtract_ciphertexts(ct, &ct_sin_scaled, params)?;

    println!("      ✓ EvalMod complete");

    Ok(ct_result)
}

/// Multiply ciphertext by a constant
///
/// Multiplies all slots by the same scalar value.
///
/// # Arguments
///
/// * `ct` - Input ciphertext
/// * `constant` - Scalar multiplier
/// * `params` - FHE parameters
///
/// # Returns
///
/// Ciphertext with all slots multiplied by constant
fn multiply_by_constant(
    ct: &Ciphertext,
    constant: f64,
    params: &CliffordFHEParams,
) -> Result<Ciphertext, String> {
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;

    // Create plaintext with constant in all slots
    let num_slots = params.n / 2;
    let constant_vec = vec![constant; num_slots];

    // Encode as plaintext at the same level as the ciphertext
    let pt_constant = encode_constant_plaintext(&constant_vec, params, ct.level)?;

    // Use V2's multiply_plain which properly handles rescale
    let ckks_ctx = CkksContext::new(params.clone());
    Ok(ct.multiply_plain(&pt_constant, &ckks_ctx))
}

/// Encode constant vector as plaintext at a specific level
///
/// **CRITICAL FIX**: Uses `encode_at_level` to ensure the plaintext has the same
/// RNS basis as the ciphertext it will operate with. This prevents the moduli
/// mismatch error that was causing bootstrap to fail.
///
/// # Arguments
/// * `values` - Float values to encode
/// * `params` - FHE parameters (with full moduli list)
/// * `level` - Target level (determines which moduli to use: [0..=level])
///
/// # Returns
/// Plaintext with RNS representation matching the target level
fn encode_constant_plaintext(
    values: &[f64],
    params: &CliffordFHEParams,
    level: usize,
) -> Result<Plaintext, String> {
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;

    // **EXPERT'S FIX**: Use encode_at_level to match ciphertext's RNS basis exactly
    // This replaces the old approach of creating temporary params with truncated moduli
    let pt = Plaintext::encode_at_level(values, params.scale, params, level);

    Ok(pt)
}

/// Evaluate sine polynomial homomorphically using Baby-Step Giant-Step (BSGS)
///
/// Computes sin(x) ≈ c₁x + c₃x³ + c₅x⁵ + ... using polynomial approximation.
///
/// Uses **Baby-Step Giant-Step** algorithm to minimize multiplication depth:
/// - For degree n polynomial, uses O(√n) multiplicative depth instead of O(n)
/// - Baby-step: Precompute powers x, x², ..., x^k where k = ⌈√n⌉
/// - Giant-step: Evaluate polynomial in blocks using precomputed powers
///
/// For degree-23 polynomial:
/// - Baby steps: 5 (√23 ≈ 5)
/// - Giant steps: 5
/// - Multiplicative depth: O(√23) ≈ 5 levels instead of 23 with Horner
///
/// # Arguments
///
/// * `ct` - Input ciphertext (x)
/// * `sin_coeffs` - Polynomial coefficients [c₀, c₁, c₂, ...]
/// * `evk` - Evaluation key (for relinearization)
/// * `params` - FHE parameters
/// * `key_ctx` - Key context (for NTT operations)
///
/// # Returns
///
/// Ciphertext containing sin(x)
fn eval_sine_polynomial(
    ct: &Ciphertext,
    sin_coeffs: &[f64],
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
    key_ctx: &KeyContext,
) -> Result<Ciphertext, String> {
    // Find the highest non-zero coefficient to determine effective degree
    let mut degree = sin_coeffs.len() - 1;
    while degree > 0 && sin_coeffs[degree].abs() < 1e-10 {
        degree -= 1;
    }

    // Handle trivial cases
    if degree == 0 {
        return create_constant_ciphertext(sin_coeffs[0], ct.level, params);
    }

    if degree == 1 && sin_coeffs[0].abs() < 1e-10 {
        // Pure linear term: c₁·x
        return multiply_by_constant(ct, sin_coeffs[1], params);
    }

    // For small degree polynomials, Horner's method is simpler and sufficient
    if degree <= 4 {
        return eval_polynomial_horner(ct, sin_coeffs, evk, params, key_ctx);
    }

    // Baby-Step Giant-Step algorithm
    let baby_steps = ((degree as f64).sqrt().ceil() as usize).max(2);
    let giant_steps = (degree + baby_steps) / baby_steps;

    println!("        Using BSGS: degree={}, baby_steps={}, giant_steps={}", degree, baby_steps, giant_steps);

    // Step 1: Precompute powers x, x², x³, ..., x^baby_steps
    let mut x_powers = vec![ct.clone()]; // x_powers[i] = x^(i+1)
    for i in 1..baby_steps {
        let x_next = multiply_ciphertexts(&x_powers[i - 1], ct, evk, params, key_ctx)?;
        x_powers.push(x_next);
    }

    // x_giant = x^baby_steps (the last precomputed power)
    let x_giant = x_powers[baby_steps - 1].clone();

    // Step 2: Evaluate polynomial in blocks
    // P(x) = Σ_{g=0}^{giant_steps-1} (x^baby_steps)^g * Σ_{b=0}^{baby_steps-1} c_{g*baby_steps+b} * x^b
    let mut result: Option<Ciphertext> = None;
    let mut x_giant_power: Option<Ciphertext> = None; // (x^baby_steps)^g

    for g in 0..giant_steps {
        // Evaluate baby polynomial for this giant step:
        // baby_sum = c_{g*k} + c_{g*k+1}*x + c_{g*k+2}*x² + ... + c_{g*k+k-1}*x^{k-1}
        let mut baby_sum: Option<Ciphertext> = None;

        for b in 0..baby_steps {
            let idx = g * baby_steps + b;
            if idx > degree {
                break;
            }

            let coeff = sin_coeffs[idx];
            if coeff.abs() > 1e-10 {
                let term = if b == 0 {
                    // Constant term: just the coefficient
                    create_constant_ciphertext(coeff, ct.level, params)?
                } else {
                    // x^b * coeff
                    multiply_by_constant(&x_powers[b - 1], coeff, params)?
                };

                baby_sum = match baby_sum {
                    None => Some(term),
                    Some(sum) => Some(add_ciphertexts(&sum, &term, params)?),
                };
            }
        }

        // Skip if all coefficients in this block were zero
        let baby_sum = match baby_sum {
            Some(sum) => sum,
            None => continue,
        };

        // Multiply baby_sum by (x^baby_steps)^g
        let term = match &x_giant_power {
            None => baby_sum, // g=0, so (x^k)^0 = 1
            Some(xgp) => multiply_ciphertexts(&baby_sum, xgp, evk, params, key_ctx)?,
        };

        // Add to result
        result = match result {
            None => Some(term),
            Some(r) => Some(add_ciphertexts(&r, &term, params)?),
        };

        // Update x_giant_power for next iteration: (x^k)^{g+1} = (x^k)^g * x^k
        if g < giant_steps - 1 {
            x_giant_power = match x_giant_power {
                None => Some(x_giant.clone()),
                Some(xgp) => Some(multiply_ciphertexts(&xgp, &x_giant, evk, params, key_ctx)?),
            };
        }
    }

    result.ok_or_else(|| "Polynomial evaluation produced no result".to_string())
}

/// Evaluate polynomial using Horner's method (for small degree polynomials)
///
/// For degree n polynomial, uses O(n) multiplicative depth.
/// Simple and numerically stable, but depth-inefficient for large n.
fn eval_polynomial_horner(
    ct: &Ciphertext,
    coeffs: &[f64],
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
    key_ctx: &KeyContext,
) -> Result<Ciphertext, String> {
    let mut max_degree = coeffs.len() - 1;
    while max_degree > 0 && coeffs[max_degree].abs() < 1e-10 {
        max_degree -= 1;
    }

    // Start with highest coefficient
    let mut result = multiply_by_constant(ct, coeffs[max_degree], params)?;

    // Horner's method: result = (...((c_n·x + c_{n-1})·x + c_{n-2})·x + ...)
    for i in (0..max_degree).rev() {
        // result = result · x
        result = multiply_ciphertexts(&result, ct, evk, params, key_ctx)?;

        // result = result + c_i
        if coeffs[i].abs() > 1e-10 {
            let ct_coeff = create_constant_ciphertext(coeffs[i], result.level, params)?;
            result = add_ciphertexts(&result, &ct_coeff, params)?;
        }
    }

    Ok(result)
}

/// Multiply two ciphertexts (with relinearization)
fn multiply_ciphertexts(
    ct1: &Ciphertext,
    ct2: &Ciphertext,
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
    key_ctx: &KeyContext,
) -> Result<Ciphertext, String> {
    // Use V2 multiplication
    use crate::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts;

    Ok(multiply_ciphertexts(ct1, ct2, evk, key_ctx))
}

/// Add two ciphertexts (automatically handles level mismatch)
fn add_ciphertexts(
    ct1: &Ciphertext,
    ct2: &Ciphertext,
    params: &CliffordFHEParams,
) -> Result<Ciphertext, String> {
    // If levels don't match, mod-switch the higher-level ciphertext down
    let (ct1_aligned, ct2_aligned) = if ct1.level != ct2.level {
        println!("      [add] Level mismatch detected: ct1.level={}, ct2.level={}", ct1.level, ct2.level);
        if ct1.level > ct2.level {
            println!("      [add] Mod-switching ct1 from level {} down to {}", ct1.level, ct2.level);
            let ct1_down = mod_switch_to_level(ct1, ct2.level, params)?;
            (ct1_down, ct2.clone())
        } else {
            println!("      [add] Mod-switching ct2 from level {} down to {}", ct2.level, ct1.level);
            let ct2_down = mod_switch_to_level(ct2, ct1.level, params)?;
            (ct1.clone(), ct2_down)
        }
    } else {
        (ct1.clone(), ct2.clone())
    };

    // Component-wise addition using RnsRepresentation::add
    let c0_sum: Vec<_> = ct1_aligned.c0.iter().zip(&ct2_aligned.c0)
        .map(|(a, b)| a.add(b))
        .collect();

    let c1_sum: Vec<_> = ct1_aligned.c1.iter().zip(&ct2_aligned.c1)
        .map(|(a, b)| a.add(b))
        .collect();

    Ok(Ciphertext {
        c0: c0_sum,
        c1: c1_sum,
        n: ct1_aligned.n,
        level: ct1_aligned.level,
        scale: ct1_aligned.scale,  // Assuming same scale
    })
}

/// Subtract two ciphertexts (automatically handles level mismatch)
fn subtract_ciphertexts(
    ct1: &Ciphertext,
    ct2: &Ciphertext,
    params: &CliffordFHEParams,
) -> Result<Ciphertext, String> {
    // If levels don't match, mod-switch the higher-level ciphertext down
    let (ct1_aligned, ct2_aligned) = if ct1.level != ct2.level {
        println!("      [subtract] Level mismatch detected: ct1.level={}, ct2.level={}", ct1.level, ct2.level);
        if ct1.level > ct2.level {
            println!("      [subtract] Mod-switching ct1 from level {} down to {}", ct1.level, ct2.level);
            let ct1_down = mod_switch_to_level(ct1, ct2.level, params)?;
            (ct1_down, ct2.clone())
        } else {
            println!("      [subtract] Mod-switching ct2 from level {} down to {}", ct2.level, ct1.level);
            let ct2_down = mod_switch_to_level(ct2, ct1.level, params)?;
            (ct1.clone(), ct2_down)
        }
    } else {
        (ct1.clone(), ct2.clone())
    };

    // Component-wise subtraction using RnsRepresentation::sub
    let c0_diff: Vec<_> = ct1_aligned.c0.iter().zip(&ct2_aligned.c0)
        .map(|(a, b)| a.sub(b))
        .collect();

    let c1_diff: Vec<_> = ct1_aligned.c1.iter().zip(&ct2_aligned.c1)
        .map(|(a, b)| a.sub(b))
        .collect();

    Ok(Ciphertext {
        c0: c0_diff,
        c1: c1_diff,
        n: ct1_aligned.n,
        level: ct1_aligned.level,
        scale: ct1_aligned.scale,
    })
}

/// Mod-switch ciphertext down to a target level
///
/// Drops the top (level - target_level) primes from the RNS representation.
/// This is a "cheap" operation that doesn't change the underlying message,
/// just reduces the modulus chain.
fn mod_switch_to_level(
    ct: &Ciphertext,
    target_level: usize,
    params: &CliffordFHEParams,
) -> Result<Ciphertext, String> {
    if target_level > ct.level {
        return Err(format!(
            "Cannot mod-switch up: ct.level={} < target_level={}",
            ct.level, target_level
        ));
    }

    if target_level == ct.level {
        return Ok(ct.clone());
    }

    use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

    // Drop the top primes by truncating the RNS representation
    let target_moduli: Vec<u64> = params.moduli[..=target_level].to_vec();

    let c0_switched: Vec<_> = ct.c0.iter().map(|rns| {
        let truncated_values = rns.values[..=target_level].to_vec();
        RnsRepresentation::new(truncated_values, target_moduli.clone())
    }).collect();

    let c1_switched: Vec<_> = ct.c1.iter().map(|rns| {
        let truncated_values = rns.values[..=target_level].to_vec();
        RnsRepresentation::new(truncated_values, target_moduli.clone())
    }).collect();

    Ok(Ciphertext {
        c0: c0_switched,
        c1: c1_switched,
        n: ct.n,
        level: target_level,
        scale: ct.scale,  // Scale doesn't change in mod-switch
    })
}

/// Create ciphertext encoding a constant
fn create_constant_ciphertext(
    constant: f64,
    level: usize,
    params: &CliffordFHEParams,
) -> Result<Ciphertext, String> {
    // Create plaintext with constant
    let n = params.n;
    let scale = params.scale;

    let mut coeffs = vec![0i64; n];
    coeffs[0] = (constant * scale).round() as i64;

    use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

    // Convert coefficients to RNS representation at the specified level
    let moduli = &params.moduli[..=level];
    let mut rns_coeffs = Vec::with_capacity(n);

    for &coeff in &coeffs {
        let values: Vec<u64> = moduli.iter().map(|&q| {
            if coeff >= 0 {
                (coeff as u64) % q
            } else {
                let abs_val = (-coeff) as u64;
                let remainder = abs_val % q;
                if remainder == 0 { 0 } else { q - remainder }
            }
        }).collect();

        rns_coeffs.push(RnsRepresentation::new(values, moduli.to_vec()));
    }

    // Create "trivial" ciphertext: (m, 0)
    // This is an encryption of m with no noise
    let c0 = rns_coeffs;

    // Create zero polynomial for c1
    let zero_values = vec![0u64; moduli.len()];
    let c1 = vec![RnsRepresentation::new(zero_values.clone(), moduli.to_vec()); n];

    Ok(Ciphertext {
        c0,
        c1,
        n,
        level,
        scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v3::bootstrapping::sin_approx::taylor_sin_coeffs;

    #[test]
    fn test_multiply_by_constant() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();

        // Encrypt value 5.0
        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{Plaintext, CkksContext};
        let ckks_ctx = CkksContext::new(params.clone());
        let pt = Plaintext::encode(&[5.0], params.scale, &params);
        let ct = ckks_ctx.encrypt(&pt, &pk);

        // Multiply by 3.0
        let ct_result = multiply_by_constant(&ct, 3.0, &params).unwrap();

        // Decrypt and decode
        let pt_result = ckks_ctx.decrypt(&ct_result, &sk);
        let result_vec = pt_result.decode(&params);
        let result = result_vec[0];

        // Should be 15.0
        assert!((result - 15.0).abs() < 1.0, "Expected 15, got {}", result);
    }

    #[test]
    fn test_add_ciphertexts() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();

        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{Plaintext, CkksContext};
        let ckks_ctx = CkksContext::new(params.clone());

        let pt1 = Plaintext::encode(&[5.0], params.scale, &params);
        let ct1 = ckks_ctx.encrypt(&pt1, &pk);

        let pt2 = Plaintext::encode(&[3.0], params.scale, &params);
        let ct2 = ckks_ctx.encrypt(&pt2, &pk);

        let ct_sum = add_ciphertexts(&ct1, &ct2, &params).unwrap();

        let pt_result = ckks_ctx.decrypt(&ct_sum, &sk);
        let result_vec = pt_result.decode(&params);
        let result = result_vec[0];

        assert!((result - 8.0).abs() < 1.0, "Expected 8, got {}", result);
    }

    #[test]
    fn test_eval_sine_polynomial_simple() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, evk) = key_ctx.keygen();

        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{Plaintext, CkksContext};
        let ckks_ctx = CkksContext::new(params.clone());

        // Test with linear approximation: sin(x) ≈ x (just c₁ = 1, others = 0)
        let mut sin_coeffs = vec![0.0; 8];
        sin_coeffs[1] = 1.0;  // Linear term only

        // Encrypt x = 0.5
        let pt = Plaintext::encode(&[0.5], params.scale, &params);
        let ct = ckks_ctx.encrypt(&pt, &pk);

        // Evaluate: should return 0.5
        let ct_result = eval_sine_polynomial(&ct, &sin_coeffs, &evk, &params, &key_ctx).unwrap();

        let pt_result = ckks_ctx.decrypt(&ct_result, &sk);
        let result_vec = pt_result.decode(&params);
        let result = result_vec[0];

        assert!((result - 0.5).abs() < 0.1, "Linear sine: expected 0.5, got {}", result);
    }
}
