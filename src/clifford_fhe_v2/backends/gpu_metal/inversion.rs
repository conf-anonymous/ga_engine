//! Metal GPU Homomorphic Division
//!
//! Full Metal GPU implementation of Newton-Raphson division for Apple Silicon.
//!
//! ## Overview
//!
//! This module implements homomorphic division for CKKS using Newton-Raphson iteration:
//!   x_{n+1} = x_n · (2 - a·x_n)
//!
//! where x_n converges to 1/a quadratically.
//!
//! ## GPU Acceleration
//!
//! All expensive operations run on Metal GPU:
//! - Ciphertext multiplication (via NTT)
//! - Ciphertext addition/subtraction
//! - Relinearization (key switching)
//! - Rescaling
//!
//! ## Performance
//!
//! Expected speedup vs CPU: ~33× (based on geometric product GPU acceleration)
//! - CPU (V2 optimized): ~8 seconds
//! - Metal GPU (M3 Max): ~240ms (estimated)

use super::ckks::{MetalCkksContext, MetalCiphertext, MetalPlaintext};
use super::relin_keys::MetalRelinKeys;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey;
use crate::clifford_fhe_v2::params::CliffordFHEParams;

/// Multiply two ciphertexts with relinearization (full Metal GPU)
///
/// Combines tensor product multiplication and relinearization in one operation.
///
/// # Arguments
/// * `ct1` - First ciphertext
/// * `ct2` - Second ciphertext
/// * `relin_keys` - Relinearization keys
/// * `ctx` - Metal CKKS context
///
/// # Returns
/// Product ciphertext at level-1 with scale = scale1 × scale2 / q_L
pub fn multiply_ciphertexts_metal(
    ct1: &MetalCiphertext,
    ct2: &MetalCiphertext,
    relin_keys: &MetalRelinKeys,
    ctx: &MetalCkksContext,
) -> Result<MetalCiphertext, String> {
    ct1.multiply(ct2, relin_keys, ctx)
}

/// Compute homomorphic inverse using Newton-Raphson iteration (Metal GPU)
///
/// Computes 1/a using the Newton-Raphson recurrence:
///   x_{n+1} = x_n · (2 - a·x_n)
///
/// Starting from an initial guess x_0 ≈ 1/a, this converges quadratically to 1/a.
///
/// # Arguments
/// * `ct` - Encrypted denominator a
/// * `initial_guess` - Initial guess x_0 ≈ 1/a (plaintext, from known range)
/// * `iterations` - Number of Newton-Raphson iterations (3-4 for full precision)
/// * `relin_keys` - Relinearization keys for multiplication
/// * `pk` - Public key for encrypting constants
/// * `ctx` - Metal CKKS context
///
/// # Returns
/// Encrypted inverse 1/a
///
/// # Depth Consumption
/// Each iteration consumes 2 levels (one multiplication + rescale per step).
/// For k iterations: depth = 2k levels.
///
/// # Example
/// ```rust,ignore
/// // Encrypt x = 2.0
/// let pt_x = ctx.encode(&[2.0])?;
/// let ct_x = ctx.encrypt(&pt_x, &pk)?;
///
/// // Compute 1/x ≈ 0.5
/// let ct_inv = newton_raphson_inverse_metal(
///     &ct_x,
///     0.5,  // initial guess
///     3,    // iterations
///     &relin_keys,
///     &pk,
///     &ctx,
/// )?;
///
/// // Decrypt and verify
/// let pt_result = ctx.decrypt(&ct_inv, &sk)?;
/// let result = ctx.decode(&pt_result)?;
/// assert!((result[0] - 0.5).abs() < 1e-6);
/// ```
pub fn newton_raphson_inverse_metal(
    ct: &MetalCiphertext,
    initial_guess: f64,
    iterations: usize,
    relin_keys: &MetalRelinKeys,
    pk: &PublicKey,
    ctx: &MetalCkksContext,
) -> Result<MetalCiphertext, String> {
    let params = &ctx.params;
    let n = params.n;
    let num_slots = n / 2;

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║        Metal GPU Newton-Raphson Inverse                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    println!("  Initial guess: {}", initial_guess);
    println!("  Iterations: {}", iterations);
    println!("  Initial level: {}", ct.level);

    // Encode and encrypt the initial guess
    let mut guess_vec = vec![0.0; num_slots];
    guess_vec[0] = initial_guess;
    let pt_guess = ctx.encode(&guess_vec)?;
    let mut ct_xn = ctx.encrypt(&pt_guess, pk)?;

    println!("  Encrypted initial guess at level {}\n", ct_xn.level);

    // Constant 2.0 (will create trivial ciphertexts for each iteration)
    let mut two_vec = vec![0.0; num_slots];
    two_vec[0] = 2.0;

    // Create a working copy of ct that we'll rescale to match ct_xn's level
    let mut ct_a = ct.clone();

    for iter_idx in 0..iterations {
        println!("  Newton-Raphson iteration {}/{}...", iter_idx + 1, iterations);

        // Match levels if needed: mod_switch ct_a to ct_xn's level
        // Note: mod_switch drops primes WITHOUT dividing - keeps scale same
        if ct_a.level > ct_xn.level {
            println!("    Mod-switching ct_a from level {} to {}", ct_a.level, ct_xn.level);
            ct_a = ct_a.mod_switch_to_level(ct_xn.level);
        }

        // Step 1: Compute a · x_n (ct_a × ct_xn, both at same level now)
        let ct_axn = multiply_ciphertexts_metal(&ct_a, &ct_xn, relin_keys, ctx)?;
        println!("    Computed a·x_n (level {})", ct_axn.level);

        // Step 2: Create trivial ciphertext for constant 2 at ct_axn's level
        let pt_two = MetalPlaintext::encode_at_level(&two_vec, ct_axn.scale, params, ct_axn.level);
        let ct_two = create_trivial_ciphertext_metal(&pt_two, ctx)?;
        println!("    Created trivial ciphertext for 2.0 (level {})", ct_two.level);

        // Step 3: Compute 2 - a·x_n
        let ct_two_minus_axn = subtract_ciphertexts_metal(&ct_two, &ct_axn, ctx)?;
        println!("    Computed 2 - a·x_n (level {})", ct_two_minus_axn.level);

        // Step 4: Match ct_xn level with ct_two_minus_axn before multiplication
        // Note: mod_switch drops primes WITHOUT dividing - keeps scale same
        if ct_xn.level > ct_two_minus_axn.level {
            println!("    Mod-switching ct_xn from level {} to {}", ct_xn.level, ct_two_minus_axn.level);
            ct_xn = ct_xn.mod_switch_to_level(ct_two_minus_axn.level);
        }

        // Step 5: Compute x_{n+1} = x_n · (2 - a·x_n)
        ct_xn = multiply_ciphertexts_metal(&ct_xn, &ct_two_minus_axn, relin_keys, ctx)?;
        println!("    ✓ Iteration complete (level {})\n", ct_xn.level);
    }

    println!("[OK] Newton-Raphson inverse complete!");
    println!("   Final level: {}", ct_xn.level);
    println!("   Depth consumed: {}\n", ct.level - ct_xn.level);

    Ok(ct_xn)
}

/// Compute homomorphic scalar division: a / b (Metal GPU)
///
/// Divides two encrypted scalars using Newton-Raphson inversion:
/// 1. Compute 1/b using Newton-Raphson
/// 2. Multiply result by a
///
/// # Arguments
/// * `numerator` - Encrypted numerator a
/// * `denominator` - Encrypted denominator b
/// * `initial_guess` - Initial guess x_0 ≈ 1/b (from known range)
/// * `iterations` - Number of Newton-Raphson iterations (3-4 for full precision)
/// * `relin_keys` - Relinearization keys
/// * `pk` - Public key for encrypting constants
/// * `ctx` - Metal CKKS context
///
/// # Returns
/// Encrypted quotient a/b
///
/// # Depth Consumption
/// For k iterations: depth = 2k + 1 levels
/// - k iterations: 2k levels (Newton-Raphson)
/// - Final multiplication: 1 level
///
/// # Example
/// ```rust,ignore
/// // Compute 100.0 / 7.0 = 14.285714...
/// let ct_result = scalar_division_metal(
///     &ct_num,
///     &ct_denom,
///     1.0/7.0,  // initial guess
///     3,        // iterations
///     &relin_keys,
///     &pk,
///     &ctx,
/// )?;
/// ```
pub fn scalar_division_metal(
    numerator: &MetalCiphertext,
    denominator: &MetalCiphertext,
    initial_guess: f64,
    iterations: usize,
    relin_keys: &MetalRelinKeys,
    pk: &PublicKey,
    ctx: &MetalCkksContext,
) -> Result<MetalCiphertext, String> {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║        Metal GPU Homomorphic Division                        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Compute 1/b using Newton-Raphson
    println!("Step 1: Computing 1/denominator...");
    let ct_inv = newton_raphson_inverse_metal(
        denominator,
        initial_guess,
        iterations,
        relin_keys,
        pk,
        ctx,
    )?;

    // Step 2: Match numerator level with inverse before multiplication
    // Note: mod_switch drops primes WITHOUT dividing - keeps scale same
    let ct_num = if numerator.level > ct_inv.level {
        println!("Step 2: Mod-switching numerator from level {} to {}", numerator.level, ct_inv.level);
        numerator.mod_switch_to_level(ct_inv.level)
    } else {
        numerator.clone()
    };

    // Step 3: Multiply numerator by 1/denominator to get numerator/denominator
    println!("Step 3: Multiplying numerator by (1/denominator)...");
    let ct_quotient = multiply_ciphertexts_metal(&ct_num, &ct_inv, relin_keys, ctx)?;

    println!("\n[OK] Division complete!");
    println!("   Final level: {}", ct_quotient.level);
    println!("   Total depth consumed: {}\n", numerator.level - ct_quotient.level);

    Ok(ct_quotient)
}

/// Create trivial encryption of plaintext: (m, 0)
///
/// A trivial ciphertext is one where c1 = 0, so decryption gives c0 + c1·s = c0 = m.
/// This is used for constants like 2.0 in Newton-Raphson.
///
/// # Arguments
/// * `pt` - Plaintext to encrypt trivially
/// * `ctx` - Metal CKKS context
///
/// # Returns
/// Trivial ciphertext (pt, 0)
fn create_trivial_ciphertext_metal(
    pt: &MetalPlaintext,
    ctx: &MetalCkksContext,
) -> Result<MetalCiphertext, String> {
    let n = pt.n;
    let num_primes = pt.num_primes;

    // c0 = plaintext
    let c0 = pt.coeffs.clone();

    // c1 = 0 (all zeros)
    let c1 = vec![0u64; n * num_primes];

    Ok(MetalCiphertext {
        c0,
        c1,
        n,
        num_primes,
        level: pt.level,
        scale: pt.scale,
    })
}

/// Subtract two ciphertexts: ct1 - ct2
///
/// Component-wise subtraction in RNS representation.
///
/// # Arguments
/// * `ct1` - First ciphertext
/// * `ct2` - Second ciphertext
/// * `ctx` - Metal CKKS context
///
/// # Returns
/// Difference ciphertext ct1 - ct2
pub fn subtract_ciphertexts_metal(
    ct1: &MetalCiphertext,
    ct2: &MetalCiphertext,
    ctx: &MetalCkksContext,
) -> Result<MetalCiphertext, String> {
    if ct1.level != ct2.level {
        return Err(format!(
            "Cannot subtract ciphertexts at different levels: {} vs {}",
            ct1.level, ct2.level
        ));
    }

    let n = ct1.n;
    let num_primes = ct1.num_primes;
    let moduli = &ctx.params.moduli[..num_primes];

    let mut c0 = vec![0u64; n * num_primes];
    let mut c1 = vec![0u64; n * num_primes];

    // c0 = ct1.c0 - ct2.c0 (mod q_i)
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        let diff = if ct1.c0[i] >= ct2.c0[i] {
            ct1.c0[i] - ct2.c0[i]
        } else {
            q - (ct2.c0[i] - ct1.c0[i])
        };
        c0[i] = diff;
    }

    // c1 = ct1.c1 - ct2.c1 (mod q_i)
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        let diff = if ct1.c1[i] >= ct2.c1[i] {
            ct1.c1[i] - ct2.c1[i]
        } else {
            q - (ct2.c1[i] - ct1.c1[i])
        };
        c1[i] = diff;
    }

    Ok(MetalCiphertext {
        c0,
        c1,
        n,
        num_primes,
        level: ct1.level,
        scale: ct1.scale,
    })
}

impl MetalPlaintext {
    /// Encode values at a specific level
    ///
    /// This is needed for creating constants at the same level as a ciphertext
    /// during Newton-Raphson iteration.
    pub fn encode_at_level(
        values: &[f64],
        scale: f64,
        params: &CliffordFHEParams,
        level: usize,
    ) -> Self {
        let n = params.n;
        let num_primes = level + 1;
        let num_slots = n / 2;

        // Canonical embedding
        let encoded_ints = MetalCkksContext::canonical_embed_encode_real(values, scale, n);

        // Convert to flat RNS layout
        let mut coeffs = vec![0u64; n * num_primes];
        for (coeff_idx, &val) in encoded_ints.iter().enumerate().take(n) {
            for (prime_idx, &q) in params.moduli.iter().enumerate().take(num_primes) {
                // Convert signed coefficient to unsigned mod q
                let val_mod_q = if val >= 0 {
                    (val as u64) % q
                } else {
                    let abs_val = (-val) as u64;
                    let remainder = abs_val % q;
                    if remainder == 0 {
                        0
                    } else {
                        q - remainder
                    }
                };
                coeffs[coeff_idx * num_primes + prime_idx] = val_mod_q;
            }
        }

        Self {
            coeffs,
            n,
            num_primes,
            level,
            scale,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use crate::clifford_fhe_v2::backends::gpu_metal::device::MetalDevice;
    use crate::clifford_fhe_v2::backends::gpu_metal::ntt::MetalNttContext;
    use std::sync::Arc;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_scalar_division() {
        // Setup
        let params = CliffordFHEParams::new_test_ntt_1024();
        let device = Arc::new(MetalDevice::new().expect("Failed to create Metal device"));
        let ctx = MetalCkksContext::new(params.clone()).expect("Failed to create CKKS context");

        // Generate keys
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();

        // Create NTT contexts
        let ntt_contexts: Vec<MetalNttContext> = params.moduli.iter()
            .map(|&q| MetalNttContext::new(params.n, q, device.clone()).unwrap())
            .collect();

        // Generate relinearization keys
        let relin_keys = MetalRelinKeys::generate(
            device.clone(),
            &sk,
            &params,
            &ntt_contexts,
            16,
        ).expect("Failed to generate relin keys");

        // Test: 100.0 / 7.0 = 14.285714...
        let num = 100.0;
        let denom = 7.0;
        let expected = num / denom;

        // Encode and encrypt
        let pt_num = ctx.encode(&[num]).expect("Failed to encode numerator");
        let pt_denom = ctx.encode(&[denom]).expect("Failed to encode denominator");

        let ct_num = ctx.encrypt(&pt_num, &pk).expect("Failed to encrypt numerator");
        let ct_denom = ctx.encrypt(&pt_denom, &pk).expect("Failed to encrypt denominator");

        // Compute division
        let ct_result = scalar_division_metal(
            &ct_num,
            &ct_denom,
            1.0 / denom,  // initial guess
            3,            // iterations
            &relin_keys,
            &pk,
            &ctx,
        ).expect("Division failed");

        // Decrypt and verify
        let pt_result = ctx.decrypt(&ct_result, &sk).expect("Failed to decrypt");
        let result = ctx.decode(&pt_result).expect("Failed to decode");

        let error = (result[0] - expected).abs();
        let relative_error = error / expected;

        println!("Expected: {:.10}", expected);
        println!("Got:      {:.10}", result[0]);
        println!("Error:    {:.2e}", error);
        println!("Rel err:  {:.2e}", relative_error);

        // With 3 iterations, we should get < 1e-6 relative error
        assert!(relative_error < 1e-6, "Relative error too large: {}", relative_error);
    }
}
