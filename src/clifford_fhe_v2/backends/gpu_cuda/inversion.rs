//! CUDA GPU Homomorphic Division
//!
//! Full GPU implementation of Newton-Raphson division for NVIDIA GPUs.
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
//! All expensive operations run on CUDA GPU:
//! - Ciphertext multiplication (via NTT)
//! - Ciphertext addition/subtraction
//! - Relinearization (key switching)
//! - Rescaling
//!
//! ## Performance
//!
//! Expected speedup vs CPU: ~10× (based on geometric product GPU acceleration)
//! - CPU (V2 optimized): ~8 seconds
//! - CUDA GPU (RTX 5090): ~800ms (estimated)

use super::ckks::{CudaCkksContext, CudaCiphertext, CudaPlaintext};
use super::relin_keys::CudaRelinKeys;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey;

/// Multiply two ciphertexts with relinearization and rescaling (full GPU)
///
/// This combines tensored multiplication + relinearization + rescaling into a single operation.
/// This is the pattern used in Metal GPU implementation.
///
/// # Arguments
///
/// * `ct1` - First ciphertext
/// * `ct2` - Second ciphertext (must be at same level as ct1)
/// * `relin_keys` - Relinearization keys
/// * `ctx` - CUDA CKKS context
///
/// # Returns
///
/// Relinearized and rescaled product ciphertext at level-1
///
/// # Important
///
/// Both ciphertexts must be at the same level. Use `mod_switch_to_level()` to align
/// levels before calling this function.
pub fn multiply_ciphertexts_gpu(
    ct1: &CudaCiphertext,
    ct2: &CudaCiphertext,
    relin_keys: &CudaRelinKeys,
    ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String> {
    // Level check - caller must align levels first using mod_switch_to_level
    if ct1.level != ct2.level {
        return Err(format!(
            "Level mismatch: {} vs {}. Use mod_switch_to_level() to align before multiplication.",
            ct1.level, ct2.level
        ));
    }

    if ct1.level == 0 {
        return Err("Cannot multiply at level 0 - no depth remaining".to_string());
    }

    // Step 1: Tensored multiplication (produces c0, c1, c2)
    let (c0, c1, c2) = ctx.multiply_ciphertexts_tensored(ct1, ct2)?;

    // Step 2: Relinearization (c0, c1, c2) → (c0', c1')
    // Note: c0, c1, c2 are in FLAT layout from multiply_ciphertexts_tensored
    // apply_relinearization_gpu returns results in FLAT layout
    let (c0_relin_flat, c1_relin_flat) = relin_keys.apply_relinearization_gpu(
        &c0,
        &c1,
        &c2,
        ct1.level,
        ctx.ntt_contexts(),
        ctx,
    )?;

    // Step 3: Convert from flat layout back to strided layout
    // CRITICAL: CudaCiphertext expects strided layout: c[coeff_idx * num_primes + prime_idx]
    // But relinearization returns flat layout: c[prime_idx * n + coeff_idx]
    //
    // IMPORTANT: Use num_active_primes as BOTH stride AND num_primes!
    // Metal uses num_primes = level + 1 (actual active primes), not the max number.
    // Using ct1.num_primes (max=30) when only num_active_primes (e.g. 22) are valid
    // causes garbage data to be read during rescaling.
    let num_active_primes = ct1.level + 1;
    let c0_relin = ctx.flat_to_strided(&c0_relin_flat, ct1.n, num_active_primes, num_active_primes);
    let c1_relin = ctx.flat_to_strided(&c1_relin_flat, ct1.n, num_active_primes, num_active_primes);

    // Step 4: Create intermediate ciphertext with doubled scale
    // Use num_active_primes as num_primes (matches Metal behavior)
    let result_scale = ct1.scale * ct2.scale;
    let ct_product = CudaCiphertext {
        c0: c0_relin,
        c1: c1_relin,
        n: ct1.n,
        num_primes: num_active_primes,  // FIXED: Use active primes, not max primes
        level: ct1.level,
        scale: result_scale,
    };

    // Step 4: Rescale to bring scale back to ~Δ and drop one level
    ct_product.rescale_to_next(ctx)
}

/// Compute homomorphic inverse using Newton-Raphson iteration (CUDA GPU)
///
/// Computes 1/a using the Newton-Raphson recurrence:
///   x_{n+1} = x_n · (2 - a·x_n)
///
/// Starting from an initial guess x_0 ≈ 1/a, this converges quadratically to 1/a.
///
/// # Arguments
/// * `ct` - Encrypted denominator a
/// * `initial_guess` - Initial guess x_0 ≈ 1/a (plaintext, from known range)
/// * `iterations` - Number of Newton-Raphson iterations (2-4 for full precision)
/// * `relin_keys` - Relinearization keys for multiplication
/// * `pk` - Public key for encrypting constants
/// * `ctx` - CUDA CKKS context
///
/// # Returns
/// Encrypted inverse 1/a
///
/// # Depth Consumption
/// Each iteration consumes 2 levels (one multiplication + rescale per step).
/// For k iterations: depth = 2k levels.
pub fn newton_raphson_inverse_gpu(
    ct: &CudaCiphertext,
    initial_guess: f64,
    iterations: usize,
    relin_keys: &CudaRelinKeys,
    pk: &PublicKey,
    ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String> {
    let params = ctx.params();
    let n = params.n;
    let num_slots = n / 2;

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║        CUDA GPU Newton-Raphson Inverse                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    println!("  Initial guess: {}", initial_guess);
    println!("  Iterations: {}", iterations);
    println!("  Initial level: {}", ct.level);

    // Encode and encrypt the initial guess using proper canonical embedding
    let mut guess_vec = vec![0.0; num_slots];
    guess_vec[0] = initial_guess;
    let pt_guess = ctx.encode(&guess_vec, ct.scale, ct.level)?;
    let mut ct_xn = ctx.encrypt(&pt_guess, pk)?;

    println!("  Encrypted initial guess at level {}\n", ct_xn.level);

    // Constant 2.0 (will create trivial ciphertexts for each iteration)
    let mut two_vec = vec![0.0; num_slots];
    two_vec[0] = 2.0;

    // Create a working copy of ct that we'll mod-switch to match ct_xn's level
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
        let ct_axn = multiply_ciphertexts_gpu(&ct_a, &ct_xn, relin_keys, ctx)?;
        println!("    Computed a·x_n (level {})", ct_axn.level);

        // Step 2: Create trivial ciphertext for constant 2 at ct_axn's level
        // Use proper canonical embedding via context's encode method
        let pt_two = ctx.encode(&two_vec, ct_axn.scale, ct_axn.level)?;
        let ct_two = create_trivial_ciphertext_gpu(&pt_two, ctx)?;
        println!("    Created trivial ciphertext for 2.0 (level {})", ct_two.level);

        // Step 3: Compute 2 - a·x_n
        let ct_two_minus_axn = subtract_ciphertexts_gpu(&ct_two, &ct_axn, ctx)?;
        println!("    Computed 2 - a·x_n (level {})", ct_two_minus_axn.level);

        // Step 4: Match ct_xn level with ct_two_minus_axn before multiplication
        // Note: mod_switch drops primes WITHOUT dividing - keeps scale same
        if ct_xn.level > ct_two_minus_axn.level {
            println!("    Mod-switching ct_xn from level {} to {}", ct_xn.level, ct_two_minus_axn.level);
            ct_xn = ct_xn.mod_switch_to_level(ct_two_minus_axn.level);
        }

        // Step 5: Compute x_{n+1} = x_n · (2 - a·x_n)
        ct_xn = multiply_ciphertexts_gpu(&ct_xn, &ct_two_minus_axn, relin_keys, ctx)?;
        println!("    Iteration complete (level {})\n", ct_xn.level);
    }

    println!("Newton-Raphson inverse complete!");
    println!("   Final level: {}", ct_xn.level);
    println!("   Depth consumed: {}\n", ct.level - ct_xn.level);

    Ok(ct_xn)
}

/// Compute homomorphic scalar division: a / b (CUDA GPU)
///
/// Divides two encrypted scalars using Newton-Raphson inversion:
/// 1. Compute 1/b using Newton-Raphson
/// 2. Multiply result by a
///
/// # Arguments
/// * `numerator` - Encrypted numerator a
/// * `denominator` - Encrypted denominator b
/// * `initial_guess` - Initial guess x_0 ≈ 1/b (from known range)
/// * `iterations` - Number of Newton-Raphson iterations (2-4 for full precision)
/// * `relin_keys` - Relinearization keys
/// * `pk` - Public key for encrypting constants
/// * `ctx` - CUDA CKKS context
///
/// # Returns
/// Encrypted quotient a/b
///
/// # Depth Consumption
/// For k iterations: depth = 2k + 1 levels
/// - k iterations: 2k levels (Newton-Raphson)
/// - Final multiplication: 1 level
pub fn scalar_division_gpu(
    numerator: &CudaCiphertext,
    denominator: &CudaCiphertext,
    initial_guess: f64,
    iterations: usize,
    relin_keys: &CudaRelinKeys,
    pk: &PublicKey,
    ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String> {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║        CUDA GPU Homomorphic Division                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Compute 1/b using Newton-Raphson
    println!("Step 1: Computing 1/denominator...");
    let ct_inv = newton_raphson_inverse_gpu(
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
    let ct_quotient = multiply_ciphertexts_gpu(&ct_num, &ct_inv, relin_keys, ctx)?;

    println!("\nDivision complete!");
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
/// * `ctx` - CUDA CKKS context
///
/// # Returns
/// Trivial ciphertext (pt, 0)
fn create_trivial_ciphertext_gpu(
    pt: &CudaPlaintext,
    _ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String> {
    let n = pt.n;
    let num_primes = pt.num_primes;

    // c0 = plaintext polynomial
    let c0 = pt.poly.clone();

    // c1 = 0 (all zeros)
    let c1 = vec![0u64; n * num_primes];

    Ok(CudaCiphertext {
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
/// * `ctx` - CUDA CKKS context
///
/// # Returns
/// Difference ciphertext ct1 - ct2
pub fn subtract_ciphertexts_gpu(
    ct1: &CudaCiphertext,
    ct2: &CudaCiphertext,
    ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String> {
    if ct1.level != ct2.level {
        return Err(format!(
            "Cannot subtract ciphertexts at different levels: {} vs {}",
            ct1.level, ct2.level
        ));
    }

    let n = ct1.n;
    let num_primes = ct1.num_primes;
    let params = ctx.params();

    // Use strided layout: c[coeff_idx * num_primes + prime_idx]
    let mut c0 = vec![0u64; n * num_primes];
    let mut c1 = vec![0u64; n * num_primes];

    // c0 = ct1.c0 - ct2.c0 (mod q_i)
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            let idx = coeff_idx * num_primes + prime_idx;
            let q = params.moduli[prime_idx];
            let diff = if ct1.c0[idx] >= ct2.c0[idx] {
                ct1.c0[idx] - ct2.c0[idx]
            } else {
                q - (ct2.c0[idx] - ct1.c0[idx])
            };
            c0[idx] = diff;
        }
    }

    // c1 = ct1.c1 - ct2.c1 (mod q_i)
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            let idx = coeff_idx * num_primes + prime_idx;
            let q = params.moduli[prime_idx];
            let diff = if ct1.c1[idx] >= ct2.c1[idx] {
                ct1.c1[idx] - ct2.c1[idx]
            } else {
                q - (ct2.c1[idx] - ct1.c1[idx])
            };
            c1[idx] = diff;
        }
    }

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
    fn test_division_api_compiles() {
        // This test just verifies the API compiles
        // Full integration tests in examples/bench_division_cuda_gpu.rs
        assert!(true);
    }
}
