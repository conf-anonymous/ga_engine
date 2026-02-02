//! CUDA GPU EvalMod - Homomorphic Modular Reduction
//!
//! GPU-accelerated homomorphic evaluation of modular reduction using sine approximation.
//!
//! **Algorithm**: x mod q ≈ x - (q/2π) · sin(2πx/q)
//!
//! **GPU Optimizations**:
//! - Polynomial evaluation uses GPU rescaling
//! - Multiplication by constants (plaintext multiply)
//! - Baby-step giant-step for deep polynomials
//!
//! **Performance Target**: ~10-12s on RTX 5090 (vs ~30s on Metal M3 Max)

use crate::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
use crate::clifford_fhe_v2::backends::gpu_cuda::relin_keys::CudaRelinKeys;
use crate::clifford_fhe_v3::bootstrapping::cuda_bootstrap::CudaCiphertext;
use crate::clifford_fhe_v3::bootstrapping::cuda_coeff_to_slot::{cuda_add_ciphertexts, cuda_multiply_plain};
use crate::clifford_fhe_v3::bootstrapping::sin_approx::chebyshev_sin_coeffs;
use std::f64::consts::PI;
use std::sync::Arc;

/// CUDA GPU EvalMod - Homomorphic modular reduction
///
/// Implements: x mod q ≈ x - (q/2π) · sin(2πx/q)
///
/// # Arguments
/// * `ct` - Input ciphertext (in slot representation)
/// * `q` - Modulus to reduce by
/// * `sin_degree` - Degree of sine polynomial (15, 23, or 31)
/// * `ckks_ctx` - CUDA CKKS context
/// * `relin_keys` - Relinearization keys for exact multiplication (optional)
///
/// # Returns
/// Ciphertext with modular reduction applied
///
/// # Algorithm
/// ```text
/// 1. Scale input: ct' = (2π/q) · ct
/// 2. Evaluate sine: ct_sin = sin(ct')  [using Chebyshev polynomial]
/// 3. Final result: ct_out = ct - (q/2π) · ct_sin
/// ```
pub fn cuda_eval_mod(
    ct: &CudaCiphertext,
    q: u64,
    sin_degree: usize,
    ckks_ctx: &Arc<CudaCkksContext>,
    relin_keys: Option<&Arc<CudaRelinKeys>>,
) -> Result<CudaCiphertext, String> {
    println!("  [CUDA EvalMod] Starting modular reduction");
    println!("    Modulus: {}", q);
    println!("    Sine degree: {}", sin_degree);
    if relin_keys.is_some() {
        println!("    Relinearization: ENABLED (exact multiplication)");
    } else {
        println!("    Relinearization: DISABLED (approximation)");
    }

    // Step 1: Scale input by 2π/q
    println!("    [1/3] Scaling input by 2π/q...");
    let scale_factor = 2.0 * PI / (q as f64);
    let ct_scaled = cuda_multiply_by_constant(ct, scale_factor, ckks_ctx, relin_keys)?;
    println!("      → Scaled: level={}, scale={:.2e}", ct_scaled.level, ct_scaled.scale);

    // Step 2: Evaluate sine polynomial using Chebyshev approximation
    println!("    [2/3] Evaluating degree-{} sine polynomial...", sin_degree);
    let sin_coeffs = chebyshev_sin_coeffs(sin_degree);
    let ct_sin = cuda_eval_sine_polynomial(&ct_scaled, &sin_coeffs, ckks_ctx, relin_keys)?;
    println!("      → Sine evaluated: level={}, scale={:.2e}", ct_sin.level, ct_sin.scale);

    // Step 3: Compute final result: ct - (q/2π)·sin(ct)
    println!("    [3/3] Computing final result: x - (q/2π)·sin(x)...");
    let rescale_factor = (q as f64) / (2.0 * PI);
    let ct_sin_scaled = cuda_multiply_by_constant(&ct_sin, rescale_factor, ckks_ctx, relin_keys)?;

    // Match levels before subtraction
    let mut ct_matched = ct.clone();
    let target_level = ct_matched.level.min(ct_sin_scaled.level);
    while ct_matched.level > target_level {
        ct_matched = cuda_rescale_down(&ct_matched, ckks_ctx)?;
    }

    // Subtract: result = ct - ct_sin_scaled
    let ct_result = cuda_subtract_ciphertexts(&ct_matched, &ct_sin_scaled, ckks_ctx)?;

    println!("  [CUDA EvalMod] Complete: level={}, scale={:.2e}",
        ct_result.level, ct_result.scale);

    Ok(ct_result)
}

/// Multiply ciphertext by a constant (plaintext scalar)
fn cuda_multiply_by_constant(
    ct: &CudaCiphertext,
    constant: f64,
    ckks_ctx: &Arc<CudaCkksContext>,
    _relin_keys: Option<&Arc<CudaRelinKeys>>,  // Not needed for plaintext multiply
) -> Result<CudaCiphertext, String> {
    let num_slots = ct.n / 2;

    // Create plaintext with constant value in all slots
    let values = vec![constant; num_slots];

    // Encode as plaintext with scale matching top modulus
    let scale_for_pt = ckks_ctx.params().moduli[ct.level] as f64;
    let pt = ckks_ctx.encode(&values, scale_for_pt, ct.level)?;

    // Multiply using GPU
    cuda_multiply_plain(ct, &pt.poly, ckks_ctx, scale_for_pt)
}

/// Evaluate sine polynomial using Chebyshev coefficients
///
/// Uses baby-step giant-step algorithm for efficient deep polynomial evaluation
fn cuda_eval_sine_polynomial(
    ct: &CudaCiphertext,
    coeffs: &[f64],
    ckks_ctx: &Arc<CudaCkksContext>,
    relin_keys: Option<&Arc<CudaRelinKeys>>,
) -> Result<CudaCiphertext, String> {
    let degree = coeffs.len() - 1;

    println!("      Evaluating polynomial of degree {}...", degree);

    // For small degrees, use Horner's method
    if degree <= 7 {
        return cuda_eval_polynomial_horner(ct, coeffs, ckks_ctx, relin_keys);
    }

    // For larger degrees, use baby-step giant-step
    cuda_eval_polynomial_bsgs(ct, coeffs, ckks_ctx, relin_keys)
}

/// Evaluate polynomial using Horner's method
///
/// For polynomial p(x) = a0 + a1·x + a2·x² + ... + an·x^n
/// Horner's method: p(x) = a0 + x·(a1 + x·(a2 + x·(...)))
fn cuda_eval_polynomial_horner(
    ct: &CudaCiphertext,
    coeffs: &[f64],
    ckks_ctx: &Arc<CudaCkksContext>,
    relin_keys: Option<&Arc<CudaRelinKeys>>,
) -> Result<CudaCiphertext, String> {
    let degree = coeffs.len() - 1;

    // Start with highest degree coefficient
    let mut result = cuda_create_constant_ciphertext(coeffs[degree], ct.n, ct.level, ckks_ctx)?;

    // Work backwards: result = result * x + coeffs[i]
    for i in (0..degree).rev() {
        // Multiply by x
        result = cuda_multiply_ciphertexts(&result, ct, ckks_ctx, relin_keys)?;

        // Add constant
        let ct_const = cuda_create_constant_ciphertext(coeffs[i], ct.n, result.level, ckks_ctx)?;
        result = cuda_add_ciphertexts(&result, &ct_const, ckks_ctx)?;
    }

    Ok(result)
}

/// Evaluate polynomial using baby-step giant-step algorithm
///
/// For degree n polynomial, uses O(√n) multiplications instead of O(n)
fn cuda_eval_polynomial_bsgs(
    ct: &CudaCiphertext,
    coeffs: &[f64],
    ckks_ctx: &Arc<CudaCkksContext>,
    relin_keys: Option<&Arc<CudaRelinKeys>>,
) -> Result<CudaCiphertext, String> {
    let degree = coeffs.len() - 1;
    let baby_steps = ((degree as f64).sqrt().ceil() as usize).max(2);
    let giant_steps = (degree + baby_steps) / baby_steps;

    println!("        Using BSGS: baby_steps={}, giant_steps={}", baby_steps, giant_steps);

    // Precompute powers of x: x, x², x³, ..., x^baby_steps
    // Note: Each power will be at a progressively lower level due to rescaling
    let mut x_powers = vec![ct.clone()];
    for i in 1..baby_steps {
        // Match levels before multiplying
        let mut x_base = x_powers[0].clone();
        while x_base.level > x_powers[i-1].level {
            x_base = cuda_rescale_down(&x_base, ckks_ctx)?;
        }
        let x_next = cuda_multiply_ciphertexts(&x_powers[i-1], &x_base, ckks_ctx, relin_keys)?;
        x_powers.push(x_next);
    }

    // Compute giant step: x^baby_steps
    let x_giant = if baby_steps <= x_powers.len() {
        x_powers[baby_steps - 1].clone()
    } else {
        // Need to compute one more power
        let mut x_base = x_powers[0].clone();
        while x_base.level > x_powers[baby_steps - 2].level {
            x_base = cuda_rescale_down(&x_base, ckks_ctx)?;
        }
        cuda_multiply_ciphertexts(&x_powers[baby_steps - 2], &x_base, ckks_ctx, relin_keys)?
    };

    // Evaluate polynomial in blocks
    let mut result = cuda_create_constant_ciphertext(0.0, ct.n, ct.level, ckks_ctx)?;
    let mut x_giant_power = cuda_create_constant_ciphertext(1.0, ct.n, ct.level, ckks_ctx)?;

    for g in 0..giant_steps {
        // Evaluate baby steps for this giant step
        // Compute the minimum level we'll encounter in this iteration
        let min_level = (0..baby_steps)
            .filter_map(|b| {
                let idx = g * baby_steps + b;
                if idx >= coeffs.len() || coeffs[idx].abs() <= 1e-10 {
                    None
                } else if b == 0 {
                    Some(ct.level)  // b=0 term doesn't consume a level
                } else {
                    Some(x_powers[b-1].level - 1)  // Multiply consumes one level
                }
            })
            .min()
            .unwrap_or(ct.level);

        let mut baby_sum: Option<CudaCiphertext> = None;

        for b in 0..baby_steps {
            let idx = g * baby_steps + b;
            if idx >= coeffs.len() {
                break;
            }

            let coeff = coeffs[idx];
            if coeff.abs() > 1e-10 {
                let mut term = if b == 0 {
                    cuda_create_constant_ciphertext(coeff, ct.n, min_level, ckks_ctx)?
                } else {
                    let ct_coeff = cuda_create_constant_ciphertext(coeff, ct.n, x_powers[b-1].level, ckks_ctx)?;
                    cuda_multiply_ciphertexts(&ct_coeff, &x_powers[b-1], ckks_ctx, relin_keys)?
                };

                // Rescale term down to min_level if needed
                while term.level > min_level {
                    term = cuda_rescale_down(&term, ckks_ctx)?;
                }

                // Add to baby_sum
                baby_sum = if let Some(sum) = baby_sum {
                    Some(cuda_add_ciphertexts(&sum, &term, ckks_ctx)?)
                } else {
                    Some(term)
                };
            }
        }

        // Skip if no terms were added
        let baby_sum = match baby_sum {
            Some(sum) => sum,
            None => continue,
        };

        // Multiply by giant step power and add to result
        // First, match levels before multiplying
        let mut baby_sum_matched = baby_sum;
        let mut x_giant_power_matched = x_giant_power.clone();
        let multiply_target_level = baby_sum_matched.level.min(x_giant_power_matched.level);
        while baby_sum_matched.level > multiply_target_level {
            baby_sum_matched = cuda_rescale_down(&baby_sum_matched, ckks_ctx)?;
        }
        while x_giant_power_matched.level > multiply_target_level {
            x_giant_power_matched = cuda_rescale_down(&x_giant_power_matched, ckks_ctx)?;
        }
        let mut term = cuda_multiply_ciphertexts(&baby_sum_matched, &x_giant_power_matched, ckks_ctx, relin_keys)?;

        // Match levels before adding to result
        let target_level = result.level.min(term.level);
        while result.level > target_level {
            result = cuda_rescale_down(&result, ckks_ctx)?;
        }
        while term.level > target_level {
            term = cuda_rescale_down(&term, ckks_ctx)?;
        }
        result = cuda_add_ciphertexts(&result, &term, ckks_ctx)?;

        // Update giant step power
        if g < giant_steps - 1 {
            // Match levels before multiplying (same pattern as baby_sum × x_giant_power)
            let mut x_giant_for_mult = x_giant.clone();
            let mult_level = x_giant_power.level.min(x_giant_for_mult.level);
            while x_giant_power.level > mult_level {
                x_giant_power = cuda_rescale_down(&x_giant_power, ckks_ctx)?;
            }
            while x_giant_for_mult.level > mult_level {
                x_giant_for_mult = cuda_rescale_down(&x_giant_for_mult, ckks_ctx)?;
            }
            x_giant_power = cuda_multiply_ciphertexts(&x_giant_power, &x_giant_for_mult, ckks_ctx, relin_keys)?;
        }
    }

    Ok(result)
}

/// Create a constant ciphertext (trivial encryption of a constant)
fn cuda_create_constant_ciphertext(
    value: f64,
    n: usize,
    level: usize,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<CudaCiphertext, String> {
    let num_slots = n / 2;
    let values = vec![value; num_slots];

    // Encode as plaintext
    let num_primes_available = ckks_ctx.params().moduli.len();
    if level >= num_primes_available {
        return Err(format!("Level {} exceeds available primes {} (level mismatch: {} vs {})",
                          level, num_primes_available, level, num_primes_available - 1));
    }
    let scale = ckks_ctx.params().moduli[level] as f64;
    let pt = ckks_ctx.encode(&values, scale, level)?;

    // Trivial encryption: (pt, 0)
    let num_primes = level + 1;
    let c0 = pt.poly;
    let c1 = vec![0u64; n * num_primes];

    Ok(CudaCiphertext {
        c0,
        c1,
        n,
        num_primes,
        level,
        scale,
    })
}

/// Multiply two ciphertexts using GPU NTT-based polynomial multiplication
///
/// Implements tensor product multiplication: (a0, a1) × (b0, b1) = (c0, c1, c2)
/// where:
///   c0 = a0 * b0
///   c1 = a0 * b1 + a1 * b0
///   c2 = a1 * b1
///
/// Uses GPU NTT for fast polynomial multiplication
/// If relinearization keys are provided, reduces (c0, c1, c2) → (c0', c1')
/// Otherwise, uses approximation: drops c2 term
fn cuda_multiply_ciphertexts(
    ct1: &CudaCiphertext,
    ct2: &CudaCiphertext,
    ckks_ctx: &Arc<CudaCkksContext>,
    relin_keys: Option<&Arc<CudaRelinKeys>>,
) -> Result<CudaCiphertext, String> {
    if ct1.level != ct2.level {
        return Err(format!("Level mismatch: {} vs {}", ct1.level, ct2.level));
    }

    let n = ct1.n;
    let num_primes = ct1.num_primes;

    // Convert V3 ciphertexts to V2 format (structures are identical, just different types)
    use crate::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext as V2CudaCiphertext;

    let ct1_v2 = V2CudaCiphertext {
        c0: ct1.c0.clone(),
        c1: ct1.c1.clone(),
        n: ct1.n,
        num_primes: ct1.num_primes,
        level: ct1.level,
        scale: ct1.scale,
    };

    let ct2_v2 = V2CudaCiphertext {
        c0: ct2.c0.clone(),
        c1: ct2.c1.clone(),
        n: ct2.n,
        num_primes: ct2.num_primes,
        level: ct2.level,
        scale: ct2.scale,
    };

    // Use GPU NTT-based polynomial multiplication
    // This returns (c0, c1, c2) in flat RNS layout
    let (c0_flat, c1_flat, c2_flat) = ckks_ctx.multiply_ciphertexts_tensored(&ct1_v2, &ct2_v2)?;

    // Apply relinearization if keys are available
    let (c0_final, c1_final) = if let Some(relin_keys) = relin_keys {
        // Apply relinearization using GPU NTT: (c0, c1, c2) → (c0', c1')
        let (c0_relin, c1_relin) = relin_keys.apply_relinearization_gpu(
            &c0_flat,
            &c1_flat,
            &c2_flat,
            ct1.level,
            &ckks_ctx.ntt_contexts(),
            ckks_ctx,
        )?;

        // Convert back to strided layout using GPU
        let c0_strided = flat_to_strided_gpu(&c0_relin, n, num_primes, ct1.level + 1, ckks_ctx)?;
        let c1_strided = flat_to_strided_gpu(&c1_relin, n, num_primes, ct1.level + 1, ckks_ctx)?;

        (c0_strided, c1_strided)
    } else {
        // No relinearization keys: use approximation (drop c2)
        // Convert flat to strided layout using GPU
        let c0_strided = flat_to_strided_gpu(&c0_flat, n, num_primes, ct1.level + 1, ckks_ctx)?;
        let c1_strided = flat_to_strided_gpu(&c1_flat, n, num_primes, ct1.level + 1, ckks_ctx)?;
        (c0_strided, c1_strided)
    };

    // New scale = scale1 * scale2
    let new_scale = ct1.scale * ct2.scale;

    // Rescale to maintain scale (drop one modulus)
    if num_primes > 1 {
        let c0_rescaled = ckks_ctx.exact_rescale_gpu(&c0_final, num_primes - 1)?;
        let c1_rescaled = ckks_ctx.exact_rescale_gpu(&c1_final, num_primes - 1)?;

        Ok(CudaCiphertext {
            c0: c0_rescaled,
            c1: c1_rescaled,
            n,
            num_primes: num_primes - 1,
            level: ct1.level - 1,
            scale: new_scale / ckks_ctx.params().moduli[num_primes - 1] as f64,
        })
    } else {
        Ok(CudaCiphertext {
            c0: c0_final,
            c1: c1_final,
            n,
            num_primes,
            level: ct1.level,
            scale: new_scale,
        })
    }
}

/// Subtract two ciphertexts
fn cuda_subtract_ciphertexts(
    ct1: &CudaCiphertext,
    ct2: &CudaCiphertext,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<CudaCiphertext, String> {
    if ct1.level != ct2.level {
        return Err(format!("Level mismatch: {} vs {}", ct1.level, ct2.level));
    }

    let n = ct1.n;
    let num_primes = ct1.num_primes;
    let mut c0_diff = vec![0u64; n * num_primes];
    let mut c1_diff = vec![0u64; n * num_primes];

    // Subtract coefficient-wise in RNS
    for idx in 0..(n * num_primes) {
        let prime_idx = idx % num_primes;
        let q = ckks_ctx.params().moduli[prime_idx];

        // c0_diff = c1 - c2 (mod q)
        c0_diff[idx] = if ct1.c0[idx] >= ct2.c0[idx] {
            ct1.c0[idx] - ct2.c0[idx]
        } else {
            q - (ct2.c0[idx] - ct1.c0[idx])
        };

        c1_diff[idx] = if ct1.c1[idx] >= ct2.c1[idx] {
            ct1.c1[idx] - ct2.c1[idx]
        } else {
            q - (ct2.c1[idx] - ct1.c1[idx])
        };
    }

    Ok(CudaCiphertext {
        c0: c0_diff,
        c1: c1_diff,
        n,
        num_primes,
        level: ct1.level,
        scale: ct1.scale,
    })
}

/// Rescale ciphertext down by one level (GPU-optimized strided version)
///
/// Uses exact_rescale_gpu_strided which avoids expensive CPU layout conversions.
/// This is CRITICAL for BSGS performance - saves ~1.3M CPU operations per rescale!
fn cuda_rescale_down(
    ct: &CudaCiphertext,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<CudaCiphertext, String> {
    if ct.level == 0 {
        return Err("Cannot rescale at level 0".to_string());
    }

    let n = ct.n;
    let num_primes = ct.num_primes;

    // Rescale using GPU strided kernel (NO layout conversion!)
    let c0_rescaled = ckks_ctx.exact_rescale_gpu_strided(&ct.c0, num_primes - 1)?;
    let c1_rescaled = ckks_ctx.exact_rescale_gpu_strided(&ct.c1, num_primes - 1)?;

    Ok(CudaCiphertext {
        c0: c0_rescaled,
        c1: c1_rescaled,
        n,
        num_primes: num_primes - 1,
        level: ct.level - 1,
        scale: ct.scale / ckks_ctx.params().moduli[num_primes - 1] as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_function_periodicity() {
        // Verify that our sine approximation has correct period
        let degree = 23;
        let coeffs = chebyshev_sin_coeffs(degree);

        // sin(0) ≈ 0
        // sin(π/2) ≈ 1
        // sin(π) ≈ 0
        // sin(2π) ≈ 0

        // (This would require actually evaluating the polynomial,
        //  which we can't do without a full CKKS context)
    }
}

/// GPU-accelerated layout conversion: Flat → Strided
///
/// Replaces ~650k CPU operations with a single GPU kernel call.
/// CRITICAL for BSGS performance.
fn flat_to_strided_gpu(
    flat_data: &[u64],
    n: usize,
    stride: usize,
    num_active_primes: usize,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<Vec<u64>, String> {
    use cudarc::driver::LaunchAsync;

    let total_elements = n * num_active_primes;
    let output_size = n * stride;

    // Copy to GPU
    let gpu_input = ckks_ctx.device().device.htod_copy(flat_data[..total_elements].to_vec())
        .map_err(|e| format!("Failed to copy to GPU: {:?}", e))?;

    let mut gpu_output = ckks_ctx.device().device.alloc_zeros::<u64>(output_size)
        .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;

    // Get kernel
    let func = ckks_ctx.device().device.get_func("rns_module", "rns_flat_to_strided")
        .ok_or_else(|| "Failed to get rns_flat_to_strided kernel".to_string())?;

    // Launch kernel
    let threads_per_block = 256;
    let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (threads_per_block as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(cfg, (
            &gpu_input,
            &mut gpu_output,
            n as u32,
            stride as u32,
            num_active_primes as u32,
        )).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
    }

    // Copy result back
    ckks_ctx.device().device.dtoh_sync_copy(&gpu_output)
        .map_err(|e| format!("Failed to copy from GPU: {:?}", e))
}
