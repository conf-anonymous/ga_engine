//! Ciphertext Multiplication with NTT-based Relinearization
//!
//! This module implements the core ciphertext multiplication operation for CKKS,
//! which is the foundation for all geometric algebra operations.
//!
//! **Algorithm:**
//! 1. Tensor product: (c0, c1) ⊗ (c0', c1') → (d0, d1, d2)
//! 2. Relinearization: (d0, d1, d2) → (c0'', c1'') using evaluation key
//! 3. Rescaling: divide by last prime to maintain scale
//!
//! **Key optimizations:**
//! - NTT for O(n log n) polynomial multiplication
//! - Gadget decomposition for low-noise relinearization
//! - Exact rescaling with proper rounding

use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{EvaluationKey, KeyContext};
use crate::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use crate::clifford_fhe_v2::params::CliffordFHEParams;

use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};

/// Multiply two ciphertexts homomorphically
///
/// This is the core operation for geometric product.
/// Computes Enc(m1) ⊗ Enc(m2) = Enc(m1 * m2)
///
/// # Arguments
/// * `ct1` - First ciphertext
/// * `ct2` - Second ciphertext
/// * `evk` - Evaluation key for relinearization
/// * `key_ctx` - Key context with NTT precomputation
///
/// # Returns
/// Result ciphertext encrypting m1 * m2
pub fn multiply_ciphertexts(
    ct1: &Ciphertext,
    ct2: &Ciphertext,
    evk: &EvaluationKey,
    key_ctx: &KeyContext,
) -> Ciphertext {
    assert_eq!(ct1.n, ct2.n, "Ciphertexts must have same dimension");

    // Align levels: both ciphertexts must be at the same level
    // If they differ, switch the higher-level one down to match the lower
    let (ct1_aligned, ct2_aligned) = if ct1.level != ct2.level {
        let min_level = ct1.level.min(ct2.level);
        let ct1_new = if ct1.level > min_level {
            ct1.mod_switch_to_level(min_level)
        } else {
            ct1.clone()
        };
        let ct2_new = if ct2.level > min_level {
            ct2.mod_switch_to_level(min_level)
        } else {
            ct2.clone()
        };
        (ct1_new, ct2_new)
    } else {
        (ct1.clone(), ct2.clone())
    };

    let n = ct1_aligned.n;
    let level = ct1_aligned.level;
    let moduli: Vec<u64> = key_ctx.params.moduli[..=level].to_vec();

    // Step 1: Tensor product
    // (c0, c1) ⊗ (c0', c1') = (d0, d1, d2)
    // where d0 = c0*c0', d1 = c0*c1' + c1*c0', d2 = c1*c1'
    let d0 = multiply_polynomials(&ct1_aligned.c0, &ct2_aligned.c0, key_ctx, &moduli);
    let c0_c1_prime = multiply_polynomials(&ct1_aligned.c0, &ct2_aligned.c1, key_ctx, &moduli);
    let c1_c0_prime = multiply_polynomials(&ct1_aligned.c1, &ct2_aligned.c0, key_ctx, &moduli);
    let d2 = multiply_polynomials(&ct1_aligned.c1, &ct2_aligned.c1, key_ctx, &moduli);

    // d1 = c0*c1' + c1*c0'
    let d1: Vec<RnsRepresentation> = c0_c1_prime
        .iter()
        .zip(&c1_c0_prime)
        .map(|(a, b)| a.add(b))
        .collect();

    // Step 2: Relinearization (degree 2 → degree 1)
    let (new_c0, new_c1) = relinearize_degree2(&d0, &d1, &d2, evk, key_ctx, &moduli);

    if std::env::var("MULT_DEBUG").is_ok() {
        println!("\n[MULT_DEBUG] After relinearization, before rescale:");
        println!("  new_c0[0]: {:?}", new_c0[0].values);
        println!("  new_c1[0]: {:?}", new_c1[0].values);
    }

    // Step 3: Rescaling (divide by last prime)
    let (rescaled_c0, rescaled_c1) = rescale_ciphertext(&new_c0, &new_c1, &moduli);

    // New scale after multiplication and rescaling
    let new_scale = (ct1_aligned.scale * ct2_aligned.scale) / (moduli[moduli.len() - 1] as f64);

    // After rescaling, we dropped the last prime, so level decreases by 1
    // Level semantics: level L means "use moduli[0..=L]"
    // Example with 3 primes (indices 0,1,2):
    //   Fresh ct: level=2, uses moduli[0..=2] (all 3 primes)
    //   After 1 mult: level=1, uses moduli[0..=1] (2 primes, dropped last)
    //   After 2 mult: level=0, uses moduli[0..=0] (1 prime, dropped last two)
    let new_level = if level > 0 {
        level - 1
    } else {
        // Already at minimum level, cannot rescale further
        panic!("Cannot multiply ciphertexts at level 0 - no more primes to drop");
    };

    Ciphertext::new(rescaled_c0, rescaled_c1, new_level, new_scale)
}

/// Multiply two RNS polynomials using NTT
fn multiply_polynomials(
    a: &[RnsRepresentation],
    b: &[RnsRepresentation],
    key_ctx: &KeyContext,
    moduli: &[u64],
) -> Vec<RnsRepresentation> {
    let n = a.len();
    assert_eq!(b.len(), n);

    let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

    // Multiply for each prime separately using NTT
    for (prime_idx, &q) in moduli.iter().enumerate() {
        // Extract coefficients for this prime
        let a_mod_q: Vec<u64> = a.iter().map(|rns| rns.values[prime_idx]).collect();
        let b_mod_q: Vec<u64> = b.iter().map(|rns| rns.values[prime_idx]).collect();

        // Debug: print inputs before multiplication
        if std::env::var("POLY_DEBUG").is_ok() && prime_idx == 0 {
            print!("[POLY_DEBUG CPU] BEFORE mult - a[0:2] prime 0: ");
            for i in 0..2.min(a_mod_q.len()) {
                print!("{} ", a_mod_q[i]);
            }
            println!();
            print!("[POLY_DEBUG CPU] BEFORE mult - b[0:2] prime 0: ");
            for i in 0..2.min(b_mod_q.len()) {
                print!("{} ", b_mod_q[i]);
            }
            println!();
        }

        // Find the NTT context for this prime
        let ntt_ctx = key_ctx
            .ntt_contexts
            .iter()
            .find(|ctx| ctx.q == q)
            .expect("NTT context not found for prime");

        // Multiply using NTT
        let product_mod_q = ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q);

        // Debug: print output after multiplication
        if std::env::var("POLY_DEBUG").is_ok() && prime_idx == 0 {
            print!("[POLY_DEBUG CPU] AFTER mult - product[0:2] prime 0: ");
            for i in 0..2.min(product_mod_q.len()) {
                print!("{} ", product_mod_q[i]);
            }
            println!();
        }

        // Store results
        for (i, &val) in product_mod_q.iter().enumerate() {
            result[i].values[prime_idx] = val;
        }
    }

    result
}

/// Relinearize degree-2 ciphertext to degree-1 using evaluation key
///
/// Input: (d0, d1, d2) where decryption is d0 + d1*s + d2*s²
/// Output: (c0, c1) where decryption is c0 + c1*s
///
/// Uses gadget decomposition with base B = 2^w
fn relinearize_degree2(
    d0: &[RnsRepresentation],
    d1: &[RnsRepresentation],
    d2: &[RnsRepresentation],
    evk: &EvaluationKey,
    key_ctx: &KeyContext,
    moduli: &[u64],
) -> (Vec<RnsRepresentation>, Vec<RnsRepresentation>) {
    let n = d0.len();
    let base_w = evk.base_w;

    // Initialize output ciphertexts
    let mut c0 = d0.to_vec();
    let mut c1 = d1.to_vec();

    // Debug: print values before relinearization
    if std::env::var("RELIN_DEBUG").is_ok() {
        println!("[RELIN_DEBUG CPU] Before relinearization:");
        print!("  c0[0] across {} primes: ", moduli.len());
        for j in 0..moduli.len() {
            print!("{} ", c0[0].values[j]);
        }
        println!();
        print!("  c1[0] across {} primes: ", moduli.len());
        for j in 0..moduli.len() {
            print!("{} ", c1[0].values[j]);
        }
        println!();
        print!("  d2[0] across {} primes: ", moduli.len());
        for j in 0..moduli.len() {
            print!("{} ", d2[0].values[j]);
        }
        println!();
    }

    // Debug: print d2 values before gadget decomposition
    if std::env::var("C2_DEBUG").is_ok() {
        print!("[C2_DEBUG CPU] d2[0] across primes: ");
        for j in 0..moduli.len() {
            print!("{} ", d2[0].values[j]);
        }
        println!();
    }

    // Decompose d2 using gadget decomposition
    let d2_decomposed = gadget_decompose(d2, base_w, moduli);

    // Debug: print first few values of first digit
    if std::env::var("MULT_DEBUG").is_ok() && !d2_decomposed.is_empty() {
        println!("[MULT_DEBUG] CPU gadget decomposition:");
        println!("  num_digits: {}", d2_decomposed.len());
        println!("  first digit, first 4 coeffs × {} primes:", n.min(4));
        for i in 0..n.min(4) {
            print!("    coeff[{}]: ", i);
            for j in 0..moduli.len() {
                print!("{} ", d2_decomposed[0][i].values[j]);
            }
            println!();
        }
    }

    // For each digit in the decomposition
    for (t, d2_digit) in d2_decomposed.iter().enumerate() {
        if t >= evk.evk0.len() {
            break; // No more evaluation key components
        }

        // Multiply d2_digit by evk[t] and accumulate
        // The EVK encrypts -B^t·s², so we need to SUBTRACT term0 and ADD term1
        // This follows from: evk0[t] - evk1[t]·s = -B^t·s² + noise
        // Therefore: c0 -= d2_digit * evk0[t], c1 += d2_digit * evk1[t]

        if std::env::var("DETAIL_DEBUG").is_ok() && t == 0 {
            print!("[DETAIL_DEBUG CPU] digit[0] coeff[0] primes: ");
            for j in 0..moduli.len() {
                print!("{} ", d2_digit[0].values[j]);
            }
            println!();
            print!("[DETAIL_DEBUG CPU] evk0[0] coeff[0] primes: ");
            for j in 0..moduli.len() {
                print!("{} ", evk.evk0[t][0].values[j]);
            }
            println!();
        }

        let term0 = multiply_polynomials(d2_digit, &evk.evk0[t], key_ctx, moduli);
        let term1 = multiply_polynomials(d2_digit, &evk.evk1[t], key_ctx, moduli);

        if std::env::var("DETAIL_DEBUG").is_ok() && t == 0 {
            print!("[DETAIL_DEBUG CPU] term0 coeff[0] primes: ");
            for j in 0..moduli.len() {
                print!("{} ", term0[0].values[j]);
            }
            println!();
            print!("[DETAIL_DEBUG CPU] term1 coeff[0] primes: ");
            for j in 0..moduli.len() {
                print!("{} ", term1[0].values[j]);
            }
            println!();
            print!("[DETAIL_DEBUG CPU] c0[0] BEFORE subtract: ");
            for j in 0..moduli.len() {
                print!("{} ", c0[0].values[j]);
            }
            println!();
            print!("[DETAIL_DEBUG CPU] c1[0] BEFORE add: ");
            for j in 0..moduli.len() {
                print!("{} ", c1[0].values[j]);
            }
            println!();
        }

        for i in 0..n {
            c0[i] = c0[i].sub(&term0[i]);  // SUBTRACT term0
            c1[i] = c1[i].add(&term1[i]);  // ADD term1
        }

        if std::env::var("DETAIL_DEBUG").is_ok() && t == 0 {
            print!("[DETAIL_DEBUG CPU] c0[0] AFTER subtract: ");
            for j in 0..moduli.len() {
                print!("{} ", c0[0].values[j]);
            }
            println!();
            print!("[DETAIL_DEBUG CPU] c1[0] AFTER add: ");
            for j in 0..moduli.len() {
                print!("{} ", c1[0].values[j]);
            }
            println!();
        }
    }

    // Debug: print values after relinearization
    if std::env::var("RELIN_DEBUG").is_ok() {
        println!("[RELIN_DEBUG CPU] After relinearization:");
        print!("  c0[0] across {} primes: ", moduli.len());
        for j in 0..moduli.len() {
            print!("{} ", c0[0].values[j]);
        }
        println!();
        print!("  c1[0] across {} primes: ", moduli.len());
        for j in 0..moduli.len() {
            print!("{} ", c1[0].values[j]);
        }
        println!();
    }

    (c0, c1)
}

/// Gadget decomposition for relinearization
///
/// **CRITICAL**: Uses CRT-consistent decomposition with BigInt!
///
/// For each coefficient:
/// 1. Reconstruct the integer x ∈ [0, Q) via CRT from all residues (using BigInt for large Q)
/// 2. Center-lift to x_c ∈ (-Q/2, Q/2]
/// 3. Balanced decomposition in Z: x_c = Σ dt·B^t where dt ∈ [-B/2, B/2)
/// 4. Map each digit back to RNS identically across all primes
///
/// This ensures Σ dt·B^t ≡ x (mod qi) for EVERY prime qi, maintaining the
/// EVK cancellation property even when noise is present.
///
/// **Why this matters**: Per-prime decomposition breaks relinearization because
/// digits don't represent the same value mod each prime, so EVK cancellation fails!
fn gadget_decompose(
    poly: &[RnsRepresentation],
    base_w: u32,
    moduli: &[u64],
) -> Vec<Vec<RnsRepresentation>> {
    let n = poly.len();
    let num_primes = moduli.len();

    // Compute Q = product of all primes using BigInt (Q can be 140+ bits!)
    let q_prod_big: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();
    let q_half_big = &q_prod_big / 2;
    let base_big = BigInt::one() << base_w;
    let half_base_big = &base_big / 2;

    // Determine number of digits needed
    let q_bits = q_prod_big.bits() as u32;
    let num_digits = ((q_bits + base_w - 1) / base_w) as usize;

    if std::env::var("DECOMP_DEBUG").is_ok() {
        println!("[DECOMP_DEBUG] Q has {} bits, base_w={}, num_digits={}", q_bits, base_w, num_digits);
        println!("[DECOMP_DEBUG] Moduli ({} primes): {:?}", num_primes, moduli);
    }

    let mut digits = vec![vec![RnsRepresentation::new(vec![0; num_primes], moduli.to_vec()); n]; num_digits];

    // Decompose each coefficient using CRT
    for i in 0..n {
        // Step 1: CRT reconstruct to get x ∈ [0, Q)
        let residues: Vec<u64> = poly[i].values.clone();

        // Debug: print input residues for first coefficient
        if std::env::var("CRT_DEBUG").is_ok() && i == 0 {
            print!("[CRT_DEBUG CPU] coeff[0] input residues: ");
            for &r in residues.iter() {
                print!("{} ", r);
            }
            println!();
        }

        let x_big = crt_reconstruct_bigint(&residues, moduli);

        // Debug: print reconstructed value for first coefficient
        if std::env::var("CRT_DEBUG").is_ok() && i == 0 {
            println!("[CRT_DEBUG CPU] coeff[0] after CRT: {}", x_big);
        }

        // Step 2: Center-lift to x_c ∈ (-Q/2, Q/2]
        let x_centered_big = if x_big > q_half_big {
            x_big - &q_prod_big
        } else {
            x_big
        };

        // Debug: print centered value for first coefficient
        if std::env::var("CRT_DEBUG").is_ok() && i == 0 {
            println!("[CRT_DEBUG CPU] coeff[0] after centering: {}", x_centered_big);
        }

        // Step 3: Balanced decomposition in Z
        let mut remainder_big = x_centered_big;

        for t in 0..num_digits {
            // Extract digit dt ∈ (-B/2, B/2] (balanced, matching V1's logic)
            let dt_unbalanced = &remainder_big % &base_big;
            let dt_big = if dt_unbalanced > half_base_big {
                &dt_unbalanced - &base_big  // Shift to negative range
            } else {
                dt_unbalanced
            };

            // Debug: print first digit for first coefficient
            if std::env::var("GADGET_DEBUG").is_ok() && i == 0 && t == 0 {
                println!("[GADGET_DEBUG CPU] coeff[0] digit[0]: dt_big={}", dt_big);
            }

            // Convert dt to residues mod each prime
            for (j, &q) in moduli.iter().enumerate() {
                let q_big = BigInt::from(q);
                let mut dt_mod_q_big = &dt_big % &q_big;
                if dt_mod_q_big.sign() == num_bigint::Sign::Minus {
                    dt_mod_q_big += &q_big;
                }
                digits[t][i].values[j] = dt_mod_q_big.to_u64().unwrap();

                // Debug: print conversion
                if std::env::var("GADGET_DEBUG").is_ok() && i == 0 && t == 0 {
                    println!("[GADGET_DEBUG CPU] coeff[0] digit[0] prime[{}]: dt_mod_q={}", j, digits[t][i].values[j]);
                }
            }

            // Update remainder: (x_c - dt) / B (exact division)
            remainder_big = (remainder_big - &dt_big) / &base_big;
        }
    }

    digits
}

/// CRT reconstruction using BigInt (handles large Q > i128::MAX)
fn crt_reconstruct_bigint(residues: &[u64], moduli: &[u64]) -> BigInt {
    let q_prod: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();

    let mut x = BigInt::zero();
    for (i, &ri) in residues.iter().enumerate() {
        let qi = BigInt::from(moduli[i]);
        let q_i = &q_prod / &qi;

        // Compute q_i^(-1) mod qi using extended GCD (matching V1)
        let qi_inv = mod_inverse_bigint(&q_i, &qi);

        let ri_big = BigInt::from(ri);
        // Compute: basis = (Q/qi) * inv mod Q, then term = ri * basis mod Q
        let basis = (&q_i * &qi_inv) % &q_prod;
        let term = (ri_big * basis) % &q_prod;
        x = (x + term) % &q_prod;
    }

    // Ensure result is positive
    if x.sign() == num_bigint::Sign::Minus {
        x += &q_prod;
    }

    x
}

/// Modular inverse using extended GCD (matching V1's implementation)
fn mod_inverse_bigint(a: &BigInt, m: &BigInt) -> BigInt {
    // Extended GCD to find x such that a*x ≡ 1 (mod m)
    let (gcd, x, _) = extended_gcd_bigint(a, m);

    if gcd != BigInt::one() {
        panic!("Modular inverse does not exist (gcd != 1)");
    }

    // Ensure positive result
    let mut result = x % m;
    if result.sign() == num_bigint::Sign::Minus {
        result += m;
    }

    result
}

/// Extended GCD for BigInt: returns (gcd, x, y) such that gcd = a*x + b*y
fn extended_gcd_bigint(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if b == &BigInt::zero() {
        return (a.clone(), BigInt::one(), BigInt::zero());
    }

    let (gcd, x1, y1) = extended_gcd_bigint(b, &(a % b));
    let x = y1.clone();
    let y = x1 - (a / b) * &y1;

    (gcd, x, y)
}

/// Modular exponentiation for BigInt
fn mod_pow_bigint(base: &BigInt, exp: &BigInt, m: &BigInt) -> BigInt {
    if exp.sign() == num_bigint::Sign::Minus {
        panic!("Negative exponent not supported");
    }

    let mut result = BigInt::one();
    let mut base = base.clone() % m;
    let mut exp = exp.clone();

    while exp > BigInt::zero() {
        if (&exp % 2) == BigInt::one() {
            result = (result * &base) % m;
        }
        base = (&base * &base) % m;
        exp /= 2;
    }

    result
}

/// Rescale ciphertext by dividing by last prime
///
/// After multiplication, scale is Δ². Dividing by q_last brings it back to ~Δ.
/// Uses exact division with proper rounding.
fn rescale_ciphertext(
    c0: &[RnsRepresentation],
    c1: &[RnsRepresentation],
    moduli: &[u64],
) -> (Vec<RnsRepresentation>, Vec<RnsRepresentation>) {
    let n = c0.len();
    let num_primes = moduli.len();

    if num_primes == 1 {
        // Cannot rescale further, just return copies
        return (c0.to_vec(), c1.to_vec());
    }

    let q_last = moduli[num_primes - 1];
    let new_moduli = moduli[..num_primes - 1].to_vec();

    let rescaled_c0 = rescale_polynomial(c0, q_last, &new_moduli);
    let rescaled_c1 = rescale_polynomial(c1, q_last, &new_moduli);

    (rescaled_c0, rescaled_c1)
}

/// Rescale a single polynomial by dividing by q_last
fn rescale_polynomial(
    poly: &[RnsRepresentation],
    q_last: u64,
    new_moduli: &[u64],
) -> Vec<RnsRepresentation> {
    let n = poly.len();
    let mut result = vec![RnsRepresentation::new(vec![0; new_moduli.len()], new_moduli.to_vec()); n];

    // Precompute q_last^(-1) mod qi for each remaining prime
    let q_last_inv: Vec<u64> = new_moduli
        .iter()
        .map(|&qi| {
            // Compute q_last^(-1) mod qi using Fermat's little theorem
            mod_pow(q_last % qi, qi - 2, qi)
        })
        .collect();

    // Debug output for first coefficient only
    let debug = std::env::var("RESCALE_DEBUG").is_ok() && n > 0;

    if debug {
        println!("\n[RESCALE_DEBUG] rescale_polynomial called:");
        println!("  n = {}", n);
        println!("  q_last = {}", q_last);
        println!("  new_moduli = {:?}", new_moduli);
        println!("  poly[0].moduli = {:?}", poly[0].moduli);
        println!("  poly[0].values = {:?}", poly[0].values);
    }

    for i in 0..n {
        // Get the value mod q_last (from last position in original poly)
        let num_primes_orig = poly[i].moduli.len();
        let val_mod_qlast = poly[i].values[num_primes_orig - 1];

        // Centered lift: convert to signed value
        let val_centered = if val_mod_qlast > q_last / 2 {
            val_mod_qlast as i64 - q_last as i64
        } else {
            val_mod_qlast as i64
        };

        if debug && i == 0 {
            println!("  num_primes_orig = {}", num_primes_orig);
            println!("  val_mod_qlast = poly[0].values[{}] = {}", num_primes_orig - 1, val_mod_qlast);
            println!("  val_centered = {}", val_centered);
        }

        // For each remaining prime qi:
        // new_val_i = (old_val_i - val_centered) * q_last^(-1) mod qi
        for (j, &qi) in new_moduli.iter().enumerate() {
            let old_val = poly[i].values[j];

            // Compute (old_val - val_centered) mod qi
            let diff = if val_centered >= 0 {
                let vc = (val_centered as u64) % qi;
                if old_val >= vc {
                    old_val - vc
                } else {
                    qi - (vc - old_val)
                }
            } else {
                let vc = ((-val_centered) as u64) % qi;
                (old_val + vc) % qi
            };

            // Multiply by q_last^(-1) mod qi
            let new_val = ((diff as u128) * (q_last_inv[j] as u128)) % (qi as u128);
            result[i].values[j] = new_val as u64;

            if debug && i == 0 {
                println!("  Prime j={}, qi={}:", j, qi);
                println!("    old_val = {}", old_val);
                println!("    diff = {}", diff);
                println!("    q_last_inv[{}] = {}", j, q_last_inv[j]);
                println!("    new_val = {}", new_val);
            }
        }
    }

    if debug {
        println!("  result[0].values = {:?}", result[0].values);
    }

    result
}

/// Modular exponentiation: base^exp mod m
fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base = base % m;

    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128) * (base as u128) % (m as u128)) as u64;
        }
        base = ((base as u128) * (base as u128) % (m as u128)) as u64;
        exp >>= 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;

    #[test]
    fn test_mod_pow() {
        // Test a^(p-2) ≡ a^(-1) mod p (Fermat's little theorem)
        let p = 97u64;
        let a = 5u64;
        let a_inv = mod_pow(a, p - 2, p);
        let product = (a * a_inv) % p;
        assert_eq!(product, 1);
    }

    #[test]
    fn test_gadget_decompose() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let moduli = params.moduli[..=params.max_level()].to_vec();
        let n = params.n;

        // Create a simple polynomial
        let poly: Vec<RnsRepresentation> = (0..n)
            .map(|i| {
                let val = if i == 0 { 1000000u64 } else { 0 };
                RnsRepresentation::from_u64(val, &moduli)
            })
            .collect();

        let base_w = 20;
        let digits = gadget_decompose(&poly, base_w, &moduli);

        // Should have ceil(Q_bits/20) digits where Q = product of moduli
        // For 3 primes of ~60, 40, 40 bits: Q ~ 140 bits → ceil(140/20) = 8 digits
        assert!(digits.len() >= 3, "Should have at least 3 digits");

        // Each digit should be a polynomial of length n
        for digit in &digits {
            assert_eq!(digit.len(), n);
        }
    }

    #[test]
    fn test_multiply_polynomials() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let moduli = params.moduli[..=params.max_level()].to_vec();
        let n = params.n;

        // Create two simple polynomials: [1, 0, 0, ...] and [2, 0, 0, ...]
        let mut a = vec![RnsRepresentation::from_u64(0, &moduli); n];
        a[0] = RnsRepresentation::from_u64(1, &moduli);

        let mut b = vec![RnsRepresentation::from_u64(0, &moduli); n];
        b[0] = RnsRepresentation::from_u64(2, &moduli);

        let result = multiply_polynomials(&a, &b, &key_ctx, &moduli);

        // Verify result has correct dimension
        assert_eq!(result.len(), n);

        // NTT multiplication should be commutative
        let result_ba = multiply_polynomials(&b, &a, &key_ctx, &moduli);
        for i in 0..n {
            assert_eq!(result[i].values[0], result_ba[i].values[0]);
        }
    }
}
