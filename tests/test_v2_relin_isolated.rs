//! Test relinearization in complete isolation
//! Manually construct d0, d1, d2 such that we know the exact expected output
//! Run with: cargo test --test test_v2_relin_isolated --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use num_bigint::BigInt;
use num_traits::{One, Zero, ToPrimitive};

#[test]
fn test_relinearization_isolated() {
    println!("\n=== RELINEARIZATION ISOLATION TEST ===\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (_pk, sk, evk) = key_ctx.keygen();

    let n = params.n;
    let moduli: Vec<u64> = params.moduli[..=params.max_level()].to_vec();

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  Moduli: {:?}", moduli);
    println!("  EVK has {} components", evk.evk0.len());

    // Construct simple degree-2 ciphertext: d0 + d1*s + d2*s²
    // We want to choose d0, d1, d2 such that we know what the result should be
    //
    // Strategy: Use very small values so we can manually compute the expected result
    //
    // Let's use:
    //   d0[0] = 1000
    //   d1[0] = 2000
    //   d2[0] = 3000  (and d2[i] = 0 for i > 0)
    //
    // Then after relinearization:
    //   m = d0 + d1*s + d2*s²
    //
    // If relinearization is correct, we should be able to decrypt this.

    let mut d0 = vec![RnsRepresentation::from_u64(0, &moduli); n];
    d0[0] = RnsRepresentation::from_u64(1000, &moduli);

    let mut d1 = vec![RnsRepresentation::from_u64(0, &moduli); n];
    d1[0] = RnsRepresentation::from_u64(2000, &moduli);

    let mut d2 = vec![RnsRepresentation::from_u64(0, &moduli); n];
    d2[0] = RnsRepresentation::from_u64(3000, &moduli);

    println!("\nConstructed degree-2 ciphertext:");
    println!("  d0[0] = {:?}", d0[0].values);
    println!("  d1[0] = {:?}", d1[0].values);
    println!("  d2[0] = {:?}", d2[0].values);

    // Compute expected value before relinearization
    // m = d0[0] + d1[0]*s[0] + d2[0]*s²[0]
    let s0 = &sk.coeffs[0];
    println!("\nSecret key:");
    println!("  s[0] = {:?}", s0.values);

    // Compute s² (full polynomial, not just first coefficient)
    let s_squared = mult_scalar_polys(&sk.coeffs, &sk.coeffs, &key_ctx, &moduli);
    println!("  s²[0] = {:?}", s_squared[0].values);

    // Compute expected m = d0[0] + d1[0]*s[0] + d2[0]*s²[0]
    let term1 = d1[0].mul_scalar(s0.values[0]);  // d1[0] * s[0] (scalar multiplication mod each prime)
    let term2 = d2[0].mul_scalar(s_squared[0].values[0]);  // d2[0] * s²[0]

    let expected_m = d0[0].add(&term1).add(&term2);

    println!("\nExpected before relinearization:");
    println!("  m = d0[0] + d1[0]*s[0] + d2[0]*s²[0]");
    println!("  m[0] = {:?}", expected_m.values);

    // NOW: Apply relinearization
    println!("\n=== APPLYING RELINEARIZATION ===");

    let (c0_relin, c1_relin) = manual_relinearize(&d0, &d1, &d2, &evk, &key_ctx, &moduli);

    println!("\nAfter relinearization:");
    println!("  c0[0] = {:?}", c0_relin[0].values);
    println!("  c1[0] = {:?}", c1_relin[0].values);

    // Decrypt the relinearized ciphertext
    // m' = c0[0] + c1[0]*s[0]
    let c1_times_s = mult_scalar_polys(&c1_relin, &sk.coeffs, &key_ctx, &moduli);
    println!("\nDEBUG: c1_times_s[0] = {:?}", c1_times_s[0].values);
    let decrypted_m = c0_relin[0].add(&c1_times_s[0]);

    println!("\nDecrypted after relinearization:");
    println!("  m' = c0[0] + c1[0]*s[0]");
    println!("  m'[0] = {:?}", decrypted_m.values);

    println!("\n=== COMPARISON ===");
    println!("Expected m[0]: {:?}", expected_m.values);
    println!("Actual  m'[0]: {:?}", decrypted_m.values);

    let mut matches = true;
    for i in 0..moduli.len() {
        let diff = if expected_m.values[i] >= decrypted_m.values[i] {
            expected_m.values[i] - decrypted_m.values[i]
        } else {
            moduli[i] - (decrypted_m.values[i] - expected_m.values[i])
        };

        println!("  Prime {}: diff = {} (mod {})", i, diff, moduli[i]);

        // Allow for small noise (< 1000)
        if diff > 1000 && diff < moduli[i] - 1000 {
            matches = false;
        }
    }

    if matches {
        println!("\n✓ RELINEARIZATION WORKS! (within noise tolerance)");
    } else {
        println!("\n✗ RELINEARIZATION FAILED!");
    }
}

/// Manual relinearization to inspect every step
fn manual_relinearize(
    d0: &[RnsRepresentation],
    d1: &[RnsRepresentation],
    d2: &[RnsRepresentation],
    evk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::EvaluationKey,
    key_ctx: &KeyContext,
    moduli: &[u64],
) -> (Vec<RnsRepresentation>, Vec<RnsRepresentation>) {
    let _n = d0.len();
    let base_w = evk.base_w;

    println!("Relinearization:");
    println!("  base_w = {}", base_w);
    println!("  EVK components = {}", evk.evk0.len());

    // Decompose d2
    let digits = gadget_decompose_simple(d2, base_w, moduli);
    println!("  Decomposed d2 into {} digits", digits.len());

    for (t, digit) in digits.iter().enumerate() {
        println!("    Digit {}: coeff[0] = {:?}", t, digit[0].values);
    }

    // Initialize c0, c1
    let mut c0 = d0.to_vec();
    let mut c1 = d1.to_vec();

    println!("\nInitial:");
    println!("  c0[0] = {:?}", c0[0].values);
    println!("  c1[0] = {:?}", c1[0].values);

    // Apply each digit
    for (t, digit) in digits.iter().enumerate() {
        if t >= evk.evk0.len() {
            println!("\n  Digit {}: skipping (no EVK component)", t);
            break;
        }

        println!("\n  Digit {}:", t);
        println!("    digit[0] = {:?}", digit[0].values);
        println!("    evk0[{}][0] = {:?}", t, evk.evk0[t][0].values);
        println!("    evk1[{}][0] = {:?}", t, evk.evk1[t][0].values);

        // term0 = digit * evk0[t]
        let term0 = mult_scalar_polys(digit, &evk.evk0[t], key_ctx, moduli);
        println!("    term0[0] = digit[0] * evk0[{}][0] = {:?}", t, term0[0].values);

        // term1 = digit * evk1[t]
        let term1 = mult_scalar_polys(digit, &evk.evk1[t], key_ctx, moduli);
        println!("    term1[0] = digit[0] * evk1[{}][0] = {:?}", t, term1[0].values);

        // c0 -= term0
        let old_c0 = c0[0].values.clone();
        c0[0] = c0[0].sub(&term0[0]);
        println!("    c0[0] -= term0[0]: {:?} - {:?} = {:?}", old_c0, term0[0].values, c0[0].values);

        // c1 += term1
        let old_c1 = c1[0].values.clone();
        c1[0] = c1[0].add(&term1[0]);
        println!("    c1[0] += term1[0]: {:?} + {:?} = {:?}", old_c1, term1[0].values, c1[0].values);
    }

    println!("\nFinal:");
    println!("  c0[0] = {:?}", c0[0].values);
    println!("  c1[0] = {:?}", c1[0].values);

    (c0, c1)
}

fn gadget_decompose_simple(
    poly: &[RnsRepresentation],
    base_w: u32,
    moduli: &[u64],
) -> Vec<Vec<RnsRepresentation>> {
    let n = poly.len();
    let num_primes = moduli.len();

    let q_prod_big: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();
    let q_half_big = &q_prod_big / 2;
    let base_big = BigInt::one() << base_w;
    let half_base_big = &base_big / 2;

    let q_bits = q_prod_big.bits() as u32;
    let num_digits = ((q_bits + base_w - 1) / base_w) as usize;

    let mut digits = vec![vec![RnsRepresentation::new(vec![0; num_primes], moduli.to_vec()); n]; num_digits];

    for i in 0..n {
        let residues: Vec<u64> = poly[i].values.clone();
        let x_big = crt_reconstruct_bigint(&residues, moduli);

        let x_centered_big = if x_big > q_half_big {
            x_big - &q_prod_big
        } else {
            x_big
        };

        let mut remainder_big = x_centered_big;

        for t in 0..num_digits {
            let dt_unbalanced = &remainder_big % &base_big;
            let dt_big = if dt_unbalanced > half_base_big {
                &dt_unbalanced - &base_big
            } else {
                dt_unbalanced
            };

            for (j, &q) in moduli.iter().enumerate() {
                let q_big = BigInt::from(q);
                let mut dt_mod_q_big = &dt_big % &q_big;
                if dt_mod_q_big.sign() == num_bigint::Sign::Minus {
                    dt_mod_q_big += &q_big;
                }
                digits[t][i].values[j] = dt_mod_q_big.to_u64().unwrap();
            }

            remainder_big = (remainder_big - &dt_big) / &base_big;
        }
    }

    digits
}

fn mult_scalar_polys(
    a: &[RnsRepresentation],
    b: &[RnsRepresentation],
    key_ctx: &KeyContext,
    moduli: &[u64],
) -> Vec<RnsRepresentation> {
    let n = a.len();
    let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = key_ctx.ntt_contexts.iter().find(|ctx| ctx.q == q).unwrap();
        let a_mod_q: Vec<u64> = a.iter().map(|rns| rns.values[prime_idx]).collect();
        let b_mod_q: Vec<u64> = b.iter().map(|rns| rns.values[prime_idx]).collect();
        let product_mod_q = ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q);

        for i in 0..n {
            result[i].values[prime_idx] = product_mod_q[i];
        }
    }

    result
}

fn crt_reconstruct_bigint(residues: &[u64], moduli: &[u64]) -> BigInt {
    let q_prod: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();

    let mut x = BigInt::zero();
    for (i, &ri) in residues.iter().enumerate() {
        let qi = BigInt::from(moduli[i]);
        let q_i = &q_prod / &qi;

        let qi_inv = mod_inverse_bigint(&q_i, &qi);

        let ri_big = BigInt::from(ri);
        let basis = (&q_i * &qi_inv) % &q_prod;
        let term = (ri_big * basis) % &q_prod;
        x = (x + term) % &q_prod;
    }

    if x.sign() == num_bigint::Sign::Minus {
        x += &q_prod;
    }

    x
}

fn mod_inverse_bigint(a: &BigInt, m: &BigInt) -> BigInt {
    let (gcd, x, _) = extended_gcd_bigint(a, m);

    if gcd != BigInt::one() {
        panic!("Modular inverse does not exist");
    }

    let mut result = x % m;
    if result.sign() == num_bigint::Sign::Minus {
        result += m;
    }

    result
}

fn extended_gcd_bigint(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if b == &BigInt::zero() {
        return (a.clone(), BigInt::one(), BigInt::zero());
    }

    let (gcd, x1, y1) = extended_gcd_bigint(b, &(a % b));
    let x = y1.clone();
    let y = x1 - (a / b) * &y1;

    (gcd, x, y)
}
