//! Test: Does gadget decomposition reconstruct correctly?
//! Verify: Σ d_t * B^t = d2
//! Run with: cargo test --test test_v2_decomposition_reconstruction --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};

#[test]
fn test_decomposition_reconstruction() {
    println!("\n=== GADGET DECOMPOSITION RECONSTRUCTION TEST ===\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let moduli: Vec<u64> = params.moduli[..=params.max_level()].to_vec();
    let n = params.n;
    let base_w = 20u32;

    println!("Testing with {} primes: {:?}", moduli.len(), moduli);
    println!("Base: 2^{} = {}", base_w, 1u64 << base_w);

    // Create a simple test value for coefficient 0
    // Use a value that's easy to track: 1,000,000,000
    let test_val = 1_000_000_000u64;

    let mut d2 = vec![RnsRepresentation::from_u64(0, &moduli); n];
    d2[0] = RnsRepresentation::from_u64(test_val, &moduli);

    println!("\nOriginal d2[0]: {:?}", d2[0].values);

    // Manually call gadget_decompose by computing it inline
    let digits = gadget_decompose(&d2, base_w, &moduli);

    println!("\nDecomposition produced {} digits", digits.len());
    for (t, digit) in digits.iter().enumerate() {
        println!("  Digit {}: coeff[0] = {:?}", t, digit[0].values);
    }

    // Reconstruct: Σ d_t * B^t
    println!("\n=== RECONSTRUCTION CHECK ===");

    let base = 1u64 << base_w;
    let mut reconstructed = RnsRepresentation::from_u64(0, &moduli);

    for (t, digit) in digits.iter().enumerate() {
        // Compute B^t
        let b_power = base.pow(t as u32);
        println!("\nDigit {}: B^{} = {}", t, t, b_power);
        println!("  digit[0] = {:?}", digit[0].values);

        // Compute digit[0] * B^t
        let term = digit[0].mul_scalar(b_power);
        println!("  digit[0] * B^{} = {:?}", t, term.values);

        reconstructed = reconstructed.add(&term);
        println!("  Running sum = {:?}", reconstructed.values);
    }

    println!("\n=== FINAL COMPARISON ===");
    println!("Original   d2[0]: {:?}", d2[0].values);
    println!("Reconstructed[0]: {:?}", reconstructed.values);

    // Check if they match
    let mut matches = true;
    for i in 0..moduli.len() {
        if d2[0].values[i] != reconstructed.values[i] {
            matches = false;
            println!("  ✗ Mismatch at prime {}: {} != {}", i, d2[0].values[i], reconstructed.values[i]);
        }
    }

    if matches {
        println!("\n✓ PERFECT RECONSTRUCTION! Σ d_t * B^t = d2");
    } else {
        println!("\n✗ RECONSTRUCTION FAILED!");

        // Show the difference
        println!("\nDifference analysis:");
        for i in 0..moduli.len() {
            let q = moduli[i];
            let orig = d2[0].values[i];
            let recon = reconstructed.values[i];

            let diff = if orig >= recon {
                orig - recon
            } else {
                q - (recon - orig)
            };

            println!("  Prime {}: diff = {} (mod {})", i, diff, q);
        }
    }
}

/// Manual gadget decomposition (CRT-consistent)
fn gadget_decompose(
    poly: &[RnsRepresentation],
    base_w: u32,
    moduli: &[u64],
) -> Vec<Vec<RnsRepresentation>> {
    let n = poly.len();
    let num_primes = moduli.len();

    // Compute Q = product of all primes using BigInt
    let q_prod_big: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();
    let q_half_big = &q_prod_big / 2;
    let base_big = BigInt::one() << base_w;
    let half_base_big = &base_big / 2;

    let q_bits = q_prod_big.bits() as u32;
    let num_digits = ((q_bits + base_w - 1) / base_w) as usize;

    let mut digits = vec![vec![RnsRepresentation::new(vec![0; num_primes], moduli.to_vec()); n]; num_digits];

    // Decompose coefficient 0 only for this test
    let i = 0;

    // Step 1: CRT reconstruct
    let residues: Vec<u64> = poly[i].values.clone();
    let x_big = crt_reconstruct_bigint(&residues, moduli);

    println!("\nCRT reconstruction:");
    println!("  Residues: {:?}", residues);
    println!("  Reconstructed X = {}", x_big);

    // Step 2: Center-lift
    let x_centered_big = if x_big > q_half_big {
        x_big - &q_prod_big
    } else {
        x_big
    };

    println!("  Centered X = {}", x_centered_big);

    // Step 3: Balanced decomposition
    let mut remainder_big = x_centered_big;

    for t in 0..num_digits {
        let dt_unbalanced = &remainder_big % &base_big;
        let dt_big = if dt_unbalanced > half_base_big {
            &dt_unbalanced - &base_big
        } else {
            dt_unbalanced
        };

        println!("\n  Digit {}: dt = {}", t, dt_big);

        // Convert to RNS
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

    digits
}

fn crt_reconstruct_bigint(residues: &[u64], moduli: &[u64]) -> BigInt {
    use num_traits::Zero;

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
    use num_traits::One;

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
    use num_traits::{Zero, One};

    if b == &BigInt::zero() {
        return (a.clone(), BigInt::one(), BigInt::zero());
    }

    let (gcd, x1, y1) = extended_gcd_bigint(b, &(a % b));
    let x = y1.clone();
    let y = x1 - (a / b) * &y1;

    (gcd, x, y)
}
