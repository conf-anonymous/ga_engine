//! Test Gadget Decomposition Round-Trip
//!
//! Verifies that decomposition correctly reconstructs the original value.

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use num_bigint::BigInt;
    use num_traits::{Zero, One, ToPrimitive};
    use num_integer::Integer;

    println!("Testing Gadget Decomposition Round-Trip");
    println!("========================================\n");

    // Use real FHE parameters
    let params = CliffordFHEParams::new_v3_bootstrap_metal()?;
    let moduli = &params.moduli[..=19]; // Use 20 primes
    let n = params.n;
    let base_w = 30u32;

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  num_primes = {}", moduli.len());
    println!("  base_w = {} (B = 2^{})", base_w, base_w);
    println!();

    // Compute Q = product of primes
    let q_prod: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();
    let q_half = &q_prod / 2;
    let base_big = BigInt::one() << base_w;

    println!("Q (product of primes) has {} bits\n", q_prod.bits());

    // Test with several values
    let test_values = vec![
        42i64,
        -100i64,
        1000000i64,
        -5000000i64,
    ];

    for &test_val in &test_values {
        println!("Testing value: {}", test_val);

        // Convert to centered BigInt mod Q
        let mut val_big = BigInt::from(test_val);
        val_big = val_big.mod_floor(&q_prod);
        if val_big > q_half {
            val_big -= &q_prod;
        }

        println!("  Centered value: {}", val_big);

        // Decompose into base-B digits
        let mut remaining = val_big.clone();
        let half_base = &base_big / 2;
        let mut digits_big: Vec<BigInt> = Vec::new();

        while !remaining.is_zero() {
            let mut digit_t = remaining.mod_floor(&base_big);
            if digit_t > half_base {
                digit_t -= &base_big;
            }
            digits_big.push(digit_t.clone());
            remaining = (&remaining - &digit_t).div_floor(&base_big);
        }

        println!("  Decomposed into {} digits", digits_big.len());

        // Reconstruct from digits
        let mut reconstructed = BigInt::zero();
        let mut power = BigInt::one();
        for (t, d) in digits_big.iter().enumerate() {
            reconstructed += d * &power;
            power *= &base_big;
            if t < 5 {
                println!("    digit[{}] = {}", t, d);
            }
        }

        println!("  Reconstructed: {}", reconstructed);

        // Check equality
        if reconstructed == val_big {
            println!("  ✅ PASS: Decomposition round-trip correct!\n");
        } else {
            println!("  ❌ FAIL: Expected {}, got {}\n", val_big, reconstructed);
            return Err(format!("Decomposition round-trip failed for value {}", test_val));
        }
    }

    // Now test with RNS conversion
    println!("Testing RNS conversion round-trip:");
    println!("===================================\n");

    for &test_val in &test_values {
        println!("Testing value: {}", test_val);

        // Convert to centered BigInt mod Q
        let mut val_big = BigInt::from(test_val);
        val_big = val_big.mod_floor(&q_prod);
        if val_big > q_half {
            val_big -= &q_prod;
        }

        // Convert to RNS (flat layout)
        let num_primes = moduli.len();
        let mut rns_flat = vec![0u64; num_primes];
        for (j, &q) in moduli.iter().enumerate() {
            let residue = if val_big < BigInt::zero() {
                let x = (-&val_big).mod_floor(&BigInt::from(q)).to_u64().unwrap();
                if x == 0 { 0 } else { q - x }
            } else {
                val_big.mod_floor(&BigInt::from(q)).to_u64().unwrap()
            };
            rns_flat[j] = residue;
        }

        // CRT reconstruct
        let mut reconstructed = BigInt::zero();
        for (j, &q) in moduli.iter().enumerate() {
            let residue = rns_flat[j];
            let q_big = BigInt::from(q);
            let m_j = &q_prod / &q_big;

            // Compute modular inverse
            let m_j_inv = mod_inverse(&m_j, &q_big)?;

            let term = BigInt::from(residue) * m_j * m_j_inv;
            reconstructed += term;
        }
        reconstructed = reconstructed.mod_floor(&q_prod);
        if reconstructed > q_half {
            reconstructed -= &q_prod;
        }

        println!("  Original:      {}", val_big);
        println!("  Reconstructed: {}", reconstructed);

        if reconstructed == val_big {
            println!("  ✅ PASS: RNS round-trip correct!\n");
        } else {
            println!("  ❌ FAIL: RNS reconstruction mismatch\n");
            return Err(format!("RNS round-trip failed for value {}", test_val));
        }
    }

    println!("✅ All decomposition tests PASSED!");
    Ok(())
}

fn mod_inverse(a: &num_bigint::BigInt, modulus: &num_bigint::BigInt) -> Result<num_bigint::BigInt, String> {
    use num_traits::{Zero, One};
    use num_bigint::BigInt;

    let mut t = BigInt::zero();
    let mut newt = BigInt::one();
    let mut r = modulus.clone();
    let mut newr = a.clone();

    while !newr.is_zero() {
        let quotient = &r / &newr;
        let temp_t = t.clone();
        t = newt.clone();
        newt = temp_t - &quotient * &newt;

        let temp_r = r.clone();
        r = newr.clone();
        newr = temp_r - quotient * newr;
    }

    if r > BigInt::one() {
        return Err(format!("Not invertible"));
    }
    if t < BigInt::zero() {
        t += modulus;
    }

    Ok(t)
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3")))]
fn main() {
    eprintln!("This example requires features: v2,v2-gpu-metal,v3");
    std::process::exit(1);
}
