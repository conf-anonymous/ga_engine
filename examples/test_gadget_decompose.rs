//! Test Gadget Decomposition Round-Trip
//!
//! Verifies that gadget decomposition correctly decomposes and reconstructs values.

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use num_bigint::BigInt;
    use num_traits::{Zero, One, ToPrimitive};

    println!("Testing Gadget Decomposition Round-Trip");
    println!("========================================\n");

    // Simple test parameters
    let moduli = vec![17u64, 19u64, 23u64];  // Small primes for easier debugging
    let n = 4;  // Small ring dimension
    let base_w = 4;  // Small base: B = 2^4 = 16

    // Compute Q = product of primes
    let q_prod: u64 = moduli.iter().product();
    println!("Moduli: {:?}", moduli);
    println!("Q = {} (product of primes)", q_prod);
    println!("Base: B = 2^{} = {}", base_w, 1u64 << base_w);

    // Create a simple test value: just coefficient 0 = 42, rest = 0
    let test_value = 42u64;
    let mut poly_flat = vec![0u64; n * moduli.len()];

    // Set first coefficient to test_value in all RNS components
    for (j, &q) in moduli.iter().enumerate() {
        poly_flat[j] = test_value % q;
    }

    println!("\nTest polynomial (flat RNS):");
    println!("  Coeff 0: {:?}", &poly_flat[0..moduli.len()]);

    // Now decompose and reconstruct
    println!("\nDecomposing...");

    // Manual decomposition to verify
    let base_big = BigInt::from(1u64 << base_w);
    let q_prod_big = BigInt::from(q_prod);

    // Reconstruct test_value using CRT
    let mut reconstructed = BigInt::zero();
    for (j, &q) in moduli.iter().enumerate() {
        let residue = poly_flat[j];
        let q_big = BigInt::from(q);
        let q_inv = &q_prod_big / &q_big;

        // Find modular inverse manually
        let q_inv_mod = mod_inverse(&q_inv, &q_big)?;
        let term = BigInt::from(residue) * q_inv * q_inv_mod;
        reconstructed += term;
    }
    reconstructed %= &q_prod_big;

    println!("  Reconstructed value: {}", reconstructed);
    println!("  Original value: {}", test_value);

    if reconstructed == BigInt::from(test_value) {
        println!("  ✅ CRT reconstruction correct!");
    } else {
        println!("  ❌ CRT reconstruction WRONG!");
        return Err("CRT reconstruction failed".to_string());
    }

    // Decompose into base-B digits
    let mut remaining = reconstructed.clone();
    let mut digits = Vec::new();

    let q_bits = q_prod_big.bits() as u32;
    let num_digits = ((q_bits + base_w - 1) / base_w) as usize;
    println!("\nDecomposing into {} digits (base B={})...", num_digits, 1u64 << base_w);

    for t in 0..num_digits {
        let digit = &remaining % &base_big;
        digits.push(digit.clone());
        println!("  Digit {}: {}", t, digit);
        remaining = (remaining - &digit) >> base_w;
    }

    // Reconstruct from digits
    let mut reconstructed2 = BigInt::zero();
    let mut power = BigInt::one();
    for (t, digit) in digits.iter().enumerate() {
        reconstructed2 += digit * &power;
        power *= &base_big;
    }

    println!("\nReconstructed from digits: {}", reconstructed2);
    println!("Original: {}", test_value);

    if reconstructed2 == BigInt::from(test_value) {
        println!("✅ Gadget decomposition round-trip SUCCESS!");
        Ok(())
    } else {
        println!("❌ Gadget decomposition round-trip FAILED!");
        Err(format!("Mismatch: got {}, expected {}", reconstructed2, test_value))
    }
}

#[cfg(all(feature = "v2", feature = "v2-gpu-metal", feature = "v3"))]
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
