//! Basic Clifford-FHE encryption/decryption example
//!
//! Demonstrates:
//! - Parameter setup
//! - Key generation
//! - Encrypting a 3D multivector (componentwise)
//! - Decryption and verification
//!
//! This is the simplest example showcasing Clifford FHE fundamentals.
//! V1-specific example - only compile when v1 is enabled and v2 is NOT enabled

#[cfg(all(feature = "v1", not(feature = "v2")))]
use ga_engine::clifford_fhe::{
    ckks_rns::{rns_decrypt, rns_encrypt, RnsPlaintext},
    keys_rns::rns_keygen,
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v1", not(feature = "v2")))]
fn main() {
    println!("=================================================================");
    println!("Clifford-FHE: Basic Encryption/Decryption Demo");
    println!("=================================================================\n");

    // 1. Setup parameters
    println!("1. Setting up parameters...");
    let params = CliffordFHEParams::new_rns_mult();
    println!("   ✓ Ring dimension (N): {}", params.n);
    println!("   ✓ Modulus chain: {} primes", params.moduli.len());
    println!("   ✓ Scaling factor: 2^{}", params.scale.log2() as u32);
    println!("   ✓ Security: ~128 bits (NIST Level 1)");
    println!("   ✓ First 3 primes: {}, {}, {}\n", params.moduli[0], params.moduli[1], params.moduli[2]);

    // 2. Generate keys
    println!("2. Generating keys...");
    let (pk, sk, _evk) = rns_keygen(&params);
    println!("   ✓ Public key, secret key, and evaluation key generated\n");

    // 3. Create a multivector (Cl(3,0): 8 components)
    println!("3. Creating multivector...");
    let mv = [1.5, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    println!("   Multivector: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
             mv[0], mv[1], mv[2], mv[3], mv[4], mv[5], mv[6], mv[7]);
    println!("   (1.5 + 2.0*e1 + 3.0*e2)\n");

    // 4. Encrypt multivector componentwise (8 ciphertexts)
    //    This is the standard encoding used in Clifford FHE
    println!("4. Encrypting multivector...");
    let mut ciphertexts = Vec::new();
    for i in 0..8 {
        // Each component gets its own ciphertext
        // Component value is placed at polynomial coefficient 0
        let mut coeffs = vec![0i64; params.n];
        coeffs[0] = (mv[i] * params.scale).round() as i64;
        let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
        ciphertexts.push(rns_encrypt(&pk, &pt, &params));
    }
    println!("   ✓ Multivector encrypted componentwise (8 ciphertexts)\n");

    // 5. Decrypt and verify (using single-prime decoding)
    println!("5. Decrypting and verifying...");
    let mut decrypted_mv = [0.0; 8];
    for i in 0..8 {
        let decrypted_pt = rns_decrypt(&sk, &ciphertexts[i], &params);

        // Use single-prime decoding (first prime) to avoid CRT overflow
        // This works well when message + noise << any single prime
        let val = decrypted_pt.coeffs.rns_coeffs[0][0];
        let q = params.moduli[0];
        let centered = if val > q / 2 { val - q } else { val };
        decrypted_mv[i] = (centered as f64) / ciphertexts[i].scale;
    }

    println!("   Decrypted:  [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
             decrypted_mv[0], decrypted_mv[1], decrypted_mv[2], decrypted_mv[3],
             decrypted_mv[4], decrypted_mv[5], decrypted_mv[6], decrypted_mv[7]);
    println!("   Original:   [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
             mv[0], mv[1], mv[2], mv[3], mv[4], mv[5], mv[6], mv[7]);

    // 6. Compute errors
    println!("\n6. Error analysis:");
    let mut max_error = 0.0;
    for i in 0..8 {
        let error = (decrypted_mv[i] - mv[i]).abs();
        let relative_error = if mv[i].abs() > 1e-10 {
            error / mv[i].abs()
        } else {
            error
        };
        if error > max_error {
            max_error = error;
        }
        println!("   Component {}: absolute error = {:.6}, relative error = {:.6}",
                 i, error, relative_error);
    }
    println!("   Maximum absolute error: {:.6}", max_error);

    // 7. Verification
    println!("\n7. Verification:");
    let threshold = 1e-3;
    if max_error < threshold {
        println!("   ✅ SUCCESS: Error = {:.6} < threshold {:.6}", max_error, threshold);
    } else {
        println!("   ⚠️  WARNING: Error = {:.6} exceeds threshold {:.6}", max_error, threshold);
    }

    println!("\n=================================================================");
    println!("Demo complete!");
    println!("=================================================================");
}

#[cfg(not(all(feature = "v1", not(feature = "v2"))))]
fn main() {
    println!("This example requires the 'v1' feature and must NOT have the 'v2' feature enabled.");
    println!("Run with: cargo run --example clifford_fhe_basic --features v1 --no-default-features");
}
