//! Test Galois automorphism alone (without key-switching)

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;
use ga_engine::clifford_fhe_v3::bootstrapping::keys::{apply_galois_automorphism, galois_element_for_rotation};

fn main() {
    println!("=== Galois Automorphism Test ===\n");

    let params = CliffordFHEParams::new_128bit();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Create message
    let mut message = vec![0.0; params.n / 2];
    message[0] = 100.0;
    message[1] = 200.0;
    message[2] = 300.0;

    println!("Original message: [{}, {}, {}]", message[0], message[1], message[2]);

    // Encrypt
    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    // Decrypt to verify
    let pt_dec = ckks_ctx.decrypt(&ct, &sk);
    let dec = ckks_ctx.decode(&pt_dec);
    println!("Decrypted: [{:.2}, {:.2}, {:.2}]", dec[0], dec[1], dec[2]);

    // Apply Galois automorphism (WITHOUT key-switching)
    // This will make it decrypt under s(X^5) instead of s(X)
    let g = galois_element_for_rotation(1, params.n);
    println!("\nApplying Galois automorphism with g={}...", g);

    let c0_auto = apply_galois_automorphism(&ct.c0, g, params.n);
    let c1_auto = apply_galois_automorphism(&ct.c1, g, params.n);

    let ct_auto = Ciphertext {
        c0: c0_auto,
        c1: c1_auto,
        level: ct.level,
        scale: ct.scale,
        n: ct.n,
    };

    // Decrypt with original secret key
    // This will be WRONG because ciphertext now encrypts under s(X^5)
    println!("Decrypting with s(X) (will be wrong)...");
    let pt_wrong = ckks_ctx.decrypt(&ct_auto, &sk);
    let dec_wrong = ckks_ctx.decode(&pt_wrong);
    println!("Wrong decryption: [{:.2}, {:.2}, {:.2}]", dec_wrong[0], dec_wrong[1], dec_wrong[2]);

    // Apply automorphism to secret key
    println!("\nApplying Galois automorphism to secret key...");
    let sk_auto_coeffs = apply_galois_automorphism(&sk.coeffs, g, params.n);

    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey;
    let sk_auto = SecretKey::new(sk_auto_coeffs, sk.level);

    // Decrypt with automorphic secret key
    println!("Decrypting with s(X^5)...");
    let pt_auto = ckks_ctx.decrypt(&ct_auto, &sk_auto);
    let dec_auto = ckks_ctx.decode(&pt_auto);
    println!("Decryption with s(X^5): [{:.2}, {:.2}, {:.2}]", dec_auto[0], dec_auto[1], dec_auto[2]);

    println!("\nExpected: Rotated message (some permutation of original)");

    // Check if at least one value is non-zero and large
    let has_large_value = dec_auto.iter().take(5).any(|&v| v.abs() > 50.0);
    if has_large_value {
        println!("✅ Galois automorphism produces non-trivial output");
    } else {
        println!("❌ Galois automorphism seems broken");
    }
}
