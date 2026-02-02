//! Test CPU ciphertext multiplication to verify it works

use ga_engine::clifford_fhe_v2::{
    backends::cpu_optimized::{
        ckks::CkksContext,
        keys::KeyContext,
        multiplication::multiply_ciphertexts,
    },
    params::CliffordFHEParams,
};

fn main() {
    println!("Testing CPU ciphertext multiplication");
    println!("=====================================\n");

    // Setup
    let params = CliffordFHEParams::new_test_ntt_4096();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ctx = CkksContext::new(params.clone());

    // Test: 6.0 × 7.0 = 42.0
    let a = 6.0;
    let b = 7.0;
    let expected = a * b;

    println!("Computing {} × {} = {}", a, b, expected);

    // Encrypt
    let pt_a = ctx.encode(&[a]);
    let pt_b = ctx.encode(&[b]);
    let ct_a = ctx.encrypt(&pt_a, &pk);
    let ct_b = ctx.encrypt(&pt_b, &pk);

    println!("Initial level: {}", ct_a.level);
    println!("Initial scale: {}", ct_a.scale);

    // Multiply
    let ct_result = multiply_ciphertexts(&ct_a, &ct_b, &evk, &key_ctx);

    println!("After multiply level: {}", ct_result.level);
    println!("After multiply scale: {}", ct_result.scale);

    // Decrypt and decode
    let pt_result = ctx.decrypt(&ct_result, &sk);
    let result = ctx.decode(&pt_result);

    println!("\nExpected: {}", expected);
    println!("Got:      {}", result[0]);

    let error = (result[0] - expected).abs();
    let rel_error = error / expected;
    println!("Absolute error: {:.2e}", error);
    println!("Relative error: {:.2e}", rel_error);

    if rel_error < 1e-6 {
        println!("\n✅ PASS: CPU multiplication works correctly!");
    } else {
        println!("\n❌ FAIL: CPU multiplication has errors!");
        panic!("Relative error too large: {}", rel_error);
    }
}
