//! Simple test of gadget decomposition
//! Run with: cargo test --test test_v2_gadget_decomp_simple --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;

#[test]
fn test_gadget_decomposition_simple() {
    println!("\n=== SIMPLE GADGET DECOMPOSITION TEST ===\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Encrypt 2.0
    let pt = ckks_ctx.encode(&[2.0]);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    println!("Encrypted 2.0");
    println!("ct.c0[0]: {:?}", ct.c0[0].values);
    println!("ct.c1[0]: {:?}", ct.c1[0].values);

    // Decrypt to verify
    let dec = ckks_ctx.decrypt(&ct, &sk);
    let val = decode(&dec, ct.scale);
    println!("\nDecrypt(ct) = {:.10} (expected 2.0)", val);

    // Now test multiplication
    let ct2 = ckks_ctx.encrypt(&ckks_ctx.encode(&[3.0]), &pk);

    println!("\n=== MULTIPLICATION ===");
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts;
    let ct_prod = multiply_ciphertexts(&ct, &ct2, &evk, &key_ctx);

    let dec_prod = ckks_ctx.decrypt(&ct_prod, &sk);
    let result = decode(&dec_prod, ct_prod.scale);

    println!("\nResult: {:.10} (expected 6.0)", result);
    println!("Error: {:.2e}", (result - 6.0).abs());

    if (result - 6.0).abs() < 0.1 {
        println!("\n✓ SUCCESS!");
    } else {
        println!("\n✗ FAILED!");
    }
}

fn decode(pt: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext, scale: f64) -> f64 {
    let val = pt.coeffs[0].values[0];
    let q = pt.coeffs[0].moduli[0];
    let centered = if val > q / 2 { val as i64 - q as i64 } else { val as i64 };
    (centered as f64) / scale
}
