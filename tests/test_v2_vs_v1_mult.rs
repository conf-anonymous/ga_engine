//! Compare V1 vs V2 multiplication step by step
//! Run with: cargo test --test test_v2_vs_v1_mult --features v1,v2 -- --nocapture

use ga_engine::clifford_fhe_v1::params::CliffordFHEParams as V1Params;
use ga_engine::clifford_fhe_v1::keys_rns::rns_keygen as v1_keygen;
use ga_engine::clifford_fhe_v1::ckks_rns::{rns_encrypt as v1_encrypt, rns_decrypt as v1_decrypt, RnsPlaintext as V1Plaintext};

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams as V2Params;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext as V2KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext as V2CkksContext;

#[test]
fn test_v1_vs_v2_multiplication() {
    println!("\n=== V1 vs V2 MULTIPLICATION COMPARISON ===\n");

    // V1 setup
    println!("Setting up V1...");
    let v1_params = V1Params::new_rns_mult_depth2_safe();
    let (v1_pk, v1_sk, v1_evk) = v1_keygen(&v1_params);

    // V2 setup
    println!("Setting up V2...");
    let v2_params = V2Params::new_test_ntt_1024();
    let v2_key_ctx = V2KeyContext::new(v2_params.clone());
    let (v2_pk, v2_sk, v2_evk) = v2_key_ctx.keygen();
    let v2_ckks_ctx = V2CkksContext::new(v2_params.clone());

    println!("\n=== PARAMETERS ===");
    println!("V1: N={}, {} primes, scale=2^{}", v1_params.n, v1_params.moduli.len(), (v1_params.scale as f64).log2() as u32);
    println!("V2: N={}, {} primes, scale=2^{}", v2_params.n, v2_params.moduli.len(), (v2_params.scale as f64).log2() as u32);

    // Encrypt 2 and 3
    println!("\n=== ENCRYPTION ===");

    // V1
    let mut v1_coeffs_a = vec![0i64; v1_params.n];
    v1_coeffs_a[0] = (2.0 * v1_params.scale).round() as i64;
    let v1_pt_a = V1Plaintext::from_coeffs(v1_coeffs_a, v1_params.scale, &v1_params.moduli, 0);
    let v1_ct_a = v1_encrypt(&v1_pk, &v1_pt_a, &v1_params);

    let mut v1_coeffs_b = vec![0i64; v1_params.n];
    v1_coeffs_b[0] = (3.0 * v1_params.scale).round() as i64;
    let v1_pt_b = V1Plaintext::from_coeffs(v1_coeffs_b, v1_params.scale, &v1_params.moduli, 0);
    let v1_ct_b = v1_encrypt(&v1_pk, &v1_pt_b, &v1_params);

    println!("V1 encrypted 2.0 and 3.0");

    // V2
    let v2_pt_a = v2_ckks_ctx.encode(&[2.0]);
    let v2_ct_a = v2_ckks_ctx.encrypt(&v2_pt_a, &v2_pk);
    let v2_pt_b = v2_ckks_ctx.encode(&[3.0]);
    let v2_ct_b = v2_ckks_ctx.encrypt(&v2_pt_b, &v2_pk);

    println!("V2 encrypted 2.0 and 3.0");

    // Decrypt to verify encryption
    let v1_dec_a = v1_decrypt(&v1_sk, &v1_ct_a, &v1_params);
    let v1_val_a = decode_v1(&v1_dec_a, v1_ct_a.scale, &v1_params.moduli);
    println!("\nV1 decrypt(ct_a) = {:.10} (expected 2.0)", v1_val_a);

    let v2_dec_a = v2_ckks_ctx.decrypt(&v2_ct_a, &v2_sk);
    let v2_val_a = decode_v2(&v2_dec_a, v2_ct_a.scale);
    println!("V2 decrypt(ct_a) = {:.10} (expected 2.0)", v2_val_a);

    // Now multiply
    println!("\n=== MULTIPLICATION ===");

    use ga_engine::clifford_fhe_v1::ckks_rns::rns_multiply_ciphertexts as v1_mult;
    let v1_ct_prod = v1_mult(&v1_ct_a, &v1_ct_b, &v1_evk, &v1_params);
    let v1_dec_prod = v1_decrypt(&v1_sk, &v1_ct_prod, &v1_params);
    let v1_result = decode_v1(&v1_dec_prod, v1_ct_prod.scale, &v1_params.moduli);

    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts as v2_mult;
    let v2_ct_prod = v2_mult(&v2_ct_a, &v2_ct_b, &v2_evk, &v2_key_ctx);
    let v2_dec_prod = v2_ckks_ctx.decrypt(&v2_ct_prod, &v2_sk);
    let v2_result = decode_v2(&v2_dec_prod, v2_ct_prod.scale);

    println!("\n=== RESULTS ===");
    println!("V1: {:.10} (expected 6.0)", v1_result);
    println!("V2: {:.10} (expected 6.0)", v2_result);
    println!("\nV1 error: {:.2e}", (v1_result - 6.0).abs());
    println!("V2 error: {:.2e}", (v2_result - 6.0).abs());

    if (v1_result - 6.0).abs() < 0.1 {
        println!("\n✓ V1 multiplication works!");
    } else {
        println!("\n✗ V1 multiplication FAILED!");
    }

    if (v2_result - 6.0).abs() < 0.1 {
        println!("\n✓ V2 multiplication works!");
    } else {
        println!("\n✗ V2 multiplication FAILED!");
    }
}

fn center(val: u64, q: u64) -> i64 {
    if val > q / 2 {
        val as i64 - q as i64
    } else {
        val as i64
    }
}

fn decode_v1(pt: &V1Plaintext, scale: f64, primes: &[i64]) -> f64 {
    let val = pt.coeffs.rns_coeffs[0][0];
    let q = primes[0] as u64;
    let centered = center(val as u64, q);
    (centered as f64) / scale
}

fn decode_v2(pt: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext, scale: f64) -> f64 {
    let val = pt.coeffs[0].values[0];
    let q = pt.coeffs[0].moduli[0];
    let centered = center(val, q);
    (centered as f64) / scale
}
