//! Minimal test to isolate rescaling bug
//!
//! Run with: cargo test --test test_v2_rescale_debug --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts;

#[test]
fn test_rescale_manually() {
    println!("\n========== MANUAL RESCALE TEST ==========");

    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Primes: {:?}", params.moduli);
    println!("  Scale = {:.2e}", params.scale);

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Encrypt 3.0 and 4.0
    let a_val = 3.0;
    let b_val = 4.0;
    println!("\nEncrypting {} and {}", a_val, b_val);

    let ct_a = encrypt_value(a_val, &ckks_ctx, &pk);
    let ct_b = encrypt_value(b_val, &ckks_ctx, &pk);

    println!("\nEncryption verification:");
    println!("  ct_a.c0[0]: {:?}", ct_a.c0[0].values.iter().map(|&v| v).collect::<Vec<_>>());
    let expected_a_scaled = (a_val * params.scale).round() as i64;
    println!("  expected a*scale = {}", expected_a_scaled);

    // Decrypt ct_a to verify
    let pt_a = ckks_ctx.decrypt(&ct_a, &sk);
    let val_a = pt_a.coeffs[0].values[0];
    let q_a = pt_a.coeffs[0].moduli[0];
    let centered_a = if val_a > q_a / 2 { val_a as i64 - q_a as i64 } else { val_a as i64 };
    let decrypted_a = (centered_a as f64) / ct_a.scale;
    println!("  Decrypted ct_a: {} (expected {})", decrypted_a, a_val);

    println!("  ct_b.c0[0]: {:?}", ct_b.c0[0].values.iter().map(|&v| v).collect::<Vec<_>>());
    let expected_b_scaled = (b_val * params.scale).round() as i64;
    println!("  expected b*scale = {}", expected_b_scaled);

    // Decrypt ct_b to verify
    let pt_b = ckks_ctx.decrypt(&ct_b, &sk);
    let val_b = pt_b.coeffs[0].values[0];
    let q_b = pt_b.coeffs[0].moduli[0];
    let centered_b = if val_b > q_b / 2 { val_b as i64 - q_b as i64 } else { val_b as i64 };
    let decrypted_b = (centered_b as f64) / ct_b.scale;
    println!("  Decrypted ct_b: {} (expected {})", decrypted_b, b_val);

    println!("\nBefore multiplication:");
    println!("  ct_a: level={}, scale={:.2e}", ct_a.level, ct_a.scale);
    println!("  ct_b: level={}, scale={:.2e}", ct_b.level, ct_b.scale);

    // Multiply (rescaling happens automatically inside multiply_ciphertexts)
    println!("\n--- Multiplying ciphertexts (with automatic rescale) ---");
    let ct_result = multiply_ciphertexts(&ct_a, &ct_b, &_evk, &key_ctx);

    println!("\nAfter multiplication+rescale:");
    println!("  ct_result: level={}, scale={:.2e}", ct_result.level, ct_result.scale);
    println!("  ct_result.c0[0]: {:?}", ct_result.c0[0].values.iter().map(|&v| v).collect::<Vec<_>>());

    // Check what we expect
    let q_last = params.moduli[ct_a.level]; // The prime we dropped
    let expected_scale = (ct_a.scale * ct_b.scale) / (q_last as f64);
    let expected_scaled = (a_val * b_val * expected_scale).round() as i64;

    println!("\nExpected:");
    println!("  result = {} * {} = {}", a_val, b_val, a_val * b_val);
    println!("  original scale² = {:.2e}", ct_a.scale * ct_b.scale);
    println!("  q_last (dropped) = {}", q_last);
    println!("  expected_scale after rescale = {:.2e}", expected_scale);
    println!("  expected_scaled value = {}", expected_scaled);

    let actual_mod_q0 = ct_result.c0[0].values[0];
    let q0 = ct_result.c0[0].moduli[0];
    let centered_actual = if actual_mod_q0 > q0 / 2 {
        actual_mod_q0 as i64 - q0 as i64
    } else {
        actual_mod_q0 as i64
    };

    println!("\nActual (after rescale):");
    println!("  ct_result.c0[0] mod q0 = {}", actual_mod_q0);
    println!("  centered = {}", centered_actual);
    println!("  ratio (actual/expected) = {:.2e}", centered_actual as f64 / expected_scaled as f64);
    println!("  ERROR: actual is {:.2e}x the expected value!", centered_actual as f64 / expected_scaled as f64);

    // Decrypt result
    let pt = ckks_ctx.decrypt(&ct_result, &sk);
    let val = pt.coeffs[0].values[0];
    let q = pt.coeffs[0].moduli[0];
    let centered = if val > q / 2 { val as i64 - q as i64 } else { val as i64 };

    let result = (centered as f64) / ct_result.scale;

    println!("\nDecrypted:");
    println!("  raw value (mod q): {}", val);
    println!("  centered value: {}", centered);
    println!("  scale: {:.2e}", ct_result.scale);
    println!("  decoded result: {}", result);
    println!("  expected: 12.0");
    println!("  error: {:.2e}", (result - 12.0).abs());

    // Now let's manually trace what rescale should do
    // We need to trace what SHOULD have happened during the multiplication
    println!("\n========== MANUAL RESCALE TRACE ==========");
    println!("We want to understand what happened during the rescaling step inside multiply_ciphertexts.");

    // The rescale happens after relinearization, working on the degree-1 ciphertext
    // For now, let's focus on understanding the rescale formula itself

    let q_last = params.moduli[ct_a.level];
    let new_moduli = &params.moduli[..ct_a.level];

    println!("Rescaling parameters:");
    println!("  q_last (dropped) = {}", q_last);
    println!("  new_moduli = {:?}", new_moduli);

    println!("\nWhat rescaling SHOULD do:");
    println!("  Input: coefficient with value v (mod q0*q1*q2)");
    println!("  Output: coefficient with value v/q2 (mod q0*q1)");
    println!("  Formula: For each remaining prime qi,");
    println!("           new_val_i = (old_val_i - v_centered) * q_last^(-1) mod qi");
    println!("           where v_centered = v mod q_last (centered in [-q_last/2, q_last/2])");

    // Let's trace a simple example with concrete numbers
    println!("\n--- Simple Manual Example ---");
    println!("Let's verify the rescaling formula with a simple example:");
    println!("Suppose we have v = 120, q_last = 10, qi = 7");
    println!("We want to compute v/q_last = 120/10 = 12");

    let v = 120i64;
    let q_last_ex = 10u64;
    let qi_ex = 7u64;

    let v_mod_qlast = (v % q_last_ex as i64) as u64;
    let v_centered = if v_mod_qlast > q_last_ex / 2 {
        v_mod_qlast as i64 - q_last_ex as i64
    } else {
        v_mod_qlast as i64
    };

    println!("  v = {}, v mod q_last = {}, v_centered = {}", v, v_mod_qlast, v_centered);

    let v_mod_qi = ((v % qi_ex as i64 + qi_ex as i64) % qi_ex as i64) as u64;
    println!("  v mod qi = {}", v_mod_qi);

    let q_last_inv = mod_pow(q_last_ex % qi_ex, qi_ex - 2, qi_ex);
    println!("  q_last^(-1) mod qi = {}", q_last_inv);

    let diff = if v_centered >= 0 {
        let vc = (v_centered as u64) % qi_ex;
        if v_mod_qi >= vc {
            v_mod_qi - vc
        } else {
            qi_ex - (vc - v_mod_qi)
        }
    } else {
        let vc = ((-v_centered) as u64) % qi_ex;
        (v_mod_qi + vc) % qi_ex
    };

    let result_mod_qi = ((diff as u128) * (q_last_inv as u128)) % (qi_ex as u128);
    let expected_mod_qi = 12u64 % qi_ex;

    println!("  diff = (v mod qi - v_centered) mod qi = {}", diff);
    println!("  result = diff * q_last_inv mod qi = {}", result_mod_qi);
    println!("  expected = 12 mod {} = {}", qi_ex, expected_mod_qi);

    if result_mod_qi as u64 == expected_mod_qi {
        println!("  ✓ FORMULA WORKS!");
    } else {
        println!("  ✗ FORMULA BROKEN! Got {} but expected {}", result_mod_qi, expected_mod_qi);
    }
}

fn encrypt_value(
    value: f64,
    ckks_ctx: &CkksContext,
    pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
) -> ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext {
    let params = &ckks_ctx.params;
    let level = params.max_level();
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let n = params.n;

    let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); n];

    let scaled_val = (value * params.scale).round() as i64;
    let values: Vec<u64> = moduli.iter().map(|&q| {
        let q_i64 = q as i64;
        let normalized = ((scaled_val % q_i64) + q_i64) % q_i64;
        normalized as u64
    }).collect();

    coeffs[0] = RnsRepresentation::new(values, moduli.clone());

    let pt = Plaintext::new(coeffs, params.scale, level);
    ckks_ctx.encrypt(&pt, pk)
}

fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base = base % m;

    while exp > 0 {
        if exp % 2 == 1 {
            result = ((result as u128 * base as u128) % m as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % m as u128) as u64;
    }

    result
}
