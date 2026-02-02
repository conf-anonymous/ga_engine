//! Unit tests to isolate multiply_plain issues
//! Each test checks ONE specific hypothesis about where the 1000× gain occurs

use super::*;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

/// H1.1: Test encode → decode identity for single value
#[test]
fn test_encode_decode_single() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let input = vec![2.0];

    let pt = Plaintext::encode(&input, params.scale, &params);
    let output = pt.decode(&params);

    let error = (output[0] - input[0]).abs();
    println!("H1.1: Input={}, Output={}, Error={}", input[0], output[0], error);

    assert!(error < 0.01, "Encode/decode single value failed: error = {}", error);
}

/// H1.2: Test encode → decode identity for large value
#[test]
fn test_encode_decode_large() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let input = vec![1000.0];

    let pt = Plaintext::encode(&input, params.scale, &params);
    let output = pt.decode(&params);

    let error = (output[0] - input[0]).abs();
    let rel_error = error / input[0];
    println!("H1.2: Input={}, Output={}, RelError={:.6}", input[0], output[0], rel_error);

    assert!(rel_error < 0.001, "Encode/decode large value failed: rel_error = {}", rel_error);
}

/// H1.3: Test encode → decode identity for all slots
#[test]
fn test_encode_decode_all_slots() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let num_slots = params.n / 2;
    let input = vec![2.0; num_slots];

    let pt = Plaintext::encode(&input, params.scale, &params);
    let output = pt.decode(&params);

    let mut max_error = 0.0;
    for i in 0..num_slots {
        let error = (output[i] - input[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }

    println!("H1.3: Max error across {} slots: {}", num_slots, max_error);
    assert!(max_error < 0.01, "Encode/decode all slots failed: max_error = {}", max_error);
}

/// H2.1: Test encrypt → decrypt → decode identity
#[test]
fn test_encrypt_decrypt_identity() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    let input = vec![2.0; params.n / 2];
    let pt = Plaintext::encode(&input, params.scale, &params);

    let ct = ckks_ctx.encrypt(&pt, &pk);
    let pt_dec = ckks_ctx.decrypt(&ct, &sk);
    let output = pt_dec.decode(&params);

    let mut max_error = 0.0;
    for i in 0..(params.n / 2) {
        let error = (output[i] - input[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }

    println!("H2.1: Encrypt→Decrypt max error: {}", max_error);
    println!("H2.1: Input scale: {}, Output scale: {}", pt.scale, pt_dec.scale);

    assert!(max_error < 10.0, "Encrypt/decrypt failed: max_error = {} (noise too high)", max_error);
}

/// H7.1: Plaintext-only multiply (no encryption) to measure κ_plaintext
#[test]
fn test_plaintext_multiply_no_encryption() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let num_slots = params.n / 2;

    // Encode two plaintexts
    let x_vals = vec![2.0; num_slots];
    let c_vals = vec![3.0; num_slots];

    let pt_x = Plaintext::encode(&x_vals, params.scale, &params);
    let pt_c = Plaintext::encode(&c_vals, params.scale, &params);

    // Manually multiply coefficients in RNS
    let mut result_coeffs = Vec::new();
    for i in 0..params.n {
        let mut values = Vec::new();
        for j in 0..pt_x.coeffs[i].values.len() {
            let q_j = pt_x.coeffs[i].moduli[j];
            let a = pt_x.coeffs[i].values[j];
            let b = pt_c.coeffs[i].values[j];

            // Multiply mod q_j
            let prod = ((a as u128) * (b as u128)) % (q_j as u128);
            values.push(prod as u64);
        }
        result_coeffs.push(RnsRepresentation::new(values, pt_x.coeffs[i].moduli.clone()));
    }

    // Create result plaintext with scale = Δ² (no rescale yet)
    let result_scale = params.scale * params.scale;
    let pt_result = Plaintext::new(result_coeffs, result_scale, pt_x.level);

    // Decode (this will divide by scale, so we get values / Δ)
    let output = pt_result.decode(&params);

    // Expected: 2 × 3 = 6, but with scale Δ² and decode dividing by Δ, we get 6 × Δ
    // Actually, decode divides by pt_result.scale = Δ², so we should get 6.0 × (embedding factor)

    let expected = 6.0;
    let mean_output: f64 = output.iter().take(num_slots).sum::<f64>() / (num_slots as f64);
    let kappa_plaintext = mean_output / expected;

    println!("H7.1: Plaintext-only multiply");
    println!("  Expected (2×3): {}", expected);
    println!("  Mean output: {}", mean_output);
    println!("  κ_plaintext = {:.2}", kappa_plaintext);
    println!("  Output[0..4]: [{:.2}, {:.2}, {:.2}, {:.2}]",
             output[0], output[1], output[2], output[3]);

    // This measures the gain from encode→multiply→decode path WITHOUT encryption or rescale
    // If κ_plaintext ≠ 1, then the embedding itself has a gain factor
}

/// H5.1: Test drop-limb rescale on known value
#[test]
fn test_rescale_drop_limb() {
    let params = CliffordFHEParams::new_test_ntt_1024();

    // Create a test value: coefficient = 1000 × Δ at level 2 (3 primes)
    let test_coeff = (1000.0 * params.scale) as i64;
    let level = 2;
    let moduli = params.moduli[..=level].to_vec();

    // Convert to RNS
    let mut rns_values = Vec::new();
    for &q in &moduli {
        let val_mod_q = if test_coeff >= 0 {
            (test_coeff as u64) % q
        } else {
            let abs_val = (-test_coeff) as u64;
            let rem = abs_val % q;
            if rem == 0 { 0 } else { q - rem }
        };
        rns_values.push(val_mod_q);
    }

    println!("H5.1: Original value: {} × Δ", 1000.0);
    println!("  RNS representation: {:?}", rns_values);

    // Drop top limb (simulate rescale)
    let new_rns_values = rns_values[..level].to_vec();
    let new_moduli = moduli[..level].to_vec();

    println!("  After drop limb: {:?}", new_rns_values);

    // Reconstruct (approximate, using first prime only)
    let q0 = new_moduli[0];
    let half_q0 = q0 / 2;
    let reconstructed_mod_q0 = new_rns_values[0];

    let reconstructed = if reconstructed_mod_q0 > half_q0 {
        (reconstructed_mod_q0 as i64) - (q0 as i64)
    } else {
        reconstructed_mod_q0 as i64
    };

    let q_top = params.moduli[level];
    let expected_after_rescale = (test_coeff as f64) / (q_top as f64);

    println!("  Reconstructed (approx): {}", reconstructed);
    println!("  Expected (1000×Δ / q_top): {:.2}", expected_after_rescale);
    println!("  Ratio: {:.6}", (reconstructed as f64) / expected_after_rescale);

    // This shows if drop-limb approximates division reasonably
}

/// H6.1: Measure encode normalization factor
#[test]
fn test_encode_normalization_factor() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let num_slots = params.n / 2;

    // Encode all ones
    let ones = vec![1.0; num_slots];
    let pt = Plaintext::encode(&ones, params.scale, &params);

    // Sum of polynomial coefficients (using first prime only for simplicity)
    let q0 = params.moduli[0];
    let half_q0 = (q0 / 2) as i128;

    let mut coeff_sum: i128 = 0;
    for i in 0..params.n {
        let val = pt.coeffs[i].values[0] as i128;
        let signed = if val > half_q0 { val - (q0 as i128) } else { val };
        coeff_sum += signed;
    }

    // Expected: if encoding is "clean", sum should be related to n and Δ
    let sigma = (coeff_sum as f64) / (params.scale * params.n as f64);

    println!("H6.1: Encode normalization");
    println!("  Input: {} slots of 1.0", num_slots);
    println!("  Sum of coeffs: {}", coeff_sum);
    println!("  σ = sum / (Δ × n) = {:.6}", sigma);
    println!("  Expected σ for unitary: ≈ 0.5 or ≈ 0.707");

    // σ tells us the forward embedding normalization constant
}

/// H4.1: Track scales through multiply_plain
#[test]
fn test_scale_tracking() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    let values = vec![2.0; params.n / 2];
    let pt1 = Plaintext::encode(&values, params.scale, &params);
    let ct = ckks_ctx.encrypt(&pt1, &pk);

    let multiplier = vec![3.0; params.n / 2];
    let pt2 = Plaintext::encode(&multiplier, params.scale, &params);

    println!("H4.1: Scale tracking");
    println!("  Before multiply:");
    println!("    CT scale: {}", ct.scale);
    println!("    PT scale: {}", pt2.scale);
    println!("    CT level: {}", ct.level);

    let ct_result = ct.multiply_plain(&pt2, &ckks_ctx);

    println!("  After multiply_plain:");
    println!("    Result scale: {}", ct_result.scale);
    println!("    Result level: {}", ct_result.level);
    println!("    Expected scale: {} / {} = {:.2}",
             ct.scale * pt2.scale,
             params.moduli[ct.level],
             (ct.scale * pt2.scale) / (params.moduli[ct.level] as f64));

    let expected_scale = (ct.scale * pt2.scale) / (params.moduli[ct.level] as f64);
    let scale_error = (ct_result.scale - expected_scale).abs() / expected_scale;

    println!("    Scale tracking error: {:.2}%", scale_error * 100.0);

    assert!(scale_error < 0.01, "Scale tracking failed: error = {:.2}%", scale_error * 100.0);
}
