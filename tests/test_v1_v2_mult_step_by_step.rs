//! Step-by-step comparison of V1 and V2 multiplication
//! Run with: cargo test --test test_v1_v2_mult_step_by_step --features v1,v2 -- --nocapture

use ga_engine::clifford_fhe_v1::params::CliffordFHEParams as V1Params;
use ga_engine::clifford_fhe_v1::keys_rns::rns_keygen as v1_keygen;
use ga_engine::clifford_fhe_v1::ckks_rns::{rns_encrypt as v1_encrypt, RnsPlaintext as V1Plaintext};

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams as V2Params;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext as V2KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext as V2CkksContext;

#[test]
fn test_v1_v2_step_by_step() {
    println!("\n=== STEP-BY-STEP COMPARISON: V1 vs V2 ===\n");

    // Use V1 parameters (5 primes) for V1
    let v1_params = V1Params::new_rns_mult_depth2_safe();
    let (v1_pk, v1_sk, v1_evk) = v1_keygen(&v1_params);

    // Use V2 parameters (3 primes) for V2
    let v2_params = V2Params::new_test_ntt_1024();
    let v2_key_ctx = V2KeyContext::new(v2_params.clone());
    let (v2_pk, v2_sk, v2_evk) = v2_key_ctx.keygen();
    let v2_ckks_ctx = V2CkksContext::new(v2_params.clone());

    println!("V1: N={}, {} primes", v1_params.n, v1_params.moduli.len());
    println!("V2: N={}, {} primes", v2_params.n, v2_params.moduli.len());

    // Encrypt same values with same scale
    let scale = v1_params.scale;

    // V1 encryption
    let mut v1_coeffs_a = vec![0i64; v1_params.n];
    v1_coeffs_a[0] = (2.0 * scale).round() as i64;
    let v1_pt_a = V1Plaintext::from_coeffs(v1_coeffs_a, scale, &v1_params.moduli, 0);
    let v1_ct_a = v1_encrypt(&v1_pk, &v1_pt_a, &v1_params);

    let mut v1_coeffs_b = vec![0i64; v1_params.n];
    v1_coeffs_b[0] = (3.0 * scale).round() as i64;
    let v1_pt_b = V1Plaintext::from_coeffs(v1_coeffs_b, scale, &v1_params.moduli, 0);
    let v1_ct_b = v1_encrypt(&v1_pk, &v1_pt_b, &v1_params);

    // V2 encryption
    let v2_pt_a = v2_ckks_ctx.encode(&[2.0]);
    let v2_ct_a = v2_ckks_ctx.encrypt(&v2_pt_a, &v2_pk);
    let v2_pt_b = v2_ckks_ctx.encode(&[3.0]);
    let v2_ct_b = v2_ckks_ctx.encrypt(&v2_pt_b, &v2_pk);

    println!("\n=== STEP 1: TENSOR PRODUCT ===");
    println!("Computing (c0_a, c1_a) ⊗ (c0_b, c1_b) = (d0, d1, d2)");

    // V1 tensor product
    use ga_engine::clifford_fhe_v1::rns::rns_multiply as v1_poly_mult;
    use ga_engine::clifford_fhe_v1::ckks_rns::polynomial_multiply_ntt as v1_ntt_mult;

    let v1_d0 = v1_poly_mult(&v1_ct_a.c0, &v1_ct_b.c0, &v1_params.moduli, v1_ntt_mult);
    let v1_c0_c1b = v1_poly_mult(&v1_ct_a.c0, &v1_ct_b.c1, &v1_params.moduli, v1_ntt_mult);
    let v1_c1a_c0b = v1_poly_mult(&v1_ct_a.c1, &v1_ct_b.c0, &v1_params.moduli, v1_ntt_mult);
    let v1_d2 = v1_poly_mult(&v1_ct_a.c1, &v1_ct_b.c1, &v1_params.moduli, v1_ntt_mult);

    use ga_engine::clifford_fhe_v1::rns::rns_add as v1_add;
    let v1_d1 = v1_add(&v1_c0_c1b, &v1_c1a_c0b, &v1_params.moduli);

    println!("\nV1 tensor product (first coefficient, first 2 residues):");
    println!("  d0[0]: {:?}", &v1_d0.rns_coeffs[0][..2]);
    println!("  d1[0]: {:?}", &v1_d1.rns_coeffs[0][..2]);
    println!("  d2[0]: {:?}", &v1_d2.rns_coeffs[0][..2]);

    // V2 tensor product
    // We need to access the internals, so let's compute it manually
    let v2_moduli: Vec<u64> = v2_params.moduli[..=v2_params.max_level()].to_vec();

    let v2_d0 = mult_polys_v2(&v2_ct_a.c0, &v2_ct_b.c0, &v2_key_ctx, &v2_moduli);
    let v2_c0_c1b = mult_polys_v2(&v2_ct_a.c0, &v2_ct_b.c1, &v2_key_ctx, &v2_moduli);
    let v2_c1a_c0b = mult_polys_v2(&v2_ct_a.c1, &v2_ct_b.c0, &v2_key_ctx, &v2_moduli);
    let v2_d2 = mult_polys_v2(&v2_ct_a.c1, &v2_ct_b.c1, &v2_key_ctx, &v2_moduli);

    let v2_d1: Vec<_> = v2_c0_c1b.iter().zip(&v2_c1a_c0b).map(|(a, b)| a.add(b)).collect();

    println!("\nV2 tensor product (first coefficient, all residues):");
    println!("  d0[0]: {:?}", &v2_d0[0].values);
    println!("  d1[0]: {:?}", &v2_d1[0].values);
    println!("  d2[0]: {:?}", &v2_d2[0].values);

    println!("\n=== STEP 2: GADGET DECOMPOSITION ===");
    println!("Decomposing d2 into base-2^20 digits");

    // V1 decomposition
    use ga_engine::clifford_fhe_v1::rns::decompose_base_pow2 as v1_decompose;
    let v1_digits = v1_decompose(&v1_d2, &v1_params.moduli, v1_evk.base_w);

    println!("\nV1 gadget decomposition:");
    println!("  Number of digits: {}", v1_digits.len());
    println!("  Digit 0, coeff[0], first 2 residues: {:?}", &v1_digits[0].rns_coeffs[0][..2]);
    if v1_digits.len() > 1 {
        println!("  Digit 1, coeff[0], first 2 residues: {:?}", &v1_digits[1].rns_coeffs[0][..2]);
    }

    // V2 decomposition - we need to call the internal function
    // Since it's private, let's check what multiply_ciphertexts produces
    println!("\n=== RUNNING FULL V2 MULTIPLICATION (to see final result) ===");

    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts as v2_mult;
    let v2_ct_prod = v2_mult(&v2_ct_a, &v2_ct_b, &v2_evk, &v2_key_ctx);

    let v2_dec = v2_ckks_ctx.decrypt(&v2_ct_prod, &v2_sk);
    let v2_result = decode_v2(&v2_dec, v2_ct_prod.scale);

    println!("V2 final result: {:.10} (expected 6.0)", v2_result);
    println!("V2 error: {:.2e}", (v2_result - 6.0).abs());

    println!("\n=== RUNNING FULL V1 MULTIPLICATION ===");
    use ga_engine::clifford_fhe_v1::ckks_rns::{rns_multiply_ciphertexts as v1_mult, rns_decrypt as v1_decrypt};
    let v1_ct_prod = v1_mult(&v1_ct_a, &v1_ct_b, &v1_evk, &v1_params);
    let v1_dec = v1_decrypt(&v1_sk, &v1_ct_prod, &v1_params);
    let v1_result = decode_v1(&v1_dec, v1_ct_prod.scale, &v1_params.moduli);

    println!("V1 final result: {:.10} (expected 6.0)", v1_result);
    println!("V1 error: {:.2e}", (v1_result - 6.0).abs());

    println!("\n=== KEY OBSERVATION ===");
    println!("V1 uses {} primes, V2 uses {} primes", v1_params.moduli.len(), v2_params.moduli.len());
    println!("Both should produce ~6.0, but V2 produces {:.2e}", v2_result);
    println!("\nThe bug must be in:");
    println!("1. Gadget decomposition (CRT-consistency)");
    println!("2. Relinearization (EVK application)");
    println!("3. Rescaling (division by last prime)");
}

fn mult_polys_v2(
    a: &[ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation],
    b: &[ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation],
    key_ctx: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext,
    moduli: &[u64],
) -> Vec<ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation> {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

    let n = a.len();
    let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = key_ctx.ntt_contexts.iter().find(|ctx| ctx.q == q).unwrap();
        let a_mod_q: Vec<u64> = a.iter().map(|rns| rns.values[prime_idx]).collect();
        let b_mod_q: Vec<u64> = b.iter().map(|rns| rns.values[prime_idx]).collect();
        let product_mod_q = ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q);

        for i in 0..n {
            result[i].values[prime_idx] = product_mod_q[i];
        }
    }

    result
}

fn decode_v1(pt: &ga_engine::clifford_fhe_v1::ckks_rns::RnsPlaintext, scale: f64, primes: &[i64]) -> f64 {
    let val = pt.coeffs.rns_coeffs[0][0];
    let q = primes[0];
    let centered = if val > q / 2 { val - q } else { val };
    (centered as f64) / scale
}

fn decode_v2(pt: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext, scale: f64) -> f64 {
    let val = pt.coeffs[0].values[0];
    let q = pt.coeffs[0].moduli[0];
    let centered = if val > q / 2 { val as i64 - q as i64 } else { val as i64 };
    (centered as f64) / scale
}
