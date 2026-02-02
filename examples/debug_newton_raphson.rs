//! Debug Newton-Raphson to see what's going wrong

#[cfg(feature = "v2")]
use ga_engine::clifford_fhe_v2::{
    backends::cpu_optimized::{
        ckks::{CkksContext, Plaintext},
        keys::KeyContext,
        multiplication::multiply_ciphertexts,
    },
    params::CliffordFHEParams,
};

#[cfg(feature = "v2")]
fn main() {
    println!("Debug Newton-Raphson Implementation");
    println!("====================================\n");

    // Setup
    let params = CliffordFHEParams::default();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Encrypt x = 2.0
    let num_slots = params.n / 2;
    let mut vec = vec![0.0; num_slots];
    vec[0] = 2.0;
    let pt_x = Plaintext::encode(&vec, params.scale, &params);
    let ct_x = ckks_ctx.encrypt(&pt_x, &pk);

    println!("Encrypted x = 2.0");
    println!("Goal: compute 1/x = 0.5\n");

    // Encrypt initial guess = 0.5
    vec[0] = 0.5;
    let pt_guess = Plaintext::encode(&vec, params.scale, &params);
    let mut ct_xn = ckks_ctx.encrypt(&pt_guess, &pk);

    println!("Initial guess x_0 = 0.5");

    // Decrypt and check
    let pt_dec = ckks_ctx.decrypt(&ct_xn, &sk);
    let val = pt_dec.decode(&params)[0];
    println!("Decrypted x_0 = {:.6}\n", val);

    // Newton-Raphson iterations
    for iter in 0..3 {
        println!("Iteration {}:", iter + 1);

        // Step 1: a · x_n
        let ct_axn = multiply_ciphertexts(&ct_x, &ct_xn, &evk, &key_ctx);
        let pt_axn = ckks_ctx.decrypt(&ct_axn, &sk);
        let axn_val = pt_axn.decode(&params)[0];
        println!("  a · x_n = {:.6}", axn_val);

        // Step 2: 2 - a·x_n
        vec[0] = 2.0;
        let pt_two = Plaintext::encode_at_level(&vec, ct_axn.scale, &params, ct_axn.level);

        // Create trivial ciphertext for 2
        use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
        let c0_two = pt_two.coeffs.clone();
        let c1_zero: Vec<RnsRepresentation> = (0..params.n).map(|_| {
            RnsRepresentation::new(vec![0u64; ct_axn.level + 1], params.moduli[..=ct_axn.level].to_vec())
        }).collect();
        let ct_two = ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext::new(
            c0_two, c1_zero, ct_axn.level, ct_axn.scale
        );

        let ct_two_minus_axn = ct_two.sub(&ct_axn);

        let pt_diff = ckks_ctx.decrypt(&ct_two_minus_axn, &sk);
        let diff_val = pt_diff.decode(&params)[0];
        println!("  2 - a·x_n = {:.6}", diff_val);

        // Step 3: x_{n+1} = x_n · (2 - a·x_n)
        ct_xn = multiply_ciphertexts(&ct_xn, &ct_two_minus_axn, &evk, &key_ctx);

        let pt_new = ckks_ctx.decrypt(&ct_xn, &sk);
        let new_val = pt_new.decode(&params)[0];
        println!("  x_{{n+1}} = {:.6}\n", new_val);
    }

    let final_val = ckks_ctx.decrypt(&ct_xn, &sk).decode(&params)[0];
    println!("Final result: {:.6}", final_val);
    println!("Expected: 0.5");
    println!("Error: {:.6}", (final_val - 0.5).abs());
}

#[cfg(not(feature = "v2"))]
fn main() {
    println!("This example requires the 'v2' feature.");
}
