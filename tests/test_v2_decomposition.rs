//! Test CRT-consistent gadget decomposition
//! Run with: cargo test --test test_v2_decomposition --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

#[test]
fn test_crt_decomposition() {
    println!("\n=== TEST CRT-CONSISTENT DECOMPOSITION ===\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let moduli: Vec<u64> = params.moduli[..=params.max_level()].to_vec();
    let base_w = 20u32;
    let base = 1i128 << base_w;

    println!("Moduli: {:?}", moduli);
    println!("Base: 2^{} = {}", base_w, base);

    // Create a test value: 1,000,000
    let test_val = 1_000_000i128;
    println!("\nTest value: {}", test_val);

    // Convert to RNS
    let rns_val = RnsRepresentation::from_u64(test_val as u64, &moduli);
    println!("RNS residues: {:?}", rns_val.values);

    // Manual CRT reconstruction
    let q_prod: i128 = moduli.iter().map(|&q| q as i128).product();
    println!("\nQ = product of moduli = {}", q_prod);

    let residues: Vec<i128> = rns_val.values.iter().map(|&v| v as i128).collect();
    let reconstructed = crt_reconstruct(&residues, &moduli);
    println!("CRT reconstructed: {}", reconstructed);
    println!("Match: {}", reconstructed == test_val);

    // Decompose using balanced digits
    println!("\n=== BALANCED DECOMPOSITION ===");
    let mut remainder = test_val;
    let half_base = base / 2;

    for t in 0..3 {
        let mut dt = remainder % base;
        if dt >= half_base {
            dt -= base;
        } else if dt < -half_base {
            dt += base;
        }

        println!("Digit {}: dt = {}", t, dt);

        // Convert to RNS
        for (j, &q) in moduli.iter().enumerate() {
            let q_i128 = q as i128;
            let dt_mod_q = ((dt % q_i128) + q_i128) % q_i128;
            println!("  mod q[{}] = {}", j, dt_mod_q);
        }

        remainder = (remainder - dt) / base;
        println!("  Remainder for next: {}", remainder);
    }

    println!("\n=== RECONSTRUCTION CHECK ===");
    let d0 = 1000000 % base;
    let d1 = (1000000 - d0) / base;
    let reconstructed_manual = d0 + d1 * base;
    println!("d0 = {}, d1 = {}", d0, d1);
    println!("Reconstruction: d0 + d1*B = {}", reconstructed_manual);
    println!("Match: {}", reconstructed_manual == test_val);
}

fn crt_reconstruct(residues: &[i128], moduli: &[u64]) -> i128 {
    let q_prod: i128 = moduli.iter().map(|&q| q as i128).product();

    let mut x = 0i128;
    for (i, &ri) in residues.iter().enumerate() {
        let qi = moduli[i] as i128;
        let q_i = q_prod / qi;

        // Compute q_i^(-1) mod qi using Fermat's little theorem
        let qi_inv = mod_pow_i128(q_i % qi, qi - 2, qi);

        let term = (ri * q_i % q_prod * qi_inv % q_prod) % q_prod;
        x = (x + term) % q_prod;
    }

    if x < 0 {
        x += q_prod;
    }

    x
}

fn mod_pow_i128(mut base: i128, mut exp: i128, m: i128) -> i128 {
    let mut result = 1i128;
    base = base % m;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        exp >>= 1;
    }

    result
}
