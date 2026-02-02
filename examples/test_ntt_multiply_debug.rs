//! Debug NTT multiplication directly

use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

fn main() -> Result<(), String> {
    let params = CliffordFHEParams::new_test_ntt_1024();

    println!("Creating Metal GPU context...");
    let mut key_ctx = MetalKeyContext::new(params.clone())?;
    let (pk, sk, _evk) = key_ctx.keygen()?;
    let ckks = MetalCkksContext::new(params.clone())?;

    // Encrypt 42
    let value = 42.0;
    let pt1 = ckks.encode(&[value])?;
    let ct1 = ckks.encrypt(&pt1, &pk)?;

    println!("Ciphertext c0 first 10 coefficients (RNS, first prime):");
    for i in 0..10 {
        print!("{} ", ct1.c0[i * ct1.num_primes]);
    }
    println!("\n");

    // Encode multiplier
    let multiplier = 2.0;
    let pt_mult = ckks.encode(&[multiplier])?;

    println!("Plaintext multiplier first 10 coefficients (RNS, first prime):");
    for i in 0..10 {
        print!("{} ", pt_mult.coeffs[i * pt_mult.num_primes]);
    }
    println!("\n");

    // Do polynomial multiplication WITHOUT rescale
    let moduli = &params.moduli[..=ct1.level];
    println!("Multiplying using Metal GPU NTT...");
    let new_c0 = ckks.multiply_polys_flat_ntt(&ct1.c0, &pt_mult.coeffs, moduli)?;

    println!("Result c0 first 10 coefficients (RNS, first prime):");
    for i in 0..10 {
        print!("{} ", new_c0[i * ct1.num_primes]);
    }
    println!("\n");

    // Now compare with CPU
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

    let cpu_key_ctx = KeyContext::new(params.clone());
    let (cpu_pk, cpu_sk, _) = cpu_key_ctx.keygen();
    let cpu_ckks = CkksContext::new(params.clone());

    let cpu_pt1 = cpu_ckks.encode(&[value]);
    let cpu_ct1 = cpu_ckks.encrypt(&cpu_pt1, &cpu_pk);

    println!("CPU Ciphertext c0 first 10 coefficients (RNS comp 0):");
    for i in 0..10 {
        print!("{} ", cpu_ct1.c0[i].values[0]);
    }
    println!("\n");

    Ok(())
}
