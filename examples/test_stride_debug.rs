//! Debug stride issue in multiply_plain

use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

fn main() -> Result<(), String> {
    let params = CliffordFHEParams::new_test_ntt_1024();

    let mut key_ctx = MetalKeyContext::new(params.clone())?;
    let (pk, sk, _evk) = key_ctx.keygen()?;
    let ckks = MetalCkksContext::new(params.clone())?;

    let value = 42.0;
    let pt1 = ckks.encode(&[value])?;
    let ct1 = ckks.encrypt(&pt1, &pk)?;

    println!("Original ct1:");
    println!("  n={}, num_primes={}, level={}", ct1.n, ct1.num_primes, ct1.level);
    println!("  c0.len()={}, c1.len()={}", ct1.c0.len(), ct1.c1.len());
    println!("  Expected c0.len() = n * num_primes = {} * {} = {}", ct1.n, ct1.num_primes, ct1.n * ct1.num_primes);

    let multiplier = 2.0;
    let pt_mult = ckks.encode(&[multiplier])?;

    println!("\nPlaintext:");
    println!("  n={}, num_primes={}, level={}", pt_mult.n, pt_mult.num_primes, pt_mult.level);
    println!("  coeffs.len()={}", pt_mult.coeffs.len());

    // Check moduli
    let moduli = &params.moduli[..=ct1.level];
    println!("\nModuli slice:");
    println!("  moduli.len()={}", moduli.len());
    println!("  ct1.level={}, so moduli[..=level] has {} primes", ct1.level, moduli.len());

    Ok(())
}
