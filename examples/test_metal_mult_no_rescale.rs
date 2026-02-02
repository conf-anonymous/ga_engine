//! Test Metal GPU multiplication WITHOUT rescale

use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::{MetalCkksContext, MetalCiphertext};
use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

fn main() -> Result<(), String> {
    let params = CliffordFHEParams::new_test_ntt_1024();

    let mut key_ctx = MetalKeyContext::new(params.clone())?;
    let (pk, sk, _evk) = key_ctx.keygen()?;
    let ckks = MetalCkksContext::new(params.clone())?;

    // Encrypt 42
    let value = 42.0;
    let pt1 = ckks.encode(&[value])?;
    let ct1 = ckks.encrypt(&pt1, &pk)?;

    println!("Original ct: level={}, scale={}", ct1.level, ct1.scale);

    // Encode multiplier
    let multiplier = 2.0;
    let pt_mult = ckks.encode(&[multiplier])?;

    // Do polynomial multiplication WITHOUT rescale - just the NTT part
    let moduli = &params.moduli[..=ct1.level];
    let new_c0 = ckks.multiply_polys_flat_ntt(&ct1.c0, &pt_mult.coeffs, moduli)?;
    let new_c1 = ckks.multiply_polys_flat_ntt(&ct1.c1, &pt_mult.coeffs, moduli)?;

    // Create result WITHOUT rescale (so scale is Δ²)
    let ct_mult = MetalCiphertext {
        c0: new_c0,
        c1: new_c1,
        n: ct1.n,
        num_primes: moduli.len(),
        level: ct1.level,
        scale: ct1.scale * pt_mult.scale,  // Δ²
    };

    println!("After mult (no rescale): level={}, scale={}", ct_mult.level, ct_mult.scale);

    // Decrypt - should get (42 * 2) * Δ encoded at scale Δ²
    let dec = ckks.decrypt(&ct_mult, &sk)?;
    println!("Decrypted plaintext scale: {}", dec.scale);

    // Manually decode with correct scale
    let decoded = ckks.decode(&dec)?;
    println!("Decoded value (expect 42*2=84): {}", decoded[0]);
    println!("Error: {}", (decoded[0] - 84.0).abs());

    Ok(())
}
