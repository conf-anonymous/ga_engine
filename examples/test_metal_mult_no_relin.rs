//! Test Metal multiplication WITHOUT relinearization/rescale
//! This isolates whether the tensor product itself is correct

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::ckks::{MetalCkksContext, MetalCiphertext},
    backends::cpu_optimized::keys::KeyContext,
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Test: Metal multiplication WITHOUT relin/rescale\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();

    let metal_ctx = MetalCkksContext::new(params.clone())?;

    // Test: 2.0 × 3.0 = 6.0
    let a = 2.0;
    let b = 3.0;
    let expected = a * b;

    println!("Test: {} × {} = {}\n", a, b, expected);

    let pt_a = metal_ctx.encode(&[a])?;
    let pt_b = metal_ctx.encode(&[b])?;
    let ct_a = metal_ctx.encrypt(&pt_a, &pk)?;
    let ct_b = metal_ctx.encrypt(&pt_b, &pk)?;

    let n = ct_a.n;
    let level = ct_a.level;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];

    println!("Input level: {}, num_primes: {}\n", level, num_primes);

    // Manually compute JUST the tensor product c0 = ct_a.c0 * ct_b.c0
    let c0_d0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(
        &ct_a.c0,
        &ct_b.c0,
        moduli,
    )?;

    println!("Tensor product c0×d0 computed");
    println!("Result c0[0] across {} primes:", num_primes);
    for j in 0..num_primes {
        println!("  prime[{}]: {}", j, c0_d0[0 * num_primes + j]);
    }

    // Create a "fake" ciphertext with just c0 (set c1=0)
    // This won't decrypt correctly, but we can see if the polynomial values make sense
    let fake_ct = MetalCiphertext {
        c0: c0_d0.clone(),
        c1: vec![0u64; c0_d0.len()],
        n,
        num_primes,
        level,
        scale: ct_a.scale * ct_b.scale, // Scale is now scale²
    };

    println!("\nFake ciphertext (c0=product, c1=0):");
    println!("  scale: {}", fake_ct.scale);
    println!("  level: {}", fake_ct.level);

    // Decrypt (this will just give us c0, since c1*s = 0)
    let pt_result = metal_ctx.decrypt(&fake_ct, &sk)?;

    println!("\nDecrypted plaintext coeff[0] across {} primes:", pt_result.num_primes);
    for j in 0..pt_result.num_primes {
        println!("  prime[{}]: {}", j, pt_result.coeffs[0 * pt_result.num_primes + j]);
    }

    // Try to decode
    let result = metal_ctx.decode(&pt_result)?;
    println!("\nDecoded value: {}", result[0]);
    println!("Expected (approximately, since we didn't decrypt c1*s): around {}", expected);

    // The result won't be exact since we're missing the c1*s term,
    // but it should be in the same ballpark if the multiplication worked
    let error_magnitude = (result[0] - expected).abs();
    println!("Error magnitude: {:.2e}", error_magnitude);

    // If error is astronomical (> 10^6), something is very wrong
    if error_magnitude > 1e6 {
        println!("\n❌ Error is astronomical - tensor product or decode is broken!");
        Err("Tensor product produces wrong values".to_string())
    } else {
        println!("\n✅ Error is reasonable - tensor product seems OK");
        Ok(())
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
