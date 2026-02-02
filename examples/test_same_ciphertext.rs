//! Test using the SAME ciphertext values for both CPU and Metal multiplication
//! This isolates whether the issue is in encryption (different random values)
//! or in relinearization itself

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::{MetalCkksContext, MetalCiphertext},
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::KeyContext,
        multiplication::multiply_ciphertexts,
        ckks::{CkksContext, Ciphertext, Plaintext},
        rns::RnsRepresentation,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Same Ciphertext Test");
    println!("====================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let device = Arc::new(MetalDevice::new()?);
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();

    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let cpu_ctx = CkksContext::new(params.clone());

    // Generate Metal relin keys from the SAME secret key
    let ntt_contexts = metal_ctx.ntt_contexts();
    let metal_relin_keys = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        20,
    )?;

    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];

    // Create a CPU ciphertext for 2.0 × 3.0
    let a = 2.0;
    let b = 3.0;
    let cpu_pt_a = cpu_ctx.encode(&[a]);
    let cpu_pt_b = cpu_ctx.encode(&[b]);
    let cpu_ct_a = cpu_ctx.encrypt(&cpu_pt_a, &pk);
    let cpu_ct_b = cpu_ctx.encrypt(&cpu_pt_b, &pk);

    println!("Using CPU-generated ciphertexts for BOTH CPU and Metal\n");

    // === CPU Multiplication ===
    println!("--- CPU Multiplication ---");
    let cpu_ct_result = multiply_ciphertexts(&cpu_ct_a, &cpu_ct_b, &evk, &key_ctx);
    let cpu_pt_result = cpu_ctx.decrypt(&cpu_ct_result, &sk);
    let cpu_result = cpu_ctx.decode(&cpu_pt_result);
    println!("CPU Result: {} (error: {:.2e})\n", cpu_result[0], (cpu_result[0] - 6.0).abs());

    // === Convert CPU ciphertexts to Metal format ===
    println!("--- Metal Multiplication (using CPU's ciphertexts) ---");

    // Convert CPU ciphertext to Metal flat format
    let metal_ct_a = cpu_to_metal_ciphertext(&cpu_ct_a, &params)?;
    let metal_ct_b = cpu_to_metal_ciphertext(&cpu_ct_b, &params)?;

    println!("Converted ciphertexts to Metal format");
    println!("  metal_ct_a: level={}, num_primes={}", metal_ct_a.level, metal_ct_a.num_primes);

    // Verify conversion: first few values should match
    print!("  CPU ct_a.c0[0] primes: ");
    for j in 0..num_primes {
        print!("{} ", cpu_ct_a.c0[0].values[j]);
    }
    println!();
    print!("  Metal ct_a.c0[0] primes: ");
    for j in 0..num_primes {
        print!("{} ", metal_ct_a.c0[0 * num_primes + j]);
    }
    println!("\n");

    // Metal multiply using the same ciphertext data
    let metal_ct_result = metal_ct_a.multiply(&metal_ct_b, &metal_relin_keys, &metal_ctx)?;

    // Convert Metal result back to CPU format for decryption
    let cpu_format_result = metal_to_cpu_ciphertext(&metal_ct_result, &params)?;
    let metal_pt_result = cpu_ctx.decrypt(&cpu_format_result, &sk);
    let metal_result = cpu_ctx.decode(&metal_pt_result);

    println!("Metal Result: {} (error: {:.2e})\n", metal_result[0], (metal_result[0] - 6.0).abs());

    // === Comparison ===
    println!("--- Comparison ---");
    println!("Expected:     6.0");
    println!("CPU Got:      {} (error: {:.2e})", cpu_result[0], (cpu_result[0] - 6.0).abs());
    println!("Metal Got:    {} (error: {:.2e})", metal_result[0], (metal_result[0] - 6.0).abs());

    let cpu_ok = (cpu_result[0] - 6.0).abs() < 1e-6;
    let metal_ok = (metal_result[0] - 6.0).abs() < 1e-6;

    println!("\nCPU:   {}", if cpu_ok { "✅ PASS" } else { "❌ FAIL" });
    println!("Metal: {}", if metal_ok { "✅ PASS" } else { "❌ FAIL" });

    if metal_ok {
        Ok(())
    } else {
        Err("Metal multiplication with same ciphertext failed".to_string())
    }
}

#[cfg(feature = "v2-gpu-metal")]
fn cpu_to_metal_ciphertext(
    cpu_ct: &Ciphertext,
    params: &CliffordFHEParams,
) -> Result<MetalCiphertext, String> {
    let n = params.n;
    let num_primes = cpu_ct.level + 1;

    // Convert to flat layout: [coeff0_q0, coeff0_q1, ..., coeff1_q0, ...]
    let mut c0_flat = vec![0u64; n * num_primes];
    let mut c1_flat = vec![0u64; n * num_primes];

    for i in 0..n {
        for j in 0..num_primes {
            c0_flat[i * num_primes + j] = cpu_ct.c0[i].values[j];
            c1_flat[i * num_primes + j] = cpu_ct.c1[i].values[j];
        }
    }

    Ok(MetalCiphertext {
        c0: c0_flat,
        c1: c1_flat,
        n,
        num_primes,
        level: cpu_ct.level,
        scale: cpu_ct.scale,
    })
}

#[cfg(feature = "v2-gpu-metal")]
fn metal_to_cpu_ciphertext(
    metal_ct: &MetalCiphertext,
    params: &CliffordFHEParams,
) -> Result<Ciphertext, String> {
    let n = metal_ct.n;
    let num_primes = metal_ct.num_primes;
    let moduli = &params.moduli[..num_primes];

    // Convert from flat layout back to Vec<RnsRepresentation>
    let mut c0 = Vec::with_capacity(n);
    let mut c1 = Vec::with_capacity(n);

    for i in 0..n {
        let mut c0_rns = vec![0u64; num_primes];
        let mut c1_rns = vec![0u64; num_primes];

        for j in 0..num_primes {
            c0_rns[j] = metal_ct.c0[i * num_primes + j];
            c1_rns[j] = metal_ct.c1[i * num_primes + j];
        }

        c0.push(RnsRepresentation::new(c0_rns, moduli.to_vec()));
        c1.push(RnsRepresentation::new(c1_rns, moduli.to_vec()));
    }

    Ok(Ciphertext {
        c0,
        c1,
        level: metal_ct.level,
        scale: metal_ct.scale,
        n,
    })
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
