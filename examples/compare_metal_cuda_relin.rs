//! Compare Metal vs CUDA relinearization step by step
//!
//! This test runs identical operations on both backends and compares outputs
//! at each step to find exactly where they diverge.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda,v2-gpu-metal --example compare_metal_cuda_relin
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
use ga_engine::clifford_fhe_v2::{
    backends::{
        cpu_optimized::keys::KeyContext,
        gpu_cuda::{
            ckks::{CudaCkksContext, CudaCiphertext as CudaCt},
            device::CudaDeviceContext,
            relin_keys::CudaRelinKeys,
        },
        gpu_metal::{
            ckks::{MetalCkksContext, MetalCiphertext as MetalCt},
            device::MetalDeviceContext,
            relin_keys::MetalRelinKeys,
        },
    },
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
use std::sync::Arc;

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
fn secret_key_to_strided(
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    num_primes: usize,
) -> Vec<u64> {
    let n = sk.n;
    let mut strided = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            strided[coeff_idx * num_primes + prime_idx] = sk.coeffs[coeff_idx].values[prime_idx];
        }
    }
    strided
}

/// Compare two flat-layout polynomials and report differences
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
fn compare_flat_polys(name: &str, metal: &[u64], cuda: &[u64], n: usize, num_primes: usize) -> bool {
    if metal.len() != cuda.len() {
        println!("  {} SIZE MISMATCH: Metal={}, CUDA={}", name, metal.len(), cuda.len());
        return false;
    }

    let mut diff_count = 0;
    let mut first_diff_idx = None;
    let mut first_diff_prime = None;
    let mut first_diff_coeff = None;

    for prime_idx in 0..num_primes {
        for coeff_idx in 0..n {
            let idx = prime_idx * n + coeff_idx;
            if metal[idx] != cuda[idx] {
                diff_count += 1;
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(idx);
                    first_diff_prime = Some(prime_idx);
                    first_diff_coeff = Some(coeff_idx);
                }
            }
        }
    }

    if diff_count > 0 {
        println!("  {} DIFFERS: {} out of {} values", name, diff_count, metal.len());
        if let (Some(idx), Some(p), Some(c)) = (first_diff_idx, first_diff_prime, first_diff_coeff) {
            println!("    First diff at idx={} (prime={}, coeff={}): Metal={}, CUDA={}",
                     idx, p, c, metal[idx], cuda[idx]);
        }
        // Show first few values
        println!("    First 5 values:");
        for i in 0..5.min(metal.len()) {
            let m = metal[i];
            let c = cuda[i];
            let marker = if m != c { " <-- DIFF" } else { "" };
            println!("      [{}]: Metal={}, CUDA={}{}", i, m, c, marker);
        }
        return false;
    }

    println!("  {} MATCH ✓ ({} values)", name, metal.len());
    true
}

/// Compare two strided-layout polynomials
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
fn compare_strided_polys(name: &str, metal: &[u64], cuda: &[u64], n: usize, num_primes: usize) -> bool {
    if metal.len() != cuda.len() {
        println!("  {} SIZE MISMATCH: Metal={}, CUDA={}", name, metal.len(), cuda.len());
        return false;
    }

    let mut diff_count = 0;
    let mut first_diff_idx = None;

    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            let idx = coeff_idx * num_primes + prime_idx;
            if idx < metal.len() && metal[idx] != cuda[idx] {
                diff_count += 1;
                if first_diff_idx.is_none() {
                    first_diff_idx = Some((idx, coeff_idx, prime_idx));
                }
            }
        }
    }

    if diff_count > 0 {
        println!("  {} DIFFERS: {} out of {} values", name, diff_count, metal.len());
        if let Some((idx, coeff, prime)) = first_diff_idx {
            println!("    First diff at idx={} (coeff={}, prime={}): Metal={}, CUDA={}",
                     idx, coeff, prime, metal[idx], cuda[idx]);
        }
        return false;
    }

    println!("  {} MATCH ✓ ({} values)", name, metal.len());
    true
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
fn main() -> Result<(), String> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     COMPARISON TEST: Metal vs CUDA Relinearization                     ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    // Setup both devices
    let metal_device = Arc::new(MetalDeviceContext::new()?);
    let cuda_device = Arc::new(CudaDeviceContext::new()?);

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();
    let max_level = num_primes - 1;
    let scale = params.scale;

    println!("Parameters: N={}, num_primes={}, max_level={}, scale=2^{}\n",
             n, num_primes, max_level, (scale.log2() as u32));

    // Generate keys once (shared between both)
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let sk_strided = secret_key_to_strided(&sk, num_primes);

    // Create contexts for both backends
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let cuda_ctx = CudaCkksContext::new(params.clone())?;

    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 0: Compare PSI values");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let metal_psi = metal_ctx.psi_per_prime();
    let cuda_psi = cuda_ctx.psi_per_prime();

    println!("  PSI values per prime:");
    let mut psi_match = true;
    for i in 0..num_primes {
        let m = metal_psi[i];
        let c = cuda_psi[i];
        let marker = if m != c { " <-- MISMATCH!" } else { "" };
        println!("    Prime {}: Metal={}, CUDA={}{}", i, m, c, marker);
        if m != c { psi_match = false; }
    }
    if psi_match {
        println!("  ✓ PSI values MATCH\n");
    } else {
        println!("  ✗ PSI values DIFFER - This is critical!\n");
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 1: Generate relinearization keys");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let base_bits = 16;

    let metal_relin = MetalRelinKeys::new_gpu(
        metal_device.clone(),
        params.clone(),
        sk_strided.clone(),
        base_bits,
        metal_ctx.ntt_contexts(),
    )?;

    let cuda_relin = CudaRelinKeys::new_gpu(
        cuda_device.clone(),
        params.clone(),
        sk_strided.clone(),
        base_bits,
        cuda_ctx.ntt_contexts(),
    )?;

    let (metal_base_bits, metal_num_digits) = metal_relin.gadget_params();
    let (cuda_base_bits, cuda_num_digits) = cuda_relin.gadget_params();

    println!("  Metal: base_bits={}, num_digits={}, base_w={}",
             metal_base_bits, metal_num_digits, metal_relin.base_w());
    println!("  CUDA:  base_bits={}, num_digits={}, base_w={}",
             cuda_base_bits, cuda_num_digits, cuda_relin.base_w());

    if metal_base_bits != cuda_base_bits || metal_num_digits != cuda_num_digits {
        println!("  ✗ Gadget params DIFFER!\n");
    } else {
        println!("  ✓ Gadget params MATCH\n");
    }

    // Compare KS components
    println!("  Comparing KS key components...");
    let metal_ks = metal_relin.get_relin_key();
    let cuda_ks = cuda_relin.get_relin_key();

    println!("    Metal: {} components, num_primes_key={}",
             metal_ks.ks_components.len(), metal_ks.num_primes_key);
    println!("    CUDA:  {} components, num_primes_key={}",
             cuda_ks.ks_components.len(), cuda_ks.num_primes_key);

    // Compare first KS component
    let (metal_b0, metal_a0) = &metal_ks.ks_components[0];
    let (cuda_b0, cuda_a0) = &cuda_ks.ks_components[0];

    compare_flat_polys("KS b[0]", metal_b0, cuda_b0, n, metal_ks.num_primes_key);
    compare_flat_polys("KS a[0]", metal_a0, cuda_a0, n, metal_ks.num_primes_key);

    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 2: Encrypt test values");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let a_val = 2.0;
    let b_val = 3.0;
    let expected = a_val * b_val;
    println!("  Test: {} × {} = {}\n", a_val, b_val, expected);

    // Encrypt using Metal context
    let metal_pt_a = metal_ctx.encode(&[a_val], scale, max_level)?;
    let metal_pt_b = metal_ctx.encode(&[b_val], scale, max_level)?;
    let metal_ct_a = metal_ctx.encrypt(&metal_pt_a, &pk)?;
    let metal_ct_b = metal_ctx.encrypt(&metal_pt_b, &pk)?;

    // Encrypt using CUDA context
    let cuda_pt_a = cuda_ctx.encode(&[a_val], scale, max_level)?;
    let cuda_pt_b = cuda_ctx.encode(&[b_val], scale, max_level)?;
    let cuda_ct_a = cuda_ctx.encrypt(&cuda_pt_a, &pk)?;
    let cuda_ct_b = cuda_ctx.encrypt(&cuda_pt_b, &pk)?;

    // Verify both decrypt correctly
    let metal_dec_a = metal_ctx.decode(&metal_ctx.decrypt(&metal_ct_a, &sk)?)?;
    let cuda_dec_a = cuda_ctx.decode(&cuda_ctx.decrypt(&cuda_ct_a, &sk)?)?;

    println!("  Metal ct_a decrypts to: {:.10}", metal_dec_a[0]);
    println!("  CUDA  ct_a decrypts to: {:.10}", cuda_dec_a[0]);
    println!("  Both should be close to: {}\n", a_val);

    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 3: Tensored multiplication");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let (metal_c0, metal_c1, metal_c2) = metal_ctx.multiply_ciphertexts_tensored(&metal_ct_a, &metal_ct_b)?;
    let (cuda_c0, cuda_c1, cuda_c2) = cuda_ctx.multiply_ciphertexts_tensored(&cuda_ct_a, &cuda_ct_b)?;

    let num_active = metal_ct_a.level + 1;
    println!("  num_active_primes = {}\n", num_active);

    // Note: These won't match exactly because encryption uses random noise
    // But we can check sizes
    println!("  Tensor output sizes:");
    println!("    Metal: c0={}, c1={}, c2={}", metal_c0.len(), metal_c1.len(), metal_c2.len());
    println!("    CUDA:  c0={}, c1={}, c2={}", cuda_c0.len(), cuda_c1.len(), cuda_c2.len());

    // Since encryption randomness differs, we'll use Metal's tensored output
    // and feed it to BOTH relinearization functions to compare
    println!("\n  Using Metal's tensored output for both to isolate relinearization...\n");

    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 4: Apply relinearization with SAME input");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Use Metal's c0, c1, c2 as input to both
    let input_c0 = metal_c0.clone();
    let input_c1 = metal_c1.clone();
    let input_c2 = metal_c2.clone();

    println!("  Input c2 (first 10 values): {:?}", &input_c2[..10.min(input_c2.len())]);

    let (metal_relin_c0, metal_relin_c1) = metal_relin.apply_relinearization_gpu(
        &input_c0,
        &input_c1,
        &input_c2,
        metal_ct_a.level,
        metal_ctx.ntt_contexts(),
        &metal_ctx,
    )?;

    let (cuda_relin_c0, cuda_relin_c1) = cuda_relin.apply_relinearization_gpu(
        &input_c0,
        &input_c1,
        &input_c2,
        cuda_ct_a.level,
        cuda_ctx.ntt_contexts(),
        &cuda_ctx,
    )?;

    println!("\n  Comparing relinearization outputs...\n");

    let c0_match = compare_flat_polys("c0_relin", &metal_relin_c0, &cuda_relin_c0, n, num_active);
    let c1_match = compare_flat_polys("c1_relin", &metal_relin_c1, &cuda_relin_c1, n, num_active);

    if c0_match && c1_match {
        println!("\n  ✓ Relinearization outputs MATCH perfectly!\n");
    } else {
        println!("\n  ✗ Relinearization outputs DIFFER!\n");

        // Show more details about the difference
        println!("  Analyzing differences...");

        // Check if it's a layout issue
        println!("\n  Check if values exist but in different positions...");

        // Look for Metal's first value in CUDA output
        if !cuda_relin_c0.is_empty() && !metal_relin_c0.is_empty() {
            let metal_val = metal_relin_c0[0];
            let found_in_cuda = cuda_relin_c0.iter().position(|&x| x == metal_val);
            println!("    Metal c0[0] = {} found in CUDA at: {:?}", metal_val, found_in_cuda);
        }
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 5: Decrypt and decode results");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Convert to strided for decryption using respective contexts
    let metal_c0_strided = metal_ctx.flat_to_strided(&metal_relin_c0, n, num_active, num_active);
    let metal_c1_strided = metal_ctx.flat_to_strided(&metal_relin_c1, n, num_active, num_active);

    let cuda_c0_strided = cuda_ctx.flat_to_strided(&cuda_relin_c0, n, num_active, num_active);
    let cuda_c1_strided = cuda_ctx.flat_to_strided(&cuda_relin_c1, n, num_active, num_active);

    let metal_ct_result = MetalCt {
        c0: metal_c0_strided,
        c1: metal_c1_strided,
        n,
        num_primes: num_active,
        level: metal_ct_a.level,
        scale: metal_ct_a.scale * metal_ct_b.scale,
    };

    let cuda_ct_result = CudaCt {
        c0: cuda_c0_strided,
        c1: cuda_c1_strided,
        n,
        num_primes: num_active,
        level: cuda_ct_a.level,
        scale: cuda_ct_a.scale * cuda_ct_b.scale,
    };

    let metal_dec = metal_ctx.decode(&metal_ctx.decrypt(&metal_ct_result, &sk)?)?;
    let cuda_dec = cuda_ctx.decode(&cuda_ctx.decrypt(&cuda_ct_result, &sk)?)?;

    println!("  Metal decrypts to: {:.10}", metal_dec[0]);
    println!("  CUDA  decrypts to: {:.10}", cuda_dec[0]);
    println!("  Expected:         {:.10}", expected);

    let metal_error = (metal_dec[0] - expected).abs();
    let cuda_error = (cuda_dec[0] - expected).abs();

    println!("\n  Metal error: {:.6e}", metal_error);
    println!("  CUDA  error: {:.6e}", cuda_error);

    if metal_error < 1.0 && cuda_error > 1.0 {
        println!("\n  ═══════════════════════════════════════════════════════════════════");
        println!("  DIAGNOSIS: Metal works, CUDA fails");
        println!("  The bug is in CUDA's apply_relinearization_gpu");
        println!("  ═══════════════════════════════════════════════════════════════════\n");
    } else if metal_error > 1.0 && cuda_error < 1.0 {
        println!("\n  Unexpected: CUDA works, Metal fails?");
    } else if metal_error > 1.0 && cuda_error > 1.0 {
        println!("\n  Both fail - issue might be in shared code or input preparation");
    } else {
        println!("\n  ✓ Both backends produce correct results!");
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("COMPARISON COMPLETE");
    println!("════════════════════════════════════════════════════════════════════════\n");

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda", feature = "v2-gpu-metal")))]
fn main() {
    println!("This example requires both 'v2-gpu-cuda' and 'v2-gpu-metal' features.");
    println!("Run with: cargo run --release --features v2,v2-gpu-cuda,v2-gpu-metal --example compare_metal_cuda_relin");
}
