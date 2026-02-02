//! Atomic Debug: CUDA Ciphertext Multiplication
//!
//! This test breaks down the multiplication pipeline into the smallest possible steps
//! to isolate exactly where the error occurs.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example debug_cuda_mult_atomic
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::{
    backends::{
        cpu_optimized::keys::KeyContext,
        gpu_cuda::{
            ckks::{CudaCkksContext, CudaCiphertext},
            device::CudaDeviceContext,
            relin_keys::CudaRelinKeys,
        },
    },
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
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

/// Print first few coefficients of a polynomial in flat layout
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn print_flat_poly(name: &str, poly: &[u64], n: usize, num_primes: usize, num_coeffs: usize) {
    println!("  {} (flat layout, first {} coeffs):", name, num_coeffs);
    for coeff_idx in 0..num_coeffs.min(n) {
        print!("    coeff[{}]: ", coeff_idx);
        for prime_idx in 0..num_primes {
            let idx = prime_idx * n + coeff_idx;
            print!("{} ", poly[idx]);
        }
        println!();
    }
}

/// Print first few coefficients of a polynomial in strided layout
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn print_strided_poly(name: &str, poly: &[u64], num_primes: usize, num_coeffs: usize) {
    println!("  {} (strided layout, first {} coeffs):", name, num_coeffs);
    for coeff_idx in 0..num_coeffs {
        print!("    coeff[{}]: ", coeff_idx);
        for prime_idx in 0..num_primes {
            let idx = coeff_idx * num_primes + prime_idx;
            if idx < poly.len() {
                print!("{} ", poly[idx]);
            }
        }
        println!();
    }
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     ATOMIC DEBUG: CUDA Ciphertext Multiplication                       ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    // Setup
    let device = Arc::new(CudaDeviceContext::new()?);
    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();
    let max_level = num_primes - 1;
    let scale = params.scale;

    println!("Parameters: N={}, num_primes={}, max_level={}, scale=2^{}\n",
             n, num_primes, max_level, (scale.log2() as u32));

    // Generate keys
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();

    // Create CUDA context
    let ctx = CudaCkksContext::new(params.clone())?;
    let sk_strided = secret_key_to_strided(&sk, num_primes);

    // Generate relin keys
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided.clone(),
        16,
        ctx.ntt_contexts(),
    )?;

    // Encrypt test values
    let a_val = 2.0;
    let b_val = 3.0;
    let expected = a_val * b_val;

    println!("Test: {} × {} = {}\n", a_val, b_val, expected);

    let pt_a = ctx.encode(&[a_val], scale, max_level)?;
    let pt_b = ctx.encode(&[b_val], scale, max_level)?;
    let ct_a = ctx.encrypt(&pt_a, &pk)?;
    let ct_b = ctx.encrypt(&pt_b, &pk)?;

    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 0: Verify inputs decrypt correctly");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let pt_a_dec = ctx.decrypt(&ct_a, &sk)?;
    let a_dec = ctx.decode(&pt_a_dec)?;
    let pt_b_dec = ctx.decrypt(&ct_b, &sk)?;
    let b_dec = ctx.decode(&pt_b_dec)?;

    println!("  ct_a decrypts to: {:.10} (expected {})", a_dec[0], a_val);
    println!("  ct_b decrypts to: {:.10} (expected {})", b_dec[0], b_val);
    println!("  ct_a: level={}, num_primes={}", ct_a.level, ct_a.num_primes);
    println!("  ct_b: level={}, num_primes={}", ct_b.level, ct_b.num_primes);

    if (a_dec[0] - a_val).abs() > 1e-6 || (b_dec[0] - b_val).abs() > 1e-6 {
        println!("  ✗ FAIL - Input decryption failed!\n");
        return Err("Input decryption failed".to_string());
    }
    println!("  ✓ PASS\n");

    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 1: Tensored multiplication (c0*d0, c0*d1+c1*d0, c1*d1)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let num_active_primes = ct_a.level + 1;
    println!("  num_active_primes = {}", num_active_primes);

    // Call tensored multiplication
    let (c0_tensor, c1_tensor, c2_tensor) = ctx.multiply_ciphertexts_tensored(&ct_a, &ct_b)?;

    println!("  Tensored results (flat layout):");
    println!("    c0_tensor.len() = {} (expected {})", c0_tensor.len(), n * num_active_primes);
    println!("    c1_tensor.len() = {} (expected {})", c1_tensor.len(), n * num_active_primes);
    println!("    c2_tensor.len() = {} (expected {})", c2_tensor.len(), n * num_active_primes);

    print_flat_poly("c0_tensor", &c0_tensor, n, num_active_primes, 3);
    print_flat_poly("c1_tensor", &c1_tensor, n, num_active_primes, 3);
    print_flat_poly("c2_tensor", &c2_tensor, n, num_active_primes, 3);

    // Verify tensored multiplication by decrypting (c0, c1) WITHOUT c2
    // If we ignore c2, decryption gives: c0 + c1*s (missing the c2*s^2 term)
    println!("\n  Intermediate check: decrypt (c0_tensor, c1_tensor) ignoring c2...");

    // Convert c0, c1 from flat to strided
    let c0_strided = ctx.flat_to_strided(&c0_tensor, n, num_active_primes, num_active_primes);
    let c1_strided = ctx.flat_to_strided(&c1_tensor, n, num_active_primes, num_active_primes);

    let ct_no_c2 = CudaCiphertext {
        c0: c0_strided.clone(),
        c1: c1_strided.clone(),
        n,
        num_primes: num_active_primes,
        level: ct_a.level,
        scale: ct_a.scale * ct_b.scale,
    };

    let pt_no_c2 = ctx.decrypt(&ct_no_c2, &sk)?;
    let val_no_c2 = ctx.decode(&pt_no_c2)?;
    println!("  (c0 + c1*s) decrypts to: {:.10}", val_no_c2[0]);
    println!("  Expected (a*b = {}): This won't match because c2*s² is missing", expected);

    println!("\n  ✓ Tensored multiplication completed\n");

    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 2: Gadget decomposition of c2");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Get gadget params
    let (base_bits, num_digits) = relin_keys.gadget_params();
    println!("  Gadget base: 2^{} = {}", base_bits, relin_keys.base_w());
    println!("  Number of digits: {}", num_digits);

    // We can't directly call gadget_decompose, but we can check the relin key structure
    println!("  Checking relin key components...");
    let relin_key = relin_keys.get_relin_key();
    println!("  Number of KS components: {}", relin_key.ks_components.len());
    println!("  Component sizes: b_0.len()={}, a_0.len()={}",
             relin_key.ks_components[0].0.len(),
             relin_key.ks_components[0].1.len());

    // Print first component values
    println!("  First KS component (b_0, a_0) first coeff:");
    let (b_0, a_0) = &relin_key.ks_components[0];
    print_flat_poly("b_0", b_0, n, relin_key.num_primes_key, 2);
    print_flat_poly("a_0", a_0, n, relin_key.num_primes_key, 2);

    println!("\n  ✓ Gadget decomposition info retrieved\n");

    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 3: Apply relinearization");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let (c0_relin_flat, c1_relin_flat) = relin_keys.apply_relinearization_gpu(
        &c0_tensor,
        &c1_tensor,
        &c2_tensor,
        ct_a.level,
        ctx.ntt_contexts(),
        &ctx,
    )?;

    println!("  Relinearized results (flat layout):");
    println!("    c0_relin.len() = {} (expected {})", c0_relin_flat.len(), n * num_active_primes);
    println!("    c1_relin.len() = {} (expected {})", c1_relin_flat.len(), n * num_active_primes);

    print_flat_poly("c0_relin", &c0_relin_flat, n, num_active_primes, 3);
    print_flat_poly("c1_relin", &c1_relin_flat, n, num_active_primes, 3);

    // Convert to strided and create ciphertext for decryption test
    let c0_relin_strided = ctx.flat_to_strided(&c0_relin_flat, n, num_active_primes, num_active_primes);
    let c1_relin_strided = ctx.flat_to_strided(&c1_relin_flat, n, num_active_primes, num_active_primes);

    println!("\n  After flat_to_strided:");
    print_strided_poly("c0_relin_strided", &c0_relin_strided, num_active_primes, 3);
    print_strided_poly("c1_relin_strided", &c1_relin_strided, num_active_primes, 3);

    let ct_relin = CudaCiphertext {
        c0: c0_relin_strided,
        c1: c1_relin_strided,
        n,
        num_primes: num_active_primes,
        level: ct_a.level,
        scale: ct_a.scale * ct_b.scale,
    };

    println!("\n  Decrypt BEFORE rescale (scale = {} = 2^{}):",
             ct_relin.scale, (ct_relin.scale.log2() as u32));
    let pt_relin = ctx.decrypt(&ct_relin, &sk)?;
    let val_relin = ctx.decode(&pt_relin)?;
    println!("  Relinearized decrypts to: {:.10}", val_relin[0]);
    println!("  Expected (a*b*scale): {} * {} = {}", expected, scale, expected * scale);

    // The value should be expected * scale because we haven't rescaled yet
    let expected_before_rescale = expected; // Actually decode handles the scale
    let error_before_rescale = (val_relin[0] - expected_before_rescale).abs();
    println!("  Error vs expected {}: {:.6e}", expected_before_rescale, error_before_rescale);

    if error_before_rescale > 1.0 {
        println!("\n  ✗ FAIL - Relinearization produced wrong result!");
        println!("  THE BUG IS IN RELINEARIZATION OR EARLIER\n");

        // Additional debug: try decrypting with different interpretations
        println!("  Additional debug info:");
        println!("    First 5 decoded slots: {:?}", &val_relin[..5.min(val_relin.len())]);

        return Err("Relinearization failed".to_string());
    }
    println!("  ✓ PASS - Relinearization correct\n");

    println!("════════════════════════════════════════════════════════════════════════");
    println!("STEP 4: Rescale");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let ct_rescaled = ct_relin.rescale_to_next(&ctx)?;

    println!("  After rescale:");
    println!("    level: {} -> {}", ct_relin.level, ct_rescaled.level);
    println!("    num_primes: {} -> {}", ct_relin.num_primes, ct_rescaled.num_primes);
    println!("    scale: {} -> {}", ct_relin.scale, ct_rescaled.scale);

    print_strided_poly("c0_rescaled", &ct_rescaled.c0, ct_rescaled.num_primes, 3);
    print_strided_poly("c1_rescaled", &ct_rescaled.c1, ct_rescaled.num_primes, 3);

    let pt_final = ctx.decrypt(&ct_rescaled, &sk)?;
    let val_final = ctx.decode(&pt_final)?;

    let error_final = (val_final[0] - expected).abs();
    let rel_error = error_final / expected;

    println!("\n  Final result: {:.10}", val_final[0]);
    println!("  Expected:     {:.10}", expected);
    println!("  Error:        {:.6e}", error_final);
    println!("  Rel Error:    {:.6e}", rel_error);

    if rel_error < 1e-3 {
        println!("\n  ✓ PASS - Full multiplication pipeline correct!\n");
    } else {
        println!("\n  ✗ FAIL - Rescaling produced wrong result!");
        println!("  THE BUG IS IN RESCALING\n");
        return Err("Rescaling failed".to_string());
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("ALL STEPS PASSED!");
    println!("════════════════════════════════════════════════════════════════════════\n");

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires the 'v2' and 'v2-gpu-cuda' features.");
    println!("Run with: cargo run --release --features v2,v2-gpu-cuda --example debug_cuda_mult_atomic");
}
