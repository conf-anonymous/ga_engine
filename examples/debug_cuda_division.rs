//! Debug CUDA Division - Step-by-step validation
//!
//! This test validates each step of the homomorphic division pipeline
//! to isolate where the error occurs.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example debug_cuda_division
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::{
    backends::{
        cpu_optimized::keys::KeyContext,
        gpu_cuda::{
            ckks::CudaCkksContext,
            device::CudaDeviceContext,
            inversion::{multiply_ciphertexts_gpu, subtract_ciphertexts_gpu},
            relin_keys::CudaRelinKeys,
        },
    },
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::sync::Arc;

/// Convert CPU SecretKey to CUDA strided format
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

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     CUDA Division Debug - Step-by-Step Validation                     ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    // Use smaller parameters for faster debugging
    // N=1024 with 5 primes gives us max level 4 (enough for 1 NR iteration + final mult)
    println!("Step 0: Setup with small parameters...");
    let device = Arc::new(CudaDeviceContext::new()?);
    let params = CliffordFHEParams::new_test_ntt_1024();
    let num_primes = params.moduli.len();
    let max_level = num_primes - 1;
    let scale = params.scale;

    println!("  N = {}", params.n);
    println!("  num_primes = {} (max_level = {})", num_primes, max_level);
    println!("  scale = 2^{}", (scale.log2() as u32));
    println!();

    // Generate keys
    println!("Step 1: Generate keys...");
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    println!("  Keys generated\n");

    // Create CUDA context
    println!("Step 2: Create CUDA CKKS context...");
    let ctx = CudaCkksContext::new(params.clone())?;
    let sk_strided = secret_key_to_strided(&sk, num_primes);
    println!("  Context ready\n");

    // Generate relin keys
    println!("Step 3: Generate relinearization keys...");
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided.clone(),
        16,
        ctx.ntt_contexts(),
    )?;
    println!("  Relin keys ready\n");

    // ========================================
    // TEST 0: Encode/Decode (no encryption)
    // ========================================
    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 0: Encode/Decode (no encryption)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let test_val0 = 3.5;
    let pt0 = ctx.encode(&[test_val0], scale, max_level)?;
    let result0 = ctx.decode(&pt0)?;

    let error0 = (result0[0] - test_val0).abs();
    println!("  Input:    {:.10}", test_val0);
    println!("  Output:   {:.10}", result0[0]);
    println!("  Error:    {:.2e}", error0);

    if error0 < 1e-6 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL - Encode/decode broken!\n");
        return Err("Encode/decode test failed".to_string());
    }

    // ========================================
    // TEST 0.5: NTT Forward/Inverse Roundtrip
    // ========================================
    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 0.5: NTT Forward/Inverse Roundtrip");
    println!("════════════════════════════════════════════════════════════════════════\n");

    {
        let ntt_ctx = &ctx.ntt_contexts()[0];
        let n = params.n;

        // Debug: Print twiddle factors
        println!("  DEBUG: Twiddles[0..4] = {:?}", &ntt_ctx.twiddles[0..4]);
        println!("  DEBUG: Twiddles_inv[0..4] = {:?}", &ntt_ctx.twiddles_inv[0..4]);
        println!("  DEBUG: n_inv = {}", ntt_ctx.n_inv);

        // Verify omega * omega_inv = 1
        let omega = ntt_ctx.twiddles[1];  // omega^1
        let omega_inv = ntt_ctx.twiddles_inv[1];  // omega^{-1}
        let product = ((omega as u128 * omega_inv as u128) % params.moduli[0] as u128) as u64;
        println!("  DEBUG: omega * omega_inv = {} (should be 1)", product);

        // Create a simple test polynomial: [1, 2, 3, 4, 0, 0, ...]
        let mut test_poly: Vec<u64> = vec![0; n];
        test_poly[0] = 1;
        test_poly[1] = 2;
        test_poly[2] = 3;
        test_poly[3] = 4;
        let original = test_poly.clone();

        // Forward NTT
        ntt_ctx.forward(&mut test_poly)?;
        let after_forward = test_poly.clone();
        println!("  After forward NTT: [0]={}, [1]={}, [2]={}, [3]={}",
            test_poly[0], test_poly[1], test_poly[2], test_poly[3]);

        // Inverse NTT
        ntt_ctx.inverse(&mut test_poly)?;
        println!("  After inverse NTT: [0]={}, [1]={}, [2]={}, [3]={}",
            test_poly[0], test_poly[1], test_poly[2], test_poly[3]);

        // Debug: Try applying forward NTT again to the inverse result
        // If inverse is broken, forward(inverse(x)) != x
        let mut test_poly2 = after_forward.clone();
        ntt_ctx.inverse(&mut test_poly2)?;
        println!("  Second inverse attempt: [0]={}, [1]={}, [2]={}, [3]={}",
            test_poly2[0], test_poly2[1], test_poly2[2], test_poly2[3]);

        // Check roundtrip
        let mut all_match = true;
        for i in 0..n {
            if test_poly[i] != original[i] {
                println!("  MISMATCH at [{}]: expected {}, got {}", i, original[i], test_poly[i]);
                all_match = false;
                if i > 10 { break; }
            }
        }

        if all_match {
            println!("  ✓ PASS - NTT roundtrip correct\n");
        } else {
            println!("  ✗ FAIL - NTT roundtrip broken!\n");
            return Err("NTT roundtrip test failed".to_string());
        }
    }

    // ========================================
    // TEST 0.6: NTT Polynomial Multiplication
    // ========================================
    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 0.6: NTT Polynomial Multiplication (negacyclic)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    {
        // Test: (1 + x) * (1 + x) = 1 + 2x + x^2 in regular polynomial ring
        // But in negacyclic ring (mod x^n + 1), x^n = -1
        // For small test, result should be same since no wraparound
        let n = params.n;
        let num_primes = 1; // Test with first prime only
        let _q = params.moduli[0]; // Unused but kept for reference

        // Create polynomials in strided layout: a = 1 + x, b = 1 + x
        let mut a_strided = vec![0u64; n * num_primes];
        let mut b_strided = vec![0u64; n * num_primes];
        a_strided[0] = 1;  // coeff 0, prime 0
        a_strided[1 * num_primes] = 1;  // coeff 1, prime 0
        b_strided[0] = 1;
        b_strided[1 * num_primes] = 1;

        // Expected: c = 1 + 2x + x^2 (coeffs: [1, 2, 1, 0, 0, ...])
        let c = ctx.test_multiply_polys_ntt(&a_strided, &b_strided, num_primes)?;

        println!("  (1+x) * (1+x) in negacyclic ring:");
        println!("  c[0]={}, c[1]={}, c[2]={}, c[3]={}",
            c[0], c[1 * num_primes], c[2 * num_primes], c[3 * num_primes]);

        // Check result
        let c0 = c[0];
        let c1 = c[1 * num_primes];
        let c2 = c[2 * num_primes];

        if c0 == 1 && c1 == 2 && c2 == 1 {
            println!("  ✓ PASS - NTT multiplication correct\n");
        } else {
            println!("  Expected: [1, 2, 1, 0, ...]");
            println!("  ✗ FAIL - NTT multiplication broken!\n");
            return Err("NTT multiplication test failed".to_string());
        }
    }

    // ========================================
    // TEST 0.65: Batched NTT Polynomial Multiplication
    // ========================================
    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 0.65: Batched NTT Polynomial Multiplication (negacyclic)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    {
        // Same test as 0.6 but using the batched GPU path
        let n = params.n;
        let num_primes = 3; // Use all 3 primes like the real multiplication

        // Create polynomials in strided layout: a = 1 + x, b = 1 + x
        let mut a_strided = vec![0u64; n * num_primes];
        let mut b_strided = vec![0u64; n * num_primes];
        for prime_idx in 0..num_primes {
            a_strided[0 * num_primes + prime_idx] = 1;  // coeff 0
            a_strided[1 * num_primes + prime_idx] = 1;  // coeff 1
            b_strided[0 * num_primes + prime_idx] = 1;
            b_strided[1 * num_primes + prime_idx] = 1;
        }

        // Test with single-prime path first (should work)
        println!("  Testing single-prime path (test_multiply_polys_ntt)...");
        let c_single = ctx.test_multiply_polys_ntt(&a_strided, &b_strided, num_primes)?;
        println!("  Single-prime result: c[0]={}, c[1]={}, c[2]={}",
            c_single[0], c_single[1 * num_primes], c_single[2 * num_primes]);

        // Convert to flat layout for batched test
        let a_flat = ctx.strided_to_flat(&a_strided, n, num_primes, num_primes);
        let b_flat = ctx.strided_to_flat(&b_strided, n, num_primes, num_primes);

        println!("  Testing batched GPU path...");

        // Manually run the batched multiplication pipeline
        let mut a_flat = a_flat;
        let mut b_flat = b_flat;

        // Apply twist
        ctx.apply_negacyclic_twist_flat(&mut a_flat, num_primes)?;
        ctx.apply_negacyclic_twist_flat(&mut b_flat, num_primes)?;

        // Forward NTT
        ctx.ntt_forward_batched_flat(&mut a_flat, num_primes)?;
        ctx.ntt_forward_batched_flat(&mut b_flat, num_primes)?;

        // Pointwise multiply
        let mut c_flat = vec![0u64; n * num_primes];
        ctx.ntt_pointwise_multiply_batched_flat(&a_flat, &b_flat, &mut c_flat, num_primes)?;

        // Inverse NTT
        ctx.ntt_inverse_batched_flat(&mut c_flat, num_primes)?;

        // Apply untwist
        ctx.apply_negacyclic_untwist_flat(&mut c_flat, num_primes)?;

        // Convert back to strided
        let c_batched = ctx.flat_to_strided(&c_flat, n, num_primes, num_primes);

        println!("  Batched result: c[0]={}, c[1]={}, c[2]={}",
            c_batched[0], c_batched[1 * num_primes], c_batched[2 * num_primes]);

        // Compare results
        let match_0 = c_single[0] == c_batched[0];
        let match_1 = c_single[1 * num_primes] == c_batched[1 * num_primes];
        let match_2 = c_single[2 * num_primes] == c_batched[2 * num_primes];

        if match_0 && match_1 && match_2 {
            println!("  ✓ PASS - Batched matches single-prime\n");
        } else {
            println!("  ✗ FAIL - Batched differs from single-prime!");
            println!("  Single: [{}, {}, {}]", c_single[0], c_single[1 * num_primes], c_single[2 * num_primes]);
            println!("  Batched: [{}, {}, {}]\n", c_batched[0], c_batched[1 * num_primes], c_batched[2 * num_primes]);
            return Err("Batched NTT multiplication test failed".to_string());
        }
    }

    // ========================================
    // TEST 0.7: Trivial Encryption (c0=m, c1=0)
    // ========================================
    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 0.7: Trivial Encryption (c0=m, c1=0)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    {
        let test_val = 3.5;
        let pt = ctx.encode(&[test_val], scale, max_level)?;

        // Create trivial ciphertext: c0 = m, c1 = 0
        // Decryption: m' = c0 + c1*s = m + 0 = m
        let ct_trivial = ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext {
            c0: pt.poly.clone(),
            c1: vec![0u64; pt.poly.len()],
            n: pt.n,
            num_primes: pt.num_primes,
            level: pt.level,
            scale: pt.scale,
        };

        let pt_dec = ctx.decrypt(&ct_trivial, &sk)?;
        let result = ctx.decode(&pt_dec)?;

        let error = (result[0] - test_val).abs();
        println!("  Input:    {:.10}", test_val);
        println!("  Output:   {:.10}", result[0]);
        println!("  Error:    {:.2e}", error);

        if error < 1e-6 {
            println!("  ✓ PASS - Trivial encryption/decryption works\n");
        } else {
            println!("  ✗ FAIL - Even trivial decryption is broken!\n");
            return Err("Trivial encryption test failed".to_string());
        }
    }

    // ========================================
    // TEST 1: Basic encryption/decryption
    // ========================================
    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 1: Basic Encryption/Decryption");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let test_val = 3.5;
    let pt = ctx.encode(&[test_val], scale, max_level)?;

    // Debug: Check plaintext before encryption
    println!("  DEBUG: pt.poly[0] (first coeff, prime 0) = {}", pt.poly[0]);
    println!("  DEBUG: pt.level = {}, pt.num_primes = {}, pt.scale = {}", pt.level, pt.num_primes, pt.scale);

    let ct = ctx.encrypt(&pt, &pk)?;

    // Debug: Check ciphertext
    println!("  DEBUG: ct.c0[0] = {}, ct.c1[0] = {}", ct.c0[0], ct.c1[0]);
    println!("  DEBUG: ct.level = {}, ct.num_primes = {}", ct.level, ct.num_primes);

    let pt_dec = ctx.decrypt(&ct, &sk)?;

    // Debug: Check decrypted plaintext
    println!("  DEBUG: pt_dec.poly[0] = {}", pt_dec.poly[0]);

    let result = ctx.decode(&pt_dec)?;

    let error = (result[0] - test_val).abs();
    println!("  Input:    {:.10}", test_val);
    println!("  Output:   {:.10}", result[0]);
    println!("  Error:    {:.2e}", error);

    if error < 1e-6 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL - Basic encryption broken!\n");
        return Err("Basic encryption test failed".to_string());
    }

    // ========================================
    // TEST 2: Ciphertext multiplication (single)
    // ========================================
    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 2: Single Ciphertext Multiplication");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let a_val = 2.0;
    let b_val = 3.0;
    let expected_product = a_val * b_val;

    let pt_a = ctx.encode(&[a_val], scale, max_level)?;
    let pt_b = ctx.encode(&[b_val], scale, max_level)?;
    let ct_a = ctx.encrypt(&pt_a, &pk)?;
    let ct_b = ctx.encrypt(&pt_b, &pk)?;

    println!("  Inputs: {:.1} × {:.1} = {:.1}", a_val, b_val, expected_product);
    println!("  ct_a level: {}, ct_b level: {}", ct_a.level, ct_b.level);

    let ct_prod = multiply_ciphertexts_gpu(&ct_a, &ct_b, &relin_keys, &ctx)?;
    println!("  ct_prod level: {}", ct_prod.level);

    let pt_prod = ctx.decrypt(&ct_prod, &sk)?;
    let result_prod = ctx.decode(&pt_prod)?;

    let error_prod = (result_prod[0] - expected_product).abs();
    let rel_error_prod = error_prod / expected_product;

    println!("  Expected: {:.10}", expected_product);
    println!("  Got:      {:.10}", result_prod[0]);
    println!("  Error:    {:.2e}", error_prod);
    println!("  Rel Err:  {:.2e}", rel_error_prod);

    if rel_error_prod < 1e-3 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL - Multiplication broken!\n");
        return Err("Multiplication test failed".to_string());
    }

    // ========================================
    // TEST 3: Ciphertext subtraction
    // ========================================
    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 3: Ciphertext Subtraction");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let c_val = 5.0;
    let d_val = 2.0;
    let expected_diff = c_val - d_val;

    let pt_c = ctx.encode(&[c_val], scale, max_level)?;
    let pt_d = ctx.encode(&[d_val], scale, max_level)?;
    let ct_c = ctx.encrypt(&pt_c, &pk)?;
    let ct_d = ctx.encrypt(&pt_d, &pk)?;

    println!("  Inputs: {:.1} - {:.1} = {:.1}", c_val, d_val, expected_diff);

    let ct_diff = subtract_ciphertexts_gpu(&ct_c, &ct_d, &ctx)?;

    let pt_diff = ctx.decrypt(&ct_diff, &sk)?;
    let result_diff = ctx.decode(&pt_diff)?;

    let error_diff = (result_diff[0] - expected_diff).abs();
    let rel_error_diff = error_diff / expected_diff;

    println!("  Expected: {:.10}", expected_diff);
    println!("  Got:      {:.10}", result_diff[0]);
    println!("  Error:    {:.2e}", error_diff);
    println!("  Rel Err:  {:.2e}", rel_error_diff);

    if rel_error_diff < 1e-3 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL - Subtraction broken!\n");
        return Err("Subtraction test failed".to_string());
    }

    // ========================================
    // TEST 4: Two consecutive multiplications
    // ========================================
    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 4: Two Consecutive Multiplications");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let x_val = 2.0;
    let y_val = 3.0;
    let z_val = 4.0;
    let expected_xyz = x_val * y_val * z_val;

    let pt_x = ctx.encode(&[x_val], scale, max_level)?;
    let pt_y = ctx.encode(&[y_val], scale, max_level)?;
    let pt_z = ctx.encode(&[z_val], scale, max_level)?;
    let ct_x = ctx.encrypt(&pt_x, &pk)?;
    let ct_y = ctx.encrypt(&pt_y, &pk)?;
    let ct_z = ctx.encrypt(&pt_z, &pk)?;

    println!("  Inputs: {:.1} × {:.1} × {:.1} = {:.1}", x_val, y_val, z_val, expected_xyz);
    println!("  Initial levels: ct_x={}, ct_y={}, ct_z={}", ct_x.level, ct_y.level, ct_z.level);

    // First multiplication: x * y
    let ct_xy = multiply_ciphertexts_gpu(&ct_x, &ct_y, &relin_keys, &ctx)?;
    println!("  After x*y: level={}", ct_xy.level);

    // Mod-switch z to match xy's level
    let ct_z_switched = ct_z.mod_switch_to_level(ct_xy.level);
    println!("  After mod_switch z: level={}", ct_z_switched.level);

    // Second multiplication: (x*y) * z
    let ct_xyz = multiply_ciphertexts_gpu(&ct_xy, &ct_z_switched, &relin_keys, &ctx)?;
    println!("  After (x*y)*z: level={}", ct_xyz.level);

    let pt_xyz = ctx.decrypt(&ct_xyz, &sk)?;
    let result_xyz = ctx.decode(&pt_xyz)?;

    let error_xyz = (result_xyz[0] - expected_xyz).abs();
    let rel_error_xyz = error_xyz / expected_xyz;

    println!("  Expected: {:.10}", expected_xyz);
    println!("  Got:      {:.10}", result_xyz[0]);
    println!("  Error:    {:.2e}", error_xyz);
    println!("  Rel Err:  {:.2e}", rel_error_xyz);

    if rel_error_xyz < 1e-2 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL - Consecutive multiplications broken!\n");
        return Err("Consecutive multiplication test failed".to_string());
    }

    // ========================================
    // TEST 5: Newton-Raphson single iteration
    // (x_{n+1} = x_n * (2 - a * x_n))
    // ========================================
    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 5: Newton-Raphson Single Iteration");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Compute 1/4 with initial guess 0.25 (perfect guess)
    let denom = 4.0;
    let x0 = 0.25;  // Initial guess = 1/denom (perfect)

    // One NR iteration: x1 = x0 * (2 - denom * x0)
    // With perfect guess: x1 = 0.25 * (2 - 4*0.25) = 0.25 * (2 - 1) = 0.25 * 1 = 0.25
    let expected_x1 = x0 * (2.0 - denom * x0);

    println!("  Computing 1/{} with initial guess {}", denom, x0);
    println!("  Expected after 1 NR iteration: {}", expected_x1);
    println!();

    // Encode and encrypt
    let pt_denom = ctx.encode(&[denom], scale, max_level)?;
    let pt_x0 = ctx.encode(&[x0], scale, max_level)?;
    let ct_denom = ctx.encrypt(&pt_denom, &pk)?;
    let ct_x0 = ctx.encrypt(&pt_x0, &pk)?;

    println!("  ct_denom level: {}", ct_denom.level);
    println!("  ct_x0 level: {}", ct_x0.level);

    // Step 1: a * x_n
    let ct_ax = multiply_ciphertexts_gpu(&ct_denom, &ct_x0, &relin_keys, &ctx)?;
    println!("  After denom*x0: level={}", ct_ax.level);

    // Decrypt to check intermediate
    let pt_ax = ctx.decrypt(&ct_ax, &sk)?;
    let result_ax = ctx.decode(&pt_ax)?;
    let expected_ax = denom * x0;
    println!("  denom*x0: expected={:.6}, got={:.6}, error={:.2e}",
             expected_ax, result_ax[0], (result_ax[0] - expected_ax).abs());

    // Step 2: Create constant 2 at the same level
    let num_slots = params.n / 2;
    let mut two_vec = vec![0.0; num_slots];
    two_vec[0] = 2.0;
    let pt_two = ctx.encode(&two_vec, ct_ax.scale, ct_ax.level)?;

    // Create trivial ciphertext (c0 = pt, c1 = 0)
    use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext;
    let ct_two = CudaCiphertext {
        c0: pt_two.poly.clone(),
        c1: vec![0u64; pt_two.poly.len()],
        n: pt_two.n,
        num_primes: pt_two.num_primes,
        level: pt_two.level,
        scale: pt_two.scale,
    };
    println!("  ct_two (trivial) level: {}", ct_two.level);

    // Step 3: 2 - a*x_n
    let ct_2_minus_ax = subtract_ciphertexts_gpu(&ct_two, &ct_ax, &ctx)?;
    println!("  After 2 - denom*x0: level={}", ct_2_minus_ax.level);

    // Decrypt to check
    let pt_2_minus_ax = ctx.decrypt(&ct_2_minus_ax, &sk)?;
    let result_2_minus_ax = ctx.decode(&pt_2_minus_ax)?;
    let expected_2_minus_ax = 2.0 - denom * x0;
    println!("  2-denom*x0: expected={:.6}, got={:.6}, error={:.2e}",
             expected_2_minus_ax, result_2_minus_ax[0], (result_2_minus_ax[0] - expected_2_minus_ax).abs());

    // Step 4: Mod-switch x0 to match level
    let ct_x0_switched = ct_x0.mod_switch_to_level(ct_2_minus_ax.level);
    println!("  ct_x0 after mod_switch: level={}", ct_x0_switched.level);

    // Step 5: x_n * (2 - a*x_n)
    let ct_x1 = multiply_ciphertexts_gpu(&ct_x0_switched, &ct_2_minus_ax, &relin_keys, &ctx)?;
    println!("  After x0 * (2 - denom*x0): level={}", ct_x1.level);

    // Final decrypt
    let pt_x1 = ctx.decrypt(&ct_x1, &sk)?;
    let result_x1 = ctx.decode(&pt_x1)?;

    let error_x1 = (result_x1[0] - expected_x1).abs();
    let rel_error_x1 = error_x1 / expected_x1.abs().max(1e-10);

    println!();
    println!("  Final Result:");
    println!("  Expected x1: {:.10}", expected_x1);
    println!("  Got x1:      {:.10}", result_x1[0]);
    println!("  Error:       {:.2e}", error_x1);
    println!("  Rel Err:     {:.2e}", rel_error_x1);

    if rel_error_x1 < 1e-2 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL - Newton-Raphson iteration broken!\n");
        return Err("Newton-Raphson test failed".to_string());
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("ALL TESTS PASSED!");
    println!("════════════════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires the 'v2' and 'v2-gpu-cuda' features.");
    println!("Run with: cargo run --release --features v2,v2-gpu-cuda --example debug_cuda_division");
}
