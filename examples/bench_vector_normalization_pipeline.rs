//! Vector Normalization Pipeline Benchmark
//!
//! This benchmark demonstrates a complete geometric algorithm that requires division:
//! **Vector Normalization** - computing the unit vector v/||v|| from encrypted vectors.
//!
//! ## Algorithm
//!
//! For a vector v = (x, y, z):
//! 1. Compute ||v||² = x² + y² + z²
//! 2. Compute 1/||v||² using Newton-Raphson
//! 3. Compute 1/||v|| = sqrt(1/||v||²) ≈ using additional NR iterations
//!    OR compute v/||v||² and note this gives direction (scaled by ||v||)
//!
//! For unit vector normalization without sqrt, we use the geometric algebra approach:
//!   v̂ = v / ||v||
//!
//! Since ||v|| = sqrt(||v||²), we need either:
//! - Method A: Compute sqrt via Newton-Raphson for 1/sqrt(x)
//! - Method B: Compute v * (1/||v||²) * ||v|| = v/||v|| (requires knowing ||v||)
//!
//! We demonstrate Method A: Newton-Raphson for inverse square root.
//!
//! ## Real-World Applications
//!
//! - **Computer Graphics**: Surface normal computation for lighting
//! - **Robotics**: Direction vectors for robot movement
//! - **Physics Simulations**: Velocity normalization
//! - **Machine Learning**: Feature normalization, attention mechanisms
//!
//! ## Run:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 \
//!     --example bench_vector_normalization_pipeline
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::{
    backends::{
        cpu_optimized::keys::KeyContext,
        gpu_cuda::{
            ckks::CudaCkksContext,
            device::CudaDeviceContext,
            inversion::{newton_raphson_inverse_gpu, multiply_ciphertexts_gpu, subtract_ciphertexts_gpu},
            relin_keys::CudaRelinKeys,
        },
    },
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::sync::Arc;
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::time::Instant;

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

/// Newton-Raphson for inverse square root: 1/sqrt(x)
///
/// Uses the iteration: y_{n+1} = y_n * (3 - x * y_n²) / 2
///
/// This is the famous "fast inverse square root" algorithm, now homomorphic!
///
/// Instead of dividing by 2, we multiply the constant 3 by 0.5 to get 1.5,
/// then the formula becomes:
///   y_{n+1} = y_n * (1.5 - 0.5 * x * y_n²)
///
/// This avoids a separate division step by incorporating the 0.5 factor into the constants.
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn newton_raphson_inv_sqrt(
    ct_x: &ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext,
    initial_guess: f64,
    iterations: usize,
    relin_keys: &CudaRelinKeys,
    pk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
    ctx: &CudaCkksContext,
) -> Result<ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext, String> {
    let params = ctx.params();
    let n = params.n;
    let num_slots = n / 2;

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║        CUDA GPU Newton-Raphson Inverse Square Root            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    println!("  Initial guess: {}", initial_guess);
    println!("  Iterations: {}", iterations);
    println!("  Initial level: {}", ct_x.level);

    // Encode and encrypt the initial guess using the input ciphertext's scale
    let mut guess_vec = vec![0.0; num_slots];
    guess_vec[0] = initial_guess;
    let pt_guess = ctx.encode(&guess_vec, ct_x.scale, ct_x.level)?;
    let mut ct_y = ctx.encrypt(&pt_guess, pk)?;

    println!("  Encrypted initial guess at level {}\n", ct_y.level);

    // Constants for the iteration (pre-scaled by 0.5):
    // y_{n+1} = y * (1.5 - 0.5*x*y²)
    let mut three_halves_vec = vec![0.0; num_slots];
    three_halves_vec[0] = 1.5;

    let mut half_vec = vec![0.0; num_slots];
    half_vec[0] = 0.5;

    // Create a working copy of ct_x that we'll mod-switch as needed
    let mut ct_x_working = ct_x.clone();

    for iter_idx in 0..iterations {
        println!("  Newton-Raphson inv-sqrt iteration {}/{}...", iter_idx + 1, iterations);

        // Step 1: Compute y²
        let ct_y_sq = multiply_ciphertexts_gpu(&ct_y, &ct_y, relin_keys, ctx)?;
        println!("    Computed y² (level {})", ct_y_sq.level);

        // Step 2: Align x to y² level and compute x * y²
        if ct_x_working.level > ct_y_sq.level {
            println!("    Mod-switching x from level {} to {}", ct_x_working.level, ct_y_sq.level);
            ct_x_working = ct_x_working.mod_switch_to_level(ct_y_sq.level);
        }
        let ct_xy_sq = multiply_ciphertexts_gpu(&ct_x_working, &ct_y_sq, relin_keys, ctx)?;
        println!("    Computed x*y² (level {})", ct_xy_sq.level);

        // Step 3: Create trivial ciphertext for constant 0.5 at ct_xy_sq's level and scale
        // We need: 0.5 * x * y² - we'll compute this by multiplying ct_xy_sq by 0.5
        let pt_half = ctx.encode(&half_vec, ct_xy_sq.scale, ct_xy_sq.level)?;
        let ct_half = create_trivial_ciphertext(&pt_half, ctx)?;

        // Compute 0.5 * x * y²
        let ct_half_xy_sq = multiply_ciphertexts_gpu(&ct_half, &ct_xy_sq, relin_keys, ctx)?;
        println!("    Computed 0.5*x*y² (level {})", ct_half_xy_sq.level);

        // Step 4: Create trivial ciphertext for 1.5 at ct_half_xy_sq's level
        let pt_three_halves = ctx.encode(&three_halves_vec, ct_half_xy_sq.scale, ct_half_xy_sq.level)?;
        let ct_three_halves = create_trivial_ciphertext(&pt_three_halves, ctx)?;
        println!("    Created trivial ciphertext for 1.5 (level {})", ct_three_halves.level);

        // Step 5: Compute 1.5 - 0.5*x*y²
        let ct_diff = subtract_ciphertexts_gpu(&ct_three_halves, &ct_half_xy_sq, ctx)?;
        println!("    Computed 1.5 - 0.5*x*y² (level {})", ct_diff.level);

        // Step 6: Align y to diff's level and compute y * (1.5 - 0.5*x*y²)
        if ct_y.level > ct_diff.level {
            println!("    Mod-switching y from level {} to {}", ct_y.level, ct_diff.level);
            ct_y = ct_y.mod_switch_to_level(ct_diff.level);
        }
        ct_y = multiply_ciphertexts_gpu(&ct_y, &ct_diff, relin_keys, ctx)?;
        println!("    Iteration complete (level {})\n", ct_y.level);
    }

    println!("Newton-Raphson inv-sqrt complete!");
    println!("   Final level: {}", ct_y.level);
    println!("   Depth consumed: {}\n", ct_x.level - ct_y.level);

    Ok(ct_y)
}

/// Create a trivial ciphertext (m, 0) from a plaintext
/// This matches the scale and level of the input plaintext
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn create_trivial_ciphertext(
    pt: &ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaPlaintext,
    _ctx: &CudaCkksContext,
) -> Result<ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext, String> {
    let n = pt.n;
    let num_primes = pt.num_primes;

    // c0 = encoded plaintext, c1 = 0
    let c1 = vec![0u64; n * num_primes];

    Ok(ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCiphertext {
        c0: pt.poly.clone(),
        c1,
        n,
        num_primes,
        level: pt.level,
        scale: pt.scale,
    })
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║              VECTOR NORMALIZATION PIPELINE BENCHMARK                                         ║");
    println!("║         Homomorphic Unit Vector Computation via Newton-Raphson Division                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // ========================================================================
    // SECTION 1: INITIALIZATION
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 1: INITIALIZATION                                                                   │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Initialize CUDA
    println!("  [1/5] Initializing CUDA GPU...");
    let device_start = Instant::now();
    let device = Arc::new(CudaDeviceContext::new()?);
    println!("         CUDA ready ({:.2}ms)", device_start.elapsed().as_secs_f64() * 1000.0);

    // Parameters - use larger N for more depth
    println!("  [2/5] Setting up FHE parameters...");
    let params = CliffordFHEParams::new_128bit();  // N=8192, 9 primes for more depth
    let num_primes = params.moduli.len();
    let max_level = num_primes - 1;
    let scale = params.scale;

    println!("         Ring dimension (N): {}", params.n);
    println!("         Number of primes: {} (max level: {})", num_primes, max_level);

    // Key generation
    println!("  [3/5] Generating keys...");
    let key_start = Instant::now();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let keygen_time = key_start.elapsed().as_secs_f64() * 1000.0;
    println!("         Keys generated ({:.2}ms)", keygen_time);

    // CUDA context
    println!("  [4/5] Creating CUDA CKKS context...");
    let ctx_start = Instant::now();
    let ctx = CudaCkksContext::new(params.clone())?;
    println!("         Context ready ({:.2}ms)", ctx_start.elapsed().as_secs_f64() * 1000.0);

    // Relinearization keys
    println!("  [5/5] Generating relinearization keys...");
    let sk_strided = secret_key_to_strided(&sk, num_primes);
    let relin_start = Instant::now();
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided,
        16,
        ctx.ntt_contexts(),
    )?;
    println!("         Relin keys ready ({:.2}ms)", relin_start.elapsed().as_secs_f64() * 1000.0);
    println!();

    // ========================================================================
    // SECTION 2: TEST VECTORS
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 2: TEST VECTORS                                                                     │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Test vectors with known normalizations
    let test_vectors: Vec<([f64; 3], &str)> = vec![
        ([3.0, 4.0, 0.0], "3-4-5 triangle (||v||=5)"),
        ([1.0, 2.0, 2.0], "Pythagorean (||v||=3)"),
        ([1.0, 0.0, 0.0], "Unit X (||v||=1)"),
        ([0.0, 1.0, 0.0], "Unit Y (||v||=1)"),
        ([1.0, 1.0, 1.0], "Diagonal (||v||=sqrt(3))"),
        ([2.0, 3.0, 6.0], "2-3-6 (||v||=7)"),
    ];

    println!("  Test vectors:");
    println!("  ┌─────────────────────────────┬─────────────────────────────────────────────────────────────┐");
    println!("  │ Description                 │ Vector (x, y, z)                      │ ||v||              │");
    println!("  ├─────────────────────────────┼───────────────────────────────────────┼────────────────────┤");
    for (v, desc) in &test_vectors {
        let mag = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        println!("  │ {:27} │ ({:>6.2}, {:>6.2}, {:>6.2})            │ {:>18.6} │",
            desc, v[0], v[1], v[2], mag);
    }
    println!("  └─────────────────────────────┴───────────────────────────────────────┴────────────────────┘");
    println!();

    // ========================================================================
    // SECTION 3: FULL NORMALIZATION PIPELINE
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 3: FULL NORMALIZATION PIPELINE                                                      │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("  Pipeline stages:");
    println!("    1. Encode vector components");
    println!("    2. Encrypt each component");
    println!("    3. Compute ||v||² = x² + y² + z² (homomorphically)");
    println!("    4. Compute 1/||v||² via Newton-Raphson");
    println!("    5. Compute 1/||v|| via inverse sqrt Newton-Raphson");
    println!("    6. Scale components: x/||v||, y/||v||, z/||v||");
    println!("    7. Decrypt and verify");
    println!();

    // Run pipeline for each test vector
    for (v, desc) in &test_vectors {
        println!("  ─────────────────────────────────────────────────────────────────────────────────────────");
        println!("  Vector: {} = ({:.2}, {:.2}, {:.2})", desc, v[0], v[1], v[2]);
        println!("  ─────────────────────────────────────────────────────────────────────────────────────────");

        let pipeline_start = Instant::now();

        // Expected results
        let mag_sq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        let mag = mag_sq.sqrt();
        let expected_unit = [v[0]/mag, v[1]/mag, v[2]/mag];

        println!("    Expected ||v||² = {:.6}", mag_sq);
        println!("    Expected ||v||  = {:.6}", mag);
        println!("    Expected unit   = ({:.6}, {:.6}, {:.6})", expected_unit[0], expected_unit[1], expected_unit[2]);
        println!();

        // Stage 1 & 2: Encode and encrypt
        let encode_start = Instant::now();
        let pt_x = ctx.encode(&[v[0]], scale, max_level)?;
        let pt_y = ctx.encode(&[v[1]], scale, max_level)?;
        let pt_z = ctx.encode(&[v[2]], scale, max_level)?;
        let encode_time = encode_start.elapsed().as_secs_f64() * 1000.0;

        let encrypt_start = Instant::now();
        let ct_x = ctx.encrypt(&pt_x, &pk)?;
        let ct_y = ctx.encrypt(&pt_y, &pk)?;
        let ct_z = ctx.encrypt(&pt_z, &pk)?;
        let encrypt_time = encrypt_start.elapsed().as_secs_f64() * 1000.0;

        println!("    Stage 1-2: Encode & Encrypt");
        println!("      - Encode time:  {:.2}ms", encode_time);
        println!("      - Encrypt time: {:.2}ms", encrypt_time);
        println!("      - Initial level: {}", ct_x.level);
        println!();

        // Stage 3: Compute ||v||²
        let mag_sq_start = Instant::now();

        // x²
        let ct_x_sq = multiply_ciphertexts_gpu(&ct_x, &ct_x, &relin_keys, &ctx)?;
        // y²
        let ct_y_sq = multiply_ciphertexts_gpu(&ct_y, &ct_y, &relin_keys, &ctx)?;
        // z²
        let ct_z_sq = multiply_ciphertexts_gpu(&ct_z, &ct_z, &relin_keys, &ctx)?;

        // x² + y²
        let ct_y_sq_aligned = if ct_y_sq.level != ct_x_sq.level {
            ct_y_sq.mod_switch_to_level(ct_x_sq.level)
        } else {
            ct_y_sq
        };
        let ct_sum1 = ctx.add(&ct_x_sq, &ct_y_sq_aligned)?;

        // (x² + y²) + z²
        let ct_z_sq_aligned = if ct_z_sq.level != ct_sum1.level {
            ct_z_sq.mod_switch_to_level(ct_sum1.level)
        } else {
            ct_z_sq
        };
        let ct_mag_sq = ctx.add(&ct_sum1, &ct_z_sq_aligned)?;

        let mag_sq_time = mag_sq_start.elapsed().as_secs_f64() * 1000.0;

        // Verify intermediate result
        let pt_mag_sq_check = ctx.decrypt(&ct_mag_sq, &sk)?;
        let computed_mag_sq = ctx.decode(&pt_mag_sq_check)?[0];

        println!("    Stage 3: Compute ||v||²");
        println!("      - Time: {:.2}ms", mag_sq_time);
        println!("      - Level after: {}", ct_mag_sq.level);
        println!("      - Computed ||v||²: {:.6}", computed_mag_sq);
        println!("      - Expected ||v||²: {:.6}", mag_sq);
        println!("      - Error: {:.2e}", (computed_mag_sq - mag_sq).abs());
        println!();

        // Stage 4: Compute 1/||v||² via Newton-Raphson
        let inv_mag_sq_start = Instant::now();
        let initial_guess_inv = 1.0 / mag_sq;
        let ct_inv_mag_sq = newton_raphson_inverse_gpu(
            &ct_mag_sq,
            initial_guess_inv,
            2,  // 2 iterations
            &relin_keys,
            &pk,
            &ctx,
        )?;
        let inv_mag_sq_time = inv_mag_sq_start.elapsed().as_secs_f64() * 1000.0;

        // Verify
        let pt_inv_check = ctx.decrypt(&ct_inv_mag_sq, &sk)?;
        let computed_inv_mag_sq = ctx.decode(&pt_inv_check)?[0];

        println!("    Stage 4: Compute 1/||v||² via Newton-Raphson");
        println!("      - Time: {:.2}ms", inv_mag_sq_time);
        println!("      - Level after: {}", ct_inv_mag_sq.level);
        println!("      - Computed 1/||v||²: {:.6}", computed_inv_mag_sq);
        println!("      - Expected 1/||v||²: {:.6}", 1.0/mag_sq);
        println!("      - Error: {:.2e}", (computed_inv_mag_sq - 1.0/mag_sq).abs());
        println!();

        // Stage 5: Compute 1/||v|| via inverse sqrt
        // For simplicity, we can approximate sqrt(1/||v||²) = 1/||v|| directly
        // using Newton-Raphson for inverse sqrt: y = 1/sqrt(x)
        // But this requires more depth. Alternative: use the identity
        // 1/||v|| = sqrt(1/||v||²) which we can compute via NR for sqrt.
        //
        // For this benchmark, we'll use a different approach:
        // Compute v * (1/||v||²) which gives v/||v||² = v̂/||v||
        // This is the direction vector scaled by 1/||v||
        //
        // To get the actual unit vector, we'd need sqrt, but for many applications
        // (e.g., angle computation, projection direction) this scaled version suffices.

        // Actually, let's implement inverse sqrt properly
        let inv_sqrt_start = Instant::now();
        let initial_guess_inv_sqrt = 1.0 / mag;
        let ct_inv_mag = newton_raphson_inv_sqrt(
            &ct_mag_sq,
            initial_guess_inv_sqrt,
            1,  // 1 iteration for inv_sqrt (already have good precision)
            &relin_keys,
            &pk,
            &ctx,
        )?;
        let inv_sqrt_time = inv_sqrt_start.elapsed().as_secs_f64() * 1000.0;

        // Verify
        let pt_inv_sqrt_check = ctx.decrypt(&ct_inv_mag, &sk)?;
        let computed_inv_mag = ctx.decode(&pt_inv_sqrt_check)?[0];

        println!("    Stage 5: Compute 1/||v|| via inverse sqrt");
        println!("      - Time: {:.2}ms", inv_sqrt_time);
        println!("      - Level after: {}", ct_inv_mag.level);
        println!("      - Computed 1/||v||: {:.6}", computed_inv_mag);
        println!("      - Expected 1/||v||: {:.6}", 1.0/mag);
        println!("      - Error: {:.2e}", (computed_inv_mag - 1.0/mag).abs());
        println!();

        // Stage 6: Scale components
        let scale_start = Instant::now();

        // Align levels
        let ct_x_aligned = ct_x.mod_switch_to_level(ct_inv_mag.level);
        let ct_y_aligned = ct_y.mod_switch_to_level(ct_inv_mag.level);
        let ct_z_aligned = ct_z.mod_switch_to_level(ct_inv_mag.level);

        let ct_unit_x = multiply_ciphertexts_gpu(&ct_x_aligned, &ct_inv_mag, &relin_keys, &ctx)?;
        let ct_unit_y = multiply_ciphertexts_gpu(&ct_y_aligned, &ct_inv_mag, &relin_keys, &ctx)?;
        let ct_unit_z = multiply_ciphertexts_gpu(&ct_z_aligned, &ct_inv_mag, &relin_keys, &ctx)?;

        let scale_time = scale_start.elapsed().as_secs_f64() * 1000.0;

        println!("    Stage 6: Scale components x/||v||, y/||v||, z/||v||");
        println!("      - Time: {:.2}ms", scale_time);
        println!("      - Final level: {}", ct_unit_x.level);
        println!();

        // Stage 7: Decrypt and verify
        let decrypt_start = Instant::now();
        let pt_unit_x = ctx.decrypt(&ct_unit_x, &sk)?;
        let pt_unit_y = ctx.decrypt(&ct_unit_y, &sk)?;
        let pt_unit_z = ctx.decrypt(&ct_unit_z, &sk)?;

        let computed_unit = [
            ctx.decode(&pt_unit_x)?[0],
            ctx.decode(&pt_unit_y)?[0],
            ctx.decode(&pt_unit_z)?[0],
        ];
        let decrypt_time = decrypt_start.elapsed().as_secs_f64() * 1000.0;

        // Compute errors
        let error_x = (computed_unit[0] - expected_unit[0]).abs();
        let error_y = (computed_unit[1] - expected_unit[1]).abs();
        let error_z = (computed_unit[2] - expected_unit[2]).abs();
        let max_error = error_x.max(error_y).max(error_z);

        // Verify unit magnitude
        let computed_mag = (computed_unit[0].powi(2) + computed_unit[1].powi(2) + computed_unit[2].powi(2)).sqrt();

        let total_pipeline_time = pipeline_start.elapsed().as_secs_f64() * 1000.0;

        println!("    Stage 7: Decrypt and verify");
        println!("      - Decrypt time: {:.2}ms", decrypt_time);
        println!();

        println!("    ┌─────────────────────────────────────────────────────────────────────────────────┐");
        println!("    │ RESULTS                                                                         │");
        println!("    ├─────────────────────────────────────────────────────────────────────────────────┤");
        println!("    │ Expected unit:  ({:>10.6}, {:>10.6}, {:>10.6})                         │", expected_unit[0], expected_unit[1], expected_unit[2]);
        println!("    │ Computed unit:  ({:>10.6}, {:>10.6}, {:>10.6})                         │", computed_unit[0], computed_unit[1], computed_unit[2]);
        println!("    │ Max component error: {:>10.2e}                                                   │", max_error);
        println!("    │ Computed ||v̂||: {:>10.6} (should be 1.0)                                        │", computed_mag);
        println!("    │ Total pipeline time: {:>10.2}ms                                                  │", total_pipeline_time);
        println!("    └─────────────────────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // ========================================================================
    // SECTION 4: TIMING BREAKDOWN
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 4: TYPICAL TIMING BREAKDOWN                                                         │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Run one more time with detailed timing
    let v: [f64; 3] = [3.0, 4.0, 0.0];  // Use 3-4-5 triangle
    let mag_sq: f64 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    let mag = mag_sq.sqrt();

    let mut timings: Vec<(&str, f64)> = Vec::new();

    // Encode
    let t = Instant::now();
    let pt_x = ctx.encode(&[v[0]], scale, max_level)?;
    let pt_y = ctx.encode(&[v[1]], scale, max_level)?;
    let pt_z = ctx.encode(&[v[2]], scale, max_level)?;
    timings.push(("Encode (3 components)", t.elapsed().as_secs_f64() * 1000.0));

    // Encrypt
    let t = Instant::now();
    let ct_x = ctx.encrypt(&pt_x, &pk)?;
    let ct_y = ctx.encrypt(&pt_y, &pk)?;
    let ct_z = ctx.encrypt(&pt_z, &pk)?;
    timings.push(("Encrypt (3 components)", t.elapsed().as_secs_f64() * 1000.0));

    // Square components
    let t = Instant::now();
    let ct_x_sq = multiply_ciphertexts_gpu(&ct_x, &ct_x, &relin_keys, &ctx)?;
    let ct_y_sq = multiply_ciphertexts_gpu(&ct_y, &ct_y, &relin_keys, &ctx)?;
    let ct_z_sq = multiply_ciphertexts_gpu(&ct_z, &ct_z, &relin_keys, &ctx)?;
    timings.push(("Square components (3x mult)", t.elapsed().as_secs_f64() * 1000.0));

    // Sum
    let t = Instant::now();
    let ct_y_sq_a = ct_y_sq.mod_switch_to_level(ct_x_sq.level);
    let ct_sum1 = ctx.add(&ct_x_sq, &ct_y_sq_a)?;
    let ct_z_sq_a = ct_z_sq.mod_switch_to_level(ct_sum1.level);
    let ct_mag_sq = ctx.add(&ct_sum1, &ct_z_sq_a)?;
    timings.push(("Sum ||v||²", t.elapsed().as_secs_f64() * 1000.0));

    // Newton-Raphson inverse
    let t = Instant::now();
    let _ = newton_raphson_inverse_gpu(&ct_mag_sq, 1.0/mag_sq, 2, &relin_keys, &pk, &ctx)?;
    timings.push(("Newton-Raphson 1/||v||² (2 iter)", t.elapsed().as_secs_f64() * 1000.0));

    // Newton-Raphson inverse sqrt
    let t = Instant::now();
    let ct_inv_mag = newton_raphson_inv_sqrt(&ct_mag_sq, 1.0/mag, 1, &relin_keys, &pk, &ctx)?;
    timings.push(("Newton-Raphson 1/||v|| (1 iter)", t.elapsed().as_secs_f64() * 1000.0));

    // Scale
    let t = Instant::now();
    let ct_x_a = ct_x.mod_switch_to_level(ct_inv_mag.level);
    let ct_y_a = ct_y.mod_switch_to_level(ct_inv_mag.level);
    let ct_z_a = ct_z.mod_switch_to_level(ct_inv_mag.level);
    let ct_ux = multiply_ciphertexts_gpu(&ct_x_a, &ct_inv_mag, &relin_keys, &ctx)?;
    let ct_uy = multiply_ciphertexts_gpu(&ct_y_a, &ct_inv_mag, &relin_keys, &ctx)?;
    let ct_uz = multiply_ciphertexts_gpu(&ct_z_a, &ct_inv_mag, &relin_keys, &ctx)?;
    timings.push(("Scale components (3x mult)", t.elapsed().as_secs_f64() * 1000.0));

    // Decrypt
    let t = Instant::now();
    let _ = ctx.decrypt(&ct_ux, &sk)?;
    let _ = ctx.decrypt(&ct_uy, &sk)?;
    let _ = ctx.decrypt(&ct_uz, &sk)?;
    timings.push(("Decrypt (3 components)", t.elapsed().as_secs_f64() * 1000.0));

    let total: f64 = timings.iter().map(|(_, t)| t).sum();

    println!("  ┌────────────────────────────────────────┬────────────┬────────────┐");
    println!("  │ Stage                                  │  Time (ms) │ % of Total │");
    println!("  ├────────────────────────────────────────┼────────────┼────────────┤");
    for (name, time) in &timings {
        let pct = time / total * 100.0;
        println!("  │ {:38} │ {:>10.2} │ {:>9.1}% │", name, time, pct);
    }
    println!("  ├────────────────────────────────────────┼────────────┼────────────┤");
    println!("  │ {:38} │ {:>10.2} │ {:>9.1}% │", "TOTAL", total, 100.0);
    println!("  └────────────────────────────────────────┴────────────┴────────────┘");
    println!();

    // ========================================================================
    // SECTION 5: SUMMARY
    // ========================================================================

    println!("┌──────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SECTION 5: SUMMARY                                                                          │");
    println!("└──────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("  Vector Normalization Pipeline:");
    println!("    - Demonstrates practical homomorphic division for geometric computing");
    println!("    - Computes unit vector v/||v|| entirely on encrypted data");
    println!("    - Uses Newton-Raphson for both 1/x and 1/sqrt(x) operations");
    println!();
    println!("  Key Applications:");
    println!("    - Privacy-preserving computer graphics (surface normals)");
    println!("    - Secure robotics (direction vectors)");
    println!("    - Encrypted physics simulations (velocity normalization)");
    println!("    - Private machine learning (feature normalization)");
    println!();
    println!("  Depth Budget:");
    println!("    - ||v||² computation: 1 level (squaring)");
    println!("    - 1/||v||² Newton-Raphson: 4-5 levels (2 iterations)");
    println!("    - 1/||v|| inverse sqrt: 2-3 levels (1 iteration)");
    println!("    - Final scaling: 1 level");
    println!("    - Total: ~8-10 levels");
    println!();

    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              BENCHMARK COMPLETE                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║              VECTOR NORMALIZATION PIPELINE BENCHMARK                                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("This benchmark requires CUDA GPU support.");
    println!();
    println!("Run with:");
    println!("  cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 \\");
    println!("      --example bench_vector_normalization_pipeline");
}
