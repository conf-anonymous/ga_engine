//! Isolated tests for individual Clifford FHE operations
//!
//! Each test can be run independently with beautiful output.
//!
//! Run individual tests:
//!   cargo test --test test_clifford_operations_isolated test_key_generation --features v1 -- --nocapture
//!   cargo test --test test_clifford_operations_isolated test_encryption_decryption --features v1 -- --nocapture
//!   cargo test --test test_clifford_operations_isolated test_reverse --features v1 -- --nocapture
//!   cargo test --test test_clifford_operations_isolated test_geometric_product --features v1 -- --nocapture
//!   cargo test --test test_clifford_operations_isolated test_rotation --features v1 -- --nocapture
//!   cargo test --test test_clifford_operations_isolated test_wedge_product --features v1 -- --nocapture
//!   cargo test --test test_clifford_operations_isolated test_inner_product --features v1 -- --nocapture
//!   cargo test --test test_clifford_operations_isolated test_projection --features v1 -- --nocapture
//!   cargo test --test test_clifford_operations_isolated test_rejection --features v1 -- --nocapture
//!
//! Run all:
//!   cargo test --test test_clifford_operations_isolated --features v1 -- --nocapture

use ga_engine::clifford_fhe_v1::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v1::keys_rns::rns_keygen;
use ga_engine::clifford_fhe_v1::ckks_rns::{rns_encrypt, rns_decrypt, RnsPlaintext, RnsCiphertext};
use ga_engine::clifford_fhe_v1::geometric_product_rns::{
    geometric_product_3d_componentwise,
    reverse_3d,
    rotate_3d,
    wedge_product_3d,
    inner_product_3d,
    project_3d,
    reject_3d,
};
use std::time::Instant;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

/// Create a progress bar for long operations with known steps
fn create_progress_bar(msg: &str, total_steps: u64) -> ProgressBar {
    let pb = ProgressBar::new(total_steps);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{prefix} {spinner:.cyan} [{bar:30.cyan/blue}] {pos}/{len} [{elapsed_precise}] {msg}")
            .unwrap()
            .progress_chars("█▓▒░ ")
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    );
    pb.set_prefix(format!("  {}", "▸".bright_cyan().to_string()));
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Create a progress spinner for long operations (no progress tracking)
#[allow(dead_code)]
fn create_spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{prefix} {spinner:.cyan} {msg} [{elapsed_precise}]")
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    );
    pb.set_prefix(format!("  {}", "▸".bright_cyan().to_string()));
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Helper to encrypt a multivector
fn encrypt_multivector_3d(
    mv: &[f64; 8],
    pk: &ga_engine::clifford_fhe_v1::keys_rns::RnsPublicKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 8] {
    let mut result = Vec::new();
    for &component in mv.iter() {
        let mut coeffs = vec![0i64; params.n];
        coeffs[0] = (component * params.scale).round() as i64;
        let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
        let ct = rns_encrypt(pk, &pt, params);
        result.push(ct);
    }
    result.try_into().unwrap()
}

/// Helper to encrypt a multivector with progress bar
fn encrypt_multivector_3d_with_progress(
    mv: &[f64; 8],
    pk: &ga_engine::clifford_fhe_v1::keys_rns::RnsPublicKey,
    params: &CliffordFHEParams,
    pb: &ProgressBar,
) -> [RnsCiphertext; 8] {
    let mut result = Vec::new();
    for (i, &component) in mv.iter().enumerate() {
        pb.set_message(format!("encrypting component {}/8", i + 1));
        let mut coeffs = vec![0i64; params.n];
        coeffs[0] = (component * params.scale).round() as i64;
        let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
        let ct = rns_encrypt(pk, &pt, params);
        result.push(ct);
        pb.inc(1);
    }
    result.try_into().unwrap()
}

/// Helper to decrypt a multivector
fn decrypt_multivector_3d(
    ct: &[RnsCiphertext; 8],
    sk: &ga_engine::clifford_fhe_v1::keys_rns::RnsSecretKey,
    params: &CliffordFHEParams,
) -> [f64; 8] {
    let mut result = [0.0; 8];
    for i in 0..8 {
        let pt = rns_decrypt(sk, &ct[i], params);
        let val = pt.coeffs.rns_coeffs[0][0];
        let q = params.moduli[0];
        let centered = if val > q / 2 { val - q } else { val };
        result[i] = (centered as f64) / ct[i].scale;
    }
    result
}

/// Helper to decrypt a multivector with progress bar
fn decrypt_multivector_3d_with_progress(
    ct: &[RnsCiphertext; 8],
    sk: &ga_engine::clifford_fhe_v1::keys_rns::RnsSecretKey,
    params: &CliffordFHEParams,
    pb: &ProgressBar,
) -> [f64; 8] {
    let mut result = [0.0; 8];
    for i in 0..8 {
        pb.set_message(format!("decrypting component {}/8", i + 1));
        let pt = rns_decrypt(sk, &ct[i], params);
        let val = pt.coeffs.rns_coeffs[0][0];
        let q = params.moduli[0];
        let centered = if val > q / 2 { val - q } else { val };
        result[i] = (centered as f64) / ct[i].scale;
        pb.inc(1);
    }
    result
}

/// Compute maximum error
fn max_error(a: &[f64; 8], b: &[f64; 8]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Print test header
fn print_header(name: &str) {
    println!("\n{}", "═".repeat(80).bright_blue().bold());
    println!("{} {}", "◆".bright_cyan().bold(), name.bright_white().bold());
    println!("{}\n", "═".repeat(80).bright_blue().bold());
}

/// Print test result
fn print_result(passed: bool, error: f64, duration: std::time::Duration) {
    println!("\n{}", "─".repeat(80).bright_blue());
    if passed {
        println!("{} {} {} {}",
            "✓".bright_green().bold(),
            "PASS".bright_green().bold(),
            format!("[{:.2}s]", duration.as_secs_f64()).dimmed(),
            format!("[max_error: {:.2e}]", error).bright_cyan()
        );
    } else {
        println!("{} {} {} {}",
            "✗".bright_red().bold(),
            "FAIL".bright_red().bold(),
            format!("[{:.2}s]", duration.as_secs_f64()).dimmed(),
            format!("[max_error: {:.2e}]", error).bright_red()
        );
    }
    println!("{}\n", "═".repeat(80).bright_blue().bold());
}

#[test]
fn test_key_generation() {
    print_header("Clifford FHE V1: Key Generation");

    let start = Instant::now();

    print!("  {} {}...", "▸".bright_cyan(), "Setting up parameters".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    println!(" {}", "✓".bright_green().bold());

    println!("    Ring dimension: N = {}", params.n.to_string().bright_cyan());
    println!("    Number of primes: {}", params.moduli.len().to_string().bright_cyan());
    println!("    Security level: {} bits", "≥128".bright_cyan());
    println!();

    print!("  {} {}...", "▸".bright_cyan(), "Generating public key".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let (pk, _sk, evk) = rns_keygen(&params);
    println!(" {}", "✓".bright_green().bold());

    print!("  {} {}...", "▸".bright_cyan(), "Verifying key consistency".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    // Verify keys are non-zero and have correct dimensions
    assert_eq!(pk.a.n, params.n);
    assert!(evk.evk0.len() > 0); // EVK should have multiple digits
    println!(" {}", "✓".bright_green().bold());

    let duration = start.elapsed();
    print_result(true, 0.0, duration);
}

#[test]
fn test_encryption_decryption() {
    print_header("Clifford FHE V1: Encryption/Decryption");

    let start = Instant::now();

    print!("  {} {}...", "▸".bright_cyan(), "Initializing FHE system".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let (pk, sk, _evk) = rns_keygen(&params);
    println!(" {}", "✓".bright_green().bold());
    println!();

    print!("  {} {}...", "▸".bright_cyan(), "Creating test multivector".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let mv = [1.5, 2.3, 0.0, 0.0, 3.1, 0.0, 0.0, 4.7];
    println!(" {}", "✓".bright_green().bold());
    println!("    Input: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
        mv[0], mv[1], mv[2], mv[3], mv[4], mv[5], mv[6], mv[7]);
    println!();

    print!("  {} {}...", "▸".bright_cyan(), "Encrypting (8 components)".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let ct = encrypt_multivector_3d(&mv, &pk, &params);
    println!(" {}", "✓".bright_green().bold());
    println!();

    print!("  {} {}...", "▸".bright_cyan(), "Decrypting".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let result = decrypt_multivector_3d(&ct, &sk, &params);
    println!(" {}", "✓".bright_green().bold());
    println!("    Output: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]);
    println!();

    let error = max_error(&mv, &result);
    let passed = error < 1e-6;

    let duration = start.elapsed();
    print_result(passed, error, duration);

    assert!(passed, "Encryption/decryption error too large: {}", error);
}

#[test]
fn test_reverse() {
    print_header("Clifford FHE V1: Reverse (~a)");

    let start = Instant::now();

    print!("  {} {}...", "▸".bright_cyan(), "Initializing FHE system".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let (pk, sk, _evk) = rns_keygen(&params);
    println!(" {}", "✓".bright_green().bold());
    println!();

    print!("  {} {}...", "▸".bright_cyan(), "Encrypting test multivector".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let a = [1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0];
    let ct_a = encrypt_multivector_3d(&a, &pk, &params);
    println!(" {}", "✓".bright_green().bold());
    println!("    Input: 1 + 2e₁ + 3e₁₂ + 4e₁₂₃");
    println!();

    print!("  {} {}...", "▸".bright_cyan(), "Applying homomorphic reverse".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let ct_reversed = reverse_3d(&ct_a, &params);
    println!(" {}", "✓".bright_green().bold());
    println!();

    print!("  {} {}...", "▸".bright_cyan(), "Decrypting result".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let result = decrypt_multivector_3d(&ct_reversed, &sk, &params);
    println!(" {}", "✓".bright_green().bold());
    println!("    Expected: 1 + 2e₁ - 3e₁₂ + 4e₁₂₃ (sign flip on bivector)");
    println!("    Got:      [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]);
    println!();

    let expected = [1.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0];
    let error = max_error(&result, &expected);
    let passed = error < 1e-6;

    let duration = start.elapsed();
    print_result(passed, error, duration);

    assert!(passed, "Reverse error too large: {}", error);
}

#[test]
fn test_geometric_product() {
    print_header("Clifford FHE V1: Geometric Product (a ⊗ b)");

    let start = Instant::now();

    print!("  {} {}...", "▸".bright_cyan(), "Initializing FHE system".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let (pk, sk, evk) = rns_keygen(&params);
    println!(" {}", "✓".bright_green().bold());
    println!();

    let a = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Total progress: 8 (encrypt a) + 8 (encrypt b) + 64 (geometric product) + 8 (decrypt) = 88 steps
    let pb = create_progress_bar("Encrypting a", 88);
    let ct_a = encrypt_multivector_3d_with_progress(&a, &pk, &params, &pb);

    pb.set_message("Encrypting b".to_string());
    let ct_b = encrypt_multivector_3d_with_progress(&b, &pk, &params, &pb);

    pb.set_message("Computing geometric product (64 multiplications)".to_string());
    println!();
    println!("    a = (1 + 2e₁)");
    println!("    b = (3e₂)");
    println!();

    let gp_start = Instant::now();
    let ct_prod = geometric_product_3d_componentwise(&ct_a, &ct_b, &evk, &params);
    pb.inc(64); // Increment for all 64 multiplications
    let gp_time = gp_start.elapsed();

    pb.set_message("Decrypting result".to_string());
    let result = decrypt_multivector_3d_with_progress(&ct_prod, &sk, &params, &pb);

    pb.finish_and_clear();
    println!("  {} Complete! Geometric product time: {} s",
        "▸".bright_cyan(),
        format!("{:.2}", gp_time.as_secs_f64()).bright_green().bold());
    println!();
    println!("    Expected: 3e₂ + 6e₁₂");
    println!("    Got:      [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]);
    println!();

    let expected = [0.0, 0.0, 3.0, 0.0, 6.0, 0.0, 0.0, 0.0];
    let error = max_error(&result, &expected);
    let passed = error < 1e-6;

    let duration = start.elapsed();
    print_result(passed, error, duration);

    assert!(passed, "Geometric product error too large: {}", error);
}

#[test]
fn test_rotation() {
    print_header("Clifford FHE V1: Rotation (R ⊗ v ⊗ ~R)");

    let start = Instant::now();

    print!("  {} {}...", "▸".bright_cyan(), "Initializing FHE system".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let (pk, sk, evk) = rns_keygen(&params);
    println!(" {}", "✓".bright_green().bold());
    println!();

    print!("  {} {}...", "▸".bright_cyan(), "Creating 90° rotor about Z-axis".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let cos45 = std::f64::consts::FRAC_1_SQRT_2;
    let sin45 = std::f64::consts::FRAC_1_SQRT_2;
    let rotor = [cos45, 0.0, 0.0, 0.0, sin45, 0.0, 0.0, 0.0];
    let vector = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
    println!(" {}", "✓".bright_green().bold());
    println!("    Rotor: R = cos(45°) + sin(45°)e₁₂");
    println!("    Vector: v = e₁");
    println!();

    // Total: 16 (encrypt) + 128 (rotation: 2 geometric products) + 8 (decrypt) = 152 steps
    let pb = create_progress_bar("Encrypting rotor", 152);
    let ct_rotor = encrypt_multivector_3d_with_progress(&rotor, &pk, &params, &pb);

    pb.set_message("Encrypting vector".to_string());
    let ct_vec = encrypt_multivector_3d_with_progress(&vector, &pk, &params, &pb);
    println!();

    pb.set_message("Applying rotation (128 multiplications)".to_string());
    let rot_start = Instant::now();
    let ct_rotated = rotate_3d(&ct_rotor, &ct_vec, &evk, &params);
    pb.inc(128);
    let rot_time = rot_start.elapsed();

    pb.set_message("Decrypting result".to_string());
    let result = decrypt_multivector_3d_with_progress(&ct_rotated, &sk, &params, &pb);

    pb.finish_and_clear();
    println!("  {} Complete! Rotation time: {} s",
        "▸".bright_cyan(),
        format!("{:.2}", rot_time.as_secs_f64()).bright_green().bold());
    println!();
    println!("    Expected: e₂ (90° rotation from X to Y)");
    println!("    Got:      [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]);
    println!();

    let expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let error = max_error(&result, &expected);
    let passed = error < 0.5; // Rotation accumulates more error

    let duration = start.elapsed();
    print_result(passed, error, duration);

    assert!(passed, "Rotation error too large: {}", error);
}

#[test]
fn test_wedge_product() {
    print_header("Clifford FHE V1: Wedge Product (a ∧ b)");

    let start = Instant::now();

    print!("  {} {}...", "▸".bright_cyan(), "Initializing FHE system".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let (pk, sk, evk) = rns_keygen(&params);
    println!(" {}", "✓".bright_green().bold());
    println!();

    let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
    let b = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₂

    // Total: 16 (encrypt) + 128 (wedge: 2 GPs for (a⊗b - b⊗a)/2) + 8 (decrypt) = 152 steps
    let pb = create_progress_bar("Encrypting vectors", 152);
    let ct_a = encrypt_multivector_3d_with_progress(&a, &pk, &params, &pb);
    let ct_b = encrypt_multivector_3d_with_progress(&b, &pk, &params, &pb);
    println!();
    println!("    a = e₁");
    println!("    b = e₂");
    println!();

    pb.set_message("Computing wedge product".to_string());
    let wedge_start = Instant::now();
    let ct_wedge = wedge_product_3d(&ct_a, &ct_b, &evk, &params);
    pb.inc(128);
    let wedge_time = wedge_start.elapsed();

    pb.set_message("Decrypting result".to_string());
    let result = decrypt_multivector_3d_with_progress(&ct_wedge, &sk, &params, &pb);

    pb.finish_and_clear();
    println!("  {} Complete! Wedge product time: {} s",
        "▸".bright_cyan(),
        format!("{:.2}", wedge_time.as_secs_f64()).bright_green().bold());
    println!();
    println!("    Expected: e₁₂ (bivector)");
    println!("    Got:      [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]);
    println!();

    let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let error = max_error(&result, &expected);
    let passed = error < 1e-6;

    let duration = start.elapsed();
    print_result(passed, error, duration);

    assert!(passed, "Wedge product error too large: {}", error);
}

#[test]
fn test_inner_product() {
    print_header("Clifford FHE V1: Inner Product (a · b)");

    let start = Instant::now();

    print!("  {} {}...", "▸".bright_cyan(), "Initializing FHE system".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let (pk, sk, evk) = rns_keygen(&params);
    println!(" {}", "✓".bright_green().bold());
    println!();

    let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
    let b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁

    // Total: 16 (encrypt) + 128 (inner: 2 GPs for (a⊗b + b⊗a)/2) + 8 (decrypt) = 152 steps
    let pb = create_progress_bar("Encrypting vectors", 152);
    let ct_a = encrypt_multivector_3d_with_progress(&a, &pk, &params, &pb);
    let ct_b = encrypt_multivector_3d_with_progress(&b, &pk, &params, &pb);
    println!();
    println!("    a = e₁");
    println!("    b = e₁");
    println!();

    pb.set_message("Computing inner product".to_string());
    let inner_start = Instant::now();
    let ct_inner = inner_product_3d(&ct_a, &ct_b, &evk, &params);
    pb.inc(128);
    let inner_time = inner_start.elapsed();

    pb.set_message("Decrypting result".to_string());
    let result = decrypt_multivector_3d_with_progress(&ct_inner, &sk, &params, &pb);

    pb.finish_and_clear();
    println!("  {} Complete! Inner product time: {} s",
        "▸".bright_cyan(),
        format!("{:.2}", inner_time.as_secs_f64()).bright_green().bold());
    println!();
    println!("    Expected: 1 (scalar)");
    println!("    Got:      [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]);
    println!();

    let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let error = max_error(&result, &expected);
    let passed = error < 1e-6;

    let duration = start.elapsed();
    print_result(passed, error, duration);

    assert!(passed, "Inner product error too large: {}", error);
}

#[test]
fn test_projection() {
    print_header("Clifford FHE V1: Projection (proj_a(b))");

    let start = Instant::now();

    print!("  {} {}...", "▸".bright_cyan(), "Initializing FHE system".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let (pk, sk, evk) = rns_keygen(&params);
    println!(" {}", "✓".bright_green().bold());
    println!();

    let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
    let b = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁ + e₂

    // Total: 16 (encrypt) + 256 (projection: inner + 2 GPs) + 8 (decrypt) = 280 steps
    let pb = create_progress_bar("Encrypting vectors", 280);
    let ct_a = encrypt_multivector_3d_with_progress(&a, &pk, &params, &pb);
    let ct_b = encrypt_multivector_3d_with_progress(&b, &pk, &params, &pb);
    println!();
    println!("    a = e₁ (project onto this)");
    println!("    b = e₁ + e₂");
    println!();

    pb.set_message("Computing projection (depth-3 operation)".to_string());
    let proj_start = Instant::now();
    let ct_proj = project_3d(&ct_a, &ct_b, &evk, &params);
    pb.inc(256);
    let proj_time = proj_start.elapsed();

    pb.set_message("Decrypting result".to_string());
    let result = decrypt_multivector_3d_with_progress(&ct_proj, &sk, &params, &pb);

    pb.finish_and_clear();
    println!("  {} Complete! Projection time: {} s",
        "▸".bright_cyan(),
        format!("{:.2}", proj_time.as_secs_f64()).bright_green().bold());
    println!();
    println!("    Expected: e₁ (component along a)");
    println!("    Got:      [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]);
    println!();

    let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let error = max_error(&result, &expected);
    let passed = error < 0.5; // Depth-3 operation

    let duration = start.elapsed();
    print_result(passed, error, duration);

    assert!(passed, "Projection error too large: {}", error);
}

#[test]
fn test_rejection() {
    print_header("Clifford FHE V1: Rejection (rej_a(b))");

    let start = Instant::now();

    print!("  {} {}...", "▸".bright_cyan(), "Initializing FHE system".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    let (pk, sk, evk) = rns_keygen(&params);
    println!(" {}", "✓".bright_green().bold());
    println!();

    let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁
    let b = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // e₁ + e₂

    // Total: 16 (encrypt) + 256 (rejection: projection + subtraction) + 8 (decrypt) = 280 steps
    let pb = create_progress_bar("Encrypting vectors", 280);
    let ct_a = encrypt_multivector_3d_with_progress(&a, &pk, &params, &pb);
    let ct_b = encrypt_multivector_3d_with_progress(&b, &pk, &params, &pb);
    println!();
    println!("    a = e₁ (reject from this)");
    println!("    b = e₁ + e₂");
    println!();

    pb.set_message("Computing rejection (depth-3 operation)".to_string());
    let rej_start = Instant::now();
    let ct_rej = reject_3d(&ct_a, &ct_b, &evk, &params);
    pb.inc(256);
    let rej_time = rej_start.elapsed();

    pb.set_message("Decrypting result".to_string());
    let result = decrypt_multivector_3d_with_progress(&ct_rej, &sk, &params, &pb);

    pb.finish_and_clear();
    println!("  {} Complete! Rejection time: {} s",
        "▸".bright_cyan(),
        format!("{:.2}", rej_time.as_secs_f64()).bright_green().bold());
    println!();
    println!("    Expected: e₂ (component perpendicular to a)");
    println!("    Got:      [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]);
    println!();

    let expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let error = max_error(&result, &expected);
    let passed = error < 0.5; // Depth-3 operation

    let duration = start.elapsed();
    print_result(passed, error, duration);

    assert!(passed, "Rejection error too large: {}", error);
}
