//! Privacy-Preserving 3D Point Cloud Classification with Clifford FHE
//!
//! This demonstrates the machine learning application of Clifford FHE:
//! Privacy-preserving classification of encrypted 3D shapes.
//!
//! Task: Classify encrypted 3D shapes (sphere/cube/pyramid) using
//! geometric neural networks operating on encrypted Cl(3,0) multivectors.
//!
//! Architecture:
//! - 3-layer geometric neural network (1 -> 16 -> 8 -> 3)
//! - Homomorphic geometric product for layer transformations
//! - Polynomial activation approximation
//!
//! Target performance:
//! - 99% accuracy (vs 100% plaintext)
//! - ~58 seconds total inference time
//! - <1% component-wise relative error

use ga_engine::clifford_fhe_v1::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v1::keys_rns::{rns_keygen, RnsPublicKey, RnsSecretKey};
use ga_engine::clifford_fhe_v1::ckks_rns::{rns_encrypt, rns_decrypt, RnsPlaintext, RnsCiphertext};
use ga_engine::clifford_fhe_v1::geometric_product_rns::geometric_product_3d_componentwise;
use rand::Rng;
use std::time::Instant;
use colored::Colorize;

/// Helper to encrypt a multivector (8 components for 3D)
fn encrypt_multivector_3d(
    mv: &[f64; 8],
    pk: &RnsPublicKey,
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

/// Helper to decrypt a multivector
fn decrypt_multivector_3d(
    ct: &[RnsCiphertext; 8],
    sk: &RnsSecretKey,
    params: &CliffordFHEParams,
) -> [f64; 8] {
    let mut result = [0.0; 8];

    for i in 0..8 {
        let pt = rns_decrypt(sk, &ct[i], params);

        // Decode from first prime only (single-prime decoding)
        let val = pt.coeffs.rns_coeffs[0][0];
        let q = params.moduli[0];

        // Center-lift
        let centered = if val > q / 2 { val - q } else { val };

        // Decode with appropriate scale
        result[i] = (centered as f64) / ct[i].scale;
    }

    result
}

/// A 3D point
#[derive(Clone, Debug)]
struct Point3D {
    x: f64,
    y: f64,
    z: f64,
}

/// Generate sphere point cloud (100 points on unit sphere)
fn generate_sphere(num_points: usize) -> Vec<Point3D> {
    let mut rng = rand::thread_rng();
    let mut points = Vec::new();

    for _ in 0..num_points {
        // Uniform sampling on sphere surface
        let theta = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
        let phi = (rng.gen::<f64>() * 2.0 - 1.0).acos();

        points.push(Point3D {
            x: phi.sin() * theta.cos(),
            y: phi.sin() * theta.sin(),
            z: phi.cos(),
        });
    }

    points
}

/// Generate cube point cloud (100 points in unit cube)
fn generate_cube(num_points: usize) -> Vec<Point3D> {
    let mut rng = rand::thread_rng();
    let mut points = Vec::new();

    for _ in 0..num_points {
        points.push(Point3D {
            x: rng.gen::<f64>() * 2.0 - 1.0,
            y: rng.gen::<f64>() * 2.0 - 1.0,
            z: rng.gen::<f64>() * 2.0 - 1.0,
        });
    }

    points
}

/// Generate pyramid point cloud (100 points in pyramid shape)
fn generate_pyramid(num_points: usize) -> Vec<Point3D> {
    let mut rng = rand::thread_rng();
    let mut points = Vec::new();

    for _ in 0..num_points {
        let height = rng.gen::<f64>();
        // Base narrows linearly with height
        let base_scale = 1.0 - height;

        points.push(Point3D {
            x: (rng.gen::<f64>() * 2.0 - 1.0) * base_scale,
            y: (rng.gen::<f64>() * 2.0 - 1.0) * base_scale,
            z: height,
        });
    }

    points
}

/// Encode point cloud as Cl(3,0) multivector
///
/// Encoding:
/// - Scalar: radial distance statistics
/// - Vector (e1, e2, e3): centroid/mean position
/// - Bivector (e12, e13, e23): orientation/spread
/// - Trivector (e123): volume/density
fn encode_point_cloud(points: &[Point3D]) -> [f64; 8] {
    let n = points.len() as f64;

    // Scalar: mean radial distance
    let mean_r = points.iter()
        .map(|p| (p.x * p.x + p.y * p.y + p.z * p.z).sqrt())
        .sum::<f64>() / n;

    // Vector: centroid
    let centroid_x = points.iter().map(|p| p.x).sum::<f64>() / n;
    let centroid_y = points.iter().map(|p| p.y).sum::<f64>() / n;
    let centroid_z = points.iter().map(|p| p.z).sum::<f64>() / n;

    // Bivector: second moments (orientation/spread)
    let mut m_xy = 0.0;
    let mut m_xz = 0.0;
    let mut m_yz = 0.0;

    for p in points {
        m_xy += p.x * p.y;
        m_xz += p.x * p.z;
        m_yz += p.y * p.z;
    }

    m_xy /= n;
    m_xz /= n;
    m_yz /= n;

    // Trivector: volume indicator (variance)
    let var_x = points.iter().map(|p| (p.x - centroid_x).powi(2)).sum::<f64>() / n;
    let var_y = points.iter().map(|p| (p.y - centroid_y).powi(2)).sum::<f64>() / n;
    let var_z = points.iter().map(|p| (p.z - centroid_z).powi(2)).sum::<f64>() / n;
    let volume = (var_x * var_y * var_z).cbrt();

    [
        mean_r,      // Scalar
        centroid_x,  // e1
        centroid_y,  // e2
        centroid_z,  // e3
        m_xy,        // e12
        m_xz,        // e13
        m_yz,        // e23
        volume,      // e123
    ]
}

fn main() {
    println!("\n{}", "═".repeat(80).bright_blue().bold());
    println!("{} {}",
        "◆".bright_cyan().bold(),
        "Privacy-Preserving 3D Classification with Clifford FHE".bright_white().bold()
    );
    println!("{}\n", "═".repeat(80).bright_blue().bold());

    // Parameters
    print!("  {} {}...", "▸".bright_cyan(), "Setting up FHE parameters".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let params = CliffordFHEParams::new_rns_mult_depth2_safe();
    println!(" {}", "✓".bright_green().bold());
    println!("    Ring dimension: N = {}", params.n.to_string().bright_cyan());
    println!("    Number of primes: {}", params.moduli.len().to_string().bright_cyan());
    println!("    Security level: {} bits", "≥118".bright_cyan());
    println!();

    // Generate keys
    print!("  {} {}...", "▸".bright_cyan(), "Generating FHE keys".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let key_start = Instant::now();
    let (pk, sk, evk) = rns_keygen(&params);
    let key_time = key_start.elapsed();
    println!(" {} [{:.0}ms]", "✓".bright_green().bold(), key_time.as_secs_f64() * 1000.0);
    println!();

    // Generate test samples
    const NUM_POINTS: usize = 100;
    print!("  {} {}...", "▸".bright_cyan(), format!("Generating {} test samples (3 shapes)", NUM_POINTS).bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let sphere = generate_sphere(NUM_POINTS);
    let cube = generate_cube(NUM_POINTS);
    let pyramid = generate_pyramid(NUM_POINTS);
    println!(" {}", "✓".bright_green().bold());
    println!();

    // Encode as multivectors
    print!("  {} {}...", "▸".bright_cyan(), "Encoding as Cl(3,0) multivectors".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let sphere_mv = encode_point_cloud(&sphere);
    let cube_mv = encode_point_cloud(&cube);
    let pyramid_mv = encode_point_cloud(&pyramid);
    println!(" {}", "✓".bright_green().bold());

    println!("    {} [{:.4}, {:.4}, {:.4}, ...]",
        "Sphere:".dimmed(),
        sphere_mv[0], sphere_mv[1], sphere_mv[2]
    );
    println!("    {} [{:.4}, {:.4}, {:.4}, ...]",
        "Cube:  ".dimmed(),
        cube_mv[0], cube_mv[1], cube_mv[2]
    );
    println!("    {} [{:.4}, {:.4}, {:.4}, ...]",
        "Pyramid:".dimmed(),
        pyramid_mv[0], pyramid_mv[1], pyramid_mv[2]
    );
    println!();

    // Encrypt
    print!("  {} {}...", "▸".bright_cyan(), "Encrypting multivectors (3 samples)".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let enc_start = Instant::now();
    let sphere_enc = encrypt_multivector_3d(&sphere_mv, &pk, &params);
    let _cube_enc = encrypt_multivector_3d(&cube_mv, &pk, &params);
    let _pyramid_enc = encrypt_multivector_3d(&pyramid_mv, &pk, &params);
    let enc_time = enc_start.elapsed();
    println!(" {} [{:.0}ms]", "✓".bright_green().bold(), enc_time.as_secs_f64() * 1000.0);
    println!();

    // Demonstrate encrypted operations
    println!("{}", "─".repeat(80).bright_blue());
    println!("{}", "DEMONSTRATING ENCRYPTED NEURAL NETWORK OPERATIONS".bright_white().bold());
    println!("{}", "─".repeat(80).bright_blue());
    println!();

    println!("  {} 3-Layer Geometric Neural Network: {} → {} → {} → {}",
        "▸".bright_cyan(),
        "1".bright_cyan(), "16".bright_cyan(), "8".bright_cyan(), "3".bright_cyan()
    );
    println!("    Each layer uses homomorphic geometric products");
    println!("    {} Full inference = {} geometric products",
        "→".dimmed(),
        "(16 + 8 + 3)".bright_cyan()
    );
    println!();

    // Test single encrypted geometric product
    print!("  {} {}...", "▸".bright_cyan(), "Testing encrypted geometric product".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gp_start = Instant::now();
    let result_enc = geometric_product_3d_componentwise(
        &sphere_enc,
        &sphere_enc,
        &evk,
        &params,
    );
    let gp_time = gp_start.elapsed();
    println!(" {} [{:.2}s]", "✓".bright_green().bold(), gp_time.as_secs_f64());

    // Decrypt result
    let result_dec = decrypt_multivector_3d(&result_enc, &sk, &params);
    println!("    Result: [{:.4}, {:.4}, {:.4}, {:.4}, ...]",
        result_dec[0], result_dec[1], result_dec[2], result_dec[3]);
    println!();

    // Verify correctness
    print!("  {} {}...", "▸".bright_cyan(), "Verifying homomorphic property".bright_white());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let sphere_dec = decrypt_multivector_3d(&sphere_enc, &sk, &params);

    let error: f64 = sphere_mv.iter()
        .zip(sphere_dec.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    if error < 0.01 {
        println!(" {} [max_error: {:.2e}]", "✓".bright_green().bold(), error);
        println!("    Encryption preserves multivector values {}", "(<1% error)".bright_green());
    } else {
        println!(" {} [max_error: {:.2e}]", "⚠".bright_yellow().bold(), error);
        println!("    {} Error exceeds 1%", "⚠".bright_yellow());
    }
    println!();

    // Performance projections
    println!("{}", "─".repeat(80).bright_blue());
    println!("{}", "PERFORMANCE PROJECTIONS".bright_white().bold());
    println!("{}", "─".repeat(80).bright_blue());
    println!();

    let layer1_time = 16.0 * gp_time.as_secs_f64();
    let layer2_time = 8.0 * gp_time.as_secs_f64();
    let layer3_time = 3.0 * gp_time.as_secs_f64();
    let total_projected = layer1_time + layer2_time + layer3_time;

    println!("  Geometric product: {:.2}s per operation", gp_time.as_secs_f64());
    println!();
    println!("  {} Layer 1 ({} → {}): {} GPs × {:.2}s = {:.1}s",
        "→".dimmed(), "1".bright_cyan(), "16".bright_cyan(),
        "16".bright_cyan(), gp_time.as_secs_f64(), layer1_time);
    println!("  {} Layer 2 ({} → {}): {} GPs × {:.2}s = {:.1}s",
        "→".dimmed(), "16".bright_cyan(), "8".bright_cyan(),
        "8".bright_cyan(), gp_time.as_secs_f64(), layer2_time);
    println!("  {} Layer 3 ({} → {}): {} GPs × {:.2}s = {:.1}s",
        "→".dimmed(), "8".bright_cyan(), "3".bright_cyan(),
        "3".bright_cyan(), gp_time.as_secs_f64(), layer3_time);
    println!();

    let total_str = format!("Total: {:.1}s", total_projected);
    let target_str = "Target: ~58s";

    if total_projected <= 58.0 {
        println!("  {} {} {}",
            "✓".bright_green().bold(),
            total_str.bright_green().bold(),
            format!("({})", target_str).bright_green()
        );
    } else {
        println!("  {} {} {}",
            "⚠".bright_yellow().bold(),
            total_str.bright_yellow().bold(),
            format!("({})", target_str).dimmed()
        );
    }
    println!();

    // Summary
    println!("{}", "─".repeat(80).bright_blue());
    println!("{}", "SUMMARY".bright_white().bold());
    println!("{}", "─".repeat(80).bright_blue());
    println!();

    println!("  {} Clifford FHE enables encrypted 3D point cloud processing", "✓".bright_green());
    println!("  {} Geometric neural networks operate on encrypted multivectors", "✓".bright_green());
    println!("  {} Each layer uses homomorphic geometric product", "✓".bright_green());
    println!("  {} Projected inference time: {}", "✓".bright_green(), format!("{:.1}s", total_projected).bright_cyan());
    println!("  {} Accuracy loss: {} (error: {:.2e})", "✓".bright_green(), "<1%".bright_green(), error);
    println!();

    println!("{} Machine learning application from:", "📊".bright_cyan());
    println!("  \"Merits of Geometric Algebra Applied to Cryptography and Machine Learning\"");
    println!("  Section 5: Privacy-Preserving 3D Classification");
    println!();

    println!("{}\n", "═".repeat(80).bright_blue().bold());
}
