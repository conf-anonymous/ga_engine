//! Privacy-Preserving Medical 3D Scan Classification
//!
//! Simulates a telemedicine scenario where a hospital sends an encrypted
//! organ/tissue point cloud to a cloud AI service for pathology classification.
//! The cloud never sees the patient's raw scan data.
//!
//! ## Scenario
//!
//! ```text
//! Hospital (Client):
//!   - Acquires 3D scan of tissue/organ (CT, MRI, ultrasound)
//!   - Converts to point cloud representation
//!   - Encrypts with patient-specific FHE keys
//!   - Sends encrypted scan to AI diagnostic service
//!
//! AI Diagnostic Service (Server):
//!   - Receives encrypted point cloud
//!   - Computes encrypted geometric features
//!   - Returns encrypted feature vector
//!   - HIPAA/GDPR compliant: never accesses raw patient data
//!
//! Hospital (Client):
//!   - Decrypts feature vector
//!   - Runs diagnostic classifier locally
//!   - Presents results to physician
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v3 --example medical_scan_classification
//! ```

#[cfg(all(feature = "v2", feature = "v3"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::{
        backends::cpu_optimized::{
            ckks::CkksContext,
            keys::KeyContext,
        },
        params::CliffordFHEParams,
    };
    use ga_engine::clifford_fhe_v3::batched::{
        encoding::{encode_batch, decode_batch},
        geometric::geometric_product_batched,
    };
    use ga_engine::clifford_fhe_v3::bootstrapping::keys::generate_rotation_keys;
    use ga_engine::clifford_pointnet::gp_classifier::{
        GPFeatureClassifier, encode_point_augmented,
        compute_gp_features, train_gp_classifier,
    };
    use ga_engine::datasets::modelnet40::generate_synthetic_modelnet40;
    use rand::Rng;
    use std::time::Instant;

    println!("========================================================================");
    println!("  Privacy-Preserving Medical 3D Scan Classification");
    println!("  HIPAA/GDPR Compliant Cloud Diagnostics");
    println!("========================================================================");
    println!();

    // Diagnostic categories
    let class_names = ["benign_cyst", "malignant_tumor", "healthy_tissue", "inflammation", "calcification"];
    let num_classes = class_names.len();
    let points_per_scan = 32;

    // ════════════════════════════════════════════════════════════════════════
    // Step 1: Train diagnostic model (done once with labeled historical data)
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 1: Training diagnostic classifier (historical labeled data)");
    println!("────────────────────────────────────────────────────────────────────────");

    let (train_split, test_split) = generate_synthetic_modelnet40(40, points_per_scan, num_classes);

    let hidden_dim = 32;
    let mut classifier = GPFeatureClassifier::new(hidden_dim, num_classes);

    let (acc, _) = train_gp_classifier(
        &mut classifier,
        &train_split.samples,
        &test_split.samples,
        150, 0.005, 50,
    );
    println!("  Diagnostic model accuracy: {:.1}%", acc * 100.0);
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // Step 2: FHE key generation (per-hospital or per-patient)
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 2: FHE key generation (hospital enrollment)");
    println!("────────────────────────────────────────────────────────────────────────");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());
    let rotations: Vec<i32> = (-7..=7).collect();
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);
    println!("  Hospital keys generated (N={}, {} primes)", n, params.moduli.len());
    println!("  Keys stored in hospital's secure enclave");
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // Step 3: Simulate patient scans
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 3: Processing patient scans");
    println!("────────────────────────────────────────────────────────────────────────");
    println!();

    let mut rng = rand::thread_rng();
    let num_patients = 8;

    let mut total_correct = 0;
    let mut total_high_risk = 0;
    let mut total_gp_ms = 0.0;

    let pipeline_start = Instant::now();

    for patient_id in 0..num_patients {
        let true_class = rng.gen_range(0..num_classes);
        let patient_name = format!("Patient-{:04}", 1000 + patient_id);

        // Generate synthetic scan
        let scan = generate_medical_scan(true_class, points_per_scan, &mut rng);

        println!("  {} | Scan: {} points | True condition: {}",
            patient_name, scan.points.len(), class_names[true_class]);

        // ── HOSPITAL: Encrypt scan ──
        let enc_start = Instant::now();
        let multivectors: Vec<[f64; 8]> = scan.points.iter()
            .map(|p| encode_point_augmented(p.x, p.y, p.z))
            .collect();
        let encrypted_scan = encode_batch(&multivectors, &ckks_ctx, &pk);
        let enc_ms = enc_start.elapsed().as_secs_f64() * 1000.0;

        // ── SERVER: Compute encrypted features ──
        let gp_start = Instant::now();
        let encrypted_features = geometric_product_batched(
            &encrypted_scan, &encrypted_scan, &rotation_keys, &evk, &ckks_ctx
        )?;
        let gp_ms = gp_start.elapsed().as_secs_f64() * 1000.0;
        total_gp_ms += gp_ms;

        // ── HOSPITAL: Decrypt and diagnose ──
        let dec_start = Instant::now();
        let decrypted = decode_batch(&encrypted_features, &ckks_ctx, &sk);

        let n_pts = scan.points.len().min(decrypted.len());
        let mut pooled = [0.0; 8];
        for mv in &decrypted[..n_pts] {
            for j in 0..8 { pooled[j] += mv[j]; }
        }
        for j in 0..8 { pooled[j] /= n_pts as f64; }

        let (pred, conf) = classifier.predict_with_confidence(&pooled);
        let dec_ms = dec_start.elapsed().as_secs_f64() * 1000.0;

        let correct = pred == true_class;
        if correct { total_correct += 1; }

        let risk_level = match pred {
            1 => { total_high_risk += 1; "HIGH RISK" },
            3 => "MODERATE",
            _ => "LOW",
        };

        let status = if correct { "CORRECT  " } else { "MISMATCH " };

        println!("    Diagnosis: {} ({:.0}%) | Risk: {} | {} | enc={:.0}ms gp={:.0}ms dec={:.0}ms",
            class_names[pred], conf * 100.0, risk_level, status, enc_ms, gp_ms, dec_ms);

        // Verify against plaintext
        let plain_features = compute_gp_features(&scan);
        let plain_pred = classifier.predict(&plain_features);
        let agreement = if pred == plain_pred { "agrees" } else { "DIFFERS" };
        println!("    Plaintext check: {} (encrypted {})", class_names[plain_pred], agreement);
        println!();
    }

    let pipeline_time = pipeline_start.elapsed();

    // ════════════════════════════════════════════════════════════════════════
    // Summary
    // ════════════════════════════════════════════════════════════════════════
    println!("========================================================================");
    println!("  DIAGNOSTIC SUMMARY");
    println!("========================================================================");
    println!();
    println!("  Patients processed:     {}", num_patients);
    println!("  Correct diagnoses:      {}/{} ({:.1}%)",
        total_correct, num_patients, total_correct as f64 / num_patients as f64 * 100.0);
    println!("  High-risk findings:     {}", total_high_risk);
    println!("  Avg GP time per scan:   {:.0}ms", total_gp_ms / num_patients as f64);
    println!("  Total pipeline time:    {:.1}s", pipeline_time.as_secs_f64());
    println!();
    println!("  Compliance:");
    println!("    - Patient data encrypted end-to-end");
    println!("    - Cloud server never accessed raw scan data");
    println!("    - Decryption keys held only by hospital");
    println!("    - HIPAA Safe Harbor: de-identification via encryption");
    println!("    - GDPR Art. 32: encryption as technical safeguard");
    println!();

    Ok(())
}

/// Generate a synthetic medical scan point cloud
#[cfg(all(feature = "v2", feature = "v3"))]
fn generate_medical_scan(class_idx: usize, num_points: usize, rng: &mut impl rand::Rng) -> ga_engine::datasets::point_cloud::PointCloud {
    use ga_engine::datasets::point_cloud::{Point3D, PointCloud};

    let mut points = Vec::with_capacity(num_points);
    let noise = 0.03;

    for _ in 0..num_points {
        let (x, y, z) = match class_idx {
            0 => { // benign cyst: smooth sphere
                let theta = rng.gen_range(0.0..std::f64::consts::TAU);
                let phi = rng.gen_range(0.0..std::f64::consts::PI);
                let r = 0.4 + rng.gen_range(-0.02..0.02); // very regular
                (r * phi.sin() * theta.cos(), r * phi.sin() * theta.sin(), r * phi.cos())
            }
            1 => { // malignant tumor: irregular spiky surface
                let theta = rng.gen_range(0.0..std::f64::consts::TAU);
                let phi = rng.gen_range(0.0..std::f64::consts::PI);
                let spikiness = 0.15 * (3.0 * theta).sin() * (2.0 * phi).cos();
                let r = 0.35 + spikiness + rng.gen_range(-0.05..0.05);
                (r * phi.sin() * theta.cos(), r * phi.sin() * theta.sin(), r * phi.cos())
            }
            2 => { // healthy tissue: flat disc
                let theta = rng.gen_range(0.0..std::f64::consts::TAU);
                let r = rng.gen_range(0.0..0.5);
                let h = rng.gen_range(-0.05..0.05);
                (r * theta.cos(), r * theta.sin(), h)
            }
            3 => { // inflammation: swollen ellipsoid
                let theta = rng.gen_range(0.0..std::f64::consts::TAU);
                let phi = rng.gen_range(0.0..std::f64::consts::PI);
                let rx = 0.5;
                let ry = 0.35;
                let rz = 0.45;
                (rx * phi.sin() * theta.cos(), ry * phi.sin() * theta.sin(), rz * phi.cos())
            }
            _ => { // calcification: dense cluster of small spheres
                let cx = rng.gen_range(-0.2..0.2);
                let cy = rng.gen_range(-0.2..0.2);
                let cz = rng.gen_range(-0.2..0.2);
                let r = 0.1;
                let theta = rng.gen_range(0.0..std::f64::consts::TAU);
                let phi = rng.gen_range(0.0..std::f64::consts::PI);
                (cx + r * phi.sin() * theta.cos(), cy + r * phi.sin() * theta.sin(), cz + r * phi.cos())
            }
        };

        points.push(Point3D::new(
            x + rng.gen_range(-noise..noise),
            y + rng.gen_range(-noise..noise),
            z + rng.gen_range(-noise..noise),
        ));
    }

    PointCloud::from_points_with_label(points, class_idx)
}

#[cfg(not(all(feature = "v2", feature = "v3")))]
fn main() {
    println!("This example requires features: v2, v3");
    println!("Run with: cargo run --release --no-default-features --features f64,nd,v2,v3 --example medical_scan_classification");
}
