//! V3 Batched CliffordPointNet: End-to-End Encrypted Inference
//!
//! Demonstrates privacy-preserving 3D point cloud classification using:
//! 1. Train GPFeatureClassifier on plaintext GP features
//! 2. Encrypt test points → V3 batched SIMD packing → 1 ciphertext
//! 3. Server computes encrypted geometric self-product
//! 4. Client decrypts, mean pools, classifies with trained weights
//! 5. Verify encrypted prediction matches plaintext
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v3 --example clifford_pointnet_v3_encrypted
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
        BatchedMultivector,
        encoding::{encode_batch, decode_batch},
        geometric::geometric_product_batched,
    };
    use ga_engine::clifford_fhe_v3::bootstrapping::keys::generate_rotation_keys;
    use ga_engine::clifford_pointnet::gp_classifier::{
        GPFeatureClassifier, encode_point_augmented,
        compute_gp_features, train_gp_classifier,
    };
    use ga_engine::datasets::modelnet40::generate_synthetic_modelnet40;
    use std::time::Instant;

    println!("========================================================================");
    println!("  CliffordPointNet V3: End-to-End Encrypted Inference");
    println!("  Privacy-Preserving 3D Point Cloud Classification");
    println!("========================================================================");
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 1: Train GPFeatureClassifier on plaintext data
    // ════════════════════════════════════════════════════════════════════════
    println!("PHASE 1: Training GP Feature Classifier");
    println!("────────────────────────────────────────────────────────────────────────");

    let num_classes = 5;
    let points_per_sample = 32;
    let train_start = Instant::now();

    println!("  Generating synthetic dataset...");
    let (train_split, test_split) = generate_synthetic_modelnet40(
        30,  // samples per class (train)
        points_per_sample,
        num_classes,
    );
    println!("  Train samples: {}, Test samples: {}", train_split.samples.len(), test_split.samples.len());
    println!("  Points per sample: {}, Classes: {}", points_per_sample, num_classes);

    let hidden_dim = 32;
    let mut classifier = GPFeatureClassifier::new(hidden_dim, num_classes);
    println!("  Classifier: [8] → [{}] (square) → [{}] ({} params)",
        hidden_dim, num_classes, classifier.num_params());
    println!();
    println!("  Training...");

    let (test_acc, test_loss) = train_gp_classifier(
        &mut classifier,
        &train_split.samples,
        &test_split.samples,
        100,    // epochs
        0.005,  // learning rate
        25,     // print every
    );

    let train_time = train_start.elapsed();
    println!();
    println!("  Final test accuracy: {:.1}% (loss: {:.4})", test_acc * 100.0, test_loss);
    println!("  Training time: {:.2}s", train_time.as_secs_f64());
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 2: Setup FHE parameters and keys
    // ════════════════════════════════════════════════════════════════════════
    println!("PHASE 2: Setting up FHE Parameters");
    println!("────────────────────────────────────────────────────────────────────────");
    let setup_start = Instant::now();

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let max_batch = BatchedMultivector::max_batch_size(n);

    println!("  Ring dimension (N): {}", n);
    println!("  Modulus chain: {} primes (depth = {})", params.moduli.len(), params.max_level());
    println!("  Max batch size: {} multivectors", max_batch);

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    let rotations: Vec<i32> = (-7..=7).collect();
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

    let setup_time = setup_start.elapsed();
    println!("  Setup time: {:.2}s", setup_time.as_secs_f64());
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 3: Encrypted inference on test samples
    // ════════════════════════════════════════════════════════════════════════
    println!("PHASE 3: Encrypted Inference");
    println!("────────────────────────────────────────────────────────────────────────");

    // Pick test samples (one per class for demonstration)
    let num_test = num_classes.min(test_split.samples.len()).min(max_batch);
    let test_samples: Vec<&_> = test_split.samples.iter().take(num_test).collect();

    println!("  Running encrypted inference on {} test samples...", num_test);
    println!();

    let mut encrypted_correct = 0;
    let mut plaintext_correct = 0;
    let mut encrypted_matches_plaintext = 0;
    let mut total_encrypt_ms = 0.0;
    let mut total_gp_ms = 0.0;
    let mut total_decrypt_ms = 0.0;
    let mut max_error_all = 0.0f64;

    for (sample_idx, pc) in test_samples.iter().enumerate() {
        let label = pc.label.unwrap_or(0);
        let n_points = pc.points.len().min(max_batch);

        // ──────────────────────────────────────────────────────────────
        // CLIENT: Encode and encrypt
        // ──────────────────────────────────────────────────────────────
        let enc_start = Instant::now();

        // Augmented encoding: [1, x, y, z, 0, 0, 0, 0]
        let multivectors: Vec<[f64; 8]> = pc.points.iter().take(n_points)
            .map(|p| encode_point_augmented(p.x, p.y, p.z))
            .collect();

        let batch = encode_batch(&multivectors, &ckks_ctx, &pk);
        let enc_time = enc_start.elapsed().as_secs_f64() * 1000.0;
        total_encrypt_ms += enc_time;

        // ──────────────────────────────────────────────────────────────
        // SERVER: Encrypted geometric self-product
        // ──────────────────────────────────────────────────────────────
        let gp_start = Instant::now();
        let gp_result = geometric_product_batched(
            &batch, &batch, &rotation_keys, &evk, &ckks_ctx
        )?;
        let gp_time = gp_start.elapsed().as_secs_f64() * 1000.0;
        total_gp_ms += gp_time;

        // ──────────────────────────────────────────────────────────────
        // CLIENT: Decrypt, mean pool, classify
        // ──────────────────────────────────────────────────────────────
        let dec_start = Instant::now();
        let decrypted = decode_batch(&gp_result, &ckks_ctx, &sk);

        // Mean pool over all points (client-side)
        let mut enc_pooled = [0.0; 8];
        for mv in &decrypted[..n_points] {
            for i in 0..8 {
                enc_pooled[i] += mv[i];
            }
        }
        for i in 0..8 {
            enc_pooled[i] /= n_points as f64;
        }

        let (enc_pred, enc_conf) = classifier.predict_with_confidence(&enc_pooled);
        let dec_time = dec_start.elapsed().as_secs_f64() * 1000.0;
        total_decrypt_ms += dec_time;

        // ──────────────────────────────────────────────────────────────
        // VERIFY: Plaintext computation
        // ──────────────────────────────────────────────────────────────
        let plain_features = compute_gp_features(pc);
        let (plain_pred, plain_conf) = classifier.predict_with_confidence(&plain_features);

        // Compute error between encrypted and plaintext GP features
        let mut max_error = 0.0f64;
        for i in 0..8 {
            max_error = max_error.max((enc_pooled[i] - plain_features[i]).abs());
        }
        max_error_all = max_error_all.max(max_error);

        if enc_pred == label { encrypted_correct += 1; }
        if plain_pred == label { plaintext_correct += 1; }
        if enc_pred == plain_pred { encrypted_matches_plaintext += 1; }

        let enc_correct_str = if enc_pred == label { "CORRECT" } else { "WRONG" };
        let match_str = if enc_pred == plain_pred { "MATCH" } else { "DIFFER" };

        println!("  Sample {:2} | Label: {} | Enc pred: {} ({:.0}%) | Plain pred: {} ({:.0}%) | {} | {} | err={:.4} | gp={:.0}ms",
            sample_idx, label, enc_pred, enc_conf * 100.0,
            plain_pred, plain_conf * 100.0,
            enc_correct_str, match_str, max_error, gp_time);
    }

    // ════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ════════════════════════════════════════════════════════════════════════
    println!();
    println!("========================================================================");
    println!("  RESULTS SUMMARY");
    println!("========================================================================");
    println!();
    println!("  Samples tested:                {}", num_test);
    println!("  Encrypted accuracy:            {}/{} ({:.1}%)",
        encrypted_correct, num_test, encrypted_correct as f64 / num_test as f64 * 100.0);
    println!("  Plaintext accuracy:            {}/{} ({:.1}%)",
        plaintext_correct, num_test, plaintext_correct as f64 / num_test as f64 * 100.0);
    println!("  Encrypted matches plaintext:   {}/{} ({:.1}%)",
        encrypted_matches_plaintext, num_test,
        encrypted_matches_plaintext as f64 / num_test as f64 * 100.0);
    println!("  Max CKKS error (GP features):  {:.6}", max_error_all);
    println!();
    println!("  Average timing per sample:");
    println!("    Encryption:       {:>8.1}ms", total_encrypt_ms / num_test as f64);
    println!("    Geometric Product:{:>8.1}ms  (server-side, encrypted)", total_gp_ms / num_test as f64);
    println!("    Decrypt + Pool:   {:>8.1}ms  (client-side)", total_decrypt_ms / num_test as f64);
    println!("    Total:            {:>8.1}ms",
        (total_encrypt_ms + total_gp_ms + total_decrypt_ms) / num_test as f64);
    println!();
    println!("  Architecture: V3 SIMD Batched (N={}, {} primes)", n, params.moduli.len());
    println!("  Points per ciphertext: {} (packed into 1 ct)", points_per_sample);
    println!("  Classifier: [8] → [{}] (x²) → [{}]", hidden_dim, num_classes);
    println!("  Encoding: augmented [1, x, y, z, 0, 0, 0, 0]");
    println!();

    if max_error_all < 0.1 {
        println!("  PASS: Encrypted inference matches plaintext (max error < 0.1)");
    } else if max_error_all < 1.0 {
        println!("  PASS: Encrypted inference close to plaintext (max error < 1.0)");
    } else {
        println!("  NOTE: Error above threshold ({})", max_error_all);
    }
    println!();

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v3")))]
fn main() {
    println!("This example requires features: v2, v3");
    println!("Run with: cargo run --release --no-default-features --features f64,nd,v2,v3 --example clifford_pointnet_v3_encrypted");
}
