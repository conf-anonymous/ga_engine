//! Encrypted Accuracy Benchmark
//!
//! Validates that encrypted inference produces the same predictions as
//! plaintext across a full test set. Reports:
//! - Overall encrypted vs plaintext accuracy
//! - Per-class accuracy breakdown
//! - CKKS noise analysis (per-component error statistics)
//! - Prediction agreement rate
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v3 --example encrypted_accuracy_benchmark
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
    println!("  Encrypted Accuracy Benchmark");
    println!("  Comprehensive Encrypted vs Plaintext Validation");
    println!("========================================================================");
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // Configuration
    // ════════════════════════════════════════════════════════════════════════
    let num_classes = 5;
    let points_per_sample = 32;
    let train_samples_per_class = 40;
    let _test_samples_per_class = 10;  // Will test all of these encrypted

    // ════════════════════════════════════════════════════════════════════════
    // Step 1: Generate data and train classifier
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 1: Training classifier");
    println!("─────────────────────────────────────────────────────────────────────");

    let (train_split, test_split) = generate_synthetic_modelnet40(
        train_samples_per_class,
        points_per_sample,
        num_classes,
    );
    println!("  Train: {} samples, Test: {} samples", train_split.samples.len(), test_split.samples.len());

    let hidden_dim = 32;
    let mut classifier = GPFeatureClassifier::new(hidden_dim, num_classes);

    let (final_acc, _) = train_gp_classifier(
        &mut classifier,
        &train_split.samples,
        &test_split.samples,
        150,
        0.005,
        50,
    );
    println!("  Final plaintext test accuracy: {:.1}%", final_acc * 100.0);
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // Step 2: Setup FHE
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 2: FHE setup");
    println!("─────────────────────────────────────────────────────────────────────");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let max_batch = BatchedMultivector::max_batch_size(n);

    println!("  N={}, {} primes, max batch={}", n, params.moduli.len(), max_batch);

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());
    let rotations: Vec<i32> = (-7..=7).collect();
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);
    println!("  Keys generated");
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // Step 3: Run encrypted inference on full test set
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 3: Encrypted inference on {} test samples", test_split.samples.len());
    println!("─────────────────────────────────────────────────────────────────────");

    let num_test = test_split.samples.len();
    let mut enc_correct = 0usize;
    let mut plain_correct = 0usize;
    let mut agreement = 0usize;
    let mut per_class_enc = vec![0usize; num_classes];
    let mut per_class_plain = vec![0usize; num_classes];
    let mut per_class_total = vec![0usize; num_classes];
    let mut per_class_agree = vec![0usize; num_classes];

    // Error statistics
    let mut all_errors: Vec<f64> = Vec::new();
    let mut component_errors: Vec<Vec<f64>> = vec![Vec::new(); 8];
    let mut total_gp_ms = 0.0;

    let benchmark_start = Instant::now();

    for (idx, pc) in test_split.samples.iter().enumerate() {
        let label = pc.label.unwrap_or(0);
        let n_points = pc.points.len().min(max_batch);

        // Encrypted pipeline
        let multivectors: Vec<[f64; 8]> = pc.points.iter().take(n_points)
            .map(|p| encode_point_augmented(p.x, p.y, p.z))
            .collect();

        let batch = encode_batch(&multivectors, &ckks_ctx, &pk);
        let gp_start = Instant::now();
        let gp_result = geometric_product_batched(&batch, &batch, &rotation_keys, &evk, &ckks_ctx)?;
        total_gp_ms += gp_start.elapsed().as_secs_f64() * 1000.0;
        let decrypted = decode_batch(&gp_result, &ckks_ctx, &sk);

        let mut enc_pooled = [0.0; 8];
        for mv in &decrypted[..n_points] {
            for i in 0..8 { enc_pooled[i] += mv[i]; }
        }
        for i in 0..8 { enc_pooled[i] /= n_points as f64; }

        let enc_pred = classifier.predict(&enc_pooled);

        // Plaintext pipeline
        let plain_features = compute_gp_features(pc);
        let plain_pred = classifier.predict(&plain_features);

        // Error analysis
        let mut max_comp_error = 0.0f64;
        for i in 0..8 {
            let err = (enc_pooled[i] - plain_features[i]).abs();
            component_errors[i].push(err);
            max_comp_error = max_comp_error.max(err);
        }
        all_errors.push(max_comp_error);

        // Tallies
        if label < num_classes { per_class_total[label] += 1; }
        if enc_pred == label { enc_correct += 1; if label < num_classes { per_class_enc[label] += 1; } }
        if plain_pred == label { plain_correct += 1; if label < num_classes { per_class_plain[label] += 1; } }
        if enc_pred == plain_pred { agreement += 1; if label < num_classes { per_class_agree[label] += 1; } }

        if (idx + 1) % 10 == 0 || idx + 1 == num_test {
            println!("  [{}/{}] enc_acc={:.1}% plain_acc={:.1}% agree={:.1}% max_err={:.4}",
                idx + 1, num_test,
                enc_correct as f64 / (idx + 1) as f64 * 100.0,
                plain_correct as f64 / (idx + 1) as f64 * 100.0,
                agreement as f64 / (idx + 1) as f64 * 100.0,
                max_comp_error);
        }
    }

    let total_time = benchmark_start.elapsed();

    // ════════════════════════════════════════════════════════════════════════
    // Results
    // ════════════════════════════════════════════════════════════════════════
    println!();
    println!("========================================================================");
    println!("  BENCHMARK RESULTS");
    println!("========================================================================");
    println!();
    println!("  Overall Accuracy:");
    println!("    Encrypted:  {}/{} ({:.1}%)", enc_correct, num_test, enc_correct as f64 / num_test as f64 * 100.0);
    println!("    Plaintext:  {}/{} ({:.1}%)", plain_correct, num_test, plain_correct as f64 / num_test as f64 * 100.0);
    println!("    Agreement:  {}/{} ({:.1}%)", agreement, num_test, agreement as f64 / num_test as f64 * 100.0);
    let acc_gap = ((enc_correct as f64 - plain_correct as f64) / num_test as f64 * 100.0).abs();
    println!("    Accuracy gap: {:.1}%", acc_gap);
    println!();

    println!("  Per-Class Breakdown:");
    println!("    +-------+-------+----------+-----------+-----------+");
    println!("    | Class | Count | Enc Acc  | Plain Acc | Agreement |");
    println!("    +-------+-------+----------+-----------+-----------+");
    for c in 0..num_classes {
        if per_class_total[c] > 0 {
            println!("    | {:>5} | {:>5} | {:>6.1}%  | {:>7.1}%  | {:>7.1}%  |",
                c, per_class_total[c],
                per_class_enc[c] as f64 / per_class_total[c] as f64 * 100.0,
                per_class_plain[c] as f64 / per_class_total[c] as f64 * 100.0,
                per_class_agree[c] as f64 / per_class_total[c] as f64 * 100.0);
        }
    }
    println!("    +-------+-------+----------+-----------+-----------+");
    println!();

    // Error statistics
    all_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_error: f64 = all_errors.iter().sum::<f64>() / all_errors.len() as f64;
    let max_error = all_errors.last().copied().unwrap_or(0.0);
    let median_error = all_errors[all_errors.len() / 2];
    let p99_idx = (all_errors.len() as f64 * 0.99) as usize;
    let p99_error = all_errors[p99_idx.min(all_errors.len() - 1)];

    println!("  CKKS Noise Analysis (max component error per sample):");
    println!("    Mean:   {:.6}", mean_error);
    println!("    Median: {:.6}", median_error);
    println!("    P99:    {:.6}", p99_error);
    println!("    Max:    {:.6}", max_error);
    println!();

    println!("  Per-Component Error (mean):");
    for i in 0..8 {
        let comp_mean: f64 = component_errors[i].iter().sum::<f64>() / component_errors[i].len() as f64;
        let comp_max = component_errors[i].iter().cloned().fold(0.0f64, f64::max);
        println!("    Component {} ({}): mean={:.6}, max={:.6}",
            i, ["scalar", "e1", "e2", "e3", "e12", "e13", "e23", "e123"][i],
            comp_mean, comp_max);
    }
    println!();

    println!("  Timing:");
    println!("    Total:              {:.1}s", total_time.as_secs_f64());
    println!("    Avg GP per sample:  {:.1}ms", total_gp_ms / num_test as f64);
    println!("    Throughput:         {:.2} samples/min", num_test as f64 / total_time.as_secs_f64() * 60.0);
    println!();

    // Final verdict
    if agreement == num_test {
        println!("  PASS: 100% encrypted-plaintext prediction agreement");
    } else {
        println!("  NOTE: {:.1}% prediction agreement ({} mismatches)",
            agreement as f64 / num_test as f64 * 100.0, num_test - agreement);
    }

    if acc_gap < 1.0 {
        println!("  PASS: Accuracy gap < 1% ({:.1}%)", acc_gap);
    } else {
        println!("  NOTE: Accuracy gap = {:.1}%", acc_gap);
    }

    if max_error < 0.1 {
        println!("  PASS: Max CKKS error < 0.1 ({:.6})", max_error);
    }
    println!();

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v3")))]
fn main() {
    println!("This example requires features: v2, v3");
    println!("Run with: cargo run --release --no-default-features --features f64,nd,v2,v3 --example encrypted_accuracy_benchmark");
}
