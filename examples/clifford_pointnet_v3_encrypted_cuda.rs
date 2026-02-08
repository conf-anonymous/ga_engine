//! V3 CUDA CliffordPointNet: Production-Path Encrypted Inference
//!
//! Same pipeline as clifford_pointnet_v3_encrypted but uses CUDA-accelerated
//! V3 batched geometric product for ~30ms/product (N=1024) or ~70ms/product (N=8192).
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda --example clifford_pointnet_v3_encrypted_cuda
//! ```

#[cfg(all(feature = "v2", feature = "v3", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::{
        backends::{
            cpu_optimized::{
                ckks::CkksContext,
                keys::KeyContext,
            },
            gpu_cuda::{
                ckks::CudaCkksContext,
                device::CudaDeviceContext,
                relin_keys::CudaRelinKeys,
                rotation_keys::CudaRotationKeys,
                rotation::CudaRotationContext,
            },
        },
        params::CliffordFHEParams,
    };
    use ga_engine::clifford_fhe_v3::batched::{
        BatchedMultivector,
        cuda_batched::{
            encode_batch_cuda,
            decode_batch_cuda,
            geometric_product_batched_cuda,
        },
    };
    use ga_engine::clifford_pointnet::gp_classifier::{
        GPFeatureClassifier, encode_point_augmented,
        compute_gp_features, train_gp_classifier,
    };
    use ga_engine::datasets::modelnet40::generate_synthetic_modelnet40;
    use std::sync::Arc;
    use std::time::Instant;
    use rand::Rng;

    println!("========================================================================");
    println!("  CliffordPointNet V3 CUDA: Production-Path Encrypted Inference");
    println!("  GPU-Accelerated Privacy-Preserving 3D Classification");
    println!("========================================================================");
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 1: Train classifier (plaintext)
    // ════════════════════════════════════════════════════════════════════════
    println!("PHASE 1: Training GP Feature Classifier");
    println!("────────────────────────────────────────────────────────────────────────");

    let num_classes = 5;
    let points_per_sample = 32;

    let (train_split, test_split) = generate_synthetic_modelnet40(30, points_per_sample, num_classes);
    println!("  Train: {}, Test: {}", train_split.samples.len(), test_split.samples.len());

    let hidden_dim = 32;
    let mut classifier = GPFeatureClassifier::new(hidden_dim, num_classes);

    let (test_acc, _) = train_gp_classifier(
        &mut classifier,
        &train_split.samples,
        &test_split.samples,
        100, 0.005, 50,
    );
    println!("  Final test accuracy: {:.1}%", test_acc * 100.0);
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 2: Setup FHE + CUDA
    // ════════════════════════════════════════════════════════════════════════
    println!("PHASE 2: Setting up FHE + CUDA");
    println!("────────────────────────────────────────────────────────────────────────");
    let setup_start = Instant::now();

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();
    let max_batch = BatchedMultivector::max_batch_size(n);

    println!("  N={}, {} primes, max batch={}", n, num_primes, max_batch);

    // CPU keys (needed for encode/decode)
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // CUDA context
    println!("  Initializing CUDA...");
    let cuda_ctx = Arc::new(CudaCkksContext::new(params.clone())?);
    let device = Arc::new(CudaDeviceContext::new()?);

    // CUDA relin keys
    let mut rng = rand::thread_rng();
    let mut secret_key = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        let bit = rng.gen::<u64>() & 1;
        for prime_idx in 0..num_primes {
            secret_key[coeff_idx * num_primes + prime_idx] = bit;
        }
    }

    println!("  Generating CUDA relin keys...");
    let relin_keys = CudaRelinKeys::new(
        device.clone(), params.clone(), secret_key.clone(), 16,
    )?;

    // CUDA rotation keys
    println!("  Generating CUDA rotation keys...");
    let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);
    let mut cuda_rot_keys = CudaRotationKeys::new(
        device.clone(), params.clone(), rotation_ctx.clone(), secret_key.clone(), 16,
    )?;
    for rot in 1..=7 {
        cuda_rot_keys.generate_rotation_key_gpu(rot, cuda_ctx.ntt_contexts())?;
        cuda_rot_keys.generate_rotation_key_gpu(-rot, cuda_ctx.ntt_contexts())?;
    }

    let setup_time = setup_start.elapsed();
    println!("  Setup time: {:.2}s ({} rotation keys)", setup_time.as_secs_f64(), cuda_rot_keys.num_keys());
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 3: CUDA encrypted inference
    // ════════════════════════════════════════════════════════════════════════
    println!("PHASE 3: CUDA Encrypted Inference");
    println!("────────────────────────────────────────────────────────────────────────");

    let num_test = num_classes.min(test_split.samples.len()).min(max_batch);
    let test_samples: Vec<&_> = test_split.samples.iter().take(num_test).collect();

    println!("  Testing {} samples with CUDA GP...", num_test);
    println!();

    // Warmup
    {
        let warmup_mvs: Vec<[f64; 8]> = vec![[1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]; 4];
        let warmup_batch = encode_batch_cuda(&warmup_mvs, &ckks_ctx, &pk);
        let _ = geometric_product_batched_cuda(
            &warmup_batch, &warmup_batch, &relin_keys, &cuda_rot_keys, &cuda_ctx
        )?;
        println!("  CUDA warmup complete");
    }

    let mut enc_correct = 0;
    let mut plain_correct = 0;
    let mut agreement = 0;
    let mut total_enc_ms = 0.0;
    let mut total_gp_ms = 0.0;
    let mut total_dec_ms = 0.0;
    let mut max_error_all = 0.0f64;

    for (idx, pc) in test_samples.iter().enumerate() {
        let label = pc.label.unwrap_or(0);
        let n_points = pc.points.len().min(max_batch);

        // CLIENT: Encode + encrypt to CUDA
        let enc_start = Instant::now();
        let multivectors: Vec<[f64; 8]> = pc.points.iter().take(n_points)
            .map(|p| encode_point_augmented(p.x, p.y, p.z))
            .collect();
        let cuda_batch = encode_batch_cuda(&multivectors, &ckks_ctx, &pk);
        let enc_time = enc_start.elapsed().as_secs_f64() * 1000.0;
        total_enc_ms += enc_time;

        // SERVER: CUDA geometric self-product
        let gp_start = Instant::now();
        let gp_result = geometric_product_batched_cuda(
            &cuda_batch, &cuda_batch, &relin_keys, &cuda_rot_keys, &cuda_ctx
        )?;
        let gp_time = gp_start.elapsed().as_secs_f64() * 1000.0;
        total_gp_ms += gp_time;

        // CLIENT: Decrypt + mean pool + classify
        let dec_start = Instant::now();
        let decrypted = decode_batch_cuda(&gp_result, &ckks_ctx, &sk, &params.moduli);

        let mut enc_pooled = [0.0; 8];
        for mv in &decrypted[..n_points] {
            for i in 0..8 { enc_pooled[i] += mv[i]; }
        }
        for i in 0..8 { enc_pooled[i] /= n_points as f64; }

        let (enc_pred, enc_conf) = classifier.predict_with_confidence(&enc_pooled);
        let dec_time = dec_start.elapsed().as_secs_f64() * 1000.0;
        total_dec_ms += dec_time;

        // VERIFY: Plaintext
        let plain_features = compute_gp_features(pc);
        let (plain_pred, plain_conf) = classifier.predict_with_confidence(&plain_features);

        let mut max_error = 0.0f64;
        for i in 0..8 {
            max_error = max_error.max((enc_pooled[i] - plain_features[i]).abs());
        }
        max_error_all = max_error_all.max(max_error);

        if enc_pred == label { enc_correct += 1; }
        if plain_pred == label { plain_correct += 1; }
        if enc_pred == plain_pred { agreement += 1; }

        let status = if enc_pred == label { "CORRECT" } else { "WRONG  " };
        let match_str = if enc_pred == plain_pred { "MATCH" } else { "DIFFER" };

        println!("  Sample {:2} | Label: {} | Enc: {} ({:.0}%) | Plain: {} ({:.0}%) | {} | {} | err={:.4} | gp={:.0}ms",
            idx, label, enc_pred, enc_conf*100.0, plain_pred, plain_conf*100.0,
            status, match_str, max_error, gp_time);
    }

    // ════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ════════════════════════════════════════════════════════════════════════
    println!();
    println!("========================================================================");
    println!("  CUDA RESULTS SUMMARY");
    println!("========================================================================");
    println!();
    println!("  Samples:     {}", num_test);
    println!("  Encrypted:   {}/{} ({:.1}%)", enc_correct, num_test, enc_correct as f64/num_test as f64*100.0);
    println!("  Plaintext:   {}/{} ({:.1}%)", plain_correct, num_test, plain_correct as f64/num_test as f64*100.0);
    println!("  Agreement:   {}/{} ({:.1}%)", agreement, num_test, agreement as f64/num_test as f64*100.0);
    println!("  Max error:   {:.6}", max_error_all);
    println!();
    println!("  Timing (per sample):");
    println!("    Encrypt:  {:>8.1}ms  (CPU encode + CPU→CUDA)", total_enc_ms / num_test as f64);
    println!("    CUDA GP:  {:>8.1}ms  (GPU geometric product)", total_gp_ms / num_test as f64);
    println!("    Decrypt:  {:>8.1}ms  (CUDA→CPU + decode + classify)", total_dec_ms / num_test as f64);
    println!("    Total:    {:>8.1}ms", (total_enc_ms + total_gp_ms + total_dec_ms) / num_test as f64);
    println!();

    let gp_per_product = total_gp_ms / num_test as f64;
    let meets = gp_per_product < 50.0;
    println!("  Production target: <50ms per geometric product");
    println!("  CUDA GP: {:.1}ms - {}", gp_per_product,
        if meets { "MEETS TARGET" } else { "ABOVE TARGET (single-sample, not batched)" });
    println!();

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v3", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires features: v2, v3, v2-gpu-cuda");
    println!("Run with: cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda --example clifford_pointnet_v3_encrypted_cuda");
}
