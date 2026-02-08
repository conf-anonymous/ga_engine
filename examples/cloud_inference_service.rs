//! Cloud Inference Service Simulation
//!
//! Simulates a multi-client privacy-preserving inference service where
//! multiple clients submit encrypted 3D point clouds for classification.
//! Measures throughput, latency, and scalability.
//!
//! ## Scenario
//!
//! ```text
//! Client A ──→ Encrypted Scan ──→ ┌───────────────────┐
//! Client B ──→ Encrypted Scan ──→ │  Cloud Inference   │ ──→ Encrypted Results
//! Client C ──→ Encrypted Scan ──→ │  Service (GPU)     │ ──→ Encrypted Results
//! Client D ──→ Encrypted Scan ──→ └───────────────────┘ ──→ Encrypted Results
//!
//! Each client has their own FHE keys.
//! Server processes requests sequentially (or batched).
//! Server never sees any client's raw data.
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v3 --example cloud_inference_service
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
    use ga_engine::datasets::point_cloud::{Point3D, PointCloud};
    use rand::Rng;
    use std::time::Instant;

    println!("========================================================================");
    println!("  Cloud Inference Service Simulation");
    println!("  Multi-Client Privacy-Preserving 3D Classification");
    println!("========================================================================");
    println!();

    let class_names = ["airplane", "chair", "table", "car", "lamp"];
    let num_classes = class_names.len();
    let points_per_sample = 32;
    let num_clients = 4;
    let requests_per_client = 3;

    // ════════════════════════════════════════════════════════════════════════
    // Step 1: Service setup (shared classifier, per-client keys)
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 1: Service initialization");
    println!("────────────────────────────────────────────────────────────────────────");

    // Train shared classifier model
    let (train_split, test_split) = generate_synthetic_modelnet40(40, points_per_sample, num_classes);
    let hidden_dim = 32;
    let mut classifier = GPFeatureClassifier::new(hidden_dim, num_classes);
    let (acc, _) = train_gp_classifier(
        &mut classifier,
        &train_split.samples,
        &test_split.samples,
        150, 0.005, 50,
    );
    println!("  Service model accuracy: {:.1}%", acc * 100.0);

    // FHE parameters (shared across all clients)
    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    println!("  FHE params: N={}, {} primes", n, params.moduli.len());
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // Step 2: Client enrollment (each client generates own keys)
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 2: Client enrollment ({} clients)", num_clients);
    println!("────────────────────────────────────────────────────────────────────────");

    struct ClientContext {
        name: String,
        pk: ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey,
        sk: ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
        evk: ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::EvaluationKey,
        rotation_keys: ga_engine::clifford_fhe_v3::bootstrapping::keys::RotationKeys,
        ckks_ctx: CkksContext,
    }

    let mut clients: Vec<ClientContext> = Vec::new();
    let enroll_start = Instant::now();

    for i in 0..num_clients {
        let name = format!("Client-{}", (b'A' + i as u8) as char);
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, evk) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());
        let rotations: Vec<i32> = (-7..=7).collect();
        let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

        println!("  {} enrolled (keys generated)", name);
        clients.push(ClientContext { name, pk, sk, evk, rotation_keys, ckks_ctx });
    }

    let enroll_time = enroll_start.elapsed();
    println!("  Enrollment time: {:.2}s ({:.0}ms per client)",
        enroll_time.as_secs_f64(), enroll_time.as_secs_f64() * 1000.0 / num_clients as f64);
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // Step 3: Process inference requests
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 3: Processing inference requests");
    println!("────────────────────────────────────────────────────────────────────────");
    println!();

    let mut rng = rand::thread_rng();
    let total_requests = num_clients * requests_per_client;

    // Generate request queue (interleaved from different clients)
    struct InferenceRequest {
        client_idx: usize,
        point_cloud: PointCloud,
    }

    let mut request_queue: Vec<InferenceRequest> = Vec::new();
    for req in 0..requests_per_client {
        for client_idx in 0..num_clients {
            let class_idx = rng.gen_range(0..num_classes);
            let pc = test_split.samples
                .iter()
                .filter(|s| s.label == Some(class_idx))
                .nth(req % 2)
                .cloned()
                .unwrap_or_else(|| {
                    let points: Vec<Point3D> = (0..points_per_sample)
                        .map(|_| Point3D::new(
                            rng.gen_range(-1.0..1.0),
                            rng.gen_range(-1.0..1.0),
                            rng.gen_range(-1.0..1.0),
                        ))
                        .collect();
                    PointCloud::from_points_with_label(points, class_idx)
                });
            request_queue.push(InferenceRequest { client_idx, point_cloud: pc });
        }
    }

    // Process requests
    let mut results: Vec<(String, usize, usize, f64, f64, f64, f64, bool)> = Vec::new();
    let service_start = Instant::now();

    for (req_idx, request) in request_queue.iter().enumerate() {
        let client = &clients[request.client_idx];
        let true_class = request.point_cloud.label.unwrap_or(0);

        // CLIENT: Encrypt
        let enc_start = Instant::now();
        let multivectors: Vec<[f64; 8]> = request.point_cloud.points.iter()
            .take(points_per_sample)
            .map(|p| encode_point_augmented(p.x, p.y, p.z))
            .collect();
        let encrypted = encode_batch(&multivectors, &client.ckks_ctx, &client.pk);
        let enc_ms = enc_start.elapsed().as_secs_f64() * 1000.0;

        // SERVER: Compute encrypted GP (using client's eval keys)
        let gp_start = Instant::now();
        let gp_result = geometric_product_batched(
            &encrypted, &encrypted, &client.rotation_keys, &client.evk, &client.ckks_ctx
        )?;
        let gp_ms = gp_start.elapsed().as_secs_f64() * 1000.0;

        // CLIENT: Decrypt and classify
        let dec_start = Instant::now();
        let decrypted = decode_batch(&gp_result, &client.ckks_ctx, &client.sk);

        let n_pts = multivectors.len().min(decrypted.len());
        let mut pooled = [0.0; 8];
        for mv in &decrypted[..n_pts] {
            for j in 0..8 { pooled[j] += mv[j]; }
        }
        for j in 0..8 { pooled[j] /= n_pts as f64; }

        let (pred, conf) = classifier.predict_with_confidence(&pooled);
        let dec_ms = dec_start.elapsed().as_secs_f64() * 1000.0;

        // Verify
        let plain_features = compute_gp_features(&request.point_cloud);
        let plain_pred = classifier.predict(&plain_features);
        let agreement = pred == plain_pred;

        let correct = pred == true_class;
        let total_ms = enc_ms + gp_ms + dec_ms;

        println!("  Req {:2} | {} | {} -> {} ({:.0}%) | {:.0}ms | {}",
            req_idx + 1, client.name, class_names[true_class], class_names[pred],
            conf * 100.0, total_ms,
            if correct { "OK" } else { "MISS" });

        results.push((client.name.clone(), true_class, pred, conf, enc_ms, gp_ms, dec_ms, agreement));
    }

    let service_time = service_start.elapsed();

    // ════════════════════════════════════════════════════════════════════════
    // Summary
    // ════════════════════════════════════════════════════════════════════════
    println!();
    println!("========================================================================");
    println!("  SERVICE METRICS");
    println!("========================================================================");
    println!();

    // Per-client accuracy
    println!("  Per-Client Results:");
    println!("    +-----------+------+----------+-----------+");
    println!("    | Client    | Reqs | Accuracy | Agreement |");
    println!("    +-----------+------+----------+-----------+");
    for i in 0..num_clients {
        let client_results: Vec<_> = results.iter().filter(|r| r.0 == clients[i].name).collect();
        let correct = client_results.iter().filter(|r| r.2 == r.1).count();
        let agree = client_results.iter().filter(|r| r.7).count();
        let total = client_results.len();
        println!("    | {:>9} | {:>4} | {:>6.1}%  | {:>7.1}%  |",
            clients[i].name, total,
            correct as f64 / total as f64 * 100.0,
            agree as f64 / total as f64 * 100.0);
    }
    println!("    +-----------+------+----------+-----------+");
    println!();

    // Overall stats
    let total_correct = results.iter().filter(|r| r.2 == r.1).count();
    let total_agree = results.iter().filter(|r| r.7).count();
    let avg_enc: f64 = results.iter().map(|r| r.4).sum::<f64>() / total_requests as f64;
    let avg_gp: f64 = results.iter().map(|r| r.5).sum::<f64>() / total_requests as f64;
    let avg_dec: f64 = results.iter().map(|r| r.6).sum::<f64>() / total_requests as f64;
    let avg_total = avg_enc + avg_gp + avg_dec;

    println!("  Overall:");
    println!("    Requests processed:  {}", total_requests);
    println!("    Accuracy:            {}/{} ({:.1}%)",
        total_correct, total_requests, total_correct as f64 / total_requests as f64 * 100.0);
    println!("    Enc/Plain agreement: {}/{} ({:.1}%)",
        total_agree, total_requests, total_agree as f64 / total_requests as f64 * 100.0);
    println!();

    println!("  Latency (per request):");
    println!("    Encryption:    {:>8.1}ms", avg_enc);
    println!("    GP (server):   {:>8.1}ms", avg_gp);
    println!("    Decryption:    {:>8.1}ms", avg_dec);
    println!("    End-to-end:    {:>8.1}ms", avg_total);
    println!();

    let throughput_per_sec = total_requests as f64 / service_time.as_secs_f64();
    let throughput_per_min = throughput_per_sec * 60.0;
    println!("  Throughput:");
    println!("    Wall time:         {:.1}s", service_time.as_secs_f64());
    println!("    Requests/second:   {:.2}", throughput_per_sec);
    println!("    Requests/minute:   {:.1}", throughput_per_min);
    println!();

    println!("  Isolation Guarantee:");
    println!("    - Each client has independent FHE keys");
    println!("    - Client A's data cannot be decrypted by Client B");
    println!("    - Server processes all requests without data access");
    println!("    - No cross-client information leakage possible");
    println!();

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v3")))]
fn main() {
    println!("This example requires features: v2, v3");
    println!("Run with: cargo run --release --no-default-features --features f64,nd,v2,v3 --example cloud_inference_service");
}
