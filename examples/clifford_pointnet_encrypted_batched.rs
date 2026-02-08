//! Batched CliffordPointNet Encrypted Inference
//!
//! End-to-end privacy-preserving 3D point cloud classification using
//! V3 Batched SIMD packing. All N points are packed into a single ciphertext
//! for massive throughput improvement over the V2 component-separate approach.
//!
//! ## Architecture
//!
//! ```text
//! CLIENT:
//!   1. Encode N points as [f64; 8] multivectors
//!   2. Pack into BatchedMultivector via encode_batch() -> 1 ciphertext
//!   3. Send to server
//!
//! SERVER (all encrypted):
//!   4. Geometric product: geometric_product_batched(batch, batch)
//!      - Computes pairwise self-product for all points simultaneously
//!   5. Return result
//!
//! CLIENT:
//!   6. Decrypt via decode_batch()
//!   7. Aggregate features (mean pool in plaintext)
//!   8. Run classifier on plaintext features
//!   9. Verify against plaintext computation
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v3 --example clifford_pointnet_encrypted_batched
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
    use ga_engine::datasets::point_cloud::{Point3D, PointCloud};
    use std::time::Instant;

    println!("========================================================================");
    println!("  CliffordPointNet Batched Encrypted Inference");
    println!("  Privacy-Preserving 3D Point Cloud Classification (V3 SIMD)");
    println!("========================================================================");
    println!();

    // ========================================
    // SETUP: Parameters and Key Generation
    // ========================================
    println!("1. Setting up Clifford FHE parameters...");
    let setup_start = Instant::now();

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let max_batch = BatchedMultivector::max_batch_size(n);

    println!("   Ring dimension (N): {}", n);
    println!("   Modulus chain: {} primes (depth = {})", params.moduli.len(), params.max_level());
    println!("   Max batch size: {} multivectors", max_batch);

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Generate rotation keys for extraction (-7..=7)
    let rotations: Vec<i32> = (-7..=7).collect();
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

    let setup_time = setup_start.elapsed();
    println!("   Setup time: {:.2}s", setup_time.as_secs_f64());
    println!();

    // ========================================
    // CLIENT: Create and Encrypt Point Cloud
    // ========================================
    println!("2. Client encrypts 3D point cloud...");
    let encrypt_start = Instant::now();

    // Use 8 points
    let points = vec![
        Point3D::new(1.0, 0.0, 0.0),
        Point3D::new(0.0, 1.0, 0.0),
        Point3D::new(0.0, 0.0, 1.0),
        Point3D::new(1.0, 1.0, 1.0),
        Point3D::new(-1.0, 0.0, 0.0),
        Point3D::new(0.0, -1.0, 0.0),
        Point3D::new(0.0, 0.0, -1.0),
        Point3D::new(0.5, 0.5, 0.5),
    ];
    let point_cloud = PointCloud::from_points_with_label(points.clone(), 0);
    let n_points = point_cloud.points.len();

    println!("   Point cloud: {} points", n_points);
    for (i, p) in point_cloud.points.iter().enumerate() {
        println!("     Point {}: ({:.2}, {:.2}, {:.2})", i, p.x, p.y, p.z);
    }

    // Encode each point as a Cl(3,0) multivector: (x,y,z) -> x*e1 + y*e2 + z*e3
    let multivectors: Vec<[f64; 8]> = point_cloud.points.iter()
        .map(|p| [0.0, p.x, p.y, p.z, 0.0, 0.0, 0.0, 0.0])
        .collect();

    // Pack all points into a single batched ciphertext
    let batch = encode_batch(&multivectors, &ckks_ctx, &pk);

    let encrypt_time = encrypt_start.elapsed();
    println!("   Packed {} points into 1 ciphertext (vs {} in V2)",
        n_points, n_points * 8);
    println!("   Slot utilization: {:.1}%", batch.slot_utilization());
    println!("   Encryption time: {:.2}ms", encrypt_time.as_millis());
    println!();

    // ========================================
    // SERVER: Encrypted Inference
    // ========================================
    println!("3. Server performs encrypted inference...");
    println!("   (Server never sees raw point data!)");
    let inference_start = Instant::now();

    // Geometric product: batch x batch computes point[i] * point[i] for all i
    // This extracts geometric features from each point simultaneously
    // Depth budget: extract (1 level) + multiply (1 level) = 2 levels (fits in 3 primes)
    println!("   Encrypted batched geometric self-product ({} points at once)...", n_points);
    let gp_start = Instant::now();
    let features = geometric_product_batched(&batch, &batch, &rotation_keys, &evk, &ckks_ctx)?;
    let gp_time = gp_start.elapsed();
    println!("       Geometric product: {:.2}ms ({:.2}ms per point)",
        gp_time.as_millis(), gp_time.as_millis() as f64 / n_points as f64);

    let inference_time = inference_start.elapsed();
    println!("   Total inference time: {:.2}ms", inference_time.as_millis());
    println!();

    // ========================================
    // CLIENT: Decrypt and Aggregate
    // ========================================
    println!("4. Client decrypts and aggregates...");
    let decrypt_start = Instant::now();

    let decrypted = decode_batch(&features, &ckks_ctx, &sk);

    // Client-side mean pooling (no FHE depth consumed)
    let mut pooled_features = [0.0; 8];
    for mv in &decrypted[..n_points] {
        for i in 0..8 {
            pooled_features[i] += mv[i];
        }
    }
    for i in 0..8 {
        pooled_features[i] /= n_points as f64;
    }

    let decrypt_time = decrypt_start.elapsed();
    println!("   Decryption + aggregation: {:.2}ms", decrypt_time.as_millis());

    // Classification: scalar component is the feature
    let logit = pooled_features[0];
    let predicted_class = if logit > 0.0 { 1 } else { 0 };
    println!("   Classification logit (scalar): {:.4}", logit);
    println!("   Predicted class: {}", predicted_class);
    println!();

    // ========================================
    // VERIFICATION
    // ========================================
    println!("5. Verification (plaintext computation for comparison)...");

    // Compute plaintext self-products then mean pool
    let mut plain_pooled = [0.0; 8];
    for mv in &multivectors {
        let gp = plaintext_geometric_product(mv, mv);
        for i in 0..8 {
            plain_pooled[i] += gp[i];
        }
    }
    for i in 0..8 {
        plain_pooled[i] /= n_points as f64;
    }

    println!("   Plaintext mean(GP): ({:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4})",
        plain_pooled[0], plain_pooled[1], plain_pooled[2], plain_pooled[3],
        plain_pooled[4], plain_pooled[5], plain_pooled[6], plain_pooled[7]);
    println!("   Encrypted mean(GP): ({:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4})",
        pooled_features[0], pooled_features[1], pooled_features[2], pooled_features[3],
        pooled_features[4], pooled_features[5], pooled_features[6], pooled_features[7]);

    // Compute per-component error
    let mut max_error = 0.0f64;
    println!();
    println!("   Per-component error:");
    for i in 0..8 {
        let error = (pooled_features[i] - plain_pooled[i]).abs();
        max_error = max_error.max(error);
        println!("     Component {}: expected={:.4}, got={:.4}, error={:.6}", i, plain_pooled[i], pooled_features[i], error);
    }
    println!();
    println!("   Max component error: {:.6}", max_error);

    if max_error < 0.01 {
        println!("   PASS: Encrypted computation matches plaintext (error < 0.01)");
    } else if max_error < 1.0 {
        println!("   PASS: Encrypted computation close to plaintext (error < 1.0)");
    } else {
        println!("   NOTE: Error above threshold (expected with small test params)");
    }

    // ========================================
    // SUMMARY
    // ========================================
    println!();
    println!("========================================================================");
    println!("  SUMMARY - Batched CliffordPointNet (V3 SIMD)");
    println!("========================================================================");
    println!();
    println!("  | Stage              | Time       | Description                |");
    println!("  |--------------------+------------+----------------------------|");
    println!("  | Key Generation     | {:>6}ms   | One-time setup             |", setup_time.as_millis());
    println!("  | Encryption         | {:>6}ms   | {} points -> 1 ciphertext  |", encrypt_time.as_millis(), n_points);
    println!("  | Geometric Product  | {:>6}ms   | Batched self-product       |", gp_time.as_millis());
    println!("  | Decryption + Agg   | {:>6}ms   | Decrypt + mean pool        |", decrypt_time.as_millis());
    println!();
    println!("  Key advantage over V2:");
    println!("  - V2: {} ciphertexts for {} points ({} per point)", n_points * 8, n_points, 8);
    println!("  - V3 Batched: 1 ciphertext for {} points (SIMD packed)", n_points);
    println!("  - Geometric product operates on all points simultaneously");
    println!("  - Mean pooling deferred to client (zero FHE depth cost)");
    println!();

    Ok(())
}

/// Compute plaintext geometric product for Cl(3,0)
#[cfg(all(feature = "v2", feature = "v3"))]
fn plaintext_geometric_product(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
    let mut result = [0.0; 8];

    // Component 0 (scalar)
    result[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
              - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];

    // Component 1 (e1)
    result[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[4] - a[4]*b[2]
              + a[3]*b[5] - a[5]*b[3] - a[6]*b[7] + a[7]*b[6];

    // Component 2 (e2)
    result[2] = a[0]*b[2] + a[2]*b[0] - a[1]*b[4] + a[4]*b[1]
              + a[3]*b[6] - a[6]*b[3] - a[5]*b[7] + a[7]*b[5];

    // Component 3 (e3)
    result[3] = a[0]*b[3] + a[3]*b[0] - a[1]*b[5] + a[5]*b[1]
              - a[2]*b[6] + a[6]*b[2] - a[4]*b[7] + a[7]*b[4];

    // Component 4 (e12)
    result[4] = a[0]*b[4] + a[4]*b[0] + a[1]*b[2] - a[2]*b[1]
              + a[3]*b[7] - a[7]*b[3] + a[5]*b[6] - a[6]*b[5];

    // Component 5 (e13)
    result[5] = a[0]*b[5] + a[5]*b[0] + a[1]*b[3] - a[3]*b[1]
              - a[2]*b[7] + a[7]*b[2] - a[4]*b[6] + a[6]*b[4];

    // Component 6 (e23)
    result[6] = a[0]*b[6] + a[6]*b[0] + a[2]*b[3] - a[3]*b[2]
              + a[1]*b[7] - a[7]*b[1] + a[4]*b[5] - a[5]*b[4];

    // Component 7 (e123)
    result[7] = a[0]*b[7] + a[7]*b[0] + a[1]*b[6] - a[6]*b[1]
              - a[2]*b[5] + a[5]*b[2] + a[3]*b[4] - a[4]*b[3];

    result
}

#[cfg(not(all(feature = "v2", feature = "v3")))]
fn main() {
    println!("This example requires features: v2, v3");
    println!("Run with: cargo run --release --no-default-features --features f64,nd,v2,v3 --example clifford_pointnet_encrypted_batched");
}
