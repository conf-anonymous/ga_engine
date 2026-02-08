//! Privacy-Preserving 3D Object Detection
//!
//! Simulates an autonomous vehicle scenario where a client sends an encrypted
//! LiDAR scan containing multiple objects. The server classifies each object
//! without ever seeing the raw point cloud data.
//!
//! ## Scenario
//!
//! ```text
//! Autonomous Vehicle (Client):
//!   - Captures LiDAR scan of a scene
//!   - Segments scene into object clusters
//!   - Encrypts each cluster separately
//!   - Sends encrypted clusters to cloud server
//!
//! Cloud Server (Untrusted):
//!   - Receives encrypted point clouds
//!   - Computes encrypted geometric features via V3 Batched GP
//!   - Returns encrypted feature vectors
//!
//! Vehicle (Client):
//!   - Decrypts feature vectors
//!   - Classifies each object using local classifier
//!   - Makes driving decisions based on detected objects
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v3 --example privacy_preserving_detection
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
        train_gp_classifier,
    };
    use ga_engine::datasets::modelnet40::generate_synthetic_modelnet40;
    use ga_engine::datasets::point_cloud::{Point3D, PointCloud};
    use rand::Rng;
    use std::time::Instant;

    println!("========================================================================");
    println!("  Privacy-Preserving 3D Object Detection");
    println!("  Autonomous Vehicle LiDAR Classification Demo");
    println!("========================================================================");
    println!();

    // Object class names for the AV scenario
    let class_names = ["car", "pedestrian", "cyclist", "truck", "barrier"];
    let num_classes = class_names.len();
    let points_per_object = 32;

    // ════════════════════════════════════════════════════════════════════════
    // Step 1: Train object classifier (offline, once)
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 1: Training object classifier (offline)");
    println!("────────────────────────────────────────────────────────────────────────");

    let (train_split, test_split) = generate_synthetic_modelnet40(40, points_per_object, num_classes);

    let hidden_dim = 32;
    let mut classifier = GPFeatureClassifier::new(hidden_dim, num_classes);

    let (acc, _) = train_gp_classifier(
        &mut classifier,
        &train_split.samples,
        &test_split.samples,
        150, 0.005, 50,
    );
    println!("  Classifier accuracy: {:.1}%", acc * 100.0);
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // Step 2: Setup FHE (one-time key generation per vehicle)
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 2: FHE key generation (one-time per vehicle)");
    println!("────────────────────────────────────────────────────────────────────────");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let max_batch = BatchedMultivector::max_batch_size(n);

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());
    let rotations: Vec<i32> = (-7..=7).collect();
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);
    println!("  Keys generated (N={}, {} primes)", n, params.moduli.len());
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // Step 3: Simulate a LiDAR scene with multiple objects
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 3: Simulating LiDAR scene");
    println!("────────────────────────────────────────────────────────────────────────");

    let mut rng = rand::thread_rng();

    // Generate a scene with 3-6 objects at different positions
    let num_objects = rng.gen_range(3..=6).min(max_batch / points_per_object);
    let mut scene_objects: Vec<(PointCloud, [f64; 3], usize)> = Vec::new();

    for i in 0..num_objects {
        let class_idx = rng.gen_range(0..num_classes);

        // Place object at a random position in the scene (simulating world coordinates)
        let pos_x = rng.gen_range(-20.0..20.0);
        let pos_y = rng.gen_range(-10.0..10.0);
        let pos_z = rng.gen_range(0.0..3.0);
        let position = [pos_x, pos_y, pos_z];

        // Generate a synthetic object and offset it
        let base_pc = generate_object(class_idx, points_per_object, &mut rng);
        let offset_points: Vec<Point3D> = base_pc.points.iter()
            .map(|p| Point3D::new(p.x + pos_x, p.y + pos_y, p.z + pos_z))
            .collect();
        let pc = PointCloud::from_points_with_class(
            offset_points, class_idx, class_names[class_idx].to_string()
        );

        println!("  Object {}: {} at ({:.1}, {:.1}, {:.1}) [{} points]",
            i, class_names[class_idx], pos_x, pos_y, pos_z, points_per_object);

        scene_objects.push((pc, position, class_idx));
    }
    println!("  Total objects in scene: {}", num_objects);
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // Step 4: CLIENT - Segment, center, encrypt each object
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 4: Client encrypts segmented objects");
    println!("────────────────────────────────────────────────────────────────────────");

    let total_start = Instant::now();
    let mut encrypted_objects = Vec::new();

    for (i, (pc, position, _)) in scene_objects.iter().enumerate() {
        let enc_start = Instant::now();

        // Client centers each object cluster before encryption (removes position info)
        let centered_points: Vec<Point3D> = pc.points.iter()
            .map(|p| Point3D::new(p.x - position[0], p.y - position[1], p.z - position[2]))
            .collect();

        // Encode as augmented multivectors
        let multivectors: Vec<[f64; 8]> = centered_points.iter()
            .map(|p| encode_point_augmented(p.x, p.y, p.z))
            .collect();

        // Encrypt into V3 batched ciphertext
        let batch = encode_batch(&multivectors, &ckks_ctx, &pk);

        let enc_ms = enc_start.elapsed().as_secs_f64() * 1000.0;
        println!("  Object {}: encrypted in {:.1}ms (1 ciphertext)", i, enc_ms);
        encrypted_objects.push(batch);
    }
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // Step 5: SERVER - Compute encrypted geometric features
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 5: Server computes encrypted features (never sees raw data)");
    println!("────────────────────────────────────────────────────────────────────────");

    let mut encrypted_results = Vec::new();

    for (i, batch) in encrypted_objects.iter().enumerate() {
        let gp_start = Instant::now();
        let gp_result = geometric_product_batched(batch, batch, &rotation_keys, &evk, &ckks_ctx)?;
        let gp_ms = gp_start.elapsed().as_secs_f64() * 1000.0;
        println!("  Object {}: GP computed in {:.0}ms", i, gp_ms);
        encrypted_results.push(gp_result);
    }
    println!();

    // ════════════════════════════════════════════════════════════════════════
    // Step 6: CLIENT - Decrypt, classify, make driving decisions
    // ════════════════════════════════════════════════════════════════════════
    println!("Step 6: Client decrypts and classifies");
    println!("────────────────────────────────────────────────────────────────────────");
    println!();

    let mut detections = Vec::new();

    for (i, (gp_result, (_, position, true_class))) in encrypted_results.iter().zip(scene_objects.iter()).enumerate() {
        // Decrypt
        let decrypted = decode_batch(gp_result, &ckks_ctx, &sk);

        // Mean pool
        let n_points = points_per_object.min(decrypted.len());
        let mut pooled = [0.0; 8];
        for mv in &decrypted[..n_points] {
            for j in 0..8 { pooled[j] += mv[j]; }
        }
        for j in 0..8 { pooled[j] /= n_points as f64; }

        // Classify
        let (pred, conf) = classifier.predict_with_confidence(&pooled);
        let correct = pred == *true_class;

        println!("  Object {} at ({:>5.1}, {:>5.1}, {:>4.1}):", i, position[0], position[1], position[2]);
        println!("    Detected:  {} ({:.0}% confidence)", class_names[pred], conf * 100.0);
        println!("    Actual:    {}", class_names[*true_class]);
        println!("    Status:    {}", if correct { "CORRECT" } else { "MISCLASSIFIED" });
        println!();

        detections.push((pred, conf, *true_class, *position));
    }

    let total_time = total_start.elapsed();

    // ════════════════════════════════════════════════════════════════════════
    // Step 7: Driving decision summary
    // ════════════════════════════════════════════════════════════════════════
    println!("========================================================================");
    println!("  DETECTION SUMMARY");
    println!("========================================================================");
    println!();

    let correct_count = detections.iter().filter(|(p, _, t, _)| p == t).count();
    println!("  Scene: {} objects detected, {}/{} correctly classified",
        num_objects, correct_count, num_objects);
    println!();

    // Driving safety analysis
    let pedestrians = detections.iter()
        .filter(|(pred, _, _, _)| *pred == 1) // pedestrian class
        .count();
    let cyclists = detections.iter()
        .filter(|(pred, _, _, _)| *pred == 2) // cyclist class
        .count();
    let vehicles = detections.iter()
        .filter(|(pred, _, _, _)| *pred == 0 || *pred == 3) // car or truck
        .count();

    println!("  Driving Decision:");
    if pedestrians > 0 || cyclists > 0 {
        println!("    CAUTION: {} vulnerable road users detected", pedestrians + cyclists);
        println!("    Action: Reduce speed, increase following distance");
    }
    if vehicles > 0 {
        println!("    {} vehicles in scene - maintain safe distance", vehicles);
    }
    println!();

    println!("  Privacy Guarantee:");
    println!("    - Server processed {} encrypted objects", num_objects);
    println!("    - Raw LiDAR data NEVER left the vehicle");
    println!("    - Object positions NEVER exposed to server");
    println!("    - Classification done client-side after decryption");
    println!();
    println!("  Timing: {:.2}s total ({:.0}ms per object)",
        total_time.as_secs_f64(), total_time.as_secs_f64() * 1000.0 / num_objects as f64);
    println!();

    Ok(())
}

/// Generate a synthetic object point cloud for a given class
#[cfg(all(feature = "v2", feature = "v3"))]
fn generate_object(class_idx: usize, num_points: usize, rng: &mut impl rand::Rng) -> ga_engine::datasets::point_cloud::PointCloud {
    use ga_engine::datasets::point_cloud::{Point3D, PointCloud};

    let mut points = Vec::with_capacity(num_points);
    let noise = 0.05;

    for _ in 0..num_points {
        let (x, y, z) = match class_idx {
            0 => { // car: elongated box ~4.5m x 1.8m x 1.5m
                let lx = rng.gen_range(-2.25..2.25);
                let ly = rng.gen_range(-0.9..0.9);
                let lz = rng.gen_range(0.0..1.5);
                (lx * 0.3, ly * 0.3, lz * 0.3) // scale to unit range
            }
            1 => { // pedestrian: tall thin cylinder ~0.5m x 0.5m x 1.7m
                let theta = rng.gen_range(0.0..std::f64::consts::TAU);
                let r = 0.15;
                let h = rng.gen_range(0.0..1.7);
                (r * theta.cos(), r * theta.sin(), h * 0.3)
            }
            2 => { // cyclist: medium shape ~1.8m x 0.5m x 1.5m
                let lx = rng.gen_range(-0.9..0.9);
                let ly = rng.gen_range(-0.25..0.25);
                let lz = rng.gen_range(0.0..1.5);
                (lx * 0.3, ly * 0.3, lz * 0.3)
            }
            3 => { // truck: large elongated box ~8m x 2.5m x 3m
                let lx = rng.gen_range(-4.0..4.0);
                let ly = rng.gen_range(-1.25..1.25);
                let lz = rng.gen_range(0.0..3.0);
                (lx * 0.15, ly * 0.15, lz * 0.15)
            }
            _ => { // barrier: low wide box ~3m x 0.3m x 0.8m
                let lx = rng.gen_range(-1.5..1.5);
                let ly = rng.gen_range(-0.15..0.15);
                let lz = rng.gen_range(0.0..0.8);
                (lx * 0.3, ly * 0.3, lz * 0.3)
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
    println!("Run with: cargo run --release --no-default-features --features f64,nd,v2,v3 --example privacy_preserving_detection");
}
