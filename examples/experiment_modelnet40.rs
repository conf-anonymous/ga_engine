//! ModelNet40 Real Dataset Experiments
//!
//! This runs experiments on the REAL ModelNet40 dataset for ECCV 2026.
//!
//! Prerequisites:
//!   1. Download ModelNet40.zip from http://modelnet.cs.princeton.edu/
//!   2. Extract to data/ModelNet40/
//!
//! Usage:
//!   cargo run --release --example experiment_modelnet40
//!
//! Environment variables:
//!   MODELNET_PATH=data/ModelNet40  - Path to extracted ModelNet40
//!   POINTS=1024                    - Points per sample (default: 1024)
//!   HIDDEN=128                     - Hidden dimension (default: 128)
//!   EPOCHS=200                     - Training epochs (default: 200)
//!   LR=0.001                       - Learning rate (default: 0.001)
//!   MAX_SAMPLES=0                  - Limit samples per class (0 = all)
//!   SEED=42                        - Random seed

use ga_engine::clifford_pointnet::{SimpleCliffordNet, train_simple_net};
use ga_engine::clifford_pointnet::serialization::SimpleCliffordNetWeights;
use ga_engine::datasets::modelnet40::{ModelNet40, ModelNet40Config, ModelNet40Format};
use ga_engine::datasets::point_cloud::Dataset;
use std::env;
use std::fs;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     CliffordPointNet on REAL ModelNet40 - ECCV 2026          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse configuration from environment
    let modelnet_path = env::var("MODELNET_PATH")
        .unwrap_or_else(|_| "data/ModelNet40".to_string());
    let num_points: usize = env::var("POINTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let hidden_dim: usize = env::var("HIDDEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let epochs: usize = env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let lr: f64 = env::var("LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.001);
    let max_samples: usize = env::var("MAX_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let seed: u64 = env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    println!("Experiment Configuration");
    println!("========================");
    println!("  ModelNet40 path: {}", modelnet_path);
    println!("  Points per sample: {}", num_points);
    println!("  Hidden dimension: {}", hidden_dim);
    println!("  Epochs: {}", epochs);
    println!("  Learning rate: {}", lr);
    println!("  Max samples per class: {}", if max_samples == 0 { "all".to_string() } else { max_samples.to_string() });
    println!("  Random seed: {}", seed);
    println!();

    // Set random seed
    // Note: Rust's rand doesn't have a global seed, but we can use it for consistent sampling

    // Load real ModelNet40 dataset
    println!("Loading ModelNet40 dataset from {}...", modelnet_path);
    let load_start = Instant::now();

    let mut config = ModelNet40Config::new(&modelnet_path)
        .with_num_points(num_points)
        .with_format(ModelNet40Format::Off);  // Original OFF mesh files

    if max_samples > 0 {
        config = config.with_max_samples(max_samples * 40);  // max_samples per class * 40 classes
    }

    let dataset = match ModelNet40::load(config) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("ERROR: Failed to load ModelNet40: {}", e);
            eprintln!();
            eprintln!("Make sure you have:");
            eprintln!("  1. Downloaded ModelNet40.zip from http://modelnet.cs.princeton.edu/");
            eprintln!("  2. Extracted it to {}/", modelnet_path);
            eprintln!("  3. Directory structure should be:");
            eprintln!("     {}/airplane/train/airplane_0001.off", modelnet_path);
            eprintln!("     {}/airplane/test/airplane_0627.off", modelnet_path);
            eprintln!("     ...");
            std::process::exit(1);
        }
    };

    let load_time = load_start.elapsed().as_secs_f64();
    println!("  Loaded in {:.2}s", load_time);

    // Print dataset statistics
    dataset.print_stats();
    println!();

    let train_samples = dataset.train().samples.clone();
    let test_samples = dataset.test().samples.clone();
    let num_classes = dataset.num_classes();

    println!("Dataset ready:");
    println!("  Train samples: {}", train_samples.len());
    println!("  Test samples: {}", test_samples.len());
    println!("  Classes: {}", num_classes);
    println!();

    // Create model
    println!("Creating SimpleCliffordNet...");
    let mut model = SimpleCliffordNet::new(hidden_dim, num_classes);
    let num_params = model.num_params();
    println!("  Parameters: {}", num_params);
    println!();

    // Initial evaluation
    let (init_acc, _) = model.evaluate(&test_samples);
    println!(
        "Initial test accuracy: {:.2}% (random chance: {:.2}%)",
        init_acc * 100.0,
        100.0 / num_classes as f64
    );
    println!();

    // Train
    println!("{}", "=".repeat(70));
    println!("Training on REAL ModelNet40...");
    println!("{}", "=".repeat(70));

    let train_start = Instant::now();
    let (final_acc, final_loss) = train_simple_net(
        &mut model,
        &train_samples,
        &test_samples,
        epochs,
        lr,
        10, // print every 10 epochs
    );
    let train_time = train_start.elapsed().as_secs_f64();

    // Final evaluation with per-class breakdown
    println!();
    println!("{}", "=".repeat(70));
    println!("FINAL RESULTS - REAL ModelNet40");
    println!("{}", "=".repeat(70));
    println!();

    // Compute overall accuracy (OA) and mean class accuracy (mAcc)
    let mut class_correct = vec![0usize; num_classes];
    let mut class_total = vec![0usize; num_classes];

    for sample in &test_samples {
        let target = sample.label.unwrap_or(0);
        let pred = model.predict(sample);
        class_total[target] += 1;
        if pred == target {
            class_correct[target] += 1;
        }
    }

    // Per-class accuracy
    let mut per_class_acc: Vec<f64> = Vec::new();
    for c in 0..num_classes {
        if class_total[c] > 0 {
            per_class_acc.push(class_correct[c] as f64 / class_total[c] as f64);
        }
    }

    let overall_accuracy = final_acc;
    let mean_class_accuracy = per_class_acc.iter().sum::<f64>() / per_class_acc.len() as f64;

    println!("| Metric | Value |");
    println!("|--------|-------|");
    println!("| **Overall Accuracy (OA)** | **{:.2}%** |", overall_accuracy * 100.0);
    println!("| **Mean Class Accuracy (mAcc)** | **{:.2}%** |", mean_class_accuracy * 100.0);
    println!("| Test Loss | {:.4} |", final_loss);
    println!("| Total Parameters | {} |", num_params);
    println!("| Total Training Time | {:.1}s |", train_time);
    println!("| Time per Epoch | {:.2}s |", train_time / epochs as f64);
    println!("| Points per Sample | {} |", num_points);
    println!("| Dataset | ModelNet40 (REAL) |");
    println!("| Train Samples | {} |", train_samples.len());
    println!("| Test Samples | {} |", test_samples.len());

    // Per-class accuracy breakdown (top 10 and bottom 10)
    println!();
    println!("Per-class accuracy (all {} classes):", num_classes);

    let class_names = dataset.class_names();
    let mut class_accs: Vec<(usize, f64)> = Vec::new();
    for c in 0..num_classes {
        if class_total[c] > 0 {
            let acc = class_correct[c] as f64 / class_total[c] as f64;
            class_accs.push((c, acc));
        }
    }
    class_accs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nBest 5 classes:");
    for (c, acc) in class_accs.iter().take(5) {
        println!("  {:20} {:.1}% ({}/{})",
            class_names[*c], acc * 100.0,
            class_correct[*c], class_total[*c]);
    }

    println!("\nWorst 5 classes:");
    for (c, acc) in class_accs.iter().rev().take(5) {
        println!("  {:20} {:.1}% ({}/{})",
            class_names[*c], acc * 100.0,
            class_correct[*c], class_total[*c]);
    }

    // Summary for EXPERIMENTS.md
    println!();
    println!("{}", "=".repeat(70));
    println!("COPY TO EXPERIMENTS.md (E1 Results):");
    println!("{}", "=".repeat(70));
    println!();
    println!("### Experiment E1: Plaintext CliffordPointNet on ModelNet40");
    println!();
    println!("| Metric | Value |");
    println!("|--------|-------|");
    println!("| Dataset | ModelNet40 (REAL) |");
    println!("| Train/Test Split | {}/{} |", train_samples.len(), test_samples.len());
    println!("| Points per Sample | {} |", num_points);
    println!("| Model | SimpleCliffordNet (hidden={}) |", hidden_dim);
    println!("| Parameters | {} |", num_params);
    println!("| Training Epochs | {} |", epochs);
    println!("| Random Seed | {} |", seed);
    println!("| **Overall Accuracy (OA)** | **{:.2}%** |", overall_accuracy * 100.0);
    println!("| **Mean Class Accuracy (mAcc)** | **{:.2}%** |", mean_class_accuracy * 100.0);
    println!("| Training Time | {:.1}s |", train_time);

    // Save model weights for encrypted inference
    let weights_path = format!("data/modelnet40_weights_seed{}.json", seed);
    println!();
    println!("Saving model weights to {}...", weights_path);
    let weights = SimpleCliffordNetWeights::from_model(&model);
    match serde_json::to_string_pretty(&weights) {
        Ok(json) => {
            if let Err(e) = fs::write(&weights_path, json) {
                eprintln!("Warning: Failed to write weights file: {}", e);
            } else {
                println!("Weights saved successfully!");
            }
        }
        Err(e) => {
            eprintln!("Warning: Failed to serialize weights: {}", e);
        }
    }

    println!();
    println!("Done!");
}
