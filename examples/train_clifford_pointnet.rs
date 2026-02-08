//! Train CliffordPointNet on ModelNet40 (or synthetic data)
//!
//! Usage:
//!   cargo run --release --example train_clifford_pointnet
//!
//! Options via environment variables:
//!   DATASET_PATH=/path/to/modelnet40  - Use real ModelNet40 data
//!   NUM_CLASSES=10                    - Number of classes (default: 10 for synthetic)
//!   NUM_POINTS=1024                   - Points per sample (default: 1024)
//!   EPOCHS=100                        - Training epochs (default: 100)
//!   BATCH_SIZE=32                     - Batch size (default: 32)
//!   QUICK=1                           - Quick mode for testing

use ga_engine::clifford_pointnet::{
    CliffordPointNet, CliffordPointNetConfig, Trainer, TrainingConfig,
};
use ga_engine::clifford_pointnet::layers::{Activation, PoolingMethod};
use ga_engine::datasets::modelnet40::generate_synthetic_modelnet40;
use std::env;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     CliffordPointNet Training                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse configuration from environment
    let quick_mode = env::var("QUICK").is_ok();
    let num_classes: usize = env::var("NUM_CLASSES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if quick_mode { 5 } else { 10 });
    let num_points: usize = env::var("NUM_POINTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if quick_mode { 256 } else { 1024 });
    let epochs: usize = env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if quick_mode { 20 } else { 100 });
    let batch_size: usize = env::var("BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if quick_mode { 16 } else { 32 });
    let samples_per_class: usize = env::var("SAMPLES_PER_CLASS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if quick_mode { 50 } else { 200 });

    println!("Configuration:");
    println!("  Mode: {}", if quick_mode { "Quick (testing)" } else { "Standard" });
    println!("  Classes: {}", num_classes);
    println!("  Points per sample: {}", num_points);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Samples per class: {}", samples_per_class);
    println!();

    // Generate synthetic dataset (replace with ModelNet40 loader for real experiments)
    println!("Generating synthetic dataset...");
    let gen_start = Instant::now();
    let (train_split, test_split) = generate_synthetic_modelnet40(
        samples_per_class,
        num_points,
        num_classes,
    );
    println!(
        "  Generated {} train, {} test samples in {:.2}s",
        train_split.len(),
        test_split.len(),
        gen_start.elapsed().as_secs_f64()
    );
    println!();

    // Create model configuration
    let model_config = if quick_mode {
        CliffordPointNetConfig {
            num_points,
            embed_dim: 16,
            num_agg_layers: 1,
            classifier_hidden: vec![64, 32],
            num_classes,
            activation: Activation::Square,
            pooling: PoolingMethod::Mean,
            use_poly_norm: true,
        }
    } else {
        CliffordPointNetConfig {
            num_points,
            embed_dim: 32,
            num_agg_layers: 2,
            classifier_hidden: vec![128, 64],
            num_classes,
            activation: Activation::Square,
            pooling: PoolingMethod::Mean,
            use_poly_norm: true,
        }
    };

    // Create model
    println!("Creating CliffordPointNet model...");
    let mut model = CliffordPointNet::new(model_config);
    model.summary();

    // Create training configuration
    let train_config = if quick_mode {
        TrainingConfig {
            learning_rate: 0.01,
            lr_decay: 0.8,
            lr_decay_epochs: 5,
            num_epochs: epochs,
            batch_size,
            epsilon: 1e-4,
            grad_clip: 1.0,
            weight_decay: 0.0,
            print_every: 5,
            eval_every: 2,
            early_stopping: 5,
            ..Default::default()
        }
    } else {
        TrainingConfig {
            learning_rate: 0.001,
            lr_decay: 0.7,
            lr_decay_epochs: 20,
            num_epochs: epochs,
            batch_size,
            epsilon: 1e-5,
            grad_clip: 1.0,
            weight_decay: 1e-4,
            print_every: 10,
            eval_every: 5,
            early_stopping: 15,
            ..Default::default()
        }
    };

    // Train
    println!("\n{}", "=".repeat(60));
    let trainer = Trainer::new(train_config);
    let metrics = trainer.train(&mut model, &train_split, &test_split);

    // Final evaluation
    println!("\n{}", "=".repeat(60));
    println!("Final Evaluation");
    println!("{}", "=".repeat(60));

    let (test_acc, test_loss) = model.evaluate(&test_split.samples);
    println!("Test Accuracy: {:.2}%", test_acc * 100.0);
    println!("Test Loss: {:.4}", test_loss);
    println!("Best Accuracy: {:.2}% (epoch {})", metrics.best_accuracy * 100.0, metrics.best_epoch);

    // Per-class accuracy
    println!("\nPer-class accuracy:");
    let mut class_correct = vec![0usize; num_classes];
    let mut class_total = vec![0usize; num_classes];

    for sample in &test_split.samples {
        let target = sample.label.unwrap_or(0);
        let pred = model.predict(sample);
        class_total[target] += 1;
        if pred == target {
            class_correct[target] += 1;
        }
    }

    for c in 0..num_classes {
        if class_total[c] > 0 {
            let acc = class_correct[c] as f64 / class_total[c] as f64;
            println!("  Class {:2}: {:.1}% ({}/{})", c, acc * 100.0, class_correct[c], class_total[c]);
        }
    }

    // Save results summary
    println!("\n{}", "=".repeat(60));
    println!("Results Summary (for EXPERIMENTS.md)");
    println!("{}", "=".repeat(60));
    println!("| Metric | Value |");
    println!("|--------|-------|");
    println!("| Plaintext Accuracy | {:.2}% |", test_acc * 100.0);
    println!("| Best Epoch | {} |", metrics.best_epoch);
    println!("| Final Train Loss | {:.4} |", metrics.train_loss.last().unwrap_or(&0.0));
    println!("| Total Parameters | {} |", model.num_params());

    // Timing info
    let total_time: f64 = metrics.epoch_times.iter().sum();
    let avg_epoch_time = total_time / metrics.epoch_times.len() as f64;
    println!("| Total Training Time | {:.1}s |", total_time);
    println!("| Avg Epoch Time | {:.2}s |", avg_epoch_time);

    println!("\nDone!");
}
