//! Plaintext CliffordPointNet Experiments
//!
//! This runs the full plaintext experiments to establish accuracy baselines.
//!
//! Usage:
//!   cargo run --release --example experiment_plaintext
//!
//! Environment variables:
//!   CLASSES=40         - Number of classes (default: 40 for ModelNet40)
//!   POINTS=1024        - Points per sample (default: 1024)
//!   SAMPLES=100        - Samples per class for training (default: 100)
//!   HIDDEN=128         - Hidden dimension (default: 128)
//!   EPOCHS=150         - Training epochs (default: 150)
//!   LR=0.005           - Learning rate (default: 0.005)

use ga_engine::clifford_pointnet::{SimpleCliffordNet, train_simple_net};
use ga_engine::datasets::modelnet40::generate_synthetic_modelnet40;
use std::env;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     CliffordPointNet Plaintext Experiments                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse configuration from environment
    let num_classes: usize = env::var("CLASSES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let num_points: usize = env::var("POINTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let samples_per_class: usize = env::var("SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let hidden_dim: usize = env::var("HIDDEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let epochs: usize = env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    let lr: f64 = env::var("LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.005);

    println!("Experiment Configuration");
    println!("========================");
    println!("  Classes: {} (ModelNet40 = 40)", num_classes);
    println!("  Points per sample: {} (standard = 1024)", num_points);
    println!("  Samples per class: {} (train)", samples_per_class);
    println!("  Hidden dimension: {}", hidden_dim);
    println!("  Epochs: {}", epochs);
    println!("  Learning rate: {}", lr);
    println!();

    // Generate dataset
    println!("Generating synthetic dataset...");
    let gen_start = Instant::now();
    let (train_split, test_split) = generate_synthetic_modelnet40(
        samples_per_class,
        num_points,
        num_classes,
    );
    let gen_time = gen_start.elapsed().as_secs_f64();
    println!(
        "  Generated {} train, {} test samples in {:.2}s",
        train_split.len(),
        test_split.len(),
        gen_time
    );
    println!();

    // Create model
    println!("Creating SimpleCliffordNet...");
    let mut model = SimpleCliffordNet::new(hidden_dim, num_classes);
    let num_params = model.num_params();
    println!("  Parameters: {}", num_params);
    println!();

    // Initial evaluation
    let (init_acc, _) = model.evaluate(&test_split.samples);
    println!(
        "Initial test accuracy: {:.2}% (random chance: {:.2}%)",
        init_acc * 100.0,
        100.0 / num_classes as f64
    );
    println!();

    // Train
    println!("{}", "=".repeat(70));
    println!("Training...");
    println!("{}", "=".repeat(70));

    let train_start = Instant::now();
    let (final_acc, final_loss) = train_simple_net(
        &mut model,
        &train_split.samples,
        &test_split.samples,
        epochs,
        lr,
        10, // print every 10 epochs
    );
    let train_time = train_start.elapsed().as_secs_f64();

    // Final evaluation
    println!();
    println!("{}", "=".repeat(70));
    println!("FINAL RESULTS");
    println!("{}", "=".repeat(70));
    println!();
    println!("| Metric | Value |");
    println!("|--------|-------|");
    println!("| Test Accuracy | {:.2}% |", final_acc * 100.0);
    println!("| Test Loss | {:.4} |", final_loss);
    println!("| Total Parameters | {} |", num_params);
    println!("| Total Training Time | {:.1}s |", train_time);
    println!("| Time per Epoch | {:.2}s |", train_time / epochs as f64);
    println!("| Points per Sample | {} |", num_points);
    println!("| Number of Classes | {} |", num_classes);

    // Per-class accuracy
    println!();
    println!("Per-class accuracy:");
    let mut class_correct = vec![0usize; num_classes];
    let mut class_total = vec![0usize; num_classes];
    let mut worst_class = (0, 1.0f64);
    let mut best_class = (0, 0.0f64);

    for sample in &test_split.samples {
        let target = sample.label.unwrap_or(0);
        let pred = model.predict(sample);
        class_total[target] += 1;
        if pred == target {
            class_correct[target] += 1;
        }
    }

    for c in 0..num_classes.min(10) {
        if class_total[c] > 0 {
            let acc = class_correct[c] as f64 / class_total[c] as f64;
            println!("  Class {:2}: {:.1}% ({}/{})", c, acc * 100.0, class_correct[c], class_total[c]);
            if acc < worst_class.1 {
                worst_class = (c, acc);
            }
            if acc > best_class.1 {
                best_class = (c, acc);
            }
        }
    }

    if num_classes > 10 {
        println!("  ... ({} more classes)", num_classes - 10);
    }

    // Compute overall stats
    let mut all_accs: Vec<f64> = Vec::new();
    for c in 0..num_classes {
        if class_total[c] > 0 {
            let acc = class_correct[c] as f64 / class_total[c] as f64;
            all_accs.push(acc);
            if acc < worst_class.1 {
                worst_class = (c, acc);
            }
            if acc > best_class.1 {
                best_class = (c, acc);
            }
        }
    }

    println!();
    println!("Best class: {} ({:.1}%)", best_class.0, best_class.1 * 100.0);
    println!("Worst class: {} ({:.1}%)", worst_class.0, worst_class.1 * 100.0);

    // Summary for EXPERIMENTS.md
    println!();
    println!("{}", "=".repeat(70));
    println!("COPY TO EXPERIMENTS.md:");
    println!("{}", "=".repeat(70));
    println!();
    println!("### Experiment E1: Plaintext Accuracy Baseline");
    println!();
    println!("| Metric | Value |");
    println!("|--------|-------|");
    println!("| Dataset | Synthetic ModelNet{} |", num_classes);
    println!("| Points | {} |", num_points);
    println!("| Model | SimpleCliffordNet (hidden={}) |", hidden_dim);
    println!("| Parameters | {} |", num_params);
    println!("| Training Epochs | {} |", epochs);
    println!("| **Test Accuracy** | **{:.2}%** |", final_acc * 100.0);
    println!("| Training Time | {:.1}s |", train_time);

    println!();
    println!("Done!");
}
