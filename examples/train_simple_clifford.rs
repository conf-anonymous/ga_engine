//! Quick training test with SimpleCliffordNet
//!
//! Usage: cargo run --release --example train_simple_clifford

use ga_engine::clifford_pointnet::{SimpleCliffordNet, train_simple_net};
use ga_engine::datasets::modelnet40::generate_synthetic_modelnet40;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     SimpleCliffordNet Quick Training Test                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let num_classes = 10;
    let num_points = 128;
    let samples_per_class = 100;
    let hidden_dim = 64;
    let epochs = 100;
    let lr = 0.005; // Balanced LR

    println!("Configuration:");
    println!("  Classes: {}", num_classes);
    println!("  Points per sample: {}", num_points);
    println!("  Samples per class: {}", samples_per_class);
    println!("  Hidden dim: {}", hidden_dim);
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
    println!(
        "  Generated {} train, {} test samples in {:.2}s",
        train_split.len(),
        test_split.len(),
        gen_start.elapsed().as_secs_f64()
    );
    println!();

    // Create model
    println!("Creating SimpleCliffordNet...");
    let mut model = SimpleCliffordNet::new(hidden_dim, num_classes);
    println!("  Parameters: {}", model.num_params());
    println!();

    // Initial evaluation
    let (init_acc, _) = model.evaluate(&test_split.samples);
    println!("Initial test accuracy: {:.1}% (random chance: {:.1}%)",
             init_acc * 100.0, 100.0 / num_classes as f64);
    println!();

    // Train
    println!("{}", "=".repeat(60));
    println!("Training...");
    println!("{}", "=".repeat(60));

    let train_start = Instant::now();
    let (final_acc, final_loss) = train_simple_net(
        &mut model,
        &train_split.samples,
        &test_split.samples,
        epochs,
        lr,
        5, // print every 5 epochs
    );
    let train_time = train_start.elapsed().as_secs_f64();

    // Final results
    println!();
    println!("{}", "=".repeat(60));
    println!("Final Results");
    println!("{}", "=".repeat(60));
    println!("Test Accuracy: {:.2}%", final_acc * 100.0);
    println!("Test Loss: {:.4}", final_loss);
    println!("Total Training Time: {:.1}s", train_time);
    println!("Time per Epoch: {:.2}s", train_time / epochs as f64);

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

    println!("\nDone!");
}
