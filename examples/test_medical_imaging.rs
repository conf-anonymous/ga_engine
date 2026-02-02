/// Test medical imaging synthetic data generation

use ga_engine::medical_imaging::{generate_dataset, encode_point_cloud};

fn main() {
    println!("Generating synthetic 3D shape dataset...\n");

    // Generate 30 samples (10 per class)
    let dataset = generate_dataset(10, 100);

    println!("Dataset generated: {} samples", dataset.len());
    println!("Each sample has ~100 points\n");

    // Show first 3 samples
    for (i, pc) in dataset.iter().take(3).enumerate() {
        println!("Sample {}: {}", i + 1, pc);

        let mv = encode_point_cloud(pc);
        println!("  Encoded: {}\n", mv);
    }

    println!("✓ Synthetic data generation working!");
}
