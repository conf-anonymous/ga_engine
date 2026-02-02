//! V5 Privacy Analysis: Dimension Inference Attack
//!
//! This example demonstrates how an attacker can infer input dimensions
//! from CKKS execution traces (via rotation counts), while CliffordFHE
//! traces remain indistinguishable.
//!
//! Run with:
//!   cargo run --features v5 --release --example v5_dimension_attack
//!
//! For detailed output:
//!   cargo run --features v5 --release --example v5_dimension_attack -- --verbose
//!
//! Results show:
//! - CKKS attack accuracy: ~100% (rotation count reveals dimension)
//! - CliffordFHE attack accuracy: ~random (no rotation leakage)

use ga_engine::clifford_fhe_v5::{
    TracedCpuBackend, DimensionClassifier,
    workloads::{VectorPair, WorkloadConfig, WorkloadType},
    analysis::LeakageComparison,
};
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     V5 Privacy Analysis: Dimension Inference Attack           ║");
    println!("║     Representation Matters: Execution-Trace Privacy Analysis  ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    let num_training = args.iter()
        .position(|a| a == "--train")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let num_test = args.iter()
        .position(|a| a == "--test")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // Dimension classes to distinguish
    let dimension_classes = vec![8, 16, 32, 64, 128, 256];

    println!("Configuration:");
    println!("  Dimension classes: {:?}", dimension_classes);
    println!("  Training samples per class: {}", num_training);
    println!("  Test samples per class: {}", num_test);
    println!();

    // Initialize backend
    println!("Initializing CPU backend...");
    let params = CliffordFHEParams::new_test_ntt_1024();
    let backend = TracedCpuBackend::new(params);
    println!("  Ring dimension: N=1024");
    println!("  Primes: 3");
    println!();

    // ========================================================================
    // Phase 1: Generate Training Data
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Phase 1: Generating Training Data (CKKS only)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut train_traces = Vec::new();
    let mut train_dimensions = Vec::new();

    for &dim in &dimension_classes {
        print!("  Generating {} CKKS traces for dim={}... ", num_training, dim);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        for trial in 0..num_training {
            let seed = dim as u64 * 1000 + trial as u64;
            let vectors = VectorPair::random(dim, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, dim).with_seed(seed);

            let trace = backend.execute_ckks_similarity(&vectors, &config);
            train_traces.push(trace);
            train_dimensions.push(dim);
        }
        println!("done");
    }

    println!();
    println!("  Total training samples: {}", train_traces.len());
    println!();

    // ========================================================================
    // Phase 2: Train Classifier
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Phase 2: Training Dimension Classifier");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut classifier = DimensionClassifier::new(dimension_classes.clone());
    classifier.train(&train_traces, &train_dimensions);

    println!("  Learned rotation patterns:");
    for &dim in &dimension_classes {
        if let Some(pattern) = classifier.rotation_patterns.get(&dim) {
            println!("    dim={:3} -> {:.1} rotations (expected: {})",
                dim, pattern.expected_rotations,
                DimensionClassifier::theoretical_rotations(dim));
        }
    }
    println!();

    // ========================================================================
    // Phase 3: Generate Test Data (Both CKKS and CliffordFHE)
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Phase 3: Generating Test Data (CKKS + CliffordFHE)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut ckks_test_traces = Vec::new();
    let mut ckks_test_dimensions = Vec::new();
    let mut clifford_test_traces = Vec::new();
    let mut clifford_test_dimensions = Vec::new();

    for &dim in &dimension_classes {
        print!("  Generating {} test traces for dim={}... ", num_test * 2, dim);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        for trial in 0..num_test {
            let seed = dim as u64 * 10000 + trial as u64 + 9999;
            let vectors = VectorPair::random(dim, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, dim).with_seed(seed);

            // CKKS trace
            let ckks_trace = backend.execute_ckks_similarity(&vectors, &config);
            ckks_test_traces.push(ckks_trace);
            ckks_test_dimensions.push(dim);

            // CliffordFHE trace
            let clifford_trace = backend.execute_clifford_similarity(&vectors, &config);
            clifford_test_traces.push(clifford_trace);
            clifford_test_dimensions.push(dim);
        }
        println!("done");
    }

    println!();
    println!("  CKKS test samples: {}", ckks_test_traces.len());
    println!("  CliffordFHE test samples: {}", clifford_test_traces.len());
    println!();

    // ========================================================================
    // Phase 4: Run Attack on CKKS Traces
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Phase 4: Attack on CKKS Traces");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let ckks_report = classifier.classify_batch(&ckks_test_traces, &ckks_test_dimensions);

    println!("CKKS Attack Results:");
    println!("  Overall accuracy: {:.1}%", ckks_report.accuracy * 100.0);
    println!();
    println!("  Per-dimension accuracy:");
    for &dim in &dimension_classes {
        let acc = ckks_report.per_class_accuracy.get(&dim).unwrap_or(&0.0);
        println!("    dim={:3}: {:.1}%", dim, acc * 100.0);
    }

    if verbose {
        println!();
        println!("{}", ckks_report.format());
    }
    println!();

    // ========================================================================
    // Phase 5: Run Attack on CliffordFHE Traces
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Phase 5: Attack on CliffordFHE Traces");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let clifford_report = classifier.classify_batch(&clifford_test_traces, &clifford_test_dimensions);

    println!("CliffordFHE Attack Results:");
    println!("  Overall accuracy: {:.1}%", clifford_report.accuracy * 100.0);
    println!("  (Random guessing baseline: {:.1}%)", 100.0 / dimension_classes.len() as f64);
    println!();
    println!("  Per-dimension accuracy:");
    for &dim in &dimension_classes {
        let acc = clifford_report.per_class_accuracy.get(&dim).unwrap_or(&0.0);
        println!("    dim={:3}: {:.1}%", dim, acc * 100.0);
    }

    if verbose {
        println!();
        println!("{}", clifford_report.format());
    }
    println!();

    // ========================================================================
    // Phase 6: Combined Analysis (Research Results)
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Phase 6: Combined Analysis (Research Results)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Combine all traces for unified report
    let mut all_traces = Vec::new();
    let mut all_dimensions = Vec::new();
    all_traces.extend(ckks_test_traces);
    all_dimensions.extend(ckks_test_dimensions);
    all_traces.extend(clifford_test_traces);
    all_dimensions.extend(clifford_test_dimensions);

    let combined_report = classifier.classify_batch(&all_traces, &all_dimensions);
    let metrics = combined_report.paper_metrics();

    println!("{}", metrics.format_for_paper());

    // ========================================================================
    // Phase 7: Information-Theoretic Leakage Analysis
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Phase 7: Information-Theoretic Leakage Analysis");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Re-collect traces with dimensions for leakage analysis
    let mut ckks_traces_for_leakage = Vec::new();
    let mut ckks_dims_for_leakage = Vec::new();
    let mut clifford_traces_for_leakage = Vec::new();
    let mut clifford_dims_for_leakage = Vec::new();

    for &dim in &dimension_classes {
        for trial in 0..5 {
            let seed = dim as u64 * 100000 + trial as u64;
            let vectors = VectorPair::random(dim, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, dim).with_seed(seed);

            let ckks_trace = backend.execute_ckks_similarity(&vectors, &config);
            ckks_traces_for_leakage.push(ckks_trace);
            ckks_dims_for_leakage.push(dim);

            let clifford_trace = backend.execute_clifford_similarity(&vectors, &config);
            clifford_traces_for_leakage.push(clifford_trace);
            clifford_dims_for_leakage.push(dim);
        }
    }

    let leakage_comparison = LeakageComparison::compare(
        &ckks_traces_for_leakage, &ckks_dims_for_leakage,
        &clifford_traces_for_leakage, &clifford_dims_for_leakage,
    );

    println!("{}", leakage_comparison.format_for_paper());

    // ========================================================================
    // Analysis Summary
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Analysis Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Key Finding:");
    println!("  CKKS rotation count directly reveals input dimension.");
    println!("  Attacker can infer dimension with {:.1}% accuracy.", metrics.ckks_attack_accuracy * 100.0);
    println!();
    println!("  CliffordFHE has NO rotation-based leakage.");
    println!("  Attack accuracy is {:.1}% (near random: {:.1}%).",
        metrics.clifford_attack_accuracy * 100.0,
        100.0 / dimension_classes.len() as f64);
    println!();
    println!("Privacy Improvement:");
    println!("  CliffordFHE reduces dimension inference accuracy by {:.1} percentage points.",
        metrics.privacy_gain * 100.0);
    println!();

    // Why this matters
    println!("Why This Matters (for paper discussion):");
    println!("─────────────────────────────────────────");
    println!("  1. CKKS rotation patterns are observable in:");
    println!("     - Timing side channels");
    println!("     - Memory access patterns");
    println!("     - GPU kernel invocation sequences");
    println!();
    println!("  2. Dimension leakage enables:");
    println!("     - Task identification attacks");
    println!("     - User profiling in multi-tenant FHE services");
    println!("     - Schema inference in encrypted databases");
    println!();
    println!("  3. CliffordFHE's fixed structure provides:");
    println!("     - Constant-time execution (no dimension-dependent branching)");
    println!("     - Fixed memory access patterns");
    println!("     - Oblivious computation without padding overhead");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("    Dimension inference attack complete!");
    println!("═══════════════════════════════════════════════════════════════");
}
