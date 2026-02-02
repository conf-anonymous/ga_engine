//! V5 Privacy Analysis: Trace Collector Example
//!
//! This example demonstrates how to collect execution traces for
//! privacy analysis experiments.
//!
//! Run with (CPU):
//!   cargo run --features v5 --example v5_trace_collector
//!
//! For Metal GPU:
//!   cargo run --features v5,v2-gpu-metal --example v5_trace_collector -- --metal
//!
//! Options:
//!   --metal      Use Metal GPU backend (Apple Silicon only)
//!   --trials N   Number of trials per experiment (default: 10)
//!   --compare    Run CPU vs Metal comparison
//!
//! Output:
//!   - traces/v5_analysis/session_*.json (individual traces)
//!   - traces/v5_analysis/features.csv (feature matrix for ML)

use ga_engine::clifford_fhe_v5::{
    TracedCpuBackend, TraceCollector,
    workloads::{VectorPair, MultivectorPair, WorkloadConfig, WorkloadType},
    analysis::{TraceFeatures, ComparisonResult, TraceDataset},
};
#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v5::TracedMetalBackend;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("    V5 Privacy Analysis: Trace Collector");
    println!("    Representation Matters: Execution-Trace Privacy Analysis");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    let use_metal = args.iter().any(|a| a == "--metal");
    let compare_backends = args.iter().any(|a| a == "--compare");
    let num_trials = args.iter()
        .position(|a| a == "--trials")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // Configuration
    let output_dir = "traces/v5_analysis";
    let params = CliffordFHEParams::new_test_ntt_1024();

    println!("Configuration:");
    println!("  Backend: {}", if use_metal { "Metal GPU" } else if compare_backends { "CPU + Metal GPU" } else { "CPU" });
    println!("  Ring dimension: N={}", params.n);
    println!("  Primes: {}", params.moduli.len());
    println!("  Trials per workload: {}", num_trials);
    println!("  Output directory: {}", output_dir);
    println!();

    // Create output directory
    std::fs::create_dir_all(output_dir).ok();

    // Create trace collector
    let collector = TraceCollector::new(output_dir).with_auto_save(true);

    // Create CPU backend (always available)
    let cpu_backend = TracedCpuBackend::new(params.clone());

    // Create Metal backend if requested and available
    #[cfg(feature = "v2-gpu-metal")]
    let metal_backend = if use_metal || compare_backends {
        match TracedMetalBackend::new(params.clone()) {
            Ok(b) => {
                println!("  Metal GPU: Initialized successfully");
                Some(b)
            },
            Err(e) => {
                println!("  Metal GPU: Not available ({})", e);
                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "v2-gpu-metal"))]
    let _metal_backend: Option<()> = {
        if use_metal || compare_backends {
            println!("  Metal GPU: Not compiled (enable with --features v2-gpu-metal)");
        }
        None
    };

    println!("═══════════════════════════════════════════════════════════════");
    println!("    Running Experiments");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Dataset for collecting all traces
    let mut dataset = TraceDataset::new();

    // ========================================================================
    // Experiment 1: CKKS vs CliffordFHE Similarity (varying input sizes)
    // ========================================================================
    println!("Experiment 1: CKKS vs CliffordFHE Similarity");
    println!("─────────────────────────────────────────────");

    let input_sizes = vec![3, 8, 32, 64, 128, 256];

    for dim in &input_sizes {
        println!("\n  Input dimension: {}", dim);

        for trial in 0..num_trials {
            let seed = 42 + trial as u64;
            let vectors = VectorPair::random(*dim, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, *dim)
                .with_seed(seed);

            // Run CPU comparison
            let (ckks_trace, clifford_trace) = cpu_backend.run_comparison(&vectors, &config);

            // Add to dataset
            dataset.add_trace(ckks_trace.clone());
            dataset.add_trace(clifford_trace.clone());

            // Print comparison for first trial of each size
            if trial == 0 {
                let comparison = ComparisonResult::from_traces(&ckks_trace, &clifford_trace);
                println!("    CPU - CKKS {:.2}ms ({} rotations) vs Clifford {:.2}ms ({} relins)",
                    comparison.ckks_features.total_duration_us / 1000.0,
                    comparison.ckks_features.rotation_count,
                    comparison.clifford_features.total_duration_us / 1000.0,
                    comparison.clifford_features.relin_count,
                );

                // Run Metal comparison if available
                #[cfg(feature = "v2-gpu-metal")]
                if let Some(ref metal) = metal_backend {
                    let (metal_ckks, metal_clifford) = metal.run_comparison(&vectors, &config);
                    dataset.add_trace(metal_ckks.clone());
                    dataset.add_trace(metal_clifford.clone());

                    let metal_cmp = ComparisonResult::from_traces(&metal_ckks, &metal_clifford);
                    println!("    Metal - CKKS {:.2}ms ({} rotations) vs Clifford {:.2}ms ({} relins)",
                        metal_cmp.ckks_features.total_duration_us / 1000.0,
                        metal_cmp.ckks_features.rotation_count,
                        metal_cmp.clifford_features.total_duration_us / 1000.0,
                        metal_cmp.clifford_features.relin_count,
                    );
                }
            }
        }
    }

    // ========================================================================
    // Experiment 2: CliffordFHE Geometric Operations
    // ========================================================================
    println!("\n\nExperiment 2: CliffordFHE Geometric Operations");
    println!("─────────────────────────────────────────────────");

    for trial in 0..num_trials {
        let seed = 1000 + trial as u64;
        let mvs = MultivectorPair::random(seed);
        let config = WorkloadConfig::new(WorkloadType::GeometricProduct, 8)
            .with_seed(seed);

        let trace = cpu_backend.execute_clifford_geometric_product(&mvs, &config);
        dataset.add_trace(trace.clone());

        if trial == 0 {
            let features = TraceFeatures::from_trace(&trace);
            println!("  CPU - Geometric product: {:.2}ms, {} relins, {} rescales",
                features.total_duration_us / 1000.0,
                features.relin_count,
                features.rescale_count,
            );

            // Run on Metal if available
            #[cfg(feature = "v2-gpu-metal")]
            if let Some(ref metal) = metal_backend {
                let metal_trace = metal.execute_clifford_geometric_product(&mvs, &config);
                dataset.add_trace(metal_trace.clone());

                let metal_features = TraceFeatures::from_trace(&metal_trace);
                println!("  Metal - Geometric product: {:.2}ms, {} relins, {} rescales",
                    metal_features.total_duration_us / 1000.0,
                    metal_features.relin_count,
                    metal_features.rescale_count,
                );
            }
        }
    }

    // ========================================================================
    // Experiment 3: Sparse vs Dense Inputs
    // ========================================================================
    println!("\n\nExperiment 3: Sparse vs Dense Inputs");
    println!("─────────────────────────────────────────");

    let sparsity_levels = vec![0.0, 0.25, 0.5, 0.75, 0.9];

    for sparsity in &sparsity_levels {
        println!("\n  Sparsity: {:.0}%", sparsity * 100.0);

        for trial in 0..num_trials {
            let seed = 2000 + trial as u64;
            let vectors = VectorPair::random_sparse(64, *sparsity, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, 64)
                .with_sparsity(*sparsity)
                .with_seed(seed);

            // CKKS
            let mut ckks_session = collector.start_trace("similarity", "ckks", "cpu", params.n, params.moduli.len());
            ckks_session.set_sparsity(*sparsity);
            ckks_session.set_category(&format!("sparse_{:.0}", sparsity * 100.0));
            let ckks_trace = cpu_backend.execute_ckks_similarity(&vectors, &config);
            dataset.add_trace(ckks_trace);

            // CliffordFHE
            let clifford_trace = cpu_backend.execute_clifford_similarity(&vectors, &config);
            dataset.add_trace(clifford_trace);
        }
    }

    // ========================================================================
    // Experiment 4: CPU vs Metal Backend Comparison (if Metal available)
    // ========================================================================
    #[cfg(feature = "v2-gpu-metal")]
    if let Some(ref metal) = metal_backend {
        println!("\n\nExperiment 4: CPU vs Metal Backend Comparison");
        println!("─────────────────────────────────────────────────");

        let comparison_sizes = vec![8, 64, 256];

        for dim in &comparison_sizes {
            println!("\n  Input dimension: {}", dim);

            let vectors = VectorPair::random(*dim, 42);
            let config = WorkloadConfig::new(WorkloadType::Similarity, *dim);

            // CPU traces
            let (cpu_ckks, cpu_clifford) = cpu_backend.run_comparison(&vectors, &config);

            // Metal traces
            let (metal_ckks, metal_clifford) = metal.run_comparison(&vectors, &config);

            // Compare CKKS
            println!("    CKKS:");
            println!("      CPU:   {:.2}ms, {} rotations, {} unique amounts",
                cpu_ckks.summary.total_duration_us as f64 / 1000.0,
                cpu_ckks.summary.total_rotations,
                cpu_ckks.summary.rotation_amounts_used.len(),
            );
            println!("      Metal: {:.2}ms, {} rotations, {} unique amounts",
                metal_ckks.summary.total_duration_us as f64 / 1000.0,
                metal_ckks.summary.total_rotations,
                metal_ckks.summary.rotation_amounts_used.len(),
            );

            // Compare CliffordFHE
            println!("    CliffordFHE:");
            println!("      CPU:   {:.2}ms, {} relins, {} rescales",
                cpu_clifford.summary.total_duration_us as f64 / 1000.0,
                cpu_clifford.summary.total_relins,
                cpu_clifford.summary.total_rescales,
            );
            println!("      Metal: {:.2}ms, {} relins, {} rescales",
                metal_clifford.summary.total_duration_us as f64 / 1000.0,
                metal_clifford.summary.total_relins,
                metal_clifford.summary.total_rescales,
            );

            // Add all to dataset
            dataset.add_trace(cpu_ckks);
            dataset.add_trace(cpu_clifford);
            dataset.add_trace(metal_ckks);
            dataset.add_trace(metal_clifford);
        }
    }

    // ========================================================================
    // Results Summary
    // ========================================================================
    println!("\n\n═══════════════════════════════════════════════════════════════");
    println!("    Results Summary");
    println!("═══════════════════════════════════════════════════════════════\n");

    let stats = dataset.statistics();
    println!("{}", stats);

    // Export feature matrix for ML
    let features_path = format!("{}/features.csv", output_dir);
    if let Err(e) = dataset.export_features_csv(&features_path) {
        eprintln!("Error exporting features: {}", e);
    } else {
        println!("\nFeature matrix exported to: {}", features_path);
    }

    // Export summary
    let summary_path = format!("{}/session_summary.json", output_dir);
    if let Err(e) = collector.export_all(&summary_path) {
        eprintln!("Error exporting summary: {}", e);
    } else {
        println!("Session summary exported to: {}", summary_path);
    }

    // ========================================================================
    // Key Observations (for research)
    // ========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("    Key Observations");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("1. CKKS requires O(log n) rotations for dot product summation");
    println!("   CliffordFHE has fixed 8-component layout (no input-dependent rotations)");
    println!();
    println!("2. CKKS rotation amounts vary with input size ({:?})", input_sizes);
    println!("   CliffordFHE always uses same 64 multiplications");
    println!();
    println!("3. Sparse inputs don't change encrypted operation counts");
    println!("   (padding already removes this leakage in CKKS too)");
    println!();
    println!("4. Next steps:");
    println!("   - Run traces through ML classifiers (task identification)");
    println!("   - Measure attack accuracy: CKKS vs CliffordFHE");
    println!("   - Generate 'killer table' for paper");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("    Trace collection complete!");
    println!("═══════════════════════════════════════════════════════════════");
}
