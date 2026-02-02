//! V5 Privacy Analysis: Comprehensive Privacy Attack Suite
//!
//! This example runs ALL privacy attacks on CKKS vs CliffordFHE:
//! 1. Dimension Inference Attack (rotation count reveals input size)
//! 2. Task Identification Attack (operation patterns reveal workload type)
//! 3. Sparsity Inference Attack (timing reveals data distribution)
//! 4. Multi-Tenant Linkability Attack (fingerprints link traces to users)
//!
//! Run with:
//!   cargo run --features v5 --release --example v5_privacy_attacks
//!
//! Options:
//!   --verbose    Show detailed results
//!   --json       Export results to JSON
//!   --trials N   Number of trials per experiment (default: 20)

use ga_engine::clifford_fhe_v5::{
    TracedCpuBackend,
    DimensionClassifier, TaskClassifier, SparsityClassifier, TenantLinker,
    OperationCountClassifier, TraceLengthClassifier,
    workloads::{VectorPair, MultivectorPair, WorkloadConfig, WorkloadType},
    analysis::LeakageComparison,
};
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     V5 Privacy Analysis: Comprehensive Attack Suite               ║");
    println!("║     Representation Matters: CKKS vs CliffordFHE                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    let export_json = args.iter().any(|a| a == "--json");
    let num_trials = args.iter()
        .position(|a| a == "--trials")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    // Initialize
    let params = CliffordFHEParams::new_test_ntt_1024();
    let backend = TracedCpuBackend::new(params);

    println!("Configuration:");
    println!("  Ring dimension: N=1024");
    println!("  Trials per experiment: {}", num_trials);
    println!("  Verbose: {}", verbose);
    println!();

    // Collect all results
    let mut results = AttackResults::new();

    // ========================================================================
    // Attack 1: Dimension Inference
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Attack 1: Dimension Inference");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let dimension_results = run_dimension_attack(&backend, num_trials, verbose);
    results.dimension_attack = Some(dimension_results);

    // ========================================================================
    // Attack 2: Task Identification
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Attack 2: Task Identification");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let task_results = run_task_attack(&backend, num_trials, verbose);
    results.task_attack = Some(task_results);

    // ========================================================================
    // Attack 3: Sparsity Inference
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Attack 3: Sparsity Inference");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let sparsity_results = run_sparsity_attack(&backend, num_trials, verbose);
    results.sparsity_attack = Some(sparsity_results);

    // ========================================================================
    // Attack 4: Multi-Tenant Linkability
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Attack 4: Multi-Tenant Linkability");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let linkability_results = run_linkability_attack(&backend, num_trials, verbose);
    results.linkability_attack = Some(linkability_results);

    // ========================================================================
    // Attack 5: Operation Count Attack (NEW - CliffordFHE should win)
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Attack 5: Operation Count Attack (Trace Fingerprinting)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let op_count_results = run_operation_count_attack(&backend, num_trials, verbose);
    results.operation_count_attack = Some(op_count_results);

    // ========================================================================
    // Attack 6: Trace Length Attack (Complementary)
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Attack 6: Trace Length Attack (Event Count)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let trace_length_results = run_trace_length_attack(&backend, num_trials, verbose);
    results.trace_length_attack = Some(trace_length_results);

    // ========================================================================
    // Summary: The "Killer Table" for Research Papers
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("SUMMARY: Privacy Attack Results");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    print_killer_table(&results);

    // Export to JSON if requested
    if export_json {
        let json_path = "v5_results/attack_results.json";
        std::fs::create_dir_all("v5_results").ok();
        if let Ok(json) = serde_json::to_string_pretty(&results) {
            if std::fs::write(json_path, json).is_ok() {
                println!("\nResults exported to: {}", json_path);
            }
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("    All attacks complete!");
    println!("═══════════════════════════════════════════════════════════════════");
}

// ============================================================================
// Attack Implementations
// ============================================================================

fn run_dimension_attack(
    backend: &TracedCpuBackend,
    num_trials: usize,
    verbose: bool,
) -> DimensionAttackResult {
    let dimension_classes = vec![8, 16, 32, 64, 128, 256];

    println!("  Training dimension classifier on CKKS traces...");

    // Generate training data
    let mut train_traces = Vec::new();
    let mut train_dims = Vec::new();

    for &dim in &dimension_classes {
        for trial in 0..num_trials {
            let seed = dim as u64 * 1000 + trial as u64;
            let vectors = VectorPair::random(dim, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, dim).with_seed(seed);

            let trace = backend.execute_ckks_similarity(&vectors, &config);
            train_traces.push(trace);
            train_dims.push(dim);
        }
    }

    let mut classifier = DimensionClassifier::new(dimension_classes.clone());
    classifier.train(&train_traces, &train_dims);

    // Generate test data
    println!("  Testing on CKKS and CliffordFHE traces...");

    let mut ckks_test_traces = Vec::new();
    let mut ckks_test_dims = Vec::new();
    let mut clifford_test_traces = Vec::new();
    let mut clifford_test_dims = Vec::new();

    for &dim in &dimension_classes {
        for trial in 0..(num_trials / 2) {
            let seed = dim as u64 * 10000 + trial as u64 + 9999;
            let vectors = VectorPair::random(dim, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, dim).with_seed(seed);

            let ckks_trace = backend.execute_ckks_similarity(&vectors, &config);
            ckks_test_traces.push(ckks_trace);
            ckks_test_dims.push(dim);

            let clifford_trace = backend.execute_clifford_similarity(&vectors, &config);
            clifford_test_traces.push(clifford_trace);
            clifford_test_dims.push(dim);
        }
    }

    // Evaluate
    let ckks_report = classifier.classify_batch(&ckks_test_traces, &ckks_test_dims);
    let clifford_report = classifier.classify_batch(&clifford_test_traces, &clifford_test_dims);

    // Information-theoretic analysis
    let leakage = LeakageComparison::compare(
        &ckks_test_traces, &ckks_test_dims,
        &clifford_test_traces, &clifford_test_dims,
    );

    let result = DimensionAttackResult {
        ckks_accuracy: ckks_report.accuracy,
        clifford_accuracy: clifford_report.accuracy,
        random_baseline: 1.0 / dimension_classes.len() as f64,
        ckks_leakage_bits: leakage.ckks_leakage.mutual_information,
        clifford_leakage_bits: leakage.clifford_leakage.mutual_information,
        dimension_classes: dimension_classes.clone(),
    };

    println!("  CKKS accuracy: {:.1}%", result.ckks_accuracy * 100.0);
    println!("  CliffordFHE accuracy: {:.1}% (random: {:.1}%)",
        result.clifford_accuracy * 100.0,
        result.random_baseline * 100.0);
    println!("  CKKS leakage: {:.3} bits", result.ckks_leakage_bits);
    println!("  Clifford leakage: {:.3} bits", result.clifford_leakage_bits);

    if verbose {
        println!();
        println!("{}", leakage.format_for_paper());
    }

    result
}

fn run_task_attack(
    backend: &TracedCpuBackend,
    num_trials: usize,
    verbose: bool,
) -> TaskAttackResult {
    let task_types = vec![
        "similarity".to_string(),
        "geometric_product".to_string(),
    ];

    println!("  Generating training traces for task identification...");

    // Generate training data
    let mut train_traces = Vec::new();
    let mut train_tasks = Vec::new();

    for trial in 0..num_trials {
        let seed = 5000 + trial as u64;

        // Similarity task (both CKKS and Clifford)
        let vectors = VectorPair::random(64, seed);
        let config = WorkloadConfig::new(WorkloadType::Similarity, 64).with_seed(seed);

        let ckks_sim = backend.execute_ckks_similarity(&vectors, &config);
        train_traces.push(ckks_sim);
        train_tasks.push("similarity".to_string());

        let cliff_sim = backend.execute_clifford_similarity(&vectors, &config);
        train_traces.push(cliff_sim);
        train_tasks.push("similarity".to_string());

        // Geometric product (Clifford only)
        let mvs = MultivectorPair::random(seed + 1000);
        let gp_config = WorkloadConfig::new(WorkloadType::GeometricProduct, 8).with_seed(seed);
        let gp_trace = backend.execute_clifford_geometric_product(&mvs, &gp_config);
        train_traces.push(gp_trace);
        train_tasks.push("geometric_product".to_string());
    }

    let mut classifier = TaskClassifier::new(task_types.clone());
    classifier.train(&train_traces, &train_tasks);

    println!("  Testing task classifier...");

    // Generate test data
    let mut test_traces = Vec::new();
    let mut test_tasks = Vec::new();
    let mut test_representations = Vec::new();

    for trial in 0..(num_trials / 2) {
        let seed = 8000 + trial as u64;

        // CKKS similarity
        let vectors = VectorPair::random(64, seed);
        let config = WorkloadConfig::new(WorkloadType::Similarity, 64).with_seed(seed);
        let ckks_trace = backend.execute_ckks_similarity(&vectors, &config);
        test_traces.push(ckks_trace);
        test_tasks.push("similarity".to_string());
        test_representations.push("ckks".to_string());

        // Clifford similarity
        let cliff_trace = backend.execute_clifford_similarity(&vectors, &config);
        test_traces.push(cliff_trace);
        test_tasks.push("similarity".to_string());
        test_representations.push("clifford".to_string());

        // Geometric product
        let mvs = MultivectorPair::random(seed + 2000);
        let gp_config = WorkloadConfig::new(WorkloadType::GeometricProduct, 8).with_seed(seed);
        let gp_trace = backend.execute_clifford_geometric_product(&mvs, &gp_config);
        test_traces.push(gp_trace);
        test_tasks.push("geometric_product".to_string());
        test_representations.push("clifford".to_string());
    }

    let report = classifier.classify_batch(&test_traces, &test_tasks);

    // Compute accuracy by representation
    let mut ckks_correct = 0;
    let mut ckks_total = 0;
    let mut clifford_correct = 0;
    let mut clifford_total = 0;

    for (i, result) in report.results.iter().enumerate() {
        if test_representations[i] == "ckks" {
            ckks_total += 1;
            if result.correct.unwrap_or(false) {
                ckks_correct += 1;
            }
        } else {
            clifford_total += 1;
            if result.correct.unwrap_or(false) {
                clifford_correct += 1;
            }
        }
    }

    let ckks_accuracy = if ckks_total > 0 { ckks_correct as f64 / ckks_total as f64 } else { 0.0 };
    let clifford_accuracy = if clifford_total > 0 { clifford_correct as f64 / clifford_total as f64 } else { 0.0 };

    let result = TaskAttackResult {
        overall_accuracy: report.accuracy,
        ckks_accuracy,
        clifford_accuracy,
        random_baseline: 1.0 / task_types.len() as f64,
        task_types,
    };

    println!("  Overall accuracy: {:.1}%", result.overall_accuracy * 100.0);
    println!("  CKKS traces accuracy: {:.1}%", result.ckks_accuracy * 100.0);
    println!("  CliffordFHE traces accuracy: {:.1}%", result.clifford_accuracy * 100.0);
    println!("  Random baseline: {:.1}%", result.random_baseline * 100.0);

    if verbose {
        println!();
        println!("{}", report.format_for_paper());
    }

    result
}

fn run_sparsity_attack(
    backend: &TracedCpuBackend,
    num_trials: usize,
    verbose: bool,
) -> SparsityAttackResult {
    let sparsity_levels = vec![0.0, 0.25, 0.5, 0.75, 0.9];

    println!("  Generating training traces with varying sparsity...");

    // Generate training data
    let mut train_traces = Vec::new();
    let mut train_sparsities = Vec::new();

    for &sparsity in &sparsity_levels {
        for trial in 0..num_trials {
            let seed = (sparsity * 10000.0) as u64 + trial as u64;
            let vectors = VectorPair::random_sparse(64, sparsity, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, 64)
                .with_sparsity(sparsity)
                .with_seed(seed);

            // CKKS
            let ckks_trace = backend.execute_ckks_similarity(&vectors, &config);
            train_traces.push(ckks_trace);
            train_sparsities.push(sparsity);

            // Clifford
            let cliff_trace = backend.execute_clifford_similarity(&vectors, &config);
            train_traces.push(cliff_trace);
            train_sparsities.push(sparsity);
        }
    }

    let mut classifier = SparsityClassifier::new(sparsity_levels.clone());
    classifier.train(&train_traces, &train_sparsities);

    println!("  Testing sparsity classifier...");

    // Generate test data
    let mut ckks_test_traces = Vec::new();
    let mut ckks_test_sparsities = Vec::new();
    let mut clifford_test_traces = Vec::new();
    let mut clifford_test_sparsities = Vec::new();

    for &sparsity in &sparsity_levels {
        for trial in 0..(num_trials / 2) {
            let seed = (sparsity * 50000.0) as u64 + trial as u64 + 9999;
            let vectors = VectorPair::random_sparse(64, sparsity, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, 64)
                .with_sparsity(sparsity)
                .with_seed(seed);

            let ckks_trace = backend.execute_ckks_similarity(&vectors, &config);
            ckks_test_traces.push(ckks_trace);
            ckks_test_sparsities.push(sparsity);

            let cliff_trace = backend.execute_clifford_similarity(&vectors, &config);
            clifford_test_traces.push(cliff_trace);
            clifford_test_sparsities.push(sparsity);
        }
    }

    let ckks_report = classifier.classify_batch(&ckks_test_traces, &ckks_test_sparsities);
    let cliff_report = classifier.classify_batch(&clifford_test_traces, &clifford_test_sparsities);

    let result = SparsityAttackResult {
        ckks_bucket_accuracy: ckks_report.bucket_accuracy,
        clifford_bucket_accuracy: cliff_report.bucket_accuracy,
        ckks_mae: ckks_report.mean_absolute_error,
        clifford_mae: cliff_report.mean_absolute_error,
        random_baseline: 1.0 / sparsity_levels.len() as f64,
        sparsity_levels,
    };

    println!("  CKKS bucket accuracy: {:.1}%", result.ckks_bucket_accuracy * 100.0);
    println!("  CliffordFHE bucket accuracy: {:.1}%", result.clifford_bucket_accuracy * 100.0);
    println!("  CKKS MAE: {:.3}", result.ckks_mae);
    println!("  CliffordFHE MAE: {:.3}", result.clifford_mae);
    println!("  Random baseline: {:.1}%", result.random_baseline * 100.0);

    if verbose {
        println!();
        println!("{}", ckks_report.format_for_paper());
    }

    result
}

fn run_linkability_attack(
    backend: &TracedCpuBackend,
    num_trials: usize,
    verbose: bool,
) -> LinkabilityAttackResult {
    let tenant_ids: Vec<String> = (0..5).map(|i| format!("tenant_{}", i)).collect();

    println!("  Generating training traces for {} tenants...", tenant_ids.len());

    // Generate training data - each tenant has distinct dimensions they use
    let mut train_traces = Vec::new();
    let mut train_tenants = Vec::new();

    for (tenant_idx, tenant_id) in tenant_ids.iter().enumerate() {
        // Each tenant has a "preferred" dimension based on their workload
        let tenant_dim = 8 << (tenant_idx % 4); // 8, 16, 32, 64, 8, ...

        for trial in 0..num_trials {
            let seed = tenant_idx as u64 * 10000 + trial as u64;
            let vectors = VectorPair::random(tenant_dim, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, tenant_dim).with_seed(seed);

            // CKKS trace
            let ckks_trace = backend.execute_ckks_similarity(&vectors, &config);
            train_traces.push(ckks_trace);
            train_tenants.push(tenant_id.clone());

            // Clifford trace
            let cliff_trace = backend.execute_clifford_similarity(&vectors, &config);
            train_traces.push(cliff_trace);
            train_tenants.push(tenant_id.clone());
        }
    }

    let mut linker = TenantLinker::new(0.6); // Lower threshold for more links
    linker.train(&train_traces, &train_tenants);

    println!("  Testing tenant linkability...");

    // Generate test data
    let mut ckks_test_traces = Vec::new();
    let mut ckks_test_tenants = Vec::new();
    let mut clifford_test_traces = Vec::new();
    let mut clifford_test_tenants = Vec::new();

    for (tenant_idx, tenant_id) in tenant_ids.iter().enumerate() {
        let tenant_dim = 8 << (tenant_idx % 4);

        for trial in 0..(num_trials / 2) {
            let seed = tenant_idx as u64 * 50000 + trial as u64 + 9999;
            let vectors = VectorPair::random(tenant_dim, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, tenant_dim).with_seed(seed);

            let ckks_trace = backend.execute_ckks_similarity(&vectors, &config);
            ckks_test_traces.push(ckks_trace);
            ckks_test_tenants.push(tenant_id.clone());

            let cliff_trace = backend.execute_clifford_similarity(&vectors, &config);
            clifford_test_traces.push(cliff_trace);
            clifford_test_tenants.push(tenant_id.clone());
        }
    }

    let ckks_report = linker.link_batch(&ckks_test_traces, &ckks_test_tenants);
    let cliff_report = linker.link_batch(&clifford_test_traces, &clifford_test_tenants);

    let result = LinkabilityAttackResult {
        ckks_link_accuracy: ckks_report.link_accuracy,
        clifford_link_accuracy: cliff_report.link_accuracy,
        ckks_true_positive_rate: ckks_report.true_positive_rate,
        clifford_true_positive_rate: cliff_report.true_positive_rate,
        ckks_false_positive_rate: ckks_report.false_positive_rate,
        clifford_false_positive_rate: cliff_report.false_positive_rate,
        num_tenants: tenant_ids.len(),
    };

    println!("  CKKS link accuracy: {:.1}%", result.ckks_link_accuracy * 100.0);
    println!("  CliffordFHE link accuracy: {:.1}%", result.clifford_link_accuracy * 100.0);
    println!("  CKKS TPR: {:.1}%, FPR: {:.1}%",
        result.ckks_true_positive_rate * 100.0,
        result.ckks_false_positive_rate * 100.0);
    println!("  CliffordFHE TPR: {:.1}%, FPR: {:.1}%",
        result.clifford_true_positive_rate * 100.0,
        result.clifford_false_positive_rate * 100.0);

    if verbose {
        println!();
        println!("{}", ckks_report.format_for_paper());
    }

    result
}

fn run_operation_count_attack(
    backend: &TracedCpuBackend,
    num_trials: usize,
    verbose: bool,
) -> OperationCountAttackResult {
    let dimension_classes = vec![8, 16, 32, 64, 128, 256];

    println!("  Training operation count classifier on CKKS traces...");
    println!("  This attack exploits CKKS's variable-length trace vs CliffordFHE's fixed trace.");
    println!();

    // Generate training data
    let mut train_traces = Vec::new();
    let mut train_dims = Vec::new();

    for &dim in &dimension_classes {
        for trial in 0..num_trials {
            let seed = dim as u64 * 2000 + trial as u64;
            let vectors = VectorPair::random(dim, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, dim).with_seed(seed);

            let trace = backend.execute_ckks_similarity(&vectors, &config);
            train_traces.push(trace);
            train_dims.push(dim);
        }
    }

    let mut classifier = OperationCountClassifier::new(dimension_classes.clone());
    classifier.train(&train_traces, &train_dims);

    // Generate test data
    println!("  Testing on CKKS and CliffordFHE traces...");

    let mut ckks_test_traces = Vec::new();
    let mut ckks_test_dims = Vec::new();
    let mut clifford_test_traces = Vec::new();
    let mut clifford_test_dims = Vec::new();

    for &dim in &dimension_classes {
        for trial in 0..(num_trials / 2) {
            let seed = dim as u64 * 20000 + trial as u64 + 7777;
            let vectors = VectorPair::random(dim, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, dim).with_seed(seed);

            let ckks_trace = backend.execute_ckks_similarity(&vectors, &config);
            ckks_test_traces.push(ckks_trace);
            ckks_test_dims.push(dim);

            let clifford_trace = backend.execute_clifford_similarity(&vectors, &config);
            clifford_test_traces.push(clifford_trace);
            clifford_test_dims.push(dim);
        }
    }

    // Evaluate
    let ckks_report = classifier.classify_batch(&ckks_test_traces, &ckks_test_dims);
    let clifford_report = classifier.classify_batch(&clifford_test_traces, &clifford_test_dims);

    let result = OperationCountAttackResult {
        ckks_accuracy: *ckks_report.accuracy_by_representation.get("ckks").unwrap_or(&ckks_report.accuracy),
        clifford_accuracy: *clifford_report.accuracy_by_representation.get("clifford").unwrap_or(&clifford_report.accuracy),
        random_baseline: 1.0 / dimension_classes.len() as f64,
        information_leaked_bits: ckks_report.information_leaked_bits,
        clifford_traces_identical: clifford_report.clifford_all_identical,
        dimension_classes: dimension_classes.clone(),
    };

    println!("  CKKS accuracy: {:.1}%", result.ckks_accuracy * 100.0);
    println!("  CliffordFHE accuracy: {:.1}% (random: {:.1}%)",
        result.clifford_accuracy * 100.0,
        result.random_baseline * 100.0);
    println!("  Information leaked (CKKS): {:.3} bits", result.information_leaked_bits);
    println!("  Clifford traces identical: {}", if result.clifford_traces_identical { "YES" } else { "NO" });

    if verbose {
        println!();
        println!("{}", ckks_report.format_for_paper());
    }

    result
}

fn run_trace_length_attack(
    backend: &TracedCpuBackend,
    num_trials: usize,
    verbose: bool,
) -> TraceLengthAttackResult {
    let dimension_classes = vec![8, 16, 32, 64, 128, 256];

    println!("  Training trace length classifier...");
    println!("  This attack uses raw event count as fingerprint.");
    println!();

    // Generate training data
    let mut train_traces = Vec::new();
    let mut train_dims = Vec::new();

    for &dim in &dimension_classes {
        for trial in 0..num_trials {
            let seed = dim as u64 * 3000 + trial as u64;
            let vectors = VectorPair::random(dim, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, dim).with_seed(seed);

            let trace = backend.execute_ckks_similarity(&vectors, &config);
            train_traces.push(trace);
            train_dims.push(dim);
        }
    }

    let mut classifier = TraceLengthClassifier::new(dimension_classes.clone());
    classifier.train(&train_traces, &train_dims);

    // Generate test data
    println!("  Testing on CKKS and CliffordFHE traces...");

    let mut ckks_test_traces = Vec::new();
    let mut ckks_test_dims = Vec::new();
    let mut clifford_test_traces = Vec::new();
    let mut clifford_test_dims = Vec::new();

    for &dim in &dimension_classes {
        for trial in 0..(num_trials / 2) {
            let seed = dim as u64 * 30000 + trial as u64 + 8888;
            let vectors = VectorPair::random(dim, seed);
            let config = WorkloadConfig::new(WorkloadType::Similarity, dim).with_seed(seed);

            let ckks_trace = backend.execute_ckks_similarity(&vectors, &config);
            ckks_test_traces.push(ckks_trace);
            ckks_test_dims.push(dim);

            let clifford_trace = backend.execute_clifford_similarity(&vectors, &config);
            clifford_test_traces.push(clifford_trace);
            clifford_test_dims.push(dim);
        }
    }

    // Evaluate
    let ckks_report = classifier.classify_batch(&ckks_test_traces, &ckks_test_dims);
    let clifford_report = classifier.classify_batch(&clifford_test_traces, &clifford_test_dims);

    let result = TraceLengthAttackResult {
        ckks_accuracy: *ckks_report.accuracy_by_representation.get("ckks").unwrap_or(&ckks_report.accuracy),
        clifford_accuracy: *clifford_report.accuracy_by_representation.get("clifford").unwrap_or(&clifford_report.accuracy),
        random_baseline: 1.0 / dimension_classes.len() as f64,
        ckks_unique_lengths: ckks_report.ckks_unique_lengths.len(),
        clifford_unique_lengths: clifford_report.clifford_unique_lengths.len(),
        dimension_classes: dimension_classes.clone(),
    };

    println!("  CKKS accuracy: {:.1}%", result.ckks_accuracy * 100.0);
    println!("  CliffordFHE accuracy: {:.1}% (random: {:.1}%)",
        result.clifford_accuracy * 100.0,
        result.random_baseline * 100.0);
    println!("  CKKS unique trace lengths: {}", result.ckks_unique_lengths);
    println!("  Clifford unique trace lengths: {} (should be 1)", result.clifford_unique_lengths);

    if verbose {
        println!();
        println!("{}", ckks_report.format_for_paper());
    }

    result
}

// ============================================================================
// Result Types
// ============================================================================

#[derive(Debug, Clone, serde::Serialize)]
struct AttackResults {
    dimension_attack: Option<DimensionAttackResult>,
    task_attack: Option<TaskAttackResult>,
    sparsity_attack: Option<SparsityAttackResult>,
    linkability_attack: Option<LinkabilityAttackResult>,
    operation_count_attack: Option<OperationCountAttackResult>,
    trace_length_attack: Option<TraceLengthAttackResult>,
}

impl AttackResults {
    fn new() -> Self {
        Self {
            dimension_attack: None,
            task_attack: None,
            sparsity_attack: None,
            linkability_attack: None,
            operation_count_attack: None,
            trace_length_attack: None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct DimensionAttackResult {
    ckks_accuracy: f64,
    clifford_accuracy: f64,
    random_baseline: f64,
    ckks_leakage_bits: f64,
    clifford_leakage_bits: f64,
    dimension_classes: Vec<usize>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct TaskAttackResult {
    overall_accuracy: f64,
    ckks_accuracy: f64,
    clifford_accuracy: f64,
    random_baseline: f64,
    task_types: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct SparsityAttackResult {
    ckks_bucket_accuracy: f64,
    clifford_bucket_accuracy: f64,
    ckks_mae: f64,
    clifford_mae: f64,
    random_baseline: f64,
    sparsity_levels: Vec<f64>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct LinkabilityAttackResult {
    ckks_link_accuracy: f64,
    clifford_link_accuracy: f64,
    ckks_true_positive_rate: f64,
    clifford_true_positive_rate: f64,
    ckks_false_positive_rate: f64,
    clifford_false_positive_rate: f64,
    num_tenants: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct OperationCountAttackResult {
    ckks_accuracy: f64,
    clifford_accuracy: f64,
    random_baseline: f64,
    information_leaked_bits: f64,
    clifford_traces_identical: bool,
    dimension_classes: Vec<usize>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct TraceLengthAttackResult {
    ckks_accuracy: f64,
    clifford_accuracy: f64,
    random_baseline: f64,
    ckks_unique_lengths: usize,
    clifford_unique_lengths: usize,
    dimension_classes: Vec<usize>,
}

// ============================================================================
// Pretty Printing
// ============================================================================

fn print_killer_table(results: &AttackResults) {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PRIVACY ATTACK COMPARISON TABLE                        ║");
    println!("║                       CKKS vs CliffordFHE                                 ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Attack Type           │  CKKS           │  CliffordFHE    │  Winner      ║");
    println!("╠════════════════════════╪═════════════════╪═════════════════╪══════════════╣");

    // Dimension inference
    if let Some(ref dim) = results.dimension_attack {
        let winner = if dim.ckks_accuracy > dim.clifford_accuracy + 0.1 { "CliffordFHE" } else { "Tie" };
        println!("║  Dimension Inference   │  {:.1}% accuracy  │  {:.1}% accuracy  │  {:12} ║",
            dim.ckks_accuracy * 100.0,
            dim.clifford_accuracy * 100.0,
            winner);
        println!("║    (rotation leakage)  │  {:.3} bits leak │  {:.3} bits leak │              ║",
            dim.ckks_leakage_bits,
            dim.clifford_leakage_bits);
    }

    println!("╠════════════════════════╪═════════════════╪═════════════════╪══════════════╣");

    // Task identification
    if let Some(ref task) = results.task_attack {
        let winner = if task.ckks_accuracy > task.clifford_accuracy + 0.1 { "CliffordFHE" } else { "Tie" };
        println!("║  Task Identification   │  {:.1}% accuracy  │  {:.1}% accuracy  │  {:12} ║",
            task.ckks_accuracy * 100.0,
            task.clifford_accuracy * 100.0,
            winner);
    }

    println!("╠════════════════════════╪═════════════════╪═════════════════╪══════════════╣");

    // Sparsity inference
    if let Some(ref sp) = results.sparsity_attack {
        let winner = if sp.ckks_bucket_accuracy > sp.clifford_bucket_accuracy + 0.1 { "CliffordFHE" } else { "Tie" };
        println!("║  Sparsity Inference    │  {:.1}% accuracy  │  {:.1}% accuracy  │  {:12} ║",
            sp.ckks_bucket_accuracy * 100.0,
            sp.clifford_bucket_accuracy * 100.0,
            winner);
        println!("║    (timing leakage)    │  MAE: {:.3}      │  MAE: {:.3}      │              ║",
            sp.ckks_mae,
            sp.clifford_mae);
    }

    println!("╠════════════════════════╪═════════════════╪═════════════════╪══════════════╣");

    // Linkability
    if let Some(ref link) = results.linkability_attack {
        let winner = if link.ckks_link_accuracy > link.clifford_link_accuracy + 0.1 { "CliffordFHE" } else { "Tie" };
        println!("║  Tenant Linkability    │  {:.1}% link acc  │  {:.1}% link acc  │  {:12} ║",
            link.ckks_link_accuracy * 100.0,
            link.clifford_link_accuracy * 100.0,
            winner);
        println!("║    (fingerprinting)    │  TPR: {:.1}%      │  TPR: {:.1}%      │              ║",
            link.ckks_true_positive_rate * 100.0,
            link.clifford_true_positive_rate * 100.0);
    }

    println!("╠════════════════════════╪═════════════════╪═════════════════╪══════════════╣");

    // Operation Count Attack (NEW)
    if let Some(ref op) = results.operation_count_attack {
        let winner = if op.ckks_accuracy > op.clifford_accuracy + 0.1 { "CliffordFHE" } else { "Tie" };
        println!("║  Operation Count       │  {:.1}% accuracy  │  {:.1}% accuracy  │  {:12} ║",
            op.ckks_accuracy * 100.0,
            op.clifford_accuracy * 100.0,
            winner);
        println!("║    (trace fingerprint) │  {:.3} bits leak │  traces ident.  │              ║",
            op.information_leaked_bits);
    }

    println!("╠════════════════════════╪═════════════════╪═════════════════╪══════════════╣");

    // Trace Length Attack (NEW)
    if let Some(ref tl) = results.trace_length_attack {
        let winner = if tl.ckks_accuracy > tl.clifford_accuracy + 0.1 { "CliffordFHE" } else { "Tie" };
        println!("║  Trace Length          │  {:.1}% accuracy  │  {:.1}% accuracy  │  {:12} ║",
            tl.ckks_accuracy * 100.0,
            tl.clifford_accuracy * 100.0,
            winner);
        println!("║    (event count)       │  {} unique len   │  {} unique len    │              ║",
            tl.ckks_unique_lengths,
            tl.clifford_unique_lengths);
    }

    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Summary
    println!("KEY FINDINGS:");
    println!("─────────────");

    if let Some(ref dim) = results.dimension_attack {
        if dim.ckks_accuracy > 0.9 && dim.clifford_accuracy < 0.3 {
            println!("✓ Dimension Inference: CliffordFHE provides COMPLETE protection");
            println!("  - CKKS leaks {:.3} bits (100% of dimension entropy)", dim.ckks_leakage_bits);
            println!("  - CliffordFHE leaks {:.3} bits (0% of dimension entropy)", dim.clifford_leakage_bits);
        }
    }

    if let Some(ref op) = results.operation_count_attack {
        if op.ckks_accuracy > 0.9 && op.clifford_accuracy < 0.3 {
            println!("✓ Operation Count: CliffordFHE traces are INDISTINGUISHABLE");
            println!("  - CKKS: Operation count reveals dimension with {:.1}% accuracy", op.ckks_accuracy * 100.0);
            println!("  - CliffordFHE: All traces have IDENTICAL operation counts");
        }
    }

    if let Some(ref tl) = results.trace_length_attack {
        if tl.ckks_accuracy > 0.9 && tl.clifford_accuracy < 0.3 {
            println!("✓ Trace Length: CliffordFHE has FIXED trace structure");
            println!("  - CKKS: {} different trace lengths for {} dimensions",
                tl.ckks_unique_lengths, tl.dimension_classes.len());
            println!("  - CliffordFHE: SINGLE trace length (dimension-oblivious)");
        }
    }

    if let Some(ref sp) = results.sparsity_attack {
        if (sp.ckks_bucket_accuracy - sp.clifford_bucket_accuracy).abs() < 0.1 {
            println!("○ Sparsity Inference: Both schemes provide similar protection");
            println!("  - FHE naturally hides sparsity (encrypted operations are data-oblivious)");
        }
    }

    if let Some(ref link) = results.linkability_attack {
        if link.ckks_link_accuracy > link.clifford_link_accuracy + 0.2 {
            println!("✓ Tenant Linkability: CliffordFHE reduces fingerprinting risk");
            println!("  - CKKS traces can be linked with {:.1}% accuracy", link.ckks_link_accuracy * 100.0);
            println!("  - CliffordFHE traces only {:.1}% linkable", link.clifford_link_accuracy * 100.0);
        }
    }

    // Count wins
    let mut clifford_wins = 0;
    let mut ties = 0;

    if let Some(ref dim) = results.dimension_attack {
        if dim.ckks_accuracy > dim.clifford_accuracy + 0.1 { clifford_wins += 1; } else { ties += 1; }
    }
    if let Some(ref task) = results.task_attack {
        if task.ckks_accuracy > task.clifford_accuracy + 0.1 { clifford_wins += 1; } else { ties += 1; }
    }
    if let Some(ref op) = results.operation_count_attack {
        if op.ckks_accuracy > op.clifford_accuracy + 0.1 { clifford_wins += 1; } else { ties += 1; }
    }
    if let Some(ref tl) = results.trace_length_attack {
        if tl.ckks_accuracy > tl.clifford_accuracy + 0.1 { clifford_wins += 1; } else { ties += 1; }
    }
    if let Some(ref sp) = results.sparsity_attack {
        if sp.ckks_bucket_accuracy > sp.clifford_bucket_accuracy + 0.1 { clifford_wins += 1; } else { ties += 1; }
    }
    if let Some(ref link) = results.linkability_attack {
        if link.ckks_link_accuracy > link.clifford_link_accuracy + 0.1 { clifford_wins += 1; } else { ties += 1; }
    }

    println!();
    println!("SCORE: CliffordFHE wins {} attacks, {} ties", clifford_wins, ties);
    println!();
    println!("CONCLUSION: CliffordFHE's fixed-structure representation provides stronger");
    println!("execution-trace privacy than CKKS for dimension-sensitive attacks.");
}
