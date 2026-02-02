//! Trace Analysis Utilities for Privacy Research
//!
//! This module provides tools for analyzing execution traces to measure
//! representation-induced distinguishability for privacy research.

use crate::clifford_fhe_v5::trace::{ExecutionTrace, OperationType, TraceSummary};
use std::collections::HashMap;

/// Feature vector extracted from an execution trace for ML classification
#[derive(Debug, Clone)]
pub struct TraceFeatures {
    // Timing features
    pub total_duration_us: f64,
    pub mean_op_duration_us: f64,
    pub std_op_duration_us: f64,

    // Operation count features
    pub total_ops: usize,
    pub rotation_count: usize,
    pub rescale_count: usize,
    pub relin_count: usize,
    pub bootstrap_count: usize,

    // Level features
    pub start_level: usize,
    pub end_level: usize,
    pub level_drop: usize,
    pub level_variance: f64,

    // Rotation pattern features
    pub unique_rotation_amounts: usize,
    pub rotation_amount_entropy: f64,

    // Operation type distribution (normalized)
    pub op_type_distribution: HashMap<String, f64>,

    // Memory features (if available)
    pub peak_memory: Option<usize>,

    // Kernel features (GPU only)
    pub kernel_count: Option<usize>,
}

impl TraceFeatures {
    /// Extract features from an execution trace
    pub fn from_trace(trace: &ExecutionTrace) -> Self {
        let summary = &trace.summary;

        // Compute timing statistics
        let durations: Vec<f64> = trace.events.iter()
            .map(|e| e.duration.as_micros() as f64)
            .collect();

        let mean_duration = if !durations.is_empty() {
            durations.iter().sum::<f64>() / durations.len() as f64
        } else {
            0.0
        };

        let std_duration = if durations.len() > 1 {
            let variance: f64 = durations.iter()
                .map(|d| (d - mean_duration).powi(2))
                .sum::<f64>() / (durations.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Level variance
        let level_trajectory = &summary.level_trajectory;
        let level_variance = if level_trajectory.len() > 1 {
            let mean_level = level_trajectory.iter().sum::<usize>() as f64
                / level_trajectory.len() as f64;
            let variance: f64 = level_trajectory.iter()
                .map(|&l| (l as f64 - mean_level).powi(2))
                .sum::<f64>() / (level_trajectory.len() - 1) as f64;
            variance
        } else {
            0.0
        };

        // Rotation amount entropy
        let rotation_entropy = compute_entropy(&summary.rotation_amounts_used);

        // Normalize operation type distribution
        let total_ops = summary.total_ops as f64;
        let op_distribution: HashMap<String, f64> = summary.op_histogram.iter()
            .map(|(k, &v)| (k.clone(), v as f64 / total_ops.max(1.0)))
            .collect();

        // Extract kernel count from metadata if available
        let kernel_count = trace.events.last()
            .and_then(|e| e.metadata.as_ref())
            .and_then(|m| m.get("total_kernels"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        Self {
            total_duration_us: summary.total_duration_us as f64,
            mean_op_duration_us: mean_duration,
            std_op_duration_us: std_duration,
            total_ops: summary.total_ops,
            rotation_count: summary.total_rotations,
            rescale_count: summary.total_rescales,
            relin_count: summary.total_relins,
            bootstrap_count: summary.bootstrap_count,
            start_level: summary.start_level,
            end_level: summary.end_level,
            level_drop: summary.start_level.saturating_sub(summary.end_level),
            level_variance,
            unique_rotation_amounts: summary.rotation_amounts_used.len(),
            rotation_amount_entropy: rotation_entropy,
            op_type_distribution: op_distribution,
            peak_memory: summary.peak_memory,
            kernel_count,
        }
    }

    /// Convert to feature vector for ML (fixed-size array)
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.total_duration_us,
            self.mean_op_duration_us,
            self.std_op_duration_us,
            self.total_ops as f64,
            self.rotation_count as f64,
            self.rescale_count as f64,
            self.relin_count as f64,
            self.bootstrap_count as f64,
            self.start_level as f64,
            self.end_level as f64,
            self.level_drop as f64,
            self.level_variance,
            self.unique_rotation_amounts as f64,
            self.rotation_amount_entropy,
            self.peak_memory.unwrap_or(0) as f64,
            self.kernel_count.unwrap_or(0) as f64,
        ]
    }

    /// Feature names for interpretation
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "total_duration_us",
            "mean_op_duration_us",
            "std_op_duration_us",
            "total_ops",
            "rotation_count",
            "rescale_count",
            "relin_count",
            "bootstrap_count",
            "start_level",
            "end_level",
            "level_drop",
            "level_variance",
            "unique_rotation_amounts",
            "rotation_amount_entropy",
            "peak_memory",
            "kernel_count",
        ]
    }
}

/// Compute Shannon entropy of a distribution
fn compute_entropy(values: &[i32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &v in values {
        *counts.entry(v).or_insert(0) += 1;
    }

    let total = values.len() as f64;
    let mut entropy = 0.0;

    for &count in counts.values() {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Comparison result between CKKS and CliffordFHE traces
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// CKKS trace features
    pub ckks_features: TraceFeatures,

    /// CliffordFHE trace features
    pub clifford_features: TraceFeatures,

    /// Feature differences (clifford - ckks)
    pub feature_diffs: Vec<f64>,

    /// Euclidean distance in feature space
    pub euclidean_distance: f64,

    /// Normalized distance (0-1 scale)
    pub normalized_distance: f64,
}

impl ComparisonResult {
    /// Compare two traces (CKKS vs CliffordFHE)
    pub fn from_traces(ckks: &ExecutionTrace, clifford: &ExecutionTrace) -> Self {
        let ckks_features = TraceFeatures::from_trace(ckks);
        let clifford_features = TraceFeatures::from_trace(clifford);

        let ckks_vec = ckks_features.to_vector();
        let clifford_vec = clifford_features.to_vector();

        let feature_diffs: Vec<f64> = clifford_vec.iter()
            .zip(&ckks_vec)
            .map(|(c, k)| c - k)
            .collect();

        let euclidean_distance = feature_diffs.iter()
            .map(|d| d.powi(2))
            .sum::<f64>()
            .sqrt();

        // Normalize by max possible distance (rough estimate)
        let max_distance = (ckks_vec.iter().map(|x| x.powi(2)).sum::<f64>()
            + clifford_vec.iter().map(|x| x.powi(2)).sum::<f64>())
            .sqrt();

        let normalized_distance = if max_distance > 0.0 {
            euclidean_distance / max_distance
        } else {
            0.0
        };

        Self {
            ckks_features,
            clifford_features,
            feature_diffs,
            euclidean_distance,
            normalized_distance,
        }
    }

    /// Get a summary of the comparison
    pub fn summary(&self) -> String {
        format!(
            "CKKS vs Clifford Comparison:\n\
             - Duration: {:.2}ms vs {:.2}ms ({:+.2}%)\n\
             - Rotations: {} vs {} ({:+})\n\
             - Rescales: {} vs {} ({:+})\n\
             - Relins: {} vs {} ({:+})\n\
             - Level drop: {} vs {} ({:+})\n\
             - Euclidean distance: {:.4}\n\
             - Normalized distance: {:.4}",
            self.ckks_features.total_duration_us / 1000.0,
            self.clifford_features.total_duration_us / 1000.0,
            100.0 * (self.clifford_features.total_duration_us - self.ckks_features.total_duration_us)
                / self.ckks_features.total_duration_us.max(1.0),
            self.ckks_features.rotation_count,
            self.clifford_features.rotation_count,
            self.clifford_features.rotation_count as i64 - self.ckks_features.rotation_count as i64,
            self.ckks_features.rescale_count,
            self.clifford_features.rescale_count,
            self.clifford_features.rescale_count as i64 - self.ckks_features.rescale_count as i64,
            self.ckks_features.relin_count,
            self.clifford_features.relin_count,
            self.clifford_features.relin_count as i64 - self.ckks_features.relin_count as i64,
            self.ckks_features.level_drop,
            self.clifford_features.level_drop,
            self.clifford_features.level_drop as i64 - self.ckks_features.level_drop as i64,
            self.euclidean_distance,
            self.normalized_distance,
        )
    }
}

/// Trace dataset for batch analysis
pub struct TraceDataset {
    /// All traces grouped by representation
    pub ckks_traces: Vec<ExecutionTrace>,
    pub clifford_traces: Vec<ExecutionTrace>,

    /// Feature matrices
    pub ckks_features: Vec<TraceFeatures>,
    pub clifford_features: Vec<TraceFeatures>,
}

impl TraceDataset {
    /// Create empty dataset
    pub fn new() -> Self {
        Self {
            ckks_traces: Vec::new(),
            clifford_traces: Vec::new(),
            ckks_features: Vec::new(),
            clifford_features: Vec::new(),
        }
    }

    /// Add a trace to the dataset
    pub fn add_trace(&mut self, trace: ExecutionTrace) {
        let features = TraceFeatures::from_trace(&trace);

        if trace.representation == "ckks" {
            self.ckks_traces.push(trace);
            self.ckks_features.push(features);
        } else {
            self.clifford_traces.push(trace);
            self.clifford_features.push(features);
        }
    }

    /// Get total number of traces
    pub fn len(&self) -> usize {
        self.ckks_traces.len() + self.clifford_traces.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Export feature matrices for external ML tools
    pub fn export_features_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Write header
        let feature_names = TraceFeatures::feature_names();
        writeln!(file, "representation,{}", feature_names.join(","))?;

        // Write CKKS features
        for features in &self.ckks_features {
            let values: Vec<String> = features.to_vector().iter()
                .map(|v| format!("{:.6}", v))
                .collect();
            writeln!(file, "ckks,{}", values.join(","))?;
        }

        // Write Clifford features
        for features in &self.clifford_features {
            let values: Vec<String> = features.to_vector().iter()
                .map(|v| format!("{:.6}", v))
                .collect();
            writeln!(file, "clifford,{}", values.join(","))?;
        }

        Ok(())
    }

    /// Compute aggregate statistics
    pub fn statistics(&self) -> DatasetStatistics {
        let ckks_vecs: Vec<Vec<f64>> = self.ckks_features.iter()
            .map(|f| f.to_vector())
            .collect();

        let clifford_vecs: Vec<Vec<f64>> = self.clifford_features.iter()
            .map(|f| f.to_vector())
            .collect();

        DatasetStatistics {
            ckks_count: self.ckks_traces.len(),
            clifford_count: self.clifford_traces.len(),
            ckks_mean: compute_mean(&ckks_vecs),
            ckks_std: compute_std(&ckks_vecs),
            clifford_mean: compute_mean(&clifford_vecs),
            clifford_std: compute_std(&clifford_vecs),
        }
    }
}

/// Statistics for a trace dataset
#[derive(Debug, Clone)]
pub struct DatasetStatistics {
    pub ckks_count: usize,
    pub clifford_count: usize,
    pub ckks_mean: Vec<f64>,
    pub ckks_std: Vec<f64>,
    pub clifford_mean: Vec<f64>,
    pub clifford_std: Vec<f64>,
}

impl std::fmt::Display for DatasetStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Dataset Statistics ===")?;
        writeln!(f, "CKKS traces: {}", self.ckks_count)?;
        writeln!(f, "Clifford traces: {}", self.clifford_count)?;

        let names = TraceFeatures::feature_names();

        writeln!(f, "\nFeature comparison (CKKS vs Clifford):")?;
        for (i, name) in names.iter().enumerate() {
            if i < self.ckks_mean.len() && i < self.clifford_mean.len() {
                writeln!(f, "  {}: {:.2} ± {:.2} vs {:.2} ± {:.2}",
                         name,
                         self.ckks_mean[i], self.ckks_std[i],
                         self.clifford_mean[i], self.clifford_std[i])?;
            }
        }

        Ok(())
    }
}

/// Compute mean of feature vectors
fn compute_mean(vecs: &[Vec<f64>]) -> Vec<f64> {
    if vecs.is_empty() {
        return Vec::new();
    }

    let n = vecs.len() as f64;
    let dim = vecs[0].len();

    let mut mean = vec![0.0; dim];
    for vec in vecs {
        for (i, &v) in vec.iter().enumerate() {
            mean[i] += v / n;
        }
    }

    mean
}

/// Compute standard deviation of feature vectors
fn compute_std(vecs: &[Vec<f64>]) -> Vec<f64> {
    if vecs.len() < 2 {
        return vec![0.0; vecs.first().map(|v| v.len()).unwrap_or(0)];
    }

    let mean = compute_mean(vecs);
    let n = vecs.len() as f64;
    let dim = mean.len();

    let mut variance = vec![0.0; dim];
    for vec in vecs {
        for (i, &v) in vec.iter().enumerate() {
            variance[i] += (v - mean[i]).powi(2) / (n - 1.0);
        }
    }

    variance.iter().map(|v| v.sqrt()).collect()
}

// ============================================================================
// Information-Theoretic Leakage Measurement
// ============================================================================

/// Information-theoretic leakage analysis for privacy research
///
/// This measures how much information about the input dimension is leaked
/// through the execution trace, using mutual information and entropy.
#[derive(Debug, Clone)]
pub struct LeakageAnalysis {
    /// Mutual information I(D; T) between dimension D and trace features T
    pub mutual_information: f64,

    /// Entropy of dimension distribution H(D)
    pub dimension_entropy: f64,

    /// Conditional entropy H(D|T) - remaining uncertainty after observing trace
    pub conditional_entropy: f64,

    /// Leakage ratio = I(D;T) / H(D) (0 = no leakage, 1 = complete leakage)
    pub leakage_ratio: f64,

    /// Per-dimension leakage (bits)
    pub per_dimension_leakage: std::collections::HashMap<usize, f64>,

    /// Number of traces analyzed
    pub trace_count: usize,
}

impl LeakageAnalysis {
    /// Analyze leakage from traces with known dimensions
    ///
    /// The key insight: CKKS rotation count is a deterministic function of
    /// input dimension, so I(D; rotation_count) ≈ H(D) for CKKS.
    /// For CliffordFHE, rotation_count = 0 always, so I(D; rotation_count) = 0.
    pub fn from_traces(traces: &[ExecutionTrace], dimensions: &[usize]) -> Self {
        assert_eq!(traces.len(), dimensions.len());

        if traces.is_empty() {
            return Self::zero();
        }

        // Count dimension occurrences
        let mut dim_counts: HashMap<usize, usize> = HashMap::new();
        for &dim in dimensions {
            *dim_counts.entry(dim).or_insert(0) += 1;
        }

        // Compute H(D) - entropy of dimension distribution
        let n = traces.len() as f64;
        let dimension_entropy: f64 = dim_counts.values()
            .map(|&count| {
                let p = count as f64 / n;
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .sum();

        // Group traces by (dimension, rotation_count) to estimate joint distribution
        let mut joint_counts: HashMap<(usize, usize), usize> = HashMap::new();
        let mut rotation_counts: HashMap<usize, usize> = HashMap::new();

        for (trace, &dim) in traces.iter().zip(dimensions.iter()) {
            let rot = trace.summary.total_rotations;
            *joint_counts.entry((dim, rot)).or_insert(0) += 1;
            *rotation_counts.entry(rot).or_insert(0) += 1;
        }

        // Compute H(T) - entropy of rotation count distribution
        let trace_entropy = rotation_counts.values()
            .map(|&count| {
                let p = count as f64 / n;
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .fold(0.0f64, |acc, x| acc + x);

        // Compute H(D, T) - joint entropy
        let joint_entropy = joint_counts.values()
            .map(|&count| {
                let p = count as f64 / n;
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .fold(0.0f64, |acc, x| acc + x);

        // Mutual information I(D; T) = H(D) + H(T) - H(D, T)
        let mutual_information: f64 = (dimension_entropy + trace_entropy - joint_entropy).max(0.0);

        // Conditional entropy H(D|T) = H(D) - I(D; T)
        let conditional_entropy: f64 = (dimension_entropy - mutual_information).max(0.0);

        // Leakage ratio
        let leakage_ratio = if dimension_entropy > 0.0 {
            mutual_information / dimension_entropy
        } else {
            0.0
        };

        // Per-dimension leakage (how distinguishable is this dimension)
        let mut per_dimension_leakage = HashMap::new();
        for &dim in dim_counts.keys() {
            // Count how many distinct rotation patterns this dimension produces
            let dim_rotations: std::collections::HashSet<_> = traces.iter()
                .zip(dimensions.iter())
                .filter(|(_, &d)| d == dim)
                .map(|(t, _)| t.summary.total_rotations)
                .collect();

            // Lower entropy = more distinguishable
            let dim_entropy = (dim_rotations.len() as f64).log2().max(0.0);
            per_dimension_leakage.insert(dim, dim_entropy);
        }

        Self {
            mutual_information,
            dimension_entropy,
            conditional_entropy,
            leakage_ratio,
            per_dimension_leakage,
            trace_count: traces.len(),
        }
    }

    /// Create zero leakage result
    fn zero() -> Self {
        Self {
            mutual_information: 0.0,
            dimension_entropy: 0.0,
            conditional_entropy: 0.0,
            leakage_ratio: 0.0,
            per_dimension_leakage: HashMap::new(),
            trace_count: 0,
        }
    }

    /// Format for paper
    pub fn format_for_paper(&self) -> String {
        let mut output = String::new();

        output.push_str("╔═══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║       Information-Theoretic Leakage Analysis (V5)            ║\n");
        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!("║  H(Dimension):           {:>6.3} bits                        ║\n",
            self.dimension_entropy));
        output.push_str(&format!("║  H(Dimension | Trace):   {:>6.3} bits                        ║\n",
            self.conditional_entropy));
        output.push_str(&format!("║  I(Dimension ; Trace):   {:>6.3} bits  (mutual information)  ║\n",
            self.mutual_information));
        output.push_str(&format!("║  Leakage Ratio:          {:>6.1}%   (I/H)                    ║\n",
            self.leakage_ratio * 100.0));
        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");

        // Interpretation
        if self.leakage_ratio > 0.9 {
            output.push_str("║  ⚠️  SEVERE: Trace reveals almost all dimension information   ║\n");
        } else if self.leakage_ratio > 0.5 {
            output.push_str("║  ⚠️  HIGH: Trace reveals significant dimension information    ║\n");
        } else if self.leakage_ratio > 0.1 {
            output.push_str("║  ⚠  MODERATE: Trace reveals some dimension information       ║\n");
        } else {
            output.push_str("║  ✓  LOW: Trace reveals minimal dimension information         ║\n");
        }

        output.push_str("╚═══════════════════════════════════════════════════════════════╝\n");

        output
    }
}

/// Compare leakage between CKKS and CliffordFHE
#[derive(Debug, Clone)]
pub struct LeakageComparison {
    /// CKKS leakage analysis
    pub ckks_leakage: LeakageAnalysis,

    /// CliffordFHE leakage analysis
    pub clifford_leakage: LeakageAnalysis,

    /// Privacy improvement (CKKS_leakage - Clifford_leakage)
    pub privacy_improvement_bits: f64,

    /// Privacy improvement ratio
    pub privacy_improvement_ratio: f64,
}

impl LeakageComparison {
    /// Compare leakage from separate CKKS and Clifford trace sets
    pub fn compare(
        ckks_traces: &[ExecutionTrace],
        ckks_dims: &[usize],
        clifford_traces: &[ExecutionTrace],
        clifford_dims: &[usize],
    ) -> Self {
        let ckks_leakage = LeakageAnalysis::from_traces(ckks_traces, ckks_dims);
        let clifford_leakage = LeakageAnalysis::from_traces(clifford_traces, clifford_dims);

        let privacy_improvement_bits = ckks_leakage.mutual_information - clifford_leakage.mutual_information;

        let privacy_improvement_ratio = if ckks_leakage.mutual_information > 0.0 {
            privacy_improvement_bits / ckks_leakage.mutual_information
        } else {
            0.0
        };

        Self {
            ckks_leakage,
            clifford_leakage,
            privacy_improvement_bits,
            privacy_improvement_ratio,
        }
    }

    /// Format for paper
    pub fn format_for_paper(&self) -> String {
        let mut output = String::new();

        output.push_str("╔═══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║       Leakage Comparison: CKKS vs CliffordFHE (V5)           ║\n");
        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");
        output.push_str("║                        CKKS        CliffordFHE               ║\n");
        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!("║  H(D|T) [bits]:     {:>8.3}       {:>8.3}                  ║\n",
            self.ckks_leakage.conditional_entropy,
            self.clifford_leakage.conditional_entropy));
        output.push_str(&format!("║  I(D;T) [bits]:     {:>8.3}       {:>8.3}                  ║\n",
            self.ckks_leakage.mutual_information,
            self.clifford_leakage.mutual_information));
        output.push_str(&format!("║  Leakage ratio:     {:>8.1}%      {:>8.1}%                 ║\n",
            self.ckks_leakage.leakage_ratio * 100.0,
            self.clifford_leakage.leakage_ratio * 100.0));
        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!("║  Privacy improvement:  {:>6.3} bits ({:.1}% reduction)        ║\n",
            self.privacy_improvement_bits,
            self.privacy_improvement_ratio * 100.0));
        output.push_str("╚═══════════════════════════════════════════════════════════════╝\n");

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v5::trace::OperationEvent;
    use std::time::Duration;

    fn create_test_trace(representation: &str, rotation_count: usize) -> ExecutionTrace {
        let mut trace = ExecutionTrace::new(
            "test",
            representation,
            "cpu",
            1024,
            3,
        );

        trace.add_event(
            OperationEvent::new(OperationType::Encrypt)
                .with_duration(Duration::from_micros(100))
                .with_levels(2, 2)
        );

        trace.add_event(
            OperationEvent::new(OperationType::MultiplyCiphertext)
                .with_duration(Duration::from_micros(500))
                .with_levels(2, 1)
                .with_relins(1)
                .with_rescales(1)
        );

        trace.add_event(
            OperationEvent::new(OperationType::Rotate)
                .with_duration(Duration::from_micros(200))
                .with_levels(1, 1)
                .with_rotations(rotation_count)
        );

        trace.compute_summary();
        trace
    }

    #[test]
    fn test_trace_features_extraction() {
        let trace = create_test_trace("ckks", 5);
        let features = TraceFeatures::from_trace(&trace);

        assert_eq!(features.total_ops, 3);
        assert_eq!(features.rotation_count, 5);
        assert_eq!(features.rescale_count, 1);
        assert_eq!(features.relin_count, 1);
    }

    #[test]
    fn test_comparison_result() {
        let ckks_trace = create_test_trace("ckks", 10);
        let clifford_trace = create_test_trace("clifford", 0);

        let comparison = ComparisonResult::from_traces(&ckks_trace, &clifford_trace);

        assert_eq!(comparison.ckks_features.rotation_count, 10);
        assert_eq!(comparison.clifford_features.rotation_count, 0);
        assert!(comparison.euclidean_distance > 0.0);
    }

    #[test]
    fn test_trace_dataset() {
        let mut dataset = TraceDataset::new();

        dataset.add_trace(create_test_trace("ckks", 5));
        dataset.add_trace(create_test_trace("ckks", 6));
        dataset.add_trace(create_test_trace("clifford", 0));
        dataset.add_trace(create_test_trace("clifford", 0));

        assert_eq!(dataset.ckks_traces.len(), 2);
        assert_eq!(dataset.clifford_traces.len(), 2);

        let stats = dataset.statistics();
        assert_eq!(stats.ckks_count, 2);
        assert_eq!(stats.clifford_count, 2);
    }

    #[test]
    fn test_entropy_calculation() {
        // Uniform distribution should have high entropy
        let uniform = vec![1, 2, 3, 4, 5];
        let entropy_uniform = compute_entropy(&uniform);

        // Single value should have zero entropy
        let single = vec![1, 1, 1, 1, 1];
        let entropy_single = compute_entropy(&single);

        assert!(entropy_uniform > entropy_single);
        assert!((entropy_single - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_leakage_analysis() {
        // Create CKKS-like traces with dimension-dependent rotations
        let mut ckks_traces = Vec::new();
        let mut ckks_dims = Vec::new();

        for &dim in &[8, 16, 32, 64] {
            for _ in 0..10 {
                // Rotation count = log2(dim)
                let rotations = (dim as f64).log2().ceil() as usize;
                ckks_traces.push(create_test_trace("ckks", rotations));
                ckks_dims.push(dim);
            }
        }

        let ckks_leakage = LeakageAnalysis::from_traces(&ckks_traces, &ckks_dims);

        // CKKS should have high leakage (rotation reveals dimension)
        assert!(ckks_leakage.leakage_ratio > 0.9,
            "CKKS leakage ratio should be > 90%, got {:.1}%",
            ckks_leakage.leakage_ratio * 100.0);

        // Create Clifford-like traces with constant rotation count (0)
        let mut clifford_traces = Vec::new();
        let mut clifford_dims = Vec::new();

        for &dim in &[8, 16, 32, 64] {
            for _ in 0..10 {
                clifford_traces.push(create_test_trace("clifford", 0));
                clifford_dims.push(dim);
            }
        }

        let clifford_leakage = LeakageAnalysis::from_traces(&clifford_traces, &clifford_dims);

        // Clifford should have low leakage (rotation is always 0)
        assert!(clifford_leakage.leakage_ratio < 0.1,
            "Clifford leakage ratio should be < 10%, got {:.1}%",
            clifford_leakage.leakage_ratio * 100.0);
    }

    #[test]
    fn test_leakage_comparison() {
        // CKKS traces
        let mut ckks_traces = Vec::new();
        let mut ckks_dims = Vec::new();
        for &dim in &[8, 16, 32] {
            let rotations = (dim as f64).log2().ceil() as usize;
            ckks_traces.push(create_test_trace("ckks", rotations));
            ckks_dims.push(dim);
        }

        // Clifford traces
        let mut clifford_traces = Vec::new();
        let mut clifford_dims = Vec::new();
        for &dim in &[8, 16, 32] {
            clifford_traces.push(create_test_trace("clifford", 0));
            clifford_dims.push(dim);
        }

        let comparison = LeakageComparison::compare(
            &ckks_traces, &ckks_dims,
            &clifford_traces, &clifford_dims,
        );

        // Privacy improvement should be positive
        assert!(comparison.privacy_improvement_bits > 0.0,
            "Privacy improvement should be positive");
        assert!(comparison.privacy_improvement_ratio > 0.5,
            "Privacy improvement ratio should be > 50%");
    }
}
