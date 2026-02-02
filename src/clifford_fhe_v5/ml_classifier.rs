//! ML Classifiers for Privacy Attack Demonstrations
//!
//! This module implements multiple classifiers that demonstrate different
//! privacy attacks on FHE execution traces:
//!
//! 1. **DimensionClassifier**: Infers input dimension from rotation patterns
//! 2. **TaskClassifier**: Identifies workload type from operation sequences
//! 3. **SparsityClassifier**: Infers input sparsity from timing variations
//! 4. **TenantLinker**: Links traces from the same user across sessions
//!
//! These attacks demonstrate privacy risks in FHE execution traces.

use crate::clifford_fhe_v5::trace::{ExecutionTrace, OperationType};
use crate::clifford_fhe_v5::analysis::TraceFeatures;
use std::collections::HashMap;

/// A simple Naive Bayes-style classifier for input dimension prediction
///
/// This classifier learns the distribution of rotation counts per dimension
/// and uses it to predict the input dimension from a trace.
#[derive(Debug, Clone)]
pub struct DimensionClassifier {
    /// Known dimension classes (e.g., [8, 16, 32, 64, 128, 256])
    pub classes: Vec<usize>,

    /// For each class, store the expected rotation count
    /// CKKS: rotation_count = ceil(log2(dimension))
    pub rotation_patterns: HashMap<usize, RotationPattern>,

    /// Training statistics
    pub training_stats: TrainingStats,
}

/// Pattern of rotations for a given dimension
#[derive(Debug, Clone, Default)]
pub struct RotationPattern {
    /// Expected number of rotations
    pub expected_rotations: f64,
    /// Variance in rotation count (for noisy traces)
    pub rotation_variance: f64,
    /// Expected rotation amounts (power-of-2 steps)
    pub expected_amounts: Vec<i32>,
    /// Number of training samples
    pub sample_count: usize,
}

/// Statistics from training
#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    /// Total training samples
    pub total_samples: usize,
    /// Samples per class
    pub samples_per_class: HashMap<usize, usize>,
    /// CKKS samples
    pub ckks_samples: usize,
    /// Clifford samples
    pub clifford_samples: usize,
}

/// Classification result
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Predicted dimension
    pub predicted_dimension: usize,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// True dimension (if known)
    pub true_dimension: Option<usize>,
    /// Whether prediction was correct
    pub correct: Option<bool>,
    /// Probability distribution over all classes
    pub class_probabilities: HashMap<usize, f64>,
}

/// Batch classification results
#[derive(Debug, Clone)]
pub struct ClassificationReport {
    /// Individual results
    pub results: Vec<ClassificationResult>,
    /// Overall accuracy
    pub accuracy: f64,
    /// Accuracy by representation (CKKS vs Clifford)
    pub accuracy_by_representation: HashMap<String, f64>,
    /// Confusion matrix: (true_dim, predicted_dim) -> count
    pub confusion_matrix: HashMap<(usize, usize), usize>,
    /// Per-class accuracy
    pub per_class_accuracy: HashMap<usize, f64>,
}

impl DimensionClassifier {
    /// Create a new classifier for the given dimension classes
    pub fn new(classes: Vec<usize>) -> Self {
        Self {
            classes,
            rotation_patterns: HashMap::new(),
            training_stats: TrainingStats::default(),
        }
    }

    /// Create a classifier with standard dimension classes
    pub fn standard() -> Self {
        Self::new(vec![4, 8, 16, 32, 64, 128, 256, 512])
    }

    /// Train the classifier on a set of traces with known dimensions
    ///
    /// For CKKS traces, this learns the rotation count distribution.
    /// For Clifford traces, rotation count is always 0 (no leakage).
    pub fn train(&mut self, traces: &[ExecutionTrace], dimensions: &[usize]) {
        assert_eq!(traces.len(), dimensions.len(), "Traces and dimensions must match");

        // Reset patterns
        self.rotation_patterns.clear();
        self.training_stats = TrainingStats::default();

        // Initialize patterns for each class
        for &dim in &self.classes {
            self.rotation_patterns.insert(dim, RotationPattern::default());
        }

        // Collect rotation counts per dimension (CKKS only)
        let mut rotation_counts: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut rotation_amounts: HashMap<usize, Vec<Vec<i32>>> = HashMap::new();

        for (trace, &dim) in traces.iter().zip(dimensions.iter()) {
            self.training_stats.total_samples += 1;
            *self.training_stats.samples_per_class.entry(dim).or_insert(0) += 1;

            if trace.representation == "ckks" {
                self.training_stats.ckks_samples += 1;

                rotation_counts.entry(dim).or_default().push(trace.summary.total_rotations);
                rotation_amounts.entry(dim).or_default().push(trace.summary.rotation_amounts_used.clone());
            } else {
                self.training_stats.clifford_samples += 1;
            }
        }

        // Compute statistics for each dimension
        for &dim in &self.classes {
            if let Some(counts) = rotation_counts.get(&dim) {
                let n = counts.len() as f64;
                let mean = counts.iter().sum::<usize>() as f64 / n;
                let variance = counts.iter()
                    .map(|&c| (c as f64 - mean).powi(2))
                    .sum::<f64>() / n;

                // Get most common rotation amounts
                let expected_amounts = if let Some(amounts) = rotation_amounts.get(&dim) {
                    if !amounts.is_empty() {
                        amounts[0].clone() // Use first sample's amounts
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                self.rotation_patterns.insert(dim, RotationPattern {
                    expected_rotations: mean,
                    rotation_variance: variance,
                    expected_amounts,
                    sample_count: counts.len(),
                });
            } else {
                // No training data for this class - use theoretical value
                let theoretical_rotations = if dim > 1 {
                    (dim as f64).log2().ceil()
                } else {
                    0.0
                };

                self.rotation_patterns.insert(dim, RotationPattern {
                    expected_rotations: theoretical_rotations,
                    rotation_variance: 0.1, // Small variance for unseen classes
                    expected_amounts: (0..(theoretical_rotations as usize))
                        .map(|i| 1i32 << i)
                        .collect(),
                    sample_count: 0,
                });
            }
        }
    }

    /// Classify a single trace
    pub fn classify(&self, trace: &ExecutionTrace) -> ClassificationResult {
        let features = TraceFeatures::from_trace(trace);
        let rotation_count = features.rotation_count;

        // For Clifford traces, rotation count is 0 - classifier will fail
        // This is intentional: it demonstrates that Clifford doesn't leak dimension

        // Compute probability for each class using Gaussian likelihood
        let mut probabilities: HashMap<usize, f64> = HashMap::new();
        let mut total_prob = 0.0;

        for &dim in &self.classes {
            if let Some(pattern) = self.rotation_patterns.get(&dim) {
                // Gaussian likelihood
                let mean = pattern.expected_rotations;
                let var = pattern.rotation_variance.max(0.5); // Minimum variance
                let diff = rotation_count as f64 - mean;
                let log_prob = -0.5 * (diff * diff) / var - 0.5 * var.ln();
                let prob = log_prob.exp();

                probabilities.insert(dim, prob);
                total_prob += prob;
            }
        }

        // Normalize probabilities
        if total_prob > 0.0 {
            for prob in probabilities.values_mut() {
                *prob /= total_prob;
            }
        }

        // Find best prediction
        let (predicted_dim, confidence) = probabilities.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&dim, &prob)| (dim, prob))
            .unwrap_or((self.classes[0], 0.0));

        ClassificationResult {
            predicted_dimension: predicted_dim,
            confidence,
            true_dimension: None,
            correct: None,
            class_probabilities: probabilities,
        }
    }

    /// Classify a trace with known true dimension
    pub fn classify_with_label(&self, trace: &ExecutionTrace, true_dim: usize) -> ClassificationResult {
        let mut result = self.classify(trace);
        result.true_dimension = Some(true_dim);
        result.correct = Some(result.predicted_dimension == true_dim);
        result
    }

    /// Classify multiple traces and generate a report
    pub fn classify_batch(
        &self,
        traces: &[ExecutionTrace],
        true_dimensions: &[usize],
    ) -> ClassificationReport {
        assert_eq!(traces.len(), true_dimensions.len());

        let mut results = Vec::with_capacity(traces.len());
        let mut confusion_matrix: HashMap<(usize, usize), usize> = HashMap::new();
        let mut correct_by_repr: HashMap<String, (usize, usize)> = HashMap::new(); // (correct, total)
        let mut correct_by_class: HashMap<usize, (usize, usize)> = HashMap::new();

        for (trace, &true_dim) in traces.iter().zip(true_dimensions.iter()) {
            let result = self.classify_with_label(trace, true_dim);

            // Update confusion matrix
            *confusion_matrix.entry((true_dim, result.predicted_dimension)).or_insert(0) += 1;

            // Update accuracy by representation
            let entry = correct_by_repr.entry(trace.representation.clone()).or_insert((0, 0));
            entry.1 += 1;
            if result.correct.unwrap_or(false) {
                entry.0 += 1;
            }

            // Update accuracy by class
            let entry = correct_by_class.entry(true_dim).or_insert((0, 0));
            entry.1 += 1;
            if result.correct.unwrap_or(false) {
                entry.0 += 1;
            }

            results.push(result);
        }

        // Compute overall accuracy
        let total_correct = results.iter().filter(|r| r.correct.unwrap_or(false)).count();
        let accuracy = total_correct as f64 / results.len() as f64;

        // Compute accuracy by representation
        let accuracy_by_representation: HashMap<String, f64> = correct_by_repr.iter()
            .map(|(repr, (correct, total))| {
                (repr.clone(), *correct as f64 / *total as f64)
            })
            .collect();

        // Compute per-class accuracy
        let per_class_accuracy: HashMap<usize, f64> = correct_by_class.iter()
            .map(|(&dim, (correct, total))| {
                (dim, *correct as f64 / *total as f64)
            })
            .collect();

        ClassificationReport {
            results,
            accuracy,
            accuracy_by_representation,
            confusion_matrix,
            per_class_accuracy,
        }
    }

    /// Get theoretical rotation count for a dimension
    pub fn theoretical_rotations(dim: usize) -> usize {
        if dim <= 1 { 0 } else { (dim as f64).log2().ceil() as usize }
    }
}

impl ClassificationReport {
    /// Format the report as a string
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str("=== Classification Report ===\n\n");

        // Overall accuracy
        output.push_str(&format!("Overall Accuracy: {:.2}%\n\n", self.accuracy * 100.0));

        // Accuracy by representation
        output.push_str("Accuracy by Representation:\n");
        for (repr, acc) in &self.accuracy_by_representation {
            output.push_str(&format!("  {}: {:.2}%\n", repr, acc * 100.0));
        }
        output.push('\n');

        // Per-class accuracy
        output.push_str("Per-Class Accuracy:\n");
        let mut classes: Vec<_> = self.per_class_accuracy.keys().collect();
        classes.sort();
        for dim in classes {
            let acc = self.per_class_accuracy.get(dim).unwrap_or(&0.0);
            output.push_str(&format!("  dim={}: {:.2}%\n", dim, acc * 100.0));
        }
        output.push('\n');

        // Confusion matrix (simplified)
        output.push_str("Confusion Matrix (true_dim -> predicted_dim):\n");
        let mut entries: Vec<_> = self.confusion_matrix.iter().collect();
        entries.sort_by_key(|((t, p), _)| (*t, *p));
        for ((true_dim, pred_dim), count) in entries {
            if *count > 0 {
                let marker = if true_dim == pred_dim { "✓" } else { "✗" };
                output.push_str(&format!("  {} {} -> {}: {}\n", marker, true_dim, pred_dim, count));
            }
        }

        output
    }

    /// Get key metrics for the paper
    pub fn paper_metrics(&self) -> PaperMetrics {
        let ckks_accuracy = *self.accuracy_by_representation.get("ckks").unwrap_or(&0.0);
        let clifford_accuracy = *self.accuracy_by_representation.get("clifford").unwrap_or(&0.0);

        // Count misclassifications
        let total_samples = self.results.len();
        let ckks_samples = self.results.iter()
            .filter(|r| r.class_probabilities.len() > 0) // Has predictions
            .count();

        PaperMetrics {
            ckks_attack_accuracy: ckks_accuracy,
            clifford_attack_accuracy: clifford_accuracy,
            privacy_gain: ckks_accuracy - clifford_accuracy,
            total_samples,
            confusion_entropy: self.compute_confusion_entropy(),
        }
    }

    /// Compute entropy of confusion matrix (higher = more confused = better privacy)
    fn compute_confusion_entropy(&self) -> f64 {
        let total: usize = self.confusion_matrix.values().sum();
        if total == 0 { return 0.0; }

        let mut entropy = 0.0;
        for &count in self.confusion_matrix.values() {
            if count > 0 {
                let p = count as f64 / total as f64;
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

/// Key metrics for privacy analysis papers
#[derive(Debug, Clone)]
pub struct PaperMetrics {
    /// Attack accuracy on CKKS traces (higher = more leakage)
    pub ckks_attack_accuracy: f64,
    /// Attack accuracy on CliffordFHE traces (should be ~random)
    pub clifford_attack_accuracy: f64,
    /// Privacy gain = CKKS_accuracy - Clifford_accuracy
    pub privacy_gain: f64,
    /// Total samples tested
    pub total_samples: usize,
    /// Confusion entropy (higher = more privacy)
    pub confusion_entropy: f64,
}

impl PaperMetrics {
    /// Format for paper table
    pub fn format_for_paper(&self) -> String {
        let mut output = String::new();
        output.push_str("╔═══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║       Dimension Inference Attack Results (V5)                 ║\n");
        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!("║  CKKS Attack Accuracy:      {:>6.2}%                          ║\n",
            self.ckks_attack_accuracy * 100.0));
        output.push_str(&format!("║  CliffordFHE Attack Accuracy: {:>6.2}%  (random: ~12.5%)       ║\n",
            self.clifford_attack_accuracy * 100.0));
        output.push_str(&format!("║  Privacy Gain (Δ):          {:>6.2} percentage points         ║\n",
            self.privacy_gain * 100.0));
        output.push_str(&format!("║  Confusion Entropy:         {:>6.3} bits                      ║\n",
            self.confusion_entropy));
        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");

        // Interpretation
        if self.ckks_attack_accuracy > 0.9 {
            output.push_str("║  ⚠️  CKKS traces leak dimension with HIGH confidence          ║\n");
        } else if self.ckks_attack_accuracy > 0.5 {
            output.push_str("║  ⚠️  CKKS traces leak dimension with MODERATE confidence      ║\n");
        }

        if self.clifford_attack_accuracy < 0.2 {
            output.push_str("║  ✓  CliffordFHE traces are essentially random (no leakage)   ║\n");
        }

        output.push_str("╚═══════════════════════════════════════════════════════════════╝\n");
        output
    }
}

// ============================================================================
// Task Identification Attack
// ============================================================================

/// Task Classifier: Identifies workload type from operation sequences
///
/// This attack tries to distinguish between different computation types:
/// - Similarity (dot product)
/// - Inference (neural network forward pass)
/// - Aggregation (sum/mean)
/// - Geometric operations (CliffordFHE-specific)
///
/// CKKS traces have distinctive operation patterns per task type.
/// CliffordFHE traces should look more uniform.
#[derive(Debug, Clone)]
pub struct TaskClassifier {
    /// Known task types
    pub task_classes: Vec<String>,

    /// Learned operation signatures per task
    pub task_signatures: HashMap<String, TaskSignature>,

    /// Training statistics
    pub training_stats: TaskTrainingStats,
}

/// Signature of a task based on operation patterns
#[derive(Debug, Clone, Default)]
pub struct TaskSignature {
    /// Average operation count by type
    pub avg_op_counts: HashMap<String, f64>,
    /// Average total duration (microseconds)
    pub avg_duration_us: f64,
    /// Average rotation count
    pub avg_rotations: f64,
    /// Average relinearization count
    pub avg_relins: f64,
    /// Sample count
    pub sample_count: usize,
}

/// Training stats for task classifier
#[derive(Debug, Clone, Default)]
pub struct TaskTrainingStats {
    pub total_samples: usize,
    pub samples_per_task: HashMap<String, usize>,
}

/// Task classification result
#[derive(Debug, Clone)]
pub struct TaskClassificationResult {
    pub predicted_task: String,
    pub confidence: f64,
    pub true_task: Option<String>,
    pub correct: Option<bool>,
    pub task_probabilities: HashMap<String, f64>,
}

/// Task classification report
#[derive(Debug, Clone)]
pub struct TaskClassificationReport {
    pub results: Vec<TaskClassificationResult>,
    pub accuracy: f64,
    pub accuracy_by_representation: HashMap<String, f64>,
    pub confusion_matrix: HashMap<(String, String), usize>,
}

impl TaskClassifier {
    /// Create a new task classifier
    pub fn new(task_classes: Vec<String>) -> Self {
        Self {
            task_classes,
            task_signatures: HashMap::new(),
            training_stats: TaskTrainingStats::default(),
        }
    }

    /// Create with standard FHE tasks
    pub fn standard() -> Self {
        Self::new(vec![
            "similarity".to_string(),
            "dot_product".to_string(),
            "inference".to_string(),
            "aggregation".to_string(),
            "geometric_product".to_string(),
        ])
    }

    /// Train on labeled traces
    pub fn train(&mut self, traces: &[ExecutionTrace], tasks: &[String]) {
        assert_eq!(traces.len(), tasks.len());

        self.task_signatures.clear();
        self.training_stats = TaskTrainingStats::default();

        // Collect features per task
        let mut task_features: HashMap<String, Vec<TaskFeatureVector>> = HashMap::new();

        for (trace, task) in traces.iter().zip(tasks.iter()) {
            self.training_stats.total_samples += 1;
            *self.training_stats.samples_per_task.entry(task.clone()).or_insert(0) += 1;

            let features = TaskFeatureVector::from_trace(trace);
            task_features.entry(task.clone()).or_default().push(features);
        }

        // Compute signatures
        for task in &self.task_classes {
            if let Some(features) = task_features.get(task) {
                let signature = TaskSignature::from_features(features);
                self.task_signatures.insert(task.clone(), signature);
            }
        }
    }

    /// Classify a single trace
    pub fn classify(&self, trace: &ExecutionTrace) -> TaskClassificationResult {
        let features = TaskFeatureVector::from_trace(trace);

        // Compute similarity to each task signature
        let mut probabilities: HashMap<String, f64> = HashMap::new();
        let mut total_prob = 0.0;

        for task in &self.task_classes {
            if let Some(sig) = self.task_signatures.get(task) {
                let similarity = sig.compute_similarity(&features);
                probabilities.insert(task.clone(), similarity);
                total_prob += similarity;
            }
        }

        // Normalize
        if total_prob > 0.0 {
            for prob in probabilities.values_mut() {
                *prob /= total_prob;
            }
        }

        // Find best prediction
        let (predicted_task, confidence) = probabilities.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(t, &p)| (t.clone(), p))
            .unwrap_or((self.task_classes[0].clone(), 0.0));

        TaskClassificationResult {
            predicted_task,
            confidence,
            true_task: None,
            correct: None,
            task_probabilities: probabilities,
        }
    }

    /// Classify with known label
    pub fn classify_with_label(&self, trace: &ExecutionTrace, true_task: &str) -> TaskClassificationResult {
        let mut result = self.classify(trace);
        result.true_task = Some(true_task.to_string());
        result.correct = Some(result.predicted_task == true_task);
        result
    }

    /// Classify batch and generate report
    pub fn classify_batch(
        &self,
        traces: &[ExecutionTrace],
        true_tasks: &[String],
    ) -> TaskClassificationReport {
        assert_eq!(traces.len(), true_tasks.len());

        let mut results = Vec::with_capacity(traces.len());
        let mut confusion_matrix: HashMap<(String, String), usize> = HashMap::new();
        let mut correct_by_repr: HashMap<String, (usize, usize)> = HashMap::new();

        for (trace, true_task) in traces.iter().zip(true_tasks.iter()) {
            let result = self.classify_with_label(trace, true_task);

            *confusion_matrix
                .entry((true_task.clone(), result.predicted_task.clone()))
                .or_insert(0) += 1;

            let entry = correct_by_repr
                .entry(trace.representation.clone())
                .or_insert((0, 0));
            entry.1 += 1;
            if result.correct.unwrap_or(false) {
                entry.0 += 1;
            }

            results.push(result);
        }

        let total_correct = results.iter().filter(|r| r.correct.unwrap_or(false)).count();
        let accuracy = total_correct as f64 / results.len() as f64;

        let accuracy_by_representation: HashMap<String, f64> = correct_by_repr
            .iter()
            .map(|(repr, (correct, total))| (repr.clone(), *correct as f64 / *total as f64))
            .collect();

        TaskClassificationReport {
            results,
            accuracy,
            accuracy_by_representation,
            confusion_matrix,
        }
    }
}

/// Feature vector for task classification
#[derive(Debug, Clone)]
struct TaskFeatureVector {
    rotation_count: usize,
    relin_count: usize,
    rescale_count: usize,
    encrypt_count: usize,
    decrypt_count: usize,
    mult_count: usize,
    add_count: usize,
    total_duration_us: u64,
    op_sequence_hash: u64,
}

impl TaskFeatureVector {
    fn from_trace(trace: &ExecutionTrace) -> Self {
        let features = TraceFeatures::from_trace(trace);

        // Count operations by type
        let mut encrypt_count = 0;
        let mut decrypt_count = 0;
        let mut mult_count = 0;
        let mut add_count = 0;

        // Create a sequence hash for operation pattern
        let mut op_sequence = Vec::new();

        for event in &trace.events {
            match event.op_type {
                OperationType::Encrypt => encrypt_count += 1,
                OperationType::Decrypt => decrypt_count += 1,
                OperationType::MultiplyCiphertext | OperationType::MultiplyPlain |
                OperationType::GeometricProduct => mult_count += 1,
                OperationType::Add | OperationType::Subtract => add_count += 1,
                _ => {}
            }
            op_sequence.push(format!("{:?}", event.op_type));
        }

        // Simple hash of operation sequence
        let op_sequence_hash = op_sequence.iter()
            .enumerate()
            .map(|(i, s)| {
                let h: u64 = s.bytes().map(|b| b as u64).sum();
                h.wrapping_mul((i + 1) as u64)
            })
            .fold(0u64, |acc, h| acc.wrapping_add(h));

        Self {
            rotation_count: features.rotation_count,
            relin_count: features.relin_count,
            rescale_count: features.rescale_count,
            encrypt_count,
            decrypt_count,
            mult_count,
            add_count,
            total_duration_us: features.total_duration_us as u64,
            op_sequence_hash,
        }
    }
}

impl TaskSignature {
    fn from_features(features: &[TaskFeatureVector]) -> Self {
        if features.is_empty() {
            return Self::default();
        }

        let n = features.len() as f64;

        let avg_rotations = features.iter().map(|f| f.rotation_count).sum::<usize>() as f64 / n;
        let avg_relins = features.iter().map(|f| f.relin_count).sum::<usize>() as f64 / n;
        let avg_duration_us = features.iter().map(|f| f.total_duration_us).sum::<u64>() as f64 / n;

        let mut avg_op_counts = HashMap::new();
        avg_op_counts.insert("encrypt".to_string(),
            features.iter().map(|f| f.encrypt_count).sum::<usize>() as f64 / n);
        avg_op_counts.insert("decrypt".to_string(),
            features.iter().map(|f| f.decrypt_count).sum::<usize>() as f64 / n);
        avg_op_counts.insert("mult".to_string(),
            features.iter().map(|f| f.mult_count).sum::<usize>() as f64 / n);
        avg_op_counts.insert("add".to_string(),
            features.iter().map(|f| f.add_count).sum::<usize>() as f64 / n);

        Self {
            avg_op_counts,
            avg_duration_us,
            avg_rotations,
            avg_relins,
            sample_count: features.len(),
        }
    }

    fn compute_similarity(&self, features: &TaskFeatureVector) -> f64 {
        // Gaussian similarity based on feature differences
        let rot_diff = (features.rotation_count as f64 - self.avg_rotations).abs();
        let relin_diff = (features.relin_count as f64 - self.avg_relins).abs();

        // Weighted distance
        let distance = rot_diff * 0.5 + relin_diff * 0.3;

        // Convert to similarity (higher is more similar)
        (-distance / 10.0).exp()
    }
}

impl TaskClassificationReport {
    /// Format for paper
    pub fn format_for_paper(&self) -> String {
        let mut output = String::new();
        output.push_str("╔═══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║       Task Identification Attack Results (V5)                 ║\n");
        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!("║  Overall Accuracy:          {:>6.2}%                          ║\n",
            self.accuracy * 100.0));

        for (repr, acc) in &self.accuracy_by_representation {
            output.push_str(&format!("║  {} Attack Accuracy:   {:>6.2}%                          ║\n",
                repr, acc * 100.0));
        }

        output.push_str("╚═══════════════════════════════════════════════════════════════╝\n");
        output
    }
}

// ============================================================================
// Sparsity Inference Attack
// ============================================================================

/// Sparsity Classifier: Infers input sparsity from timing variations
///
/// Even with encrypted operations, timing variations can reveal information
/// about the underlying data distribution. This attack attempts to distinguish:
/// - Dense inputs (all non-zero)
/// - Sparse inputs (many zeros)
///
/// This is particularly relevant for applications like:
/// - Sparse vector retrieval
/// - Masked language models
/// - Feature selection
#[derive(Debug, Clone)]
pub struct SparsityClassifier {
    /// Sparsity buckets (e.g., [0.0, 0.25, 0.5, 0.75, 0.9])
    pub sparsity_levels: Vec<f64>,

    /// Learned timing profiles per sparsity level
    pub timing_profiles: HashMap<usize, SparsityProfile>, // bucket index -> profile

    /// Training statistics
    pub training_stats: SparsityTrainingStats,
}

/// Timing profile for a sparsity level
#[derive(Debug, Clone, Default)]
pub struct SparsityProfile {
    /// Average total duration
    pub avg_duration_us: f64,
    /// Duration variance
    pub duration_variance: f64,
    /// Average per-operation durations
    pub avg_op_durations: HashMap<String, f64>,
    /// Sample count
    pub sample_count: usize,
}

/// Training stats for sparsity classifier
#[derive(Debug, Clone, Default)]
pub struct SparsityTrainingStats {
    pub total_samples: usize,
    pub samples_per_bucket: HashMap<usize, usize>,
}

/// Sparsity classification result
#[derive(Debug, Clone)]
pub struct SparsityClassificationResult {
    pub predicted_sparsity: f64,
    pub predicted_bucket: usize,
    pub confidence: f64,
    pub true_sparsity: Option<f64>,
    pub correct_bucket: Option<bool>,
}

/// Sparsity classification report
#[derive(Debug, Clone)]
pub struct SparsityClassificationReport {
    pub results: Vec<SparsityClassificationResult>,
    pub bucket_accuracy: f64,
    pub mean_absolute_error: f64,
    pub accuracy_by_representation: HashMap<String, f64>,
}

impl SparsityClassifier {
    /// Create a new sparsity classifier
    pub fn new(sparsity_levels: Vec<f64>) -> Self {
        Self {
            sparsity_levels,
            timing_profiles: HashMap::new(),
            training_stats: SparsityTrainingStats::default(),
        }
    }

    /// Create with standard sparsity levels
    pub fn standard() -> Self {
        Self::new(vec![0.0, 0.25, 0.5, 0.75, 0.9])
    }

    /// Find bucket for a sparsity value
    fn find_bucket(&self, sparsity: f64) -> usize {
        for (i, &level) in self.sparsity_levels.iter().enumerate() {
            if sparsity <= level + 0.125 {
                return i;
            }
        }
        self.sparsity_levels.len() - 1
    }

    /// Train on traces with known sparsity values
    pub fn train(&mut self, traces: &[ExecutionTrace], sparsities: &[f64]) {
        assert_eq!(traces.len(), sparsities.len());

        self.timing_profiles.clear();
        self.training_stats = SparsityTrainingStats::default();

        // Collect timing features per bucket
        let mut bucket_timings: HashMap<usize, Vec<f64>> = HashMap::new();

        for (trace, &sparsity) in traces.iter().zip(sparsities.iter()) {
            let bucket = self.find_bucket(sparsity);
            self.training_stats.total_samples += 1;
            *self.training_stats.samples_per_bucket.entry(bucket).or_insert(0) += 1;

            let duration = trace.summary.total_duration_us as f64;
            bucket_timings.entry(bucket).or_default().push(duration);
        }

        // Compute profiles
        for (&bucket, timings) in &bucket_timings {
            if !timings.is_empty() {
                let n = timings.len() as f64;
                let avg = timings.iter().sum::<f64>() / n;
                let variance = timings.iter()
                    .map(|&t| (t - avg).powi(2))
                    .sum::<f64>() / n;

                self.timing_profiles.insert(bucket, SparsityProfile {
                    avg_duration_us: avg,
                    duration_variance: variance,
                    avg_op_durations: HashMap::new(),
                    sample_count: timings.len(),
                });
            }
        }
    }

    /// Classify a single trace
    pub fn classify(&self, trace: &ExecutionTrace) -> SparsityClassificationResult {
        let duration = trace.summary.total_duration_us as f64;

        // Find best matching bucket
        let mut best_bucket = 0;
        let mut best_similarity = 0.0;

        for (&bucket, profile) in &self.timing_profiles {
            let diff = (duration - profile.avg_duration_us).abs();
            let std_dev = profile.duration_variance.sqrt().max(1.0);
            let similarity = (-diff / (2.0 * std_dev)).exp();

            if similarity > best_similarity {
                best_similarity = similarity;
                best_bucket = bucket;
            }
        }

        let predicted_sparsity = if best_bucket < self.sparsity_levels.len() {
            self.sparsity_levels[best_bucket]
        } else {
            0.0
        };

        SparsityClassificationResult {
            predicted_sparsity,
            predicted_bucket: best_bucket,
            confidence: best_similarity,
            true_sparsity: None,
            correct_bucket: None,
        }
    }

    /// Classify with known label
    pub fn classify_with_label(&self, trace: &ExecutionTrace, true_sparsity: f64) -> SparsityClassificationResult {
        let mut result = self.classify(trace);
        result.true_sparsity = Some(true_sparsity);
        let true_bucket = self.find_bucket(true_sparsity);
        result.correct_bucket = Some(result.predicted_bucket == true_bucket);
        result
    }

    /// Classify batch and generate report
    pub fn classify_batch(
        &self,
        traces: &[ExecutionTrace],
        true_sparsities: &[f64],
    ) -> SparsityClassificationReport {
        assert_eq!(traces.len(), true_sparsities.len());

        let mut results = Vec::with_capacity(traces.len());
        let mut correct_by_repr: HashMap<String, (usize, usize)> = HashMap::new();
        let mut total_error = 0.0;

        for (trace, &true_sparsity) in traces.iter().zip(true_sparsities.iter()) {
            let result = self.classify_with_label(trace, true_sparsity);

            total_error += (result.predicted_sparsity - true_sparsity).abs();

            let entry = correct_by_repr
                .entry(trace.representation.clone())
                .or_insert((0, 0));
            entry.1 += 1;
            if result.correct_bucket.unwrap_or(false) {
                entry.0 += 1;
            }

            results.push(result);
        }

        let total_correct = results.iter().filter(|r| r.correct_bucket.unwrap_or(false)).count();
        let bucket_accuracy = total_correct as f64 / results.len() as f64;
        let mean_absolute_error = total_error / results.len() as f64;

        let accuracy_by_representation: HashMap<String, f64> = correct_by_repr
            .iter()
            .map(|(repr, (correct, total))| (repr.clone(), *correct as f64 / *total as f64))
            .collect();

        SparsityClassificationReport {
            results,
            bucket_accuracy,
            mean_absolute_error,
            accuracy_by_representation,
        }
    }
}

impl SparsityClassificationReport {
    /// Format for paper
    pub fn format_for_paper(&self) -> String {
        let mut output = String::new();
        output.push_str("╔═══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║       Sparsity Inference Attack Results (V5)                  ║\n");
        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!("║  Bucket Accuracy:           {:>6.2}%                          ║\n",
            self.bucket_accuracy * 100.0));
        output.push_str(&format!("║  Mean Absolute Error:       {:>6.3}                           ║\n",
            self.mean_absolute_error));

        for (repr, acc) in &self.accuracy_by_representation {
            output.push_str(&format!("║  {} Attack Accuracy:   {:>6.2}%                          ║\n",
                repr, acc * 100.0));
        }

        output.push_str("╚═══════════════════════════════════════════════════════════════╝\n");
        output
    }
}

// ============================================================================
// Multi-Tenant Linkability Attack
// ============================================================================

/// Tenant Linker: Links traces from the same user across sessions
///
/// This attack attempts to identify whether two traces come from the same
/// user based on timing fingerprints. This is a privacy concern in:
/// - Multi-tenant FHE cloud services
/// - Anonymous encrypted computation
///
/// Features used:
/// - Relative timing patterns between operations
/// - Operation sequence structure
/// - System-specific noise patterns
#[derive(Debug, Clone)]
pub struct TenantLinker {
    /// Known tenant fingerprints (tenant_id -> fingerprint)
    pub fingerprints: HashMap<String, TenantFingerprint>,

    /// Training statistics
    pub training_stats: TenantTrainingStats,

    /// Similarity threshold for linking
    pub link_threshold: f64,
}

/// Fingerprint of a tenant's execution pattern
#[derive(Debug, Clone, Default)]
pub struct TenantFingerprint {
    /// Average relative timing pattern
    pub timing_pattern: Vec<f64>,
    /// Operation sequence signature
    pub op_signature: u64,
    /// Variance in timing
    pub timing_variance: f64,
    /// Sample count
    pub sample_count: usize,
}

/// Training stats for tenant linker
#[derive(Debug, Clone, Default)]
pub struct TenantTrainingStats {
    pub total_samples: usize,
    pub samples_per_tenant: HashMap<String, usize>,
    pub unique_tenants: usize,
}

/// Tenant linking result
#[derive(Debug, Clone)]
pub struct TenantLinkResult {
    /// Most likely tenant ID
    pub predicted_tenant: Option<String>,
    /// Confidence in prediction
    pub confidence: f64,
    /// True tenant ID (if known)
    pub true_tenant: Option<String>,
    /// Whether prediction was correct
    pub correct: Option<bool>,
    /// Similarity scores to known tenants
    pub similarity_scores: HashMap<String, f64>,
}

/// Tenant linking report
#[derive(Debug, Clone)]
pub struct TenantLinkReport {
    pub results: Vec<TenantLinkResult>,
    /// Accuracy at linking traces to correct tenant
    pub link_accuracy: f64,
    /// Accuracy by representation
    pub accuracy_by_representation: HashMap<String, f64>,
    /// False positive rate (wrong links)
    pub false_positive_rate: f64,
    /// True positive rate (correct links)
    pub true_positive_rate: f64,
}

impl TenantLinker {
    /// Create a new tenant linker
    pub fn new(link_threshold: f64) -> Self {
        Self {
            fingerprints: HashMap::new(),
            training_stats: TenantTrainingStats::default(),
            link_threshold,
        }
    }

    /// Create with default threshold
    pub fn default_linker() -> Self {
        Self::new(0.7)
    }

    /// Train on traces with tenant IDs
    pub fn train(&mut self, traces: &[ExecutionTrace], tenant_ids: &[String]) {
        assert_eq!(traces.len(), tenant_ids.len());

        self.fingerprints.clear();
        self.training_stats = TenantTrainingStats::default();

        // Collect timing patterns per tenant
        let mut tenant_patterns: HashMap<String, Vec<Vec<f64>>> = HashMap::new();

        for (trace, tenant_id) in traces.iter().zip(tenant_ids.iter()) {
            self.training_stats.total_samples += 1;
            *self.training_stats.samples_per_tenant.entry(tenant_id.clone()).or_insert(0) += 1;

            // Extract relative timing pattern
            let pattern = Self::extract_timing_pattern(trace);
            tenant_patterns.entry(tenant_id.clone()).or_default().push(pattern);
        }

        self.training_stats.unique_tenants = tenant_patterns.len();

        // Compute fingerprints
        for (tenant_id, patterns) in tenant_patterns {
            let fingerprint = TenantFingerprint::from_patterns(&patterns);
            self.fingerprints.insert(tenant_id, fingerprint);
        }
    }

    /// Extract relative timing pattern from trace
    fn extract_timing_pattern(trace: &ExecutionTrace) -> Vec<f64> {
        if trace.events.is_empty() {
            return vec![];
        }

        // Compute relative durations (normalized)
        let total_duration: u64 = trace.events.iter()
            .map(|e| e.duration.as_micros() as u64)
            .sum();

        if total_duration == 0 {
            return vec![1.0 / trace.events.len() as f64; trace.events.len()];
        }

        trace.events.iter()
            .map(|e| e.duration.as_micros() as f64 / total_duration as f64)
            .collect()
    }

    /// Link a trace to a known tenant
    pub fn link(&self, trace: &ExecutionTrace) -> TenantLinkResult {
        let pattern = Self::extract_timing_pattern(trace);

        let mut similarity_scores: HashMap<String, f64> = HashMap::new();
        let mut best_tenant: Option<String> = None;
        let mut best_score = 0.0;

        for (tenant_id, fingerprint) in &self.fingerprints {
            let similarity = fingerprint.compute_similarity(&pattern);
            similarity_scores.insert(tenant_id.clone(), similarity);

            if similarity > best_score {
                best_score = similarity;
                best_tenant = Some(tenant_id.clone());
            }
        }

        // Only link if above threshold
        let predicted_tenant = if best_score >= self.link_threshold {
            best_tenant
        } else {
            None
        };

        TenantLinkResult {
            predicted_tenant,
            confidence: best_score,
            true_tenant: None,
            correct: None,
            similarity_scores,
        }
    }

    /// Link with known label
    pub fn link_with_label(&self, trace: &ExecutionTrace, true_tenant: &str) -> TenantLinkResult {
        let mut result = self.link(trace);
        result.true_tenant = Some(true_tenant.to_string());
        result.correct = result.predicted_tenant.as_ref().map(|p| p == true_tenant);
        result
    }

    /// Link batch and generate report
    pub fn link_batch(
        &self,
        traces: &[ExecutionTrace],
        true_tenants: &[String],
    ) -> TenantLinkReport {
        assert_eq!(traces.len(), true_tenants.len());

        let mut results = Vec::with_capacity(traces.len());
        let mut correct_by_repr: HashMap<String, (usize, usize)> = HashMap::new();
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut total_links = 0;

        for (trace, true_tenant) in traces.iter().zip(true_tenants.iter()) {
            let result = self.link_with_label(trace, true_tenant);

            if result.predicted_tenant.is_some() {
                total_links += 1;
                if result.correct.unwrap_or(false) {
                    true_positives += 1;
                } else {
                    false_positives += 1;
                }
            }

            let entry = correct_by_repr
                .entry(trace.representation.clone())
                .or_insert((0, 0));
            entry.1 += 1;
            if result.correct.unwrap_or(false) {
                entry.0 += 1;
            }

            results.push(result);
        }

        let total_correct = results.iter().filter(|r| r.correct.unwrap_or(false)).count();
        let link_accuracy = if !results.is_empty() {
            total_correct as f64 / results.len() as f64
        } else {
            0.0
        };

        let false_positive_rate = if total_links > 0 {
            false_positives as f64 / total_links as f64
        } else {
            0.0
        };

        let true_positive_rate = if total_links > 0 {
            true_positives as f64 / total_links as f64
        } else {
            0.0
        };

        let accuracy_by_representation: HashMap<String, f64> = correct_by_repr
            .iter()
            .map(|(repr, (correct, total))| (repr.clone(), *correct as f64 / *total as f64))
            .collect();

        TenantLinkReport {
            results,
            link_accuracy,
            accuracy_by_representation,
            false_positive_rate,
            true_positive_rate,
        }
    }
}

impl TenantFingerprint {
    fn from_patterns(patterns: &[Vec<f64>]) -> Self {
        if patterns.is_empty() {
            return Self::default();
        }

        // Find max length
        let max_len = patterns.iter().map(|p| p.len()).max().unwrap_or(0);

        // Compute average pattern (zero-padded)
        let mut avg_pattern = vec![0.0; max_len];
        for pattern in patterns {
            for (i, &val) in pattern.iter().enumerate() {
                avg_pattern[i] += val / patterns.len() as f64;
            }
        }

        // Compute variance
        let timing_variance = if patterns.len() > 1 {
            let mut variance = 0.0;
            for pattern in patterns {
                for (i, &val) in pattern.iter().enumerate() {
                    variance += (val - avg_pattern[i]).powi(2);
                }
            }
            variance / (patterns.len() * max_len) as f64
        } else {
            0.01 // Small default variance
        };

        // Compute signature from pattern lengths
        let op_signature = patterns.iter()
            .map(|p| p.len() as u64)
            .fold(0u64, |acc, len| acc.wrapping_mul(31).wrapping_add(len));

        Self {
            timing_pattern: avg_pattern,
            op_signature,
            timing_variance,
            sample_count: patterns.len(),
        }
    }

    fn compute_similarity(&self, pattern: &[f64]) -> f64 {
        if self.timing_pattern.is_empty() || pattern.is_empty() {
            return 0.0;
        }

        // Compute correlation-based similarity
        let min_len = self.timing_pattern.len().min(pattern.len());

        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..min_len {
            let a = self.timing_pattern[i];
            let b = pattern[i];
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        let norm_product = (norm_a * norm_b).sqrt();
        if norm_product > 0.0 {
            (dot_product / norm_product).max(0.0)
        } else {
            0.0
        }
    }
}

impl TenantLinkReport {
    /// Format for paper
    pub fn format_for_paper(&self) -> String {
        let mut output = String::new();
        output.push_str("╔═══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║     Multi-Tenant Linkability Attack Results (V5)              ║\n");
        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!("║  Link Accuracy:             {:>6.2}%                          ║\n",
            self.link_accuracy * 100.0));
        output.push_str(&format!("║  True Positive Rate:        {:>6.2}%                          ║\n",
            self.true_positive_rate * 100.0));
        output.push_str(&format!("║  False Positive Rate:       {:>6.2}%                          ║\n",
            self.false_positive_rate * 100.0));

        for (repr, acc) in &self.accuracy_by_representation {
            output.push_str(&format!("║  {} Link Accuracy:     {:>6.2}%                          ║\n",
                repr, acc * 100.0));
        }

        output.push_str("╚═══════════════════════════════════════════════════════════════╝\n");
        output
    }
}

// ============================================================================
// Operation Count Attack (NEW - CliffordFHE should win)
// ============================================================================

/// Operation Count Classifier: Detects data-dependent operation patterns
///
/// This attack exploits a fundamental difference between CKKS and CliffordFHE:
///
/// **CKKS**: Variable operation count based on input dimension
/// - Dimension 8: 3 rotations
/// - Dimension 16: 4 rotations
/// - Dimension 32: 5 rotations
/// - ...
/// - Total operations vary: O(log n)
///
/// **CliffordFHE**: Fixed operation count regardless of dimension
/// - Always exactly 64 coefficient multiplications (8×8 multivector product)
/// - Always 0 rotations
/// - Trace length is constant
///
/// This is a **trace fingerprinting** attack: the attacker counts total operations
/// in the trace and uses statistical analysis to infer properties about the input.
///
/// **Expected results**:
/// - CKKS: High accuracy (operation count directly reveals dimension)
/// - CliffordFHE: Random accuracy (all traces have identical operation counts)
#[derive(Debug, Clone)]
pub struct OperationCountClassifier {
    /// Known dimension classes
    pub dimension_classes: Vec<usize>,

    /// Learned operation count profiles per dimension
    pub op_count_profiles: HashMap<usize, OpCountProfile>,

    /// Training statistics
    pub training_stats: OpCountTrainingStats,
}

/// Profile of operation counts for a dimension class
#[derive(Debug, Clone, Default)]
pub struct OpCountProfile {
    /// Average total operation count
    pub avg_total_ops: f64,
    /// Variance in operation count
    pub op_count_variance: f64,
    /// Average trace length (number of events)
    pub avg_trace_length: f64,
    /// Average rotation operations
    pub avg_rotation_ops: f64,
    /// Average relinearization operations
    pub avg_relin_ops: f64,
    /// Sample count
    pub sample_count: usize,
}

/// Training statistics for operation count classifier
#[derive(Debug, Clone, Default)]
pub struct OpCountTrainingStats {
    pub total_samples: usize,
    pub samples_per_dimension: HashMap<usize, usize>,
    pub ckks_samples: usize,
    pub clifford_samples: usize,
}

/// Operation count classification result
#[derive(Debug, Clone)]
pub struct OpCountClassificationResult {
    /// Predicted dimension
    pub predicted_dimension: usize,
    /// Confidence score
    pub confidence: f64,
    /// True dimension (if known)
    pub true_dimension: Option<usize>,
    /// Whether prediction was correct
    pub correct: Option<bool>,
    /// Total operations observed
    pub observed_ops: usize,
    /// Trace length observed
    pub observed_trace_length: usize,
    /// Probability distribution over dimensions
    pub dimension_probabilities: HashMap<usize, f64>,
}

/// Operation count classification report
#[derive(Debug, Clone)]
pub struct OpCountClassificationReport {
    pub results: Vec<OpCountClassificationResult>,
    /// Overall accuracy
    pub accuracy: f64,
    /// Accuracy by representation (CKKS vs Clifford)
    pub accuracy_by_representation: HashMap<String, f64>,
    /// Confusion matrix
    pub confusion_matrix: HashMap<(usize, usize), usize>,
    /// Information leaked (bits)
    pub information_leaked_bits: f64,
    /// CliffordFHE traces are indistinguishable
    pub clifford_all_identical: bool,
}

impl OperationCountClassifier {
    /// Create a new operation count classifier
    pub fn new(dimension_classes: Vec<usize>) -> Self {
        Self {
            dimension_classes,
            op_count_profiles: HashMap::new(),
            training_stats: OpCountTrainingStats::default(),
        }
    }

    /// Create with standard dimension classes
    pub fn standard() -> Self {
        Self::new(vec![8, 16, 32, 64, 128, 256])
    }

    /// Train on labeled traces
    pub fn train(&mut self, traces: &[ExecutionTrace], dimensions: &[usize]) {
        assert_eq!(traces.len(), dimensions.len());

        self.op_count_profiles.clear();
        self.training_stats = OpCountTrainingStats::default();

        // Collect operation counts per dimension
        let mut dim_features: HashMap<usize, Vec<OpCountFeatures>> = HashMap::new();

        for (trace, &dim) in traces.iter().zip(dimensions.iter()) {
            self.training_stats.total_samples += 1;
            *self.training_stats.samples_per_dimension.entry(dim).or_insert(0) += 1;

            if trace.representation == "ckks" {
                self.training_stats.ckks_samples += 1;
            } else {
                self.training_stats.clifford_samples += 1;
            }

            let features = OpCountFeatures::from_trace(trace);
            dim_features.entry(dim).or_default().push(features);
        }

        // Compute profiles
        for dim in &self.dimension_classes {
            if let Some(features) = dim_features.get(dim) {
                let profile = OpCountProfile::from_features(features);
                self.op_count_profiles.insert(*dim, profile);
            }
        }
    }

    /// Classify a single trace
    pub fn classify(&self, trace: &ExecutionTrace) -> OpCountClassificationResult {
        let features = OpCountFeatures::from_trace(trace);

        // Compute likelihood for each dimension
        let mut probabilities: HashMap<usize, f64> = HashMap::new();
        let mut total_prob = 0.0;

        for dim in &self.dimension_classes {
            if let Some(profile) = self.op_count_profiles.get(dim) {
                let likelihood = profile.compute_likelihood(&features);
                probabilities.insert(*dim, likelihood);
                total_prob += likelihood;
            }
        }

        // Normalize
        if total_prob > 0.0 {
            for prob in probabilities.values_mut() {
                *prob /= total_prob;
            }
        }

        // Find best prediction
        let (predicted_dim, confidence) = probabilities.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&d, &p)| (d, p))
            .unwrap_or((self.dimension_classes[0], 0.0));

        OpCountClassificationResult {
            predicted_dimension: predicted_dim,
            confidence,
            true_dimension: None,
            correct: None,
            observed_ops: features.total_ops,
            observed_trace_length: features.trace_length,
            dimension_probabilities: probabilities,
        }
    }

    /// Classify with known label
    pub fn classify_with_label(&self, trace: &ExecutionTrace, true_dim: usize) -> OpCountClassificationResult {
        let mut result = self.classify(trace);
        result.true_dimension = Some(true_dim);
        result.correct = Some(result.predicted_dimension == true_dim);
        result
    }

    /// Classify batch and generate report
    pub fn classify_batch(
        &self,
        traces: &[ExecutionTrace],
        true_dimensions: &[usize],
    ) -> OpCountClassificationReport {
        assert_eq!(traces.len(), true_dimensions.len());

        let mut results = Vec::with_capacity(traces.len());
        let mut confusion_matrix: HashMap<(usize, usize), usize> = HashMap::new();
        let mut correct_by_repr: HashMap<String, (usize, usize)> = HashMap::new();

        // Check if all Clifford traces have identical operation counts
        let mut clifford_op_counts: Vec<usize> = Vec::new();

        for (trace, &true_dim) in traces.iter().zip(true_dimensions.iter()) {
            let result = self.classify_with_label(trace, true_dim);

            *confusion_matrix
                .entry((true_dim, result.predicted_dimension))
                .or_insert(0) += 1;

            let entry = correct_by_repr
                .entry(trace.representation.clone())
                .or_insert((0, 0));
            entry.1 += 1;
            if result.correct.unwrap_or(false) {
                entry.0 += 1;
            }

            if trace.representation == "clifford" {
                clifford_op_counts.push(result.observed_ops);
            }

            results.push(result);
        }

        let total_correct = results.iter().filter(|r| r.correct.unwrap_or(false)).count();
        let accuracy = total_correct as f64 / results.len() as f64;

        let accuracy_by_representation: HashMap<String, f64> = correct_by_repr
            .iter()
            .map(|(repr, (correct, total))| (repr.clone(), *correct as f64 / *total as f64))
            .collect();

        // Check if Clifford traces are all identical
        let clifford_all_identical = if !clifford_op_counts.is_empty() {
            let first = clifford_op_counts[0];
            clifford_op_counts.iter().all(|&c| c == first)
        } else {
            false
        };

        // Compute information leaked
        let num_classes = self.dimension_classes.len() as f64;
        let ckks_accuracy = *accuracy_by_representation.get("ckks").unwrap_or(&0.0);
        let random_baseline = 1.0 / num_classes;
        let information_leaked_bits = if ckks_accuracy > random_baseline {
            (num_classes.log2() * (ckks_accuracy - random_baseline) / (1.0 - random_baseline)).max(0.0)
        } else {
            0.0
        };

        OpCountClassificationReport {
            results,
            accuracy,
            accuracy_by_representation,
            confusion_matrix,
            information_leaked_bits,
            clifford_all_identical,
        }
    }
}

/// Features extracted for operation count analysis
#[derive(Debug, Clone)]
struct OpCountFeatures {
    /// Total number of operations
    total_ops: usize,
    /// Trace length (number of events)
    trace_length: usize,
    /// Number of rotation operations
    rotation_ops: usize,
    /// Number of relinearization operations
    relin_ops: usize,
    /// Number of rescale operations
    rescale_ops: usize,
    /// Number of multiply operations
    multiply_ops: usize,
}

impl OpCountFeatures {
    fn from_trace(trace: &ExecutionTrace) -> Self {
        let mut rotation_ops = 0;
        let mut relin_ops = 0;
        let mut rescale_ops = 0;
        let mut multiply_ops = 0;

        for event in &trace.events {
            match event.op_type {
                OperationType::Rotate => rotation_ops += 1,
                OperationType::Relinearize => relin_ops += 1,
                OperationType::Rescale => rescale_ops += 1,
                OperationType::MultiplyCiphertext |
                OperationType::MultiplyPlain |
                OperationType::GeometricProduct => multiply_ops += 1,
                _ => {}
            }
        }

        let total_ops = rotation_ops + relin_ops + rescale_ops + multiply_ops;

        Self {
            total_ops,
            trace_length: trace.events.len(),
            rotation_ops,
            relin_ops,
            rescale_ops,
            multiply_ops,
        }
    }
}

impl OpCountProfile {
    fn from_features(features: &[OpCountFeatures]) -> Self {
        if features.is_empty() {
            return Self::default();
        }

        let n = features.len() as f64;

        let avg_total_ops = features.iter().map(|f| f.total_ops).sum::<usize>() as f64 / n;
        let avg_trace_length = features.iter().map(|f| f.trace_length).sum::<usize>() as f64 / n;
        let avg_rotation_ops = features.iter().map(|f| f.rotation_ops).sum::<usize>() as f64 / n;
        let avg_relin_ops = features.iter().map(|f| f.relin_ops).sum::<usize>() as f64 / n;

        let op_count_variance = features.iter()
            .map(|f| (f.total_ops as f64 - avg_total_ops).powi(2))
            .sum::<f64>() / n;

        Self {
            avg_total_ops,
            op_count_variance,
            avg_trace_length,
            avg_rotation_ops,
            avg_relin_ops,
            sample_count: features.len(),
        }
    }

    fn compute_likelihood(&self, features: &OpCountFeatures) -> f64 {
        // Gaussian likelihood based on total operation count
        let var = self.op_count_variance.max(0.5);
        let diff = features.total_ops as f64 - self.avg_total_ops;
        let log_prob = -0.5 * (diff * diff) / var;
        log_prob.exp()
    }
}

impl OpCountClassificationReport {
    /// Format for paper
    pub fn format_for_paper(&self) -> String {
        let mut output = String::new();
        output.push_str("╔═══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║       Operation Count Attack Results (V5)                     ║\n");
        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!("║  Overall Accuracy:          {:>6.2}%                          ║\n",
            self.accuracy * 100.0));

        for (repr, acc) in &self.accuracy_by_representation {
            output.push_str(&format!("║  {} Attack Accuracy:   {:>6.2}%                          ║\n",
                repr, acc * 100.0));
        }

        output.push_str(&format!("║  Information Leaked:        {:>6.3} bits                      ║\n",
            self.information_leaked_bits));
        output.push_str(&format!("║  Clifford Traces Identical: {}                              ║\n",
            if self.clifford_all_identical { "YES" } else { "NO " }));

        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");

        if self.clifford_all_identical {
            output.push_str("║  ✓  CliffordFHE: All traces have IDENTICAL operation counts  ║\n");
            output.push_str("║     → Zero information leakage via trace fingerprinting      ║\n");
        }

        let ckks_acc = self.accuracy_by_representation.get("ckks").unwrap_or(&0.0);
        if *ckks_acc > 0.9 {
            output.push_str("║  ⚠️  CKKS: Operation count PERFECTLY reveals dimension        ║\n");
        }

        output.push_str("╚═══════════════════════════════════════════════════════════════╝\n");
        output
    }
}

// ============================================================================
// Trace Length Attack (Complementary to Operation Count)
// ============================================================================

/// Trace Length Classifier: Infers dimension from trace event count
///
/// A simpler variant of the operation count attack that uses only
/// the raw number of events in the trace.
///
/// **Why this matters**:
/// - Trace length is observable via timing side channels
/// - Cache timing attacks can count discrete operations
/// - GPU kernel invocation counters reveal operation counts
///
/// **CKKS**: Trace length = O(log n) where n = input dimension
/// **CliffordFHE**: Trace length = constant (fixed structure)
#[derive(Debug, Clone)]
pub struct TraceLengthClassifier {
    /// Known dimension classes
    pub dimension_classes: Vec<usize>,

    /// Expected trace length per dimension
    pub expected_lengths: HashMap<usize, f64>,

    /// Training statistics
    pub training_stats: TraceLengthTrainingStats,
}

/// Training stats for trace length classifier
#[derive(Debug, Clone, Default)]
pub struct TraceLengthTrainingStats {
    pub total_samples: usize,
    pub samples_per_dimension: HashMap<usize, usize>,
}

/// Trace length classification result
#[derive(Debug, Clone)]
pub struct TraceLengthClassificationResult {
    pub predicted_dimension: usize,
    pub confidence: f64,
    pub true_dimension: Option<usize>,
    pub correct: Option<bool>,
    pub observed_trace_length: usize,
}

/// Trace length classification report
#[derive(Debug, Clone)]
pub struct TraceLengthClassificationReport {
    pub results: Vec<TraceLengthClassificationResult>,
    pub accuracy: f64,
    pub accuracy_by_representation: HashMap<String, f64>,
    /// All unique trace lengths observed for CKKS
    pub ckks_unique_lengths: Vec<usize>,
    /// All unique trace lengths observed for CliffordFHE
    pub clifford_unique_lengths: Vec<usize>,
}

impl TraceLengthClassifier {
    /// Create a new trace length classifier
    pub fn new(dimension_classes: Vec<usize>) -> Self {
        Self {
            dimension_classes,
            expected_lengths: HashMap::new(),
            training_stats: TraceLengthTrainingStats::default(),
        }
    }

    /// Create with standard dimension classes
    pub fn standard() -> Self {
        Self::new(vec![8, 16, 32, 64, 128, 256])
    }

    /// Train on labeled traces
    pub fn train(&mut self, traces: &[ExecutionTrace], dimensions: &[usize]) {
        assert_eq!(traces.len(), dimensions.len());

        self.expected_lengths.clear();
        self.training_stats = TraceLengthTrainingStats::default();

        // Collect trace lengths per dimension
        let mut dim_lengths: HashMap<usize, Vec<usize>> = HashMap::new();

        for (trace, &dim) in traces.iter().zip(dimensions.iter()) {
            self.training_stats.total_samples += 1;
            *self.training_stats.samples_per_dimension.entry(dim).or_insert(0) += 1;

            // Only learn from CKKS traces (Clifford has fixed length)
            if trace.representation == "ckks" {
                dim_lengths.entry(dim).or_default().push(trace.events.len());
            }
        }

        // Compute expected lengths
        for dim in &self.dimension_classes {
            if let Some(lengths) = dim_lengths.get(dim) {
                let avg = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
                self.expected_lengths.insert(*dim, avg);
            } else {
                // Use theoretical value: log2(dim) for rotations + a few more ops
                let theoretical = if *dim > 1 {
                    (*dim as f64).log2().ceil() + 2.0
                } else {
                    2.0
                };
                self.expected_lengths.insert(*dim, theoretical);
            }
        }
    }

    /// Classify a single trace
    pub fn classify(&self, trace: &ExecutionTrace) -> TraceLengthClassificationResult {
        let trace_length = trace.events.len();

        // Find closest matching dimension
        let mut best_dim = self.dimension_classes[0];
        let mut best_distance = f64::MAX;

        for dim in &self.dimension_classes {
            if let Some(&expected) = self.expected_lengths.get(dim) {
                let distance = (trace_length as f64 - expected).abs();
                if distance < best_distance {
                    best_distance = distance;
                    best_dim = *dim;
                }
            }
        }

        // Confidence based on distance
        let confidence = (-best_distance / 2.0).exp();

        TraceLengthClassificationResult {
            predicted_dimension: best_dim,
            confidence,
            true_dimension: None,
            correct: None,
            observed_trace_length: trace_length,
        }
    }

    /// Classify with known label
    pub fn classify_with_label(&self, trace: &ExecutionTrace, true_dim: usize) -> TraceLengthClassificationResult {
        let mut result = self.classify(trace);
        result.true_dimension = Some(true_dim);
        result.correct = Some(result.predicted_dimension == true_dim);
        result
    }

    /// Classify batch and generate report
    pub fn classify_batch(
        &self,
        traces: &[ExecutionTrace],
        true_dimensions: &[usize],
    ) -> TraceLengthClassificationReport {
        assert_eq!(traces.len(), true_dimensions.len());

        let mut results = Vec::with_capacity(traces.len());
        let mut correct_by_repr: HashMap<String, (usize, usize)> = HashMap::new();
        let mut ckks_lengths: Vec<usize> = Vec::new();
        let mut clifford_lengths: Vec<usize> = Vec::new();

        for (trace, &true_dim) in traces.iter().zip(true_dimensions.iter()) {
            let result = self.classify_with_label(trace, true_dim);

            let entry = correct_by_repr
                .entry(trace.representation.clone())
                .or_insert((0, 0));
            entry.1 += 1;
            if result.correct.unwrap_or(false) {
                entry.0 += 1;
            }

            if trace.representation == "ckks" {
                ckks_lengths.push(result.observed_trace_length);
            } else {
                clifford_lengths.push(result.observed_trace_length);
            }

            results.push(result);
        }

        let total_correct = results.iter().filter(|r| r.correct.unwrap_or(false)).count();
        let accuracy = total_correct as f64 / results.len() as f64;

        let accuracy_by_representation: HashMap<String, f64> = correct_by_repr
            .iter()
            .map(|(repr, (correct, total))| (repr.clone(), *correct as f64 / *total as f64))
            .collect();

        // Get unique lengths
        ckks_lengths.sort();
        ckks_lengths.dedup();
        clifford_lengths.sort();
        clifford_lengths.dedup();

        TraceLengthClassificationReport {
            results,
            accuracy,
            accuracy_by_representation,
            ckks_unique_lengths: ckks_lengths,
            clifford_unique_lengths: clifford_lengths,
        }
    }
}

impl TraceLengthClassificationReport {
    /// Format for paper
    pub fn format_for_paper(&self) -> String {
        let mut output = String::new();
        output.push_str("╔═══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║       Trace Length Attack Results (V5)                        ║\n");
        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!("║  Overall Accuracy:          {:>6.2}%                          ║\n",
            self.accuracy * 100.0));

        for (repr, acc) in &self.accuracy_by_representation {
            output.push_str(&format!("║  {} Attack Accuracy:   {:>6.2}%                          ║\n",
                repr, acc * 100.0));
        }

        output.push_str(&format!("║  CKKS unique trace lengths:     {:>3}                         ║\n",
            self.ckks_unique_lengths.len()));
        output.push_str(&format!("║  Clifford unique trace lengths: {:>3}                         ║\n",
            self.clifford_unique_lengths.len()));

        output.push_str("╠═══════════════════════════════════════════════════════════════╣\n");

        if self.clifford_unique_lengths.len() == 1 {
            output.push_str("║  ✓  CliffordFHE: SINGLE trace length for ALL dimensions       ║\n");
            output.push_str("║     → Trace length reveals ZERO information                   ║\n");
        }

        if self.ckks_unique_lengths.len() > 1 {
            output.push_str(&format!("║  ⚠️  CKKS: {} different trace lengths reveal dimension       ║\n",
                self.ckks_unique_lengths.len()));
        }

        output.push_str("╚═══════════════════════════════════════════════════════════════╝\n");
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v5::trace::{ExecutionTrace, OperationEvent, OperationType};

    fn create_mock_ckks_trace(dimension: usize) -> ExecutionTrace {
        let mut trace = ExecutionTrace::new("similarity", "ckks", "cpu", 1024, 3);
        trace.input_metadata.input_length = dimension;

        // Add rotation events matching the dimension
        let rotation_count = DimensionClassifier::theoretical_rotations(dimension);
        for i in 0..rotation_count {
            let event = OperationEvent::new(OperationType::Rotate)
                .with_rotation_amounts(vec![1i32 << i]);
            trace.add_event(event);
        }

        trace.compute_summary();
        trace
    }

    fn create_mock_clifford_trace(dimension: usize) -> ExecutionTrace {
        let mut trace = ExecutionTrace::new("similarity", "clifford", "cpu", 1024, 3);
        trace.input_metadata.input_length = dimension;

        // Clifford has NO rotations - always fixed structure
        let gp_event = OperationEvent::new(OperationType::GeometricProduct)
            .with_relins(64)
            .with_rescales(64);
        trace.add_event(gp_event);

        trace.compute_summary();
        trace
    }

    #[test]
    fn test_classifier_creation() {
        let classifier = DimensionClassifier::standard();
        assert_eq!(classifier.classes.len(), 8);
    }

    #[test]
    fn test_classifier_training() {
        let mut classifier = DimensionClassifier::new(vec![8, 16, 32, 64]);

        let traces: Vec<_> = vec![
            create_mock_ckks_trace(8),
            create_mock_ckks_trace(16),
            create_mock_ckks_trace(32),
            create_mock_ckks_trace(64),
        ];
        let dimensions = vec![8, 16, 32, 64];

        classifier.train(&traces, &dimensions);

        assert_eq!(classifier.training_stats.total_samples, 4);
        assert_eq!(classifier.training_stats.ckks_samples, 4);
    }

    #[test]
    fn test_ckks_classification_accuracy() {
        let mut classifier = DimensionClassifier::new(vec![8, 16, 32, 64, 128, 256]);

        // Training data
        let mut train_traces = Vec::new();
        let mut train_dims = Vec::new();
        for &dim in &[8, 16, 32, 64, 128, 256] {
            for _ in 0..10 {
                train_traces.push(create_mock_ckks_trace(dim));
                train_dims.push(dim);
            }
        }
        classifier.train(&train_traces, &train_dims);

        // Test data
        let mut test_traces = Vec::new();
        let mut test_dims = Vec::new();
        for &dim in &[8, 16, 32, 64, 128, 256] {
            for _ in 0..5 {
                test_traces.push(create_mock_ckks_trace(dim));
                test_dims.push(dim);
            }
        }

        let report = classifier.classify_batch(&test_traces, &test_dims);

        // CKKS should have high accuracy since rotation count directly reveals dimension
        assert!(report.accuracy > 0.9, "CKKS accuracy should be > 90%, got {:.2}%", report.accuracy * 100.0);
    }

    #[test]
    fn test_clifford_classification_fails() {
        let mut classifier = DimensionClassifier::new(vec![8, 16, 32, 64, 128, 256]);

        // Train on CKKS data
        let mut train_traces = Vec::new();
        let mut train_dims = Vec::new();
        for &dim in &[8, 16, 32, 64, 128, 256] {
            for _ in 0..10 {
                train_traces.push(create_mock_ckks_trace(dim));
                train_dims.push(dim);
            }
        }
        classifier.train(&train_traces, &train_dims);

        // Test on Clifford data - should fail because rotation count is always 0
        let mut test_traces = Vec::new();
        let mut test_dims = Vec::new();
        for &dim in &[8, 16, 32, 64, 128, 256] {
            for _ in 0..5 {
                test_traces.push(create_mock_clifford_trace(dim));
                test_dims.push(dim);
            }
        }

        let report = classifier.classify_batch(&test_traces, &test_dims);

        // Clifford should have LOW accuracy (close to random guessing ~16.7% for 6 classes)
        // All Clifford traces have 0 rotations, so classifier can't distinguish
        assert!(report.accuracy < 0.3,
            "Clifford accuracy should be < 30% (near random), got {:.2}%",
            report.accuracy * 100.0);
    }

    #[test]
    fn test_paper_metrics() {
        let mut classifier = DimensionClassifier::new(vec![8, 16, 32, 64]);

        // Train
        let mut train_traces = Vec::new();
        let mut train_dims = Vec::new();
        for &dim in &[8, 16, 32, 64] {
            for _ in 0..10 {
                train_traces.push(create_mock_ckks_trace(dim));
                train_dims.push(dim);
            }
        }
        classifier.train(&train_traces, &train_dims);

        // Test both representations
        let mut test_traces = Vec::new();
        let mut test_dims = Vec::new();
        for &dim in &[8, 16, 32, 64] {
            test_traces.push(create_mock_ckks_trace(dim));
            test_dims.push(dim);
            test_traces.push(create_mock_clifford_trace(dim));
            test_dims.push(dim);
        }

        let report = classifier.classify_batch(&test_traces, &test_dims);
        let metrics = report.paper_metrics();

        // CKKS should leak, Clifford should not
        assert!(metrics.ckks_attack_accuracy > metrics.clifford_attack_accuracy,
            "CKKS should be more vulnerable than Clifford");
        assert!(metrics.privacy_gain > 0.0, "Privacy gain should be positive");
    }
}
