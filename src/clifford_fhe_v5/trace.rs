//! Execution Trace Data Structures for Privacy Analysis
//!
//! This module defines the core data structures for collecting and representing
//! execution traces of FHE operations. These traces are used to measure
//! representation-induced distinguishability as per the V5 threat model.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Type of FHE operation being traced
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    // Core CKKS operations
    Encode,
    Encrypt,
    Decrypt,
    Add,
    Subtract,
    MultiplyPlain,
    MultiplyCiphertext,
    Rescale,
    Rotate,
    Relinearize,
    ModSwitch,
    Bootstrap,

    // Geometric algebra operations (CliffordFHE)
    GeometricProduct,
    Reverse,
    WedgeProduct,
    InnerProduct,
    GaRotate,       // R·v·R̃
    Project,
    Reject,

    // Composite operations
    Similarity,
    Normalization,
    NormalizeThenSimilarity,

    // Memory operations (GPU)
    HostToDevice,
    DeviceToHost,
    KernelLaunch,
}

impl OperationType {
    /// Returns the category of the operation for aggregation
    pub fn category(&self) -> &'static str {
        match self {
            // Core CKKS
            OperationType::Encode | OperationType::Encrypt | OperationType::Decrypt => "encoding",
            OperationType::Add | OperationType::Subtract => "addition",
            OperationType::MultiplyPlain | OperationType::MultiplyCiphertext => "multiplication",
            OperationType::Rescale | OperationType::ModSwitch => "rescale",
            OperationType::Rotate => "rotation",
            OperationType::Relinearize => "relinearization",
            OperationType::Bootstrap => "bootstrap",

            // Geometric algebra
            OperationType::GeometricProduct => "geometric",
            OperationType::Reverse => "geometric",
            OperationType::WedgeProduct | OperationType::InnerProduct => "geometric",
            OperationType::GaRotate => "geometric",
            OperationType::Project | OperationType::Reject => "geometric",

            // Composite
            OperationType::Similarity => "workload",
            OperationType::Normalization => "workload",
            OperationType::NormalizeThenSimilarity => "workload",

            // Memory
            OperationType::HostToDevice | OperationType::DeviceToHost => "memory",
            OperationType::KernelLaunch => "kernel",
        }
    }

    /// Returns whether this operation consumes a level (rescale)
    pub fn consumes_level(&self) -> bool {
        matches!(self, OperationType::Rescale | OperationType::ModSwitch)
    }
}

/// A single operation event in an execution trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationEvent {
    /// Type of operation
    pub op_type: OperationType,

    /// Duration of the operation
    #[serde(with = "duration_micros")]
    pub duration: Duration,

    /// Ciphertext level before operation
    pub level_before: usize,

    /// Ciphertext level after operation
    pub level_after: usize,

    /// Number of rotations performed (for Rotate, GeometricProduct)
    pub rotation_count: usize,

    /// Rotation amounts used (for Rotate operations)
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub rotation_amounts: Vec<i32>,

    /// Number of rescales performed
    pub rescale_count: usize,

    /// Number of relinearizations performed
    pub relin_count: usize,

    /// Whether bootstrapping occurred
    pub bootstrap_occurred: bool,

    /// Memory allocated (bytes, if tracked)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<usize>,

    /// GPU kernel name (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_name: Option<String>,

    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl OperationEvent {
    /// Create a new operation event
    pub fn new(op_type: OperationType) -> Self {
        Self {
            op_type,
            duration: Duration::ZERO,
            level_before: 0,
            level_after: 0,
            rotation_count: 0,
            rotation_amounts: Vec::new(),
            rescale_count: 0,
            relin_count: 0,
            bootstrap_occurred: false,
            memory_bytes: None,
            kernel_name: None,
            metadata: None,
        }
    }

    /// Set the duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Set level information
    pub fn with_levels(mut self, before: usize, after: usize) -> Self {
        self.level_before = before;
        self.level_after = after;
        self
    }

    /// Set rotation count
    pub fn with_rotations(mut self, count: usize) -> Self {
        self.rotation_count = count;
        self
    }

    /// Set rotation amounts
    pub fn with_rotation_amounts(mut self, amounts: Vec<i32>) -> Self {
        self.rotation_count = amounts.len();
        self.rotation_amounts = amounts;
        self
    }

    /// Set rescale count
    pub fn with_rescales(mut self, count: usize) -> Self {
        self.rescale_count = count;
        self
    }

    /// Set relinearization count
    pub fn with_relins(mut self, count: usize) -> Self {
        self.relin_count = count;
        self
    }

    /// Set bootstrap flag
    pub fn with_bootstrap(mut self, occurred: bool) -> Self {
        self.bootstrap_occurred = occurred;
        self
    }

    /// Set memory allocation
    pub fn with_memory(mut self, bytes: usize) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }

    /// Set kernel name
    pub fn with_kernel(mut self, name: String) -> Self {
        self.kernel_name = Some(name);
        self
    }
}

/// Complete execution trace for a workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Trace format version
    pub version: String,

    /// Unique trace identifier
    pub trace_id: String,

    /// Workload type (for classification)
    pub workload_type: String,

    /// Representation type: "ckks" or "clifford"
    pub representation: String,

    /// Backend used: "cpu", "metal", "cuda"
    pub backend: String,

    /// Timestamp when trace was created
    pub timestamp: String,

    /// Parameter set identifier
    pub params_id: String,

    /// Ring dimension N
    pub ring_dimension: usize,

    /// Number of RNS primes
    pub num_primes: usize,

    /// Input characteristics (for attribute inference analysis)
    pub input_metadata: InputMetadata,

    /// Sequence of operation events
    pub events: Vec<OperationEvent>,

    /// Aggregated statistics
    pub summary: TraceSummary,
}

/// Metadata about the input (for privacy analysis)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputMetadata {
    /// Input vector length (or feature count)
    pub input_length: usize,

    /// Sparsity ratio (fraction of zeros)
    pub sparsity: f64,

    /// Input category label (for classification experiments)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,

    /// Tenant/workload identifier (for fingerprinting experiments)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
}

/// Aggregated trace statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceSummary {
    /// Total wall-clock time (microseconds)
    pub total_duration_us: u64,

    /// Total number of operations
    pub total_ops: usize,

    /// Total rotation count
    pub total_rotations: usize,

    /// Total rescale count
    pub total_rescales: usize,

    /// Total relinearization count
    pub total_relins: usize,

    /// Bootstrap count
    pub bootstrap_count: usize,

    /// Starting level
    pub start_level: usize,

    /// Ending level
    pub end_level: usize,

    /// Peak memory usage (bytes)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory: Option<usize>,

    /// Level trajectory (level at each step)
    pub level_trajectory: Vec<usize>,

    /// Operation type histogram
    pub op_histogram: std::collections::HashMap<String, usize>,

    /// Unique rotation amounts used
    pub rotation_amounts_used: Vec<i32>,
}

impl ExecutionTrace {
    /// Create a new execution trace
    pub fn new(
        workload_type: &str,
        representation: &str,
        backend: &str,
        ring_dimension: usize,
        num_primes: usize,
    ) -> Self {
        Self {
            version: crate::clifford_fhe_v5::V5_TRACE_VERSION.to_string(),
            trace_id: generate_trace_id(),
            workload_type: workload_type.to_string(),
            representation: representation.to_string(),
            backend: backend.to_string(),
            timestamp: chrono_timestamp(),
            params_id: format!("N{}_L{}", ring_dimension, num_primes),
            ring_dimension,
            num_primes,
            input_metadata: InputMetadata::default(),
            events: Vec::new(),
            summary: TraceSummary::default(),
        }
    }

    /// Add an operation event
    pub fn add_event(&mut self, event: OperationEvent) {
        self.events.push(event);
    }

    /// Set input metadata
    pub fn with_input_metadata(mut self, metadata: InputMetadata) -> Self {
        self.input_metadata = metadata;
        self
    }

    /// Compute summary statistics from events
    pub fn compute_summary(&mut self) {
        let mut summary = TraceSummary::default();

        let mut op_histogram = std::collections::HashMap::new();
        let mut rotation_amounts_set = std::collections::HashSet::new();
        let mut level_trajectory = Vec::new();

        for event in &self.events {
            summary.total_duration_us += event.duration.as_micros() as u64;
            summary.total_ops += 1;
            summary.total_rotations += event.rotation_count;
            summary.total_rescales += event.rescale_count;
            summary.total_relins += event.relin_count;

            if event.bootstrap_occurred {
                summary.bootstrap_count += 1;
            }

            if let Some(mem) = event.memory_bytes {
                summary.peak_memory = Some(summary.peak_memory.unwrap_or(0).max(mem));
            }

            // Track level trajectory
            if level_trajectory.is_empty() {
                level_trajectory.push(event.level_before);
            }
            level_trajectory.push(event.level_after);

            // Track rotation amounts
            for &amt in &event.rotation_amounts {
                rotation_amounts_set.insert(amt);
            }

            // Operation histogram
            let op_name = format!("{:?}", event.op_type);
            *op_histogram.entry(op_name).or_insert(0) += 1;
        }

        // Set start/end levels
        if !level_trajectory.is_empty() {
            summary.start_level = level_trajectory[0];
            summary.end_level = *level_trajectory.last().unwrap();
        }

        summary.level_trajectory = level_trajectory;
        summary.op_histogram = op_histogram;
        summary.rotation_amounts_used = rotation_amounts_set.into_iter().collect();
        summary.rotation_amounts_used.sort();

        self.summary = summary;
    }

    /// Export trace to JSON file
    pub fn to_json(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// Load trace from JSON file
    pub fn from_json(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let trace = serde_json::from_reader(file)?;
        Ok(trace)
    }

    /// Export trace to compact binary format (for large datasets)
    pub fn to_bincode(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        // Using JSON for now; bincode would require additional dependency
        serde_json::to_writer(&mut writer, self)?;
        Ok(())
    }
}

/// Generate a unique trace ID
fn generate_trace_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("trace_{:016x}", timestamp)
}

/// Generate ISO 8601 timestamp
fn chrono_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Simple ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
    // For full accuracy, use chrono crate in production
    format!("{}Z", secs)
}

/// Serde helper for Duration serialization as microseconds
mod duration_micros {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_micros() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let micros = u64::deserialize(deserializer)?;
        Ok(Duration::from_micros(micros))
    }
}

/// Timer utility for measuring operation durations
pub struct OperationTimer {
    /// Start time (public for advanced use cases)
    pub start: Instant,
    op_type: OperationType,
    level_before: usize,
}

impl OperationTimer {
    /// Start timing an operation
    pub fn start(op_type: OperationType, level_before: usize) -> Self {
        Self {
            start: Instant::now(),
            op_type,
            level_before,
        }
    }

    /// Stop timing and create an event
    pub fn stop(self, level_after: usize) -> OperationEvent {
        OperationEvent::new(self.op_type)
            .with_duration(self.start.elapsed())
            .with_levels(self.level_before, level_after)
    }

    /// Stop timing with additional rotation info
    pub fn stop_with_rotations(self, level_after: usize, rotation_count: usize) -> OperationEvent {
        OperationEvent::new(self.op_type)
            .with_duration(self.start.elapsed())
            .with_levels(self.level_before, level_after)
            .with_rotations(rotation_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_event_creation() {
        let event = OperationEvent::new(OperationType::GeometricProduct)
            .with_duration(Duration::from_micros(1000))
            .with_levels(5, 4)
            .with_rotations(8)
            .with_rescales(1)
            .with_relins(64);

        assert_eq!(event.op_type, OperationType::GeometricProduct);
        assert_eq!(event.duration.as_micros(), 1000);
        assert_eq!(event.level_before, 5);
        assert_eq!(event.level_after, 4);
        assert_eq!(event.rotation_count, 8);
        assert_eq!(event.rescale_count, 1);
        assert_eq!(event.relin_count, 64);
    }

    #[test]
    fn test_execution_trace_creation() {
        let mut trace = ExecutionTrace::new(
            "similarity",
            "clifford",
            "cpu",
            1024,
            3,
        );

        trace.add_event(
            OperationEvent::new(OperationType::Encrypt)
                .with_duration(Duration::from_micros(500))
                .with_levels(2, 2)
        );

        trace.add_event(
            OperationEvent::new(OperationType::GeometricProduct)
                .with_duration(Duration::from_micros(2000))
                .with_levels(2, 1)
                .with_rotations(8)
                .with_relins(64)
        );

        trace.compute_summary();

        assert_eq!(trace.events.len(), 2);
        assert_eq!(trace.summary.total_ops, 2);
        assert_eq!(trace.summary.total_rotations, 8);
        assert_eq!(trace.summary.total_relins, 64);
    }

    #[test]
    fn test_operation_timer() {
        let timer = OperationTimer::start(OperationType::Rescale, 5);
        std::thread::sleep(Duration::from_micros(100));
        let event = timer.stop(4);

        assert_eq!(event.op_type, OperationType::Rescale);
        assert_eq!(event.level_before, 5);
        assert_eq!(event.level_after, 4);
        assert!(event.duration.as_micros() >= 100);
    }

    #[test]
    fn test_trace_serialization() {
        let mut trace = ExecutionTrace::new("test", "ckks", "cpu", 1024, 3);
        trace.add_event(
            OperationEvent::new(OperationType::Add)
                .with_duration(Duration::from_micros(50))
                .with_levels(2, 2)
        );
        trace.compute_summary();

        let json = serde_json::to_string_pretty(&trace).unwrap();
        assert!(json.contains("\"workload_type\": \"test\""));
        assert!(json.contains("\"representation\": \"ckks\""));
    }
}
