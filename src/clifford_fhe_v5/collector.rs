//! Trace Collector for Aggregating Execution Traces
//!
//! This module provides utilities for collecting and managing multiple execution
//! traces across different workloads and configurations.

use crate::clifford_fhe_v5::trace::{ExecutionTrace, InputMetadata, OperationEvent};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Thread-safe trace collector for gathering traces from multiple executions
#[derive(Debug)]
pub struct TraceCollector {
    /// All collected traces
    traces: Arc<Mutex<Vec<ExecutionTrace>>>,

    /// Output directory for trace files
    output_dir: PathBuf,

    /// Session identifier
    session_id: String,

    /// Whether to auto-save traces
    auto_save: bool,

    /// Current trace being built (for incremental collection)
    current_trace: Arc<Mutex<Option<ExecutionTrace>>>,
}

impl TraceCollector {
    /// Create a new trace collector
    pub fn new(output_dir: &str) -> Self {
        let session_id = generate_session_id();

        // Create output directory if it doesn't exist
        let path = PathBuf::from(output_dir);
        if !path.exists() {
            std::fs::create_dir_all(&path).ok();
        }

        Self {
            traces: Arc::new(Mutex::new(Vec::new())),
            output_dir: path,
            session_id,
            auto_save: false,
            current_trace: Arc::new(Mutex::new(None)),
        }
    }

    /// Enable auto-saving traces to disk
    pub fn with_auto_save(mut self, enabled: bool) -> Self {
        self.auto_save = enabled;
        self
    }

    /// Start a new trace session for a workload
    pub fn start_trace(
        &self,
        workload_type: &str,
        representation: &str,
        backend: &str,
        ring_dimension: usize,
        num_primes: usize,
    ) -> TraceSession {
        let trace = ExecutionTrace::new(
            workload_type,
            representation,
            backend,
            ring_dimension,
            num_primes,
        );

        TraceSession {
            collector: self,
            trace,
        }
    }

    /// Add a completed trace to the collection
    pub fn add_trace(&self, mut trace: ExecutionTrace) {
        trace.compute_summary();

        if self.auto_save {
            let filename = format!(
                "{}/{}_{}.json",
                self.output_dir.display(),
                self.session_id,
                trace.trace_id
            );
            trace.to_json(&filename).ok();
        }

        let mut traces = self.traces.lock().unwrap();
        traces.push(trace);
    }

    /// Get all collected traces
    pub fn get_traces(&self) -> Vec<ExecutionTrace> {
        self.traces.lock().unwrap().clone()
    }

    /// Get number of collected traces
    pub fn trace_count(&self) -> usize {
        self.traces.lock().unwrap().len()
    }

    /// Export all traces to a single JSON file
    pub fn export_all(&self, filename: &str) -> std::io::Result<()> {
        let traces = self.traces.lock().unwrap();
        let file = std::fs::File::create(filename)?;
        serde_json::to_writer_pretty(file, &*traces)?;
        Ok(())
    }

    /// Export traces grouped by workload type
    pub fn export_by_workload(&self) -> std::io::Result<()> {
        let traces = self.traces.lock().unwrap();
        let mut by_workload: HashMap<String, Vec<&ExecutionTrace>> = HashMap::new();

        for trace in traces.iter() {
            by_workload
                .entry(trace.workload_type.clone())
                .or_default()
                .push(trace);
        }

        for (workload, traces) in by_workload {
            let filename = format!(
                "{}/{}_{}.json",
                self.output_dir.display(),
                self.session_id,
                workload
            );
            let file = std::fs::File::create(&filename)?;
            serde_json::to_writer_pretty(file, &traces)?;
        }

        Ok(())
    }

    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get output directory
    pub fn output_dir(&self) -> &PathBuf {
        &self.output_dir
    }

    /// Generate summary statistics across all traces
    pub fn summary_stats(&self) -> CollectionSummary {
        let traces = self.traces.lock().unwrap();

        let mut summary = CollectionSummary::default();
        summary.total_traces = traces.len();

        for trace in traces.iter() {
            summary.total_events += trace.events.len();
            summary.total_duration_us += trace.summary.total_duration_us;

            // Group by representation
            *summary
                .traces_by_representation
                .entry(trace.representation.clone())
                .or_insert(0) += 1;

            // Group by workload
            *summary
                .traces_by_workload
                .entry(trace.workload_type.clone())
                .or_insert(0) += 1;

            // Group by backend
            *summary
                .traces_by_backend
                .entry(trace.backend.clone())
                .or_insert(0) += 1;
        }

        summary
    }
}

/// Summary statistics for a trace collection
#[derive(Debug, Default)]
pub struct CollectionSummary {
    pub total_traces: usize,
    pub total_events: usize,
    pub total_duration_us: u64,
    pub traces_by_representation: HashMap<String, usize>,
    pub traces_by_workload: HashMap<String, usize>,
    pub traces_by_backend: HashMap<String, usize>,
}

impl std::fmt::Display for CollectionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Trace Collection Summary ===")?;
        writeln!(f, "Total traces: {}", self.total_traces)?;
        writeln!(f, "Total events: {}", self.total_events)?;
        writeln!(
            f,
            "Total duration: {:.2}ms",
            self.total_duration_us as f64 / 1000.0
        )?;
        writeln!(f)?;

        writeln!(f, "By Representation:")?;
        for (rep, count) in &self.traces_by_representation {
            writeln!(f, "  {}: {}", rep, count)?;
        }

        writeln!(f)?;
        writeln!(f, "By Workload:")?;
        for (workload, count) in &self.traces_by_workload {
            writeln!(f, "  {}: {}", workload, count)?;
        }

        writeln!(f)?;
        writeln!(f, "By Backend:")?;
        for (backend, count) in &self.traces_by_backend {
            writeln!(f, "  {}: {}", backend, count)?;
        }

        Ok(())
    }
}

/// A session for building a single trace
pub struct TraceSession<'a> {
    collector: &'a TraceCollector,
    trace: ExecutionTrace,
}

impl<'a> TraceSession<'a> {
    /// Add an operation event to the current trace
    pub fn record(&mut self, event: OperationEvent) {
        self.trace.add_event(event);
    }

    /// Set input metadata for the trace
    pub fn set_input_metadata(&mut self, metadata: InputMetadata) {
        self.trace.input_metadata = metadata;
    }

    /// Set input length
    pub fn set_input_length(&mut self, length: usize) {
        self.trace.input_metadata.input_length = length;
    }

    /// Set sparsity
    pub fn set_sparsity(&mut self, sparsity: f64) {
        self.trace.input_metadata.sparsity = sparsity;
    }

    /// Set category label
    pub fn set_category(&mut self, category: &str) {
        self.trace.input_metadata.category = Some(category.to_string());
    }

    /// Set tenant ID
    pub fn set_tenant(&mut self, tenant_id: &str) {
        self.trace.input_metadata.tenant_id = Some(tenant_id.to_string());
    }

    /// Get the current trace (for inspection)
    pub fn trace(&self) -> &ExecutionTrace {
        &self.trace
    }

    /// Get mutable reference to the current trace
    pub fn trace_mut(&mut self) -> &mut ExecutionTrace {
        &mut self.trace
    }

    /// Finish the session and add the trace to the collector
    pub fn finish(mut self) {
        self.trace.compute_summary();
        self.collector.add_trace(self.trace);
    }

    /// Finish and return the trace (without adding to collector)
    pub fn into_trace(mut self) -> ExecutionTrace {
        self.trace.compute_summary();
        self.trace
    }
}

/// Generate a unique session ID
fn generate_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("session_{}", timestamp)
}

/// Batch trace runner for running multiple trials
pub struct BatchRunner {
    collector: TraceCollector,
    num_trials: usize,
    warmup_trials: usize,
}

impl BatchRunner {
    /// Create a new batch runner
    pub fn new(output_dir: &str, num_trials: usize) -> Self {
        Self {
            collector: TraceCollector::new(output_dir).with_auto_save(true),
            num_trials,
            warmup_trials: 3,
        }
    }

    /// Set number of warmup trials
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup_trials = warmup;
        self
    }

    /// Run a workload multiple times and collect traces
    pub fn run<F>(&self, workload_name: &str, mut workload_fn: F)
    where
        F: FnMut(usize) -> ExecutionTrace,
    {
        println!("Running {} x {} trials...", workload_name, self.num_trials);

        // Warmup runs (not recorded)
        for i in 0..self.warmup_trials {
            println!("  Warmup {}/{}...", i + 1, self.warmup_trials);
            let _ = workload_fn(i);
        }

        // Actual runs
        for i in 0..self.num_trials {
            println!("  Trial {}/{}...", i + 1, self.num_trials);
            let trace = workload_fn(i);
            self.collector.add_trace(trace);
        }
    }

    /// Get the collector
    pub fn collector(&self) -> &TraceCollector {
        &self.collector
    }

    /// Export all collected traces
    pub fn export(&self, filename: &str) -> std::io::Result<()> {
        self.collector.export_all(filename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v5::trace::OperationType;
    use std::time::Duration;

    #[test]
    fn test_trace_collector() {
        let collector = TraceCollector::new("/tmp/test_traces");

        let mut session = collector.start_trace("similarity", "ckks", "cpu", 1024, 3);

        session.record(
            OperationEvent::new(OperationType::Encrypt)
                .with_duration(Duration::from_micros(100))
                .with_levels(2, 2),
        );

        session.record(
            OperationEvent::new(OperationType::MultiplyCiphertext)
                .with_duration(Duration::from_micros(500))
                .with_levels(2, 1)
                .with_relins(1),
        );

        session.finish();

        assert_eq!(collector.trace_count(), 1);

        let traces = collector.get_traces();
        assert_eq!(traces[0].events.len(), 2);
        assert_eq!(traces[0].summary.total_relins, 1);
    }

    #[test]
    fn test_trace_session_metadata() {
        let collector = TraceCollector::new("/tmp/test_traces");

        let mut session = collector.start_trace("similarity", "clifford", "metal", 2048, 5);

        session.set_input_length(512);
        session.set_sparsity(0.1);
        session.set_category("dense_vectors");

        let trace = session.into_trace();

        assert_eq!(trace.input_metadata.input_length, 512);
        assert!((trace.input_metadata.sparsity - 0.1).abs() < 1e-10);
        assert_eq!(trace.input_metadata.category, Some("dense_vectors".to_string()));
    }

    #[test]
    fn test_collection_summary() {
        let collector = TraceCollector::new("/tmp/test_traces");

        // Add CKKS trace
        let mut session1 = collector.start_trace("similarity", "ckks", "cpu", 1024, 3);
        session1.record(
            OperationEvent::new(OperationType::Encrypt)
                .with_duration(Duration::from_micros(100))
                .with_levels(2, 2),
        );
        session1.finish();

        // Add Clifford trace
        let mut session2 = collector.start_trace("similarity", "clifford", "cpu", 1024, 3);
        session2.record(
            OperationEvent::new(OperationType::GeometricProduct)
                .with_duration(Duration::from_micros(500))
                .with_levels(2, 1),
        );
        session2.finish();

        let summary = collector.summary_stats();

        assert_eq!(summary.total_traces, 2);
        assert_eq!(summary.traces_by_representation.get("ckks"), Some(&1));
        assert_eq!(summary.traces_by_representation.get("clifford"), Some(&1));
    }
}
