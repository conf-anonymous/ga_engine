//! Clifford FHE V5: Privacy-Trace Collection and Analysis
//!
//! This module provides **instrumented wrappers** around V2 FHE operations
//! to collect execution traces for privacy analysis. It implements six
//! attack vectors to measure information leakage in FHE computations.
//!
//! ## Design Philosophy
//!
//! V5 is **non-invasive**: it wraps existing V2 operations without modifying them.
//! This ensures:
//! - Retrocompatibility with existing papers
//! - Clean isolation for privacy research experiments
//! - Freedom to experiment without affecting production code
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        V5 Trace Layer                           │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
//! │  │ ExecutionTrace│  │ TraceCollector│  │ WorkloadDefinition │  │
//! │  └──────────────┘  └──────────────┘  └──────────────────────┘  │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                     Backend Wrappers                            │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
//! │  │  TracedCPU   │  │ TracedMetal  │  │    TracedCUDA        │  │
//! │  └──────────────┘  └──────────────┘  └──────────────────────┘  │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                    V2 Backend (unchanged)                       │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
//! │  │ cpu_optimized│  │  gpu_metal   │  │     gpu_cuda         │  │
//! │  └──────────────┘  └──────────────┘  └──────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Observable Metrics (per V5 threat model)
//!
//! For each encrypted computation, V5 collects:
//! - **Wall-clock runtime** (microseconds)
//! - **Rotation count** (number of Galois automorphisms)
//! - **Rescale count** (modulus switching operations)
//! - **Relinearization count** (key switching operations)
//! - **Bootstrap flag** (whether bootstrapping occurred)
//! - **Level trajectory** (ciphertext level at each step)
//! - **Peak memory allocation** (bytes)
//! - **Kernel invocation histogram** (GPU only)
//!
//! ## Workload Types
//!
//! V5 supports both CKKS and CliffordFHE workloads for fair comparison:
//!
//! ### CKKS Workloads (Baseline)
//! - Plain similarity: `v₁·v₂ / (|v₁||v₂|)`
//! - Normalization + similarity
//! - Variable-size vector operations
//!
//! ### CliffordFHE Workloads (Proposed)
//! - Geometric similarity (same computation, GA representation)
//! - Fixed 8-component layout
//! - Unified operator set
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ga_engine::clifford_fhe_v5::{
//!     ExecutionTrace, TraceCollector, TracedCpuBackend,
//!     workloads::{CkksSimilarity, CliffordSimilarity},
//! };
//!
//! // Create traced backend (wraps V2 CPU backend)
//! let traced = TracedCpuBackend::new(params);
//!
//! // Run workload and collect trace
//! let trace = traced.execute_workload(CkksSimilarity::new(&v1, &v2));
//!
//! // Export trace for analysis
//! trace.to_json("traces/ckks_similarity_001.json")?;
//! ```
//!
//! ## Feature Flags
//!
//! - `v5`: Enable V5 trace collection
//! - `v5-cpu`: CPU backend tracing (wraps v2-cpu-optimized)
//! - `v5-metal`: Metal GPU tracing (wraps v2-gpu-metal)
//! - `v5-cuda`: CUDA GPU tracing (wraps v2-gpu-cuda)

// Core trace data structures
pub mod trace;

// Trace collector and aggregation
pub mod collector;

// Workload definitions (CKKS baseline vs CliffordFHE)
pub mod workloads;

// CPU backend with trace instrumentation
pub mod cpu;

// Metal GPU backend with trace instrumentation
#[cfg(feature = "v2-gpu-metal")]
pub mod metal;

// CUDA GPU backend with trace instrumentation
#[cfg(feature = "v2-gpu-cuda")]
pub mod cuda;

// Trace analysis utilities
pub mod analysis;

// ML classifiers for privacy attack demonstrations
pub mod ml_classifier;

// Re-exports for convenience
pub use trace::{ExecutionTrace, OperationEvent, OperationType};
pub use collector::{TraceCollector, TraceSession};
pub use workloads::{Workload, WorkloadType, WorkloadConfig};
pub use cpu::TracedCpuBackend;

// Attack classifiers
pub use ml_classifier::{
    // Dimension inference attack
    DimensionClassifier, ClassificationReport, PaperMetrics,
    // Task identification attack
    TaskClassifier, TaskClassificationReport,
    // Sparsity inference attack
    SparsityClassifier, SparsityClassificationReport,
    // Multi-tenant linkability attack
    TenantLinker, TenantLinkReport,
    // Operation count attack (NEW - CliffordFHE wins)
    OperationCountClassifier, OpCountClassificationReport,
    // Trace length attack (complementary to operation count)
    TraceLengthClassifier, TraceLengthClassificationReport,
};

#[cfg(feature = "v2-gpu-metal")]
pub use metal::TracedMetalBackend;

#[cfg(feature = "v2-gpu-cuda")]
pub use cuda::TracedCudaBackend;

/// Version identifier for V5 trace format
pub const V5_TRACE_VERSION: &str = "1.0.0";

/// Default output directory for trace files
pub const DEFAULT_TRACE_DIR: &str = "traces/v5_analysis";
