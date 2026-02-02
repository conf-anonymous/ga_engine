//! Clifford FHE V6 - parallel_lift GPU Acceleration
//!
//! V6 integrates the parallel_lift library for 25-552× speedups on key FHE operations:
//! - Gadget decomposition (relinearization/rotation): 25× faster
//! - CRT batch reconstruction: 25× faster
//! - Matrix-vector multiplication (bootstrapping): 50-552× faster
//!
//! ## Architecture
//!
//! V6 wraps V2's CUDA infrastructure and replaces bottleneck operations with
//! parallel_lift's GPU-accelerated implementations.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                         V6 Layer                            │
//! │  ┌─────────────────────┐  ┌───────────────────────────────┐ │
//! │  │ ParallelLiftContext │  │ V6RelinKeys / V6RotationKeys  │ │
//! │  │ - FheGpuContext     │  │ - gpu_gadget_decompose (25×)  │ │
//! │  │ - CudaCkksContext   │  └───────────────────────────────┘ │
//! │  └─────────────────────┘                                    │
//! ├─────────────────────────────────────────────────────────────┤
//! │                     V2 Layer (Unchanged)                    │
//! │  CudaCkksContext, CudaCiphertext, NTT kernels, etc.        │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use ga_engine::clifford_fhe_v6::{ParallelLiftContext, V6RelinKeys};
//! use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
//!
//! // Create V6 context (wraps V2 + parallel_lift)
//! let params = CliffordFHEParams::new_128bit();
//! let ctx = ParallelLiftContext::with_params(params)?;
//!
//! // Generate keys with V6 acceleration
//! let relin_keys = V6RelinKeys::new(&ctx, &secret_key)?;
//!
//! // Relinearization uses gpu_gadget_decompose (25× faster)
//! let ct_relin = relin_keys.apply_relinearization_v6(&ct_mult)?;
//! ```
//!
//! ## Feature Flags
//!
//! - `v6`: Base V6 (requires V2)
//! - `v6-cuda`: Full CUDA acceleration via parallel_lift
//! - `v6-full`: V6 + V3 bootstrapping support

#![allow(unused_imports)]
#![allow(dead_code)]

// V6 CUDA modules (require parallel_lift)
#[cfg(feature = "v6-cuda")]
pub mod context;

#[cfg(feature = "v6-cuda")]
pub mod gadget_decompose;

#[cfg(feature = "v6-cuda")]
pub mod relin_keys_v6;

#[cfg(feature = "v6-cuda")]
pub mod rotation_keys_v6;

// V6 bootstrap (requires V6-CUDA + V3)
#[cfg(all(feature = "v6-cuda", feature = "v3"))]
pub mod coeff_to_slot_v6;

// Re-exports for convenient access
#[cfg(feature = "v6-cuda")]
pub use context::ParallelLiftContext;

#[cfg(feature = "v6-cuda")]
pub use relin_keys_v6::V6RelinKeys;

#[cfg(feature = "v6-cuda")]
pub use rotation_keys_v6::V6RotationKeys;

#[cfg(feature = "v6-cuda")]
pub use gadget_decompose::gpu_gadget_decompose_v6;

#[cfg(all(feature = "v6-cuda", feature = "v3"))]
pub use coeff_to_slot_v6::coeff_to_slot_v6;

/// V6 error type
#[derive(Debug, Clone)]
pub enum V6Error {
    /// parallel_lift FheGpuContext initialization failed
    ContextInitFailed(String),

    /// GPU kernel execution failed
    GpuKernelError(String),

    /// Dimension mismatch between V2 and parallel_lift
    DimensionMismatch { expected: usize, actual: usize },

    /// V2 operation failed
    V2Error(String),

    /// Feature not available
    FeatureNotAvailable(String),
}

impl std::fmt::Display for V6Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            V6Error::ContextInitFailed(msg) => write!(f, "V6 context init failed: {}", msg),
            V6Error::GpuKernelError(msg) => write!(f, "GPU kernel error: {}", msg),
            V6Error::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            V6Error::V2Error(msg) => write!(f, "V2 error: {}", msg),
            V6Error::FeatureNotAvailable(msg) => write!(f, "Feature not available: {}", msg),
        }
    }
}

impl std::error::Error for V6Error {}

/// V6 result type
pub type V6Result<T> = Result<T, V6Error>;
