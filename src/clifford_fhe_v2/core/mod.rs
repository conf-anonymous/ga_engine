//! Core abstractions for Clifford FHE V2
//!
//! This module provides traits and types that are backend-agnostic,
//! allowing seamless switching between different implementations:
//! - CPU Optimized (NTT + SIMD)
//! - GPU CUDA
//! - GPU Metal (Apple Silicon)
//! - SIMD Batched (throughput-oriented)

pub mod traits;
pub mod types;

pub use traits::{BackendCapabilities, BackendInfo, CliffordFHE, MultivectorOps};
pub use types::{Backend, CliffordFHEError, Result, SecurityLevel};
