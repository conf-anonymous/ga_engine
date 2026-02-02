//! Clifford FHE V2: Optimized Implementation for V2
//!
//! This is the **optimized version** of Clifford FHE with:
//! - Harvey butterfly NTT (10-50× faster polynomial multiplication)
//! - GPU acceleration via CUDA/Metal (10-100× speedup)
//! - SIMD batching for throughput (8-16× more samples/second)
//! - Rotation-specific keys (2× faster rotations)
//!
//! **Target:** Close the 59x performance gap from V1 baseline
//! - V1 baseline: 13s per geometric product
//! - V2 target: <=220ms per geometric product
//!
//! ## Usage
//!
//! ```rust
//! use ga_engine::clifford_fhe_v2::{backends::CpuOptimizedBackend, core::CliffordFHE};
//!
//! // Select backend at compile time via features
//! #[cfg(feature = "v2-cpu-optimized")]
//! let backend = CpuOptimizedBackend;
//!
//! // Or select at runtime
//! let backend_choice = determine_best_backend();
//! ```
//!
//! ## Feature Flags
//!
//! - `v2`: Enable V2 (required for all V2 backends)
//! - `v2-cpu-optimized`: CPU-only optimizations (NTT + SIMD)
//! - `v2-gpu-cuda`: CUDA GPU acceleration
//! - `v2-gpu-metal`: Metal GPU acceleration (Apple Silicon)
//! - `v2-simd-batched`: SIMD slot packing for batch processing
//! - `v2-full`: All optimizations enabled
//!
//! ## Architecture
//!
//! V2 uses a **trait-based backend system**:
//! - `core::CliffordFHE` trait: Common interface
//! - `backends::*`: Different implementations (CPU, CUDA, Metal, SIMD)
//! - Feature flags: Compile only what you need
//!
//! This follows the design pattern of well-known FHE libraries:
//! - Versioned namespaces for backward compatibility
//! - Modular architecture with multiple backends (CPU, GPU)
//! - Trait abstraction for backend selection

pub mod backends;
pub mod core;
pub mod params;
pub mod inversion;

// TODO: Add after implementing optimized versions
// pub mod ckks_rns;
// pub mod keys_rns;
// pub mod rotation_keys;

// Re-export core types
pub use core::{Backend, CliffordFHE, CliffordFHEError, Result, SecurityLevel};

/// Determine best available backend at runtime
pub fn determine_best_backend() -> Backend {
    #[cfg(feature = "v2-gpu-cuda")]
    {
        if cuda_available() {
            return Backend::GpuCuda;
        }
    }

    #[cfg(feature = "v2-gpu-metal")]
    {
        if metal_available() {
            return Backend::GpuMetal;
        }
    }

    #[cfg(feature = "v2-cpu-optimized")]
    {
        return Backend::CpuOptimized;
    }

    #[cfg(not(any(feature = "v2-gpu-cuda", feature = "v2-gpu-metal", feature = "v2-cpu-optimized")))]
    {
        // Fallback to V1 if no V2 backends enabled
        return Backend::V1;
    }

    // Final fallback (if GPU features enabled but not available)
    #[cfg(any(feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
    Backend::CpuOptimized
}

#[cfg(feature = "v2-gpu-cuda")]
fn cuda_available() -> bool {
    // TODO: Check for CUDA runtime
    false
}

#[cfg(feature = "v2-gpu-metal")]
fn metal_available() -> bool {
    // TODO: Check for Metal support
    cfg!(target_os = "macos")
}
