//! SIMD Optimizations for V2 Backend
//!
//! This module provides production-grade SIMD implementations of core FHE operations.
//! Supports AVX2 (x86_64) and NEON (ARM/Apple Silicon) with runtime CPU detection.
//!
//! ## Architecture
//!
//! ```text
//! simd/
//! ├── mod.rs           - This file (module organization, feature detection)
//! ├── traits.rs        - SIMD abstraction traits
//! ├── avx2.rs          - AVX2 implementation (x86_64)
//! ├── neon.rs          - NEON implementation (ARM/Apple Silicon)
//! └── scalar.rs        - Fallback scalar implementation
//! ```
//!
//! ## Performance Targets
//!
//! | Operation | Scalar | AVX2 | NEON | Speedup |
//! |-----------|--------|------|------|---------|
//! | NTT Butterfly | 1× | 3-4× | 2-3× | 2-4× |
//! | Barrett Reduction | 1× | 4× | 2-3× | 2-4× |
//! | RNS Add/Sub/Mul | 1× | 4× | 2-4× | 2-4× |
//!
//! ## Safety
//!
//! All SIMD code uses `unsafe` blocks with detailed safety comments.
//! Runtime CPU feature detection ensures we only use supported instructions.
//!
//! ## Testing
//!
//! Extensive property-based testing ensures SIMD implementations match
//! scalar reference implementation across all platforms.

pub mod traits;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "aarch64")]
pub mod neon;

use traits::SimdBackend;

/// CPU feature detection and backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuFeatures {
    /// AVX2 support (x86_64)
    Avx2,
    /// NEON support (ARM/Apple Silicon)
    Neon,
    /// No SIMD support (fallback to scalar)
    Scalar,
}

/// Detect available CPU features at runtime
///
/// This function is called once during initialization to determine
/// which SIMD backend to use.
///
/// # Returns
/// The best available SIMD backend for the current CPU
///
/// # Examples
/// ```
/// use ga_engine::clifford_fhe_v2::backends::cpu_optimized::simd::detect_cpu_features;
///
/// let features = detect_cpu_features();
/// match features {
///     CpuFeatures::Avx2 => println!("Using AVX2 backend"),
///     CpuFeatures::Neon => println!("Using NEON backend"),
///     CpuFeatures::Scalar => println!("Using scalar fallback"),
/// }
/// ```
pub fn detect_cpu_features() -> CpuFeatures {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return CpuFeatures::Avx2;
        }
        return CpuFeatures::Scalar;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on AArch64, always available
        return CpuFeatures::Neon;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    CpuFeatures::Scalar
}

/// Get the appropriate SIMD backend for the current CPU
///
/// This returns a trait object that provides SIMD operations.
/// The backend is selected based on runtime CPU feature detection.
pub fn get_simd_backend() -> Box<dyn SimdBackend> {
    match detect_cpu_features() {
        #[cfg(target_arch = "x86_64")]
        CpuFeatures::Avx2 => Box::new(avx2::Avx2Backend::new()),

        #[cfg(target_arch = "aarch64")]
        CpuFeatures::Neon => Box::new(neon::NeonBackend::new()),

        _ => Box::new(scalar::ScalarBackend::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_detection() {
        let features = detect_cpu_features();
        println!("Detected CPU features: {:?}", features);

        // Ensure we can create a backend
        let backend = get_simd_backend();
        assert!(backend.name().len() > 0);
    }

    #[test]
    fn test_backend_name() {
        let backend = get_simd_backend();
        let name = backend.name();

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            assert_eq!(name, "AVX2");
        } else {
            assert_eq!(name, "Scalar");
        }

        #[cfg(target_arch = "aarch64")]
        assert_eq!(name, "NEON");

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert_eq!(name, "Scalar");
    }
}
