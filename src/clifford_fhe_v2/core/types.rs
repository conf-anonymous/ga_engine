//! Common types used across all Clifford FHE V2 backends

/// Security level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    /// ~128 bits post-quantum security (NIST Level 1)
    Secure128,
    /// ~192 bits post-quantum security (NIST Level 3)
    Secure192,
    /// ~256 bits post-quantum security (NIST Level 5)
    Secure256,
}

/// Backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// V1 baseline implementation
    V1,
    /// CPU optimized (NTT + SIMD, no GPU)
    CpuOptimized,
    /// CUDA GPU acceleration
    GpuCuda,
    /// Metal GPU acceleration (Apple Silicon)
    GpuMetal,
    /// SIMD slot packing for throughput
    SimdBatched,
}

impl Backend {
    pub fn name(&self) -> &'static str {
        match self {
            Backend::V1 => "V1 Baseline",
            Backend::CpuOptimized => "V2 CPU Optimized (NTT+SIMD)",
            Backend::GpuCuda => "V2 GPU CUDA",
            Backend::GpuMetal => "V2 GPU Metal",
            Backend::SimdBatched => "V2 SIMD Batched",
        }
    }

    pub fn expected_speedup(&self) -> f64 {
        match self {
            Backend::V1 => 1.0,  // Baseline
            Backend::CpuOptimized => 10.0,  // NTT + SIMD
            Backend::GpuCuda => 50.0,  // GPU acceleration
            Backend::GpuMetal => 30.0,  // Metal (Apple Silicon)
            Backend::SimdBatched => 12.0,  // Throughput (not latency)
        }
    }
}

/// Error types for Clifford FHE operations
#[derive(Debug, Clone)]
pub enum CliffordFHEError {
    /// Invalid parameter configuration
    InvalidParams(String),
    /// Ciphertext levels don't match
    LevelMismatch { expected: usize, actual: usize },
    /// Noise budget exhausted
    NoiseBudgetExhausted,
    /// Backend not available (e.g., CUDA not compiled)
    BackendNotAvailable(Backend),
    /// Generic error
    Other(String),
}

impl std::fmt::Display for CliffordFHEError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliffordFHEError::InvalidParams(msg) => write!(f, "Invalid parameters: {}", msg),
            CliffordFHEError::LevelMismatch { expected, actual } => {
                write!(f, "Level mismatch: expected {}, got {}", expected, actual)
            }
            CliffordFHEError::NoiseBudgetExhausted => {
                write!(f, "Noise budget exhausted (too many operations)")
            }
            CliffordFHEError::BackendNotAvailable(backend) => {
                write!(f, "Backend {:?} not available (check feature flags)", backend)
            }
            CliffordFHEError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for CliffordFHEError {}

pub type Result<T> = std::result::Result<T, CliffordFHEError>;
