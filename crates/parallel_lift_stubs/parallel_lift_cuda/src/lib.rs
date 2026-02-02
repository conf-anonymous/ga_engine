//! Stub crate for parallel_lift_cuda
//!
//! This is a placeholder that allows ga_engine to compile without the real parallel_lift.
//! All operations will panic at runtime with a clear error message.
//!
//! To use the real parallel_lift:
//! 1. Clone the parallel_lift repository to ../parallel_lift
//! 2. Run: ./scripts/enable_parallel_lift.sh
//! 3. Build with: cargo build --features v6-cuda

use std::fmt;

/// Stub GPU context that panics when used
///
/// This placeholder allows V6 code to compile without the real parallel_lift.
/// Any attempt to use it will panic with a helpful error message.
pub struct FheGpuContext {
    _private: (),
}

/// Error type for stub operations
#[derive(Debug)]
pub struct StubError(pub String);

impl fmt::Display for StubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parallel_lift stub error: {}", self.0)
    }
}

impl std::error::Error for StubError {}

const STUB_ERROR_MSG: &str = "parallel_lift is not available. \
    To enable: run ./scripts/enable_parallel_lift.sh \
    (requires parallel_lift repository at ../parallel_lift)";

impl FheGpuContext {
    /// Stub constructor - always returns an error
    ///
    /// To use real parallel_lift, run: ./scripts/enable_parallel_lift.sh
    pub fn new(
        _num_slots: usize,
        _num_rns_primes: usize,
        _num_crt_primes: usize,
    ) -> Result<Self, StubError> {
        Err(StubError(STUB_ERROR_MSG.to_string()))
    }

    /// Stub gadget decomposition - panics if called
    ///
    /// The real implementation provides 25× speedup via GPU CRT operations.
    pub fn gpu_gadget_decompose(
        &self,
        _poly: &[u64],
        _rns_primes: &[u64],
        _gadget_base: u64,
        _num_digits: usize,
    ) -> Vec<Vec<i64>> {
        panic!("{}", STUB_ERROR_MSG);
    }

    /// Stub batch reconstruct - panics if called
    pub fn gpu_batch_reconstruct(
        &self,
        _digits: &[Vec<i64>],
        _rns_primes: &[u64],
        _gadget_base: u64,
    ) -> Vec<u64> {
        panic!("{}", STUB_ERROR_MSG);
    }

    /// Stub batch matrix-vector multiply - panics if called
    pub fn gpu_batch_matrix_vector(
        &self,
        _matrix: &[Vec<u64>],
        _vector: &[u64],
        _modulus: u64,
    ) -> Vec<u64> {
        panic!("{}", STUB_ERROR_MSG);
    }
}
