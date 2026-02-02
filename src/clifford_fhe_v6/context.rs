//! V6 ParallelLiftContext - GPU Context Lifecycle Management
//!
//! This module provides the central context for V6 operations, wrapping both
//! V2's CudaCkksContext and parallel_lift's FheGpuContext.

use std::sync::Arc;

use parallel_lift_cuda::FheGpuContext;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;

use super::{V6Error, V6Result};

/// V6 GPU Context with parallel_lift acceleration
///
/// This context wraps both V2's `CudaCkksContext` for standard CKKS operations
/// and parallel_lift's `FheGpuContext` for accelerated CRT operations.
///
/// # Thread Safety
///
/// `ParallelLiftContext` is `Send` but not `Sync`. Each thread should have
/// its own context instance for best performance.
///
/// # Example
///
/// ```rust,ignore
/// let params = CliffordFHEParams::new_128bit();
/// let ctx = ParallelLiftContext::with_params(params)?;
///
/// // Access V2 context for standard operations
/// let encoded = ctx.v2_ctx().encode(&values, scale, level)?;
///
/// // Access parallel_lift for accelerated operations
/// let digits = ctx.parallel_lift().gpu_gadget_decompose(...);
/// ```
pub struct ParallelLiftContext {
    /// Underlying V2 CUDA CKKS context
    /// Provides: encode/decode, NTT, add/multiply, rescale
    pub ckks_ctx: Arc<CudaCkksContext>,

    /// parallel_lift GPU context
    /// Provides: gpu_gadget_decompose, gpu_batch_reconstruct, gpu_batch_matrix_vector
    pub fhe_gpu: FheGpuContext,

    /// Cached FHE parameters
    pub params: CliffordFHEParams,
}

impl ParallelLiftContext {
    /// Create a new V6 context from parameters and an existing V2 context
    ///
    /// # Arguments
    /// * `params` - FHE parameters
    /// * `ckks_ctx` - Pre-initialized V2 CUDA CKKS context
    ///
    /// # Errors
    /// Returns error if FheGpuContext initialization fails (e.g., no CUDA GPU)
    pub fn new(params: CliffordFHEParams, ckks_ctx: Arc<CudaCkksContext>) -> V6Result<Self> {
        let num_slots = params.n / 2;
        let num_rns_primes = params.moduli.len();
        // Use same number of CRT primes as RNS primes for key-level operations
        let num_crt_primes = num_rns_primes.max(16); // At least 16 for precision

        let fhe_gpu = FheGpuContext::new(num_slots, num_rns_primes, num_crt_primes)
            .map_err(|e| V6Error::ContextInitFailed(format!("{:?}", e)))?;

        Ok(Self {
            ckks_ctx,
            fhe_gpu,
            params,
        })
    }

    /// Create V6 context from an existing V2 context
    ///
    /// This is the most common initialization path when upgrading from V2.
    ///
    /// # Example
    /// ```rust,ignore
    /// let v2_ctx = Arc::new(CudaCkksContext::new(params.clone())?);
    /// let v6_ctx = ParallelLiftContext::from_v2_context(v2_ctx)?;
    /// ```
    pub fn from_v2_context(ckks_ctx: Arc<CudaCkksContext>) -> V6Result<Self> {
        let params = ckks_ctx.params().clone();
        Self::new(params, ckks_ctx)
    }

    /// Create a new V6 context with given parameters
    ///
    /// This creates a fresh V2 CudaCkksContext internally.
    ///
    /// # Example
    /// ```rust,ignore
    /// let params = CliffordFHEParams::new_128bit();
    /// let ctx = ParallelLiftContext::with_params(params)?;
    /// ```
    pub fn with_params(params: CliffordFHEParams) -> V6Result<Self> {
        let ckks_ctx = Arc::new(
            CudaCkksContext::new(params.clone())
                .map_err(|e| V6Error::V2Error(e))?
        );
        Self::new(params, ckks_ctx)
    }

    /// Access the underlying V2 CKKS context
    ///
    /// Use this for standard CKKS operations (encode, decode, add, multiply, rescale).
    #[inline]
    pub fn v2_ctx(&self) -> &CudaCkksContext {
        &self.ckks_ctx
    }

    /// Access the parallel_lift FheGpuContext
    ///
    /// Use this for accelerated operations (gpu_gadget_decompose, etc.).
    #[inline]
    pub fn parallel_lift(&self) -> &FheGpuContext {
        &self.fhe_gpu
    }

    /// Get the FHE parameters
    #[inline]
    pub fn params(&self) -> &CliffordFHEParams {
        &self.params
    }

    /// Get the polynomial degree N
    #[inline]
    pub fn n(&self) -> usize {
        self.params.n
    }

    /// Get the number of slots (N/2)
    #[inline]
    pub fn num_slots(&self) -> usize {
        self.params.n / 2
    }

    /// Get the number of RNS moduli
    #[inline]
    pub fn num_moduli(&self) -> usize {
        self.params.moduli.len()
    }

    /// Get the RNS moduli
    #[inline]
    pub fn moduli(&self) -> &[u64] {
        &self.params.moduli
    }

    /// Reinitialize the parallel_lift context with different dimensions
    ///
    /// This is useful when switching between different parameter sets.
    pub fn reinitialize(&mut self, num_slots: usize, num_rns_primes: usize) -> V6Result<()> {
        let num_crt_primes = num_rns_primes.max(16);

        self.fhe_gpu = FheGpuContext::new(num_slots, num_rns_primes, num_crt_primes)
            .map_err(|e| V6Error::ContextInitFailed(format!("{:?}", e)))?;

        Ok(())
    }
}

// ParallelLiftContext is Send (can be moved between threads)
// but NOT Sync (not safe for shared access without synchronization)
unsafe impl Send for ParallelLiftContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation_stub() {
        // This test verifies the API compiles correctly
        // Actual CUDA tests require a GPU
        let _ = CliffordFHEParams::new_test_ntt_1024();
    }
}
