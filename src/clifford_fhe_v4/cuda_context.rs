///! V4 CUDA Context - Unified wrapper for all CUDA backend components
///!
///! Provides a clean API matching Metal's design pattern where rotation context
///! is integrated into the main context.

use crate::clifford_fhe_v2::backends::gpu_cuda::{
    ckks::CudaCkksContext,
    rotation::CudaRotationContext,
    rotation_keys::CudaRotationKeys,
    device::CudaDeviceContext,
};
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use std::sync::Arc;

/// V4 CUDA Context - wraps all necessary CUDA components
///
/// This provides a unified interface similar to Metal's design,
/// where rotation context doesn't need to be passed separately.
pub struct V4CudaContext {
    /// CUDA CKKS context
    pub ckks_ctx: Arc<CudaCkksContext>,

    /// CUDA rotation context (integrated, not exposed to user)
    rotation_ctx: Arc<CudaRotationContext>,

    /// CUDA device
    device: Arc<CudaDeviceContext>,

    /// Parameters
    params: CliffordFHEParams,
}

impl V4CudaContext {
    /// Create new V4 CUDA context
    pub fn new(params: CliffordFHEParams) -> Result<Self, String> {
        let device = Arc::new(CudaDeviceContext::new()?);
        let ckks_ctx = Arc::new(CudaCkksContext::new(params.clone())?);
        let rotation_ctx = Arc::new(CudaRotationContext::new(device.clone(), params.clone())?);

        Ok(Self {
            ckks_ctx,
            rotation_ctx,
            device,
            params,
        })
    }

    /// Get CKKS context
    pub fn ckks(&self) -> &Arc<CudaCkksContext> {
        &self.ckks_ctx
    }

    /// Get rotation context (for internal use by ciphertext operations)
    pub fn rotation_ctx(&self) -> &Arc<CudaRotationContext> {
        &self.rotation_ctx
    }

    /// Get device
    pub fn device(&self) -> &Arc<CudaDeviceContext> {
        &self.device
    }

    /// Get parameters
    pub fn params(&self) -> &CliffordFHEParams {
        &self.params
    }
}
