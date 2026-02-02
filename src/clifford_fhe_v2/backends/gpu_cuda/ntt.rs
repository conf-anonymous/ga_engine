/// CUDA NTT Implementation
///
/// Rust wrapper for CUDA NTT kernels, providing Harvey Butterfly NTT on NVIDIA GPUs.

use super::device::CudaDeviceContext;
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

/// Precomputed twiddle factors and NTT context for CUDA
pub struct CudaNttContext {
    pub(crate) device: Arc<CudaDeviceContext>,
    pub(crate) n: usize,
    pub(crate) q: u64,
    pub(crate) root: u64,
    pub twiddles: Vec<u64>,        // Made public for debugging
    pub twiddles_inv: Vec<u64>,    // Made public for debugging
    pub n_inv: u64,                // Made public for debugging
    log_n: usize,
    // Cached to avoid recompilation
    _kernels_loaded: bool,
}

impl CudaNttContext {
    /// Create new CUDA NTT context
    pub fn new(n: usize, q: u64, root: u64) -> Result<Self, String> {
        let device = Arc::new(CudaDeviceContext::new()?);

        // Precompute twiddle factors
        let (twiddles, twiddles_inv, n_inv) = Self::precompute_twiddles(n, q, root)?;

        let log_n = (n as f64).log2() as usize;

        // Compile and load all kernels ONCE during initialization
        let kernel_src = include_str!("kernels/ntt.cu");
        let ptx = compile_ptx(kernel_src)
            .map_err(|e| format!("Failed to compile CUDA kernel: {:?}", e))?;

        device.device.load_ptx(ptx, "ntt_module", &[
            "bit_reverse_permutation",
            "bit_reverse_permutation_batched",
            "ntt_forward",
            "ntt_inverse",
            "ntt_inverse_final",
            "ntt_scalar_multiply",
            "ntt_pointwise_multiply",
            "ntt_forward_batched",
            "ntt_inverse_batched",
            "ntt_inverse_final_batched",
            "ntt_pointwise_multiply_batched",
        ]).map_err(|e| format!("Failed to load PTX: {:?}", e))?;

        Ok(CudaNttContext {
            device,
            n,
            q,
            root,
            twiddles,
            twiddles_inv,
            n_inv,
            log_n,
            _kernels_loaded: true,
        })
    }

    /// Precompute twiddle factors for NTT
    fn precompute_twiddles(n: usize, q: u64, root: u64) -> Result<(Vec<u64>, Vec<u64>, u64), String> {
        let mut twiddles = vec![1u64; n];
        let mut twiddles_inv = vec![1u64; n];

        // Forward twiddles: ω^i for i = 0..n
        let mut power = 1u64;
        for i in 0..n {
            twiddles[i] = power;
            power = Self::mul_mod_u128(power, root, q);
        }

        // Inverse twiddles: ω^(-i) for i = 0..n
        let root_inv = Self::mod_inverse(root, q)?;
        power = 1u64;
        for i in 0..n {
            twiddles_inv[i] = power;
            power = Self::mul_mod_u128(power, root_inv, q);
        }

        // n^(-1) mod q for final scaling in inverse NTT
        let n_inv = Self::mod_inverse(n as u64, q)?;

        Ok((twiddles, twiddles_inv, n_inv))
    }

    /// Modular multiplication using u128
    fn mul_mod_u128(a: u64, b: u64, q: u64) -> u64 {
        ((a as u128 * b as u128) % q as u128) as u64
    }

    /// Modular inverse using extended Euclidean algorithm
    fn mod_inverse(a: u64, q: u64) -> Result<u64, String> {
        let (mut t, mut new_t) = (0i128, 1i128);
        let (mut r, mut new_r) = (q as i128, a as i128);

        while new_r != 0 {
            let quotient = r / new_r;
            (t, new_t) = (new_t, t - quotient * new_t);
            (r, new_r) = (new_r, r - quotient * new_r);
        }

        if r > 1 {
            return Err(format!("{} is not invertible mod {}", a, q));
        }
        if t < 0 {
            t += q as i128;
        }

        Ok(t as u64)
    }

    /// Forward NTT on GPU
    pub fn forward(&self, coeffs: &mut [u64]) -> Result<(), String> {
        if coeffs.len() != self.n {
            return Err(format!("Expected {} coefficients, got {}", self.n, coeffs.len()));
        }

        // Copy to GPU
        let mut gpu_coeffs = self.device.device.htod_copy(coeffs.to_vec())
            .map_err(|e| format!("Failed to copy to GPU: {:?}", e))?;

        let gpu_twiddles = self.device.device.htod_copy(self.twiddles.clone())
            .map_err(|e| format!("Failed to copy twiddles: {:?}", e))?;

        // Kernels already loaded during initialization - just get function handles

        // Bit-reversal permutation (needs n threads, not n/2)
        let func_bit_reverse = self.device.device.get_func("ntt_module", "bit_reverse_permutation")
            .ok_or("Failed to get bit_reverse_permutation function")?;

        let config = self.device.get_launch_config(self.n);
        unsafe {
            func_bit_reverse.launch(config, (&mut gpu_coeffs, self.n as u32, self.log_n as u32))
                .map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        }

        // Synchronize after bit-reversal
        self.device.device.synchronize()
            .map_err(|e| format!("Sync after bit-reverse failed: {:?}", e))?;

        // NTT stages
        let mut m = 1usize;
        for stage in 0..self.log_n {
            let func_ntt = self.device.device.get_func("ntt_module", "ntt_forward")
                .ok_or("Failed to get ntt_forward function")?;

            let config = self.device.get_launch_config(self.n / 2);
            unsafe {
                func_ntt.launch(config, (&mut gpu_coeffs, &gpu_twiddles, self.n as u32, self.q, stage as u32, m as u32))
                    .map_err(|e| format!("NTT stage {} failed: {:?}", stage, e))?;
            }

            // Synchronize after each stage to ensure correct ordering
            self.device.device.synchronize()
                .map_err(|e| format!("Sync after stage {} failed: {:?}", stage, e))?;

            m *= 2;
        }

        // Copy result back (dtoh_sync_copy already synchronizes)
        let result = self.device.device.dtoh_sync_copy(&gpu_coeffs)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        coeffs.copy_from_slice(&result);
        Ok(())
    }

    /// Inverse NTT on GPU (Gentleman-Sande DIF algorithm, matching Metal)
    ///
    /// Uses the same algorithm as the working Metal implementation:
    /// 1. NO bit-reversal at the start
    /// 2. Butterfly stages run in REVERSE order (log_n-1 down to 0)
    /// 3. Gentleman-Sande DIF butterfly: (u, v) -> (u + v, (u - v) * w)
    /// 4. Bit-reversal + scaling by n^(-1) at the END
    pub fn inverse(&self, coeffs: &mut [u64]) -> Result<(), String> {
        if coeffs.len() != self.n {
            return Err(format!("Expected {} coefficients, got {}", self.n, coeffs.len()));
        }

        // Copy to GPU
        let mut gpu_coeffs = self.device.device.htod_copy(coeffs.to_vec())
            .map_err(|e| format!("Failed to copy to GPU: {:?}", e))?;

        let gpu_twiddles_inv = self.device.device.htod_copy(self.twiddles_inv.clone())
            .map_err(|e| format!("Failed to copy twiddles: {:?}", e))?;

        // Step 1: Inverse NTT butterfly stages in REVERSE order (log_n-1 down to 0)
        // This matches Metal's `for stage in (0..log_n).rev()`
        for stage in (0..self.log_n).rev() {
            let func_ntt_inv = self.device.device.get_func("ntt_module", "ntt_inverse")
                .ok_or("Failed to get ntt_inverse function")?;

            let m = 1usize << (stage + 1);  // m = 2^(stage+1), matching Metal
            let config = self.device.get_launch_config(self.n / 2);
            unsafe {
                func_ntt_inv.launch(config, (&mut gpu_coeffs, &gpu_twiddles_inv, self.n as u32, self.q, stage as u32, m as u32))
                    .map_err(|e| format!("Inverse NTT stage {} failed: {:?}", stage, e))?;
            }

            // Synchronize after each stage to ensure correct ordering
            self.device.device.synchronize()
                .map_err(|e| format!("Sync after stage {} failed: {:?}", stage, e))?;
        }

        // Step 2: Bit-reversal + scaling by n^(-1) at the END
        let func_final = self.device.device.get_func("ntt_module", "ntt_inverse_final")
            .ok_or("Failed to get ntt_inverse_final function")?;

        let config = self.device.get_launch_config(self.n);
        unsafe {
            func_final.launch(config, (&mut gpu_coeffs, self.n as u32, self.log_n as u32, self.q, self.n_inv))
                .map_err(|e| format!("Inverse NTT final step failed: {:?}", e))?;
        }

        // Copy result back (dtoh_sync_copy already synchronizes)
        let result = self.device.device.dtoh_sync_copy(&gpu_coeffs)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        coeffs.copy_from_slice(&result);
        Ok(())
    }

    /// Pointwise multiplication in NTT domain
    pub fn pointwise_multiply(&self, a: &[u64], b: &[u64], result: &mut [u64]) -> Result<(), String> {
        if a.len() != self.n || b.len() != self.n || result.len() != self.n {
            return Err("All arrays must have length n".to_string());
        }

        // Copy to GPU
        let gpu_a = self.device.device.htod_copy(a.to_vec())
            .map_err(|e| format!("Failed to copy a to GPU: {:?}", e))?;
        let gpu_b = self.device.device.htod_copy(b.to_vec())
            .map_err(|e| format!("Failed to copy b to GPU: {:?}", e))?;
        let mut gpu_c = self.device.device.alloc_zeros::<u64>(self.n)
            .map_err(|e| format!("Failed to allocate result: {:?}", e))?;

        // Kernels already loaded during initialization - just get function handle

        let func = self.device.device.get_func("ntt_module", "ntt_pointwise_multiply")
            .ok_or("Failed to get ntt_pointwise_multiply function")?;

        // Launch kernel
        let config = self.device.get_launch_config(self.n);
        unsafe {
            func.launch(config, (&gpu_a, &gpu_b, &mut gpu_c, self.n as u32, self.q))
                .map_err(|e| format!("Pointwise multiply failed: {:?}", e))?;
        }

        // Copy result back
        let res = self.device.device.dtoh_sync_copy(&gpu_c)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        result.copy_from_slice(&res);
        Ok(())
    }
}
