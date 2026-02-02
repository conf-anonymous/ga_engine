//! CUDA GPU-Accelerated Rotation Operations for CKKS
//!
//! Implements Galois automorphisms for ciphertext rotation using GPU kernels.
//!
//! **Key Concepts:**
//! - Rotation by k slots = Galois automorphism X → X^galois_elt
//! - Galois element: galois_elt = 5^k mod 2N (for cyclotomic ring Z[X]/(X^N + 1))
//! - Permutation map: new_idx = (old_idx * galois_elt) % (2N), then reduce to [0, N)
//!
//! **GPU Optimization:**
//! - Precompute permutation maps on CPU (cheap, done once)
//! - Apply permutation on GPU using dedicated kernel
//! - Works with flat RNS layout for optimal GPU memory access

use crate::clifford_fhe_v2::backends::gpu_cuda::device::CudaDeviceContext;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use cudarc::driver::LaunchAsync;
use cudarc::nvrtc::compile_ptx;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// CUDA rotation context with precomputed permutation maps
pub struct CudaRotationContext {
    device: Arc<CudaDeviceContext>,
    params: CliffordFHEParams,

    /// Permutation maps for each rotation amount
    /// Key: rotation steps (positive or negative)
    /// Value: permutation array perm[i] = index where coefficient i should move to
    /// Using Mutex for interior mutability with Arc
    rotation_maps: Mutex<HashMap<i32, Vec<u32>>>,

    /// Whether Galois kernels are loaded
    galois_kernels_loaded: bool,
}

impl CudaRotationContext {
    /// Create new CUDA rotation context
    pub fn new(device: Arc<CudaDeviceContext>, params: CliffordFHEParams) -> Result<Self, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║           Initializing CUDA Rotation Context                 ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let mut ctx = Self {
            device,
            params,
            rotation_maps: Mutex::new(HashMap::new()),
            galois_kernels_loaded: false,
        };

        // Load Galois CUDA kernels
        ctx.load_galois_kernels()?;

        // Precompute common rotation maps (powers of 2)
        let common_rotations = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

        println!("Precomputing rotation maps for common rotations...");
        for &rot in &common_rotations {
            if rot < (ctx.params.n / 2) as i32 {
                ctx.get_or_compute_rotation_map(rot)?;
                ctx.get_or_compute_rotation_map(-rot)?;
            }
        }

        let num_maps = ctx.rotation_maps.lock().unwrap().len();
        println!("  Precomputed {} rotation maps\n", num_maps);

        Ok(ctx)
    }

    /// Load Galois CUDA kernels
    fn load_galois_kernels(&mut self) -> Result<(), String> {
        println!("Loading Galois CUDA kernels...");

        let kernel_src = include_str!("kernels/galois.cu");

        let ptx = compile_ptx(kernel_src)
            .map_err(|e| format!("Failed to compile Galois kernels: {:?}", e))?;

        self.device.device.load_ptx(ptx, "galois_module", &[
            "apply_galois_automorphism",
            "rotate_with_key_switching",
        ]).map_err(|e| format!("Failed to load Galois PTX: {:?}", e))?;

        self.galois_kernels_loaded = true;
        println!("  Galois kernels loaded\n");

        Ok(())
    }

    /// Compute Galois element for rotation by k slots
    ///
    /// For the cyclotomic ring R = Z[X]/(X^N + 1), rotation by k slots
    /// corresponds to the Galois automorphism X → X^g where:
    ///   g = 5^k mod 2N  (for positive k)
    ///   g = 5^(-k) mod 2N  (for negative k, left rotation)
    fn compute_galois_element(&self, rotation_steps: i32) -> u64 {
        let n = self.params.n as i64;
        let two_n = 2 * n;

        // Handle negative rotations (left rotation = right rotation by -k)
        let k = if rotation_steps >= 0 {
            rotation_steps as i64
        } else {
            // For negative rotation, compute 5^(-k) = (5^{2N-2})^k mod 2N
            // Since 5^{φ(2N)} = 5^N ≡ 1 mod 2N, we have 5^{-1} ≡ 5^{N-1} mod 2N
            // But it's easier to compute as: -k rotation = (N - k) right rotation
            let slots = n / 2;
            (slots + rotation_steps as i64) % slots
        };

        // Compute g = 5^k mod 2N using modular exponentiation
        let base = 5i64;
        let mut result = 1i64;
        let mut b = base % two_n;
        let mut exp = k;

        while exp > 0 {
            if exp & 1 == 1 {
                result = (result * b) % two_n;
            }
            b = (b * b) % two_n;
            exp >>= 1;
        }

        result as u64
    }

    /// Compute permutation map for a given Galois element
    ///
    /// Given galois_elt g, the permutation is:
    ///   perm[i] = (i * g) % (2N), then map to [0, N):
    ///     - if result < N: keep as-is
    ///     - if result >= N: map to (2N - result) and negate coefficient
    ///
    /// For GPU kernel simplicity, we store:
    ///   perm[i] = target index (in [0, N))
    ///   sign[i] = 1 if positive, 0 if negative (need to negate)
    ///
    /// But we can pack this into a single i32:
    ///   perm[i] >= 0: normal copy
    ///   perm[i] < 0: copy and negate (store as -index - 1)
    fn compute_permutation_map(&self, galois_elt: u64) -> Vec<i32> {
        let n = self.params.n;
        let two_n = 2 * n;
        let mut perm = vec![0i32; n];

        for i in 0..n {
            // Compute (i * galois_elt) % 2N
            let raw_idx = ((i as u64 * galois_elt) % (two_n as u64)) as usize;

            // Map to [0, N) with sign
            if raw_idx < n {
                // Positive: keep as-is
                perm[i] = raw_idx as i32;
            } else {
                // Negative: map 2N - raw_idx and mark for negation
                let mapped_idx = two_n - raw_idx;
                perm[i] = -(mapped_idx as i32) - 1;  // Store as negative (will negate coeff)
            }
        }

        perm
    }

    /// Get or compute rotation map for given rotation steps
    fn get_or_compute_rotation_map(&self, rotation_steps: i32) -> Result<Vec<u32>, String> {
        // Check cache first (with read-only lock)
        {
            let maps = self.rotation_maps.lock()
                .map_err(|e| format!("Failed to lock rotation_maps: {:?}", e))?;
            if let Some(map) = maps.get(&rotation_steps) {
                return Ok(map.clone());
            }
        }

        // Compute new map
        let galois_elt = self.compute_galois_element(rotation_steps);
        let perm_map = self.compute_permutation_map(galois_elt);

        // Convert to u32 (GPU kernel expects unsigned indices with sign bit)
        let perm_u32: Vec<u32> = perm_map.iter().map(|&x| x as u32).collect();

        // Cache it (with write lock)
        {
            let mut maps = self.rotation_maps.lock()
                .map_err(|e| format!("Failed to lock rotation_maps for write: {:?}", e))?;
            maps.insert(rotation_steps, perm_u32.clone());
        }

        Ok(perm_u32)
    }

    /// Apply rotation to polynomial using GPU kernel
    ///
    /// Input/output in flat RNS layout: poly[prime_idx * n + coeff_idx]
    ///
    /// The GPU kernel:
    /// 1. Reads permutation map
    /// 2. For each coefficient position i:
    ///    - Read perm[i] (may be negative, indicating negation needed)
    ///    - Copy from source position to target position
    ///    - Negate if perm[i] was negative
    pub fn rotate_gpu(
        &self,
        poly_in: &[u64],
        rotation_steps: i32,
        num_primes: usize,
    ) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        assert_eq!(poly_in.len(), n * num_primes, "Input size mismatch");

        // Get permutation map (from cache or compute)
        let perm_map = self.get_or_compute_rotation_map(rotation_steps)?;
        assert_eq!(perm_map.len(), n, "Permutation map size mismatch");

        // Get moduli for this level
        let moduli = &self.params.moduli[..num_primes];

        // Copy inputs to GPU
        let gpu_input = self.device.device.htod_copy(poly_in.to_vec())
            .map_err(|e| format!("Failed to copy input to GPU: {:?}", e))?;

        let gpu_perm = self.device.device.htod_copy(perm_map)
            .map_err(|e| format!("Failed to copy permutation map: {:?}", e))?;

        let gpu_moduli = self.device.device.htod_copy(moduli.to_vec())
            .map_err(|e| format!("Failed to copy moduli: {:?}", e))?;

        let mut gpu_output = self.device.device.alloc_zeros::<u64>(n * num_primes)
            .map_err(|e| format!("Failed to allocate output: {:?}", e))?;

        // Get kernel function
        let func = self.device.device.get_func("galois_module", "apply_galois_automorphism")
            .ok_or("Failed to get apply_galois_automorphism function")?;

        // Launch kernel
        let config = self.device.get_launch_config(n * num_primes);
        unsafe {
            func.launch(config, (
                &gpu_input,
                &mut gpu_output,
                &gpu_perm,
                &gpu_moduli,
                n as u32,
                num_primes as u32,
            )).map_err(|e| format!("Rotation kernel launch failed: {:?}", e))?;
        }

        // Copy result back
        let result = self.device.device.dtoh_sync_copy(&gpu_output)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        Ok(result)
    }

    /// Get the device context
    pub fn device(&self) -> &Arc<CudaDeviceContext> {
        &self.device
    }

    /// Get the parameters
    pub fn params(&self) -> &CliffordFHEParams {
        &self.params
    }

    /// Get the Galois element for a given rotation
    pub fn galois_element(&self, rotation_steps: i32) -> u64 {
        self.compute_galois_element(rotation_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_galois_element_computation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let device = Arc::new(CudaDeviceContext::new(0).unwrap());
        let ctx = CudaRotationContext::new(device, params).unwrap();

        // Test rotation by 1 slot
        let g1 = ctx.compute_galois_element(1);
        assert_eq!(g1, 5, "Rotation by 1 should give galois element 5");

        // Test rotation by 2 slots
        let g2 = ctx.compute_galois_element(2);
        assert_eq!(g2, 25, "Rotation by 2 should give galois element 25");
    }

    #[test]
    fn test_permutation_map_rotation_by_1() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let device = Arc::new(CudaDeviceContext::new(0).unwrap());
        let ctx = CudaRotationContext::new(device, params).unwrap();

        let galois_elt = 5;  // Rotation by 1
        let perm = ctx.compute_permutation_map(galois_elt);

        // For small indices, check expected mapping
        // X^0 → X^0
        // X^1 → X^5
        // X^2 → X^10
        // etc.
        assert_eq!(perm[0], 0);
        assert_eq!(perm[1], 5);
        assert_eq!(perm[2], 10);
    }
}
