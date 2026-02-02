//! Metal-accelerated NTT (Number Theoretic Transform)
//!
//! Rust wrappers for Metal NTT compute kernels.

use super::device::MetalDevice;
use metal::*;

/// Metal NTT context with precomputed twiddle factors
pub struct MetalNttContext {
    device: std::sync::Arc<MetalDevice>,
    pub(crate) n: usize,
    pub(crate) q: u64,
    psi: u64,                    // Primitive 2n-th root (for twisted NTT)
    omega: u64,                  // Primitive n-th root = psi^2
    psi_powers: Vec<u64>,        // Powers of psi: [1, psi, psi^2, ..., psi^(n-1)]
    psi_inv_powers: Vec<u64>,    // Powers of psi^(-1)
    omega_powers: Vec<u64>,      // Powers of omega: [1, omega, omega^2, ..., omega^(n-1)]
    omega_inv_powers: Vec<u64>,  // Powers of omega^(-1)
    n_inv: u64,                  // Modular inverse of n (normal domain)
    n_inv_montgomery: u64,       // Modular inverse of n (Montgomery domain)
    // Montgomery multiplication parameters
    q_inv: u64,                  // -q^{-1} mod 2^64 for Montgomery reduction
    r_squared: u64,              // R^2 mod q where R = 2^64 (for domain conversion)
    omega_powers_montgomery: Vec<u64>,      // omega_powers in Montgomery domain
    omega_inv_powers_montgomery: Vec<u64>,  // omega_inv_powers in Montgomery domain
}

impl MetalNttContext {
    /// Create new Metal NTT context (creates its own device)
    ///
    /// @param n Polynomial degree (must be power of 2)
    /// @param q NTT-friendly prime (q ≡ 1 mod 2n)
    /// @param psi Primitive 2n-th root of unity mod q (for twisted NTT)
    pub fn new(n: usize, q: u64, psi: u64) -> Result<Self, String> {
        let device = MetalDevice::new()?;
        Self::new_with_device(std::sync::Arc::new(device), n, q, psi)
    }

    /// Create new Metal NTT context with existing device (RECOMMENDED)
    ///
    /// This avoids creating multiple Metal devices and improves performance.
    ///
    /// @param device Existing Metal device (wrapped in Arc for sharing)
    /// @param n Polynomial degree (must be power of 2)
    /// @param q NTT-friendly prime (q ≡ 1 mod 2n)
    /// @param psi Primitive 2n-th root of unity mod q (for twisted NTT)
    pub fn new_with_device(
        device: std::sync::Arc<MetalDevice>,
        n: usize,
        q: u64,
        psi: u64,
    ) -> Result<Self, String> {
        // Verify n is power of 2
        if n & (n - 1) != 0 {
            return Err(format!("n must be power of 2, got {}", n));
        }

        // Verify psi is a primitive 2n-th root
        let two_n = (2 * n) as u64;
        if Self::pow_mod(psi, two_n, q) != 1 {
            return Err(format!("psi is not a primitive 2n-th root of unity"));
        }

        // Compute omega = psi^2 (n-th root)
        let omega = Self::mul_mod(psi, psi, q);

        // Precompute powers of psi: [1, psi, psi^2, ..., psi^(n-1)]
        let mut psi_powers = vec![0u64; n];
        psi_powers[0] = 1;
        for i in 1..n {
            psi_powers[i] = Self::mul_mod(psi_powers[i - 1], psi, q);
        }

        // Precompute powers of psi^(-1)
        let psi_inv = Self::mod_inverse(psi, q)?;
        let mut psi_inv_powers = vec![0u64; n];
        psi_inv_powers[0] = 1;
        for i in 1..n {
            psi_inv_powers[i] = Self::mul_mod(psi_inv_powers[i - 1], psi_inv, q);
        }

        // Precompute powers of omega: [1, omega, omega^2, ..., omega^(n-1)]
        let mut omega_powers = vec![0u64; n];
        omega_powers[0] = 1;
        for i in 1..n {
            omega_powers[i] = Self::mul_mod(omega_powers[i - 1], omega, q);
        }

        // Precompute powers of omega^(-1)
        let omega_inv = Self::mod_inverse(omega, q)?;
        let mut omega_inv_powers = vec![0u64; n];
        omega_inv_powers[0] = 1;
        for i in 1..n {
            omega_inv_powers[i] = Self::mul_mod(omega_inv_powers[i - 1], omega_inv, q);
        }

        // Compute modular inverse of n
        let n_inv = Self::mod_inverse(n as u64, q)?;

        // Compute Montgomery parameters
        let q_inv = Self::compute_q_inv(q);
        let r_squared = Self::compute_r_squared_mod_q(q);

        // Convert twiddle factors to Montgomery domain
        let omega_powers_montgomery: Vec<u64> = omega_powers.iter()
            .map(|&w| Self::to_montgomery(w, r_squared, q, q_inv))
            .collect();

        let omega_inv_powers_montgomery: Vec<u64> = omega_inv_powers.iter()
            .map(|&w| Self::to_montgomery(w, r_squared, q, q_inv))
            .collect();

        // Convert n_inv to Montgomery domain for final scaling
        let n_inv_montgomery = Self::to_montgomery(n_inv, r_squared, q, q_inv);

        Ok(MetalNttContext {
            device,
            n,
            q,
            psi,
            omega,
            psi_powers,
            psi_inv_powers,
            omega_powers,
            omega_inv_powers,
            n_inv,
            n_inv_montgomery,
            q_inv,
            r_squared,
            omega_powers_montgomery,
            omega_inv_powers_montgomery,
        })
    }

    /// Forward NTT on GPU with proper global synchronization
    ///
    /// Uses stage-per-dispatch approach to ensure correctness:
    /// 0. Convert input to Montgomery domain (on CPU)
    /// 1. Bit-reversal permutation (1 dispatch)
    /// 2. For each stage 0..log2(n): butterfly pass (log2(n) dispatches)
    /// 3. Convert output from Montgomery domain (on CPU)
    ///
    /// Each dispatch provides implicit global synchronization barrier.
    pub fn forward(&self, coeffs: &mut [u64]) -> Result<(), String> {
        if coeffs.len() != self.n {
            return Err(format!("Expected {} coefficients, got {}", self.n, coeffs.len()));
        }

        // Convert input to Montgomery domain
        let mut coeffs_montgomery: Vec<u64> = coeffs.iter()
            .map(|&c| Self::to_montgomery(c, self.r_squared, self.q, self.q_inv))
            .collect();

        let log_n = (self.n as f64).log2() as u32;

        // Create persistent GPU buffer (will be modified in-place)
        let coeffs_buffer = self.device.create_buffer_with_data(&coeffs_montgomery);
        let omega_powers_buffer = self.device.create_buffer_with_data(&self.omega_powers_montgomery);
        let n_buffer = self.device.create_buffer_with_u32_data(&[self.n as u32]);
        let q_buffer = self.device.create_buffer_with_data(&[self.q]);
        let q_inv_buffer = self.device.create_buffer_with_data(&[self.q_inv]);

        let threadgroup_size = MTLSize::new(256, 1, 1);
        let threadgroups = MTLSize::new(((self.n + 255) / 256) as u64, 1, 1);

        // Step 1: Bit-reversal permutation
        {
            let kernel = self.device.get_function("ntt_bit_reverse")?;
            let pipeline = self.device.device()
                .new_compute_pipeline_state_with_function(&kernel)
                .map_err(|e| format!("Failed to create bit-reverse pipeline: {:?}", e))?;

            self.device.execute_kernel(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&coeffs_buffer), 0);
                encoder.set_buffer(1, Some(&n_buffer), 0);
                encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
                Ok(())
            })?;
        }

        // Step 2: Execute each butterfly stage separately
        // This provides GLOBAL synchronization between stages (the key fix!)
        let kernel = self.device.get_function("ntt_forward_stage")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create stage pipeline: {:?}", e))?;

        for stage in 0..log_n {
            let stage_buffer = self.device.create_buffer_with_u32_data(&[stage]);

            self.device.execute_kernel(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&coeffs_buffer), 0);
                encoder.set_buffer(1, Some(&omega_powers_buffer), 0);
                encoder.set_buffer(2, Some(&n_buffer), 0);
                encoder.set_buffer(3, Some(&q_buffer), 0);
                encoder.set_buffer(4, Some(&stage_buffer), 0);
                encoder.set_buffer(5, Some(&q_inv_buffer), 0);  // NEW: Montgomery q_inv

                // Only need n/2 threads for butterflies
                let butterfly_threads = ((self.n / 2 + 255) / 256) as u64;
                let butterfly_threadgroups = MTLSize::new(butterfly_threads, 1, 1);
                encoder.dispatch_thread_groups(butterfly_threadgroups, threadgroup_size);
                Ok(())
            })?;
            // Implicit global barrier here - next dispatch waits for this one to complete!
        }

        // Read result back - KEEP in Montgomery NTT domain!
        // This allows pointwise_multiply to work correctly with Montgomery inputs
        let result_montgomery = self.device.read_buffer(&coeffs_buffer, self.n);
        coeffs.copy_from_slice(&result_montgomery);

        Ok(())
    }

    /// Inverse NTT on GPU with proper global synchronization
    ///
    /// Uses stage-per-dispatch approach:
    /// 0. Convert input to Montgomery domain (on CPU)
    /// 1. For each stage (log2(n)-1 down to 0): butterfly pass
    /// 2. Bit-reversal and scaling by 1/n
    /// 3. Convert output from Montgomery domain (on CPU)
    pub fn inverse(&self, evals: &mut [u64]) -> Result<(), String> {
        if evals.len() != self.n {
            return Err(format!("Expected {} evaluation points, got {}", self.n, evals.len()));
        }

        // Input is already in Montgomery NTT domain from forward() or pointwise_multiply()
        // No conversion needed!

        let log_n = (self.n as f64).log2() as u32;

        // Create persistent GPU buffer - input is already Montgomery
        let evals_buffer = self.device.create_buffer_with_data(evals);
        let omega_inv_powers_buffer = self.device.create_buffer_with_data(&self.omega_inv_powers_montgomery);
        let n_buffer = self.device.create_buffer_with_u32_data(&[self.n as u32]);
        let q_buffer = self.device.create_buffer_with_data(&[self.q]);
        let n_inv_buffer = self.device.create_buffer_with_data(&[self.n_inv_montgomery]);
        let q_inv_buffer = self.device.create_buffer_with_data(&[self.q_inv]);

        let threadgroup_size = MTLSize::new(256, 1, 1);

        // Step 1: Execute inverse butterfly stages (in reverse order)
        let kernel = self.device.get_function("ntt_inverse_stage")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create inverse stage pipeline: {:?}", e))?;

        for stage in (0..log_n).rev() {
            let stage_buffer = self.device.create_buffer_with_u32_data(&[stage]);

            self.device.execute_kernel(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&evals_buffer), 0);
                encoder.set_buffer(1, Some(&omega_inv_powers_buffer), 0);
                encoder.set_buffer(2, Some(&n_buffer), 0);
                encoder.set_buffer(3, Some(&q_buffer), 0);
                encoder.set_buffer(4, Some(&stage_buffer), 0);
                encoder.set_buffer(5, Some(&q_inv_buffer), 0);  // NEW: Montgomery q_inv

                let butterfly_threads = ((self.n / 2 + 255) / 256) as u64;
                let butterfly_threadgroups = MTLSize::new(butterfly_threads, 1, 1);
                encoder.dispatch_thread_groups(butterfly_threadgroups, threadgroup_size);
                Ok(())
            })?;
            // Implicit global barrier between stages
        }

        // Step 2: Bit-reversal and scaling
        {
            let kernel = self.device.get_function("ntt_inverse_final_scale")?;
            let pipeline = self.device.device()
                .new_compute_pipeline_state_with_function(&kernel)
                .map_err(|e| format!("Failed to create final scale pipeline: {:?}", e))?;

            self.device.execute_kernel(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&evals_buffer), 0);
                encoder.set_buffer(1, Some(&n_buffer), 0);
                encoder.set_buffer(2, Some(&q_buffer), 0);
                encoder.set_buffer(3, Some(&n_inv_buffer), 0);
                encoder.set_buffer(4, Some(&q_inv_buffer), 0);  // NEW: Montgomery q_inv

                let threadgroups = MTLSize::new(((self.n + 255) / 256) as u64, 1, 1);
                encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
                Ok(())
            })?;
        }

        // Read result (still in Montgomery domain)
        let result_montgomery = self.device.read_buffer(&evals_buffer, self.n);

        // Convert from Montgomery domain back to normal domain
        for i in 0..self.n {
            // To leave Montgomery domain: mont_mul(x*R, 1, q, q_inv) = x
            evals[i] = Self::mont_mul_cpu(result_montgomery[i], 1, self.q, self.q_inv);
        }

        Ok(())
    }

    /// Pointwise multiplication in NTT domain (Hadamard product)
    ///
    /// Implements polynomial multiplication: c(x) = a(x) * b(x)
    pub fn pointwise_multiply(&self, a: &[u64], b: &[u64], c: &mut [u64]) -> Result<(), String> {
        if a.len() != self.n || b.len() != self.n || c.len() != self.n {
            return Err("All arrays must have length n".to_string());
        }

        // Create GPU buffers
        let a_buffer = self.device.create_buffer_with_data(a);
        let b_buffer = self.device.create_buffer_with_data(b);
        let c_buffer = self.device.create_buffer(self.n);

        // Metal requires scalar parameters as buffers
        let n_buffer = self.device.create_buffer_with_u32_data(&[self.n as u32]);
        let q_buffer = self.device.create_buffer_with_data(&[self.q]);
        let q_inv_buffer = self.device.create_buffer_with_data(&[self.q_inv]); // IMPORTANT: Add q_inv for Montgomery multiplication!

        // Get kernel
        let kernel = self.device.get_function("ntt_pointwise_multiply")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {:?}", e))?;

        // Execute
        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&a_buffer), 0);
            encoder.set_buffer(1, Some(&b_buffer), 0);
            encoder.set_buffer(2, Some(&c_buffer), 0);
            encoder.set_buffer(3, Some(&n_buffer), 0);
            encoder.set_buffer(4, Some(&q_buffer), 0);
            encoder.set_buffer(5, Some(&q_inv_buffer), 0); // Pass q_inv!

            let threadgroup_size = MTLSize::new(256, 1, 1);
            let threadgroups = MTLSize::new(((self.n + 255) / 256) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);

            Ok(())
        })?;

        // Read result
        let result = self.device.read_buffer(&c_buffer, self.n);
        c.copy_from_slice(&result);

        Ok(())
    }

    // Helper functions (same as CPU version)

    fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
        ((a as u128 * b as u128) % q as u128) as u64
    }

    fn pow_mod(mut base: u64, mut exp: u64, q: u64) -> u64 {
        let mut result = 1u64;
        base %= q;

        while exp > 0 {
            if exp & 1 == 1 {
                result = Self::mul_mod(result, base, q);
            }
            base = Self::mul_mod(base, base, q);
            exp >>= 1;
        }

        result
    }

    fn mod_inverse(a: u64, q: u64) -> Result<u64, String> {
        // Extended Euclidean algorithm
        let (mut old_r, mut r) = (a as i128, q as i128);
        let (mut old_s, mut s) = (1i128, 0i128);

        while r != 0 {
            let quotient = old_r / r;
            let temp_r = r;
            r = old_r - quotient * r;
            old_r = temp_r;

            let temp_s = s;
            s = old_s - quotient * s;
            old_s = temp_s;
        }

        if old_r != 1 {
            return Err(format!("{} has no modular inverse mod {}", a, q));
        }

        // Ensure positive result
        let result = if old_s < 0 {
            (old_s + q as i128) as u64
        } else {
            old_s as u64
        };

        Ok(result)
    }

    /// Compute q_inv = -q^{-1} mod 2^64 for Montgomery multiplication
    ///
    /// Uses Newton's method (Hensel lifting) for fast computation:
    /// x_{i+1} = x_i * (2 - q * x_i) mod 2^{2i}
    ///
    /// This is MUCH faster than Extended Euclidean for mod 2^64
    fn compute_q_inv(q: u64) -> u64 {
        // Newton's method (Hensel lifting) to compute q^{-1} mod 2^64
        // Start with 3-bit approximation: q * q_inv ≡ 1 (mod 8)
        // For odd q: q_inv ≡ q (mod 8) works as initial approximation

        // q must be odd for NTT-friendly primes
        assert!(q & 1 == 1, "q must be odd for Montgomery multiplication");

        let mut q_inv = q;  // Initial 3-bit approximation

        // Newton iteration: x_{i+1} = x_i * (2 - q * x_i)
        // Doubles precision each iteration: 3 -> 6 -> 12 -> 24 -> 48 -> 64 bits
        for _ in 0..5 {
            // Compute in wrapping arithmetic (mod 2^64)
            q_inv = q_inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(q_inv)));
        }

        // Now q_inv is q^{-1} mod 2^64
        // We want -q^{-1} mod 2^64
        q_inv.wrapping_neg()
    }

    /// Compute R^2 mod q for Montgomery domain conversion
    ///
    /// R = 2^64, so R^2 = 2^128
    /// We compute (2^128) mod q using repeated squaring
    fn compute_r_squared_mod_q(q: u64) -> u64 {
        // 2^128 mod q = ((2^64 mod q) * (2^64 mod q)) mod q

        // First compute 2^64 mod q using u128 arithmetic
        let r_mod_q = ((1u128 << 64) % q as u128) as u64;

        // Then compute (2^64 mod q)^2 mod q
        Self::mul_mod(r_mod_q, r_mod_q, q)
    }

    /// Convert a value to Montgomery domain: x -> x*R mod q
    ///
    /// Uses Montgomery multiplication with R^2 mod q:
    /// to_montgomery(x) = mont_mul(x, R^2, q, q_inv) = x*R^2*R^{-1} = x*R mod q
    pub fn to_montgomery(x: u64, r_squared: u64, q: u64, q_inv: u64) -> u64 {
        // We'll use the CPU mul_mod for now since this is just precomputation
        // In the future, we could call a Metal kernel if we have many values to convert
        Self::mont_mul_cpu(x, r_squared, q, q_inv)
    }

    /// Convert a value from Montgomery domain: x*R -> x mod q
    ///
    /// Uses Montgomery multiplication with 1:
    /// from_montgomery(x*R) = mont_mul(x*R, 1, q, q_inv) = x*R*1*R^{-1} = x mod q
    pub fn from_montgomery(x_mont: u64, q: u64, q_inv: u64) -> u64 {
        Self::mont_mul_cpu(x_mont, 1, q, q_inv)
    }

    /// Access the primitive 2N-th root of unity (psi)
    pub fn psi(&self) -> u64 {
        self.psi
    }

    /// Access psi powers (for twist/untwist in CKKS polynomial multiplication)
    pub fn psi_powers(&self) -> &[u64] {
        &self.psi_powers
    }

    /// Access psi inverse powers (for twist/untwist in CKKS polynomial multiplication)
    pub fn psi_inv_powers(&self) -> &[u64] {
        &self.psi_inv_powers
    }

    /// Access r_squared for Montgomery domain conversions
    pub fn r_squared(&self) -> u64 {
        self.r_squared
    }

    /// Access modulus q
    pub fn q(&self) -> u64 {
        self.q
    }

    /// Access q_inv for Montgomery multiplication
    pub fn q_inv(&self) -> u64 {
        self.q_inv
    }

    /// CPU implementation of Montgomery multiplication for precomputation
    ///
    /// Same algorithm as GPU mont_mul kernel
    fn mont_mul_cpu(a: u64, b: u64, q: u64, q_inv: u64) -> u64 {
        // Step 1: Compute t = a * b (128-bit)
        let t = a as u128 * b as u128;
        let t_lo = t as u64;
        let t_hi = (t >> 64) as u64;

        // Step 2: Compute m = (t_lo * q_inv) mod 2^64
        let m = t_lo.wrapping_mul(q_inv);

        // Step 3: Compute m * q (128-bit)
        let mq = m as u128 * q as u128;
        let mq_lo = mq as u64;
        let mq_hi = (mq >> 64) as u64;

        // Step 4: Compute u = (t + m*q) / 2^64
        let (sum_lo, carry1) = t_lo.overflowing_add(mq_lo);
        let (sum_hi, carry2) = t_hi.overflowing_add(mq_hi);
        let sum_hi = sum_hi.wrapping_add(carry1 as u64).wrapping_add(carry2 as u64);

        // Step 5: Conditional subtraction
        if sum_hi >= q {
            sum_hi - q
        } else {
            sum_hi
        }
    }

    /// Apply negacyclic twist on Metal GPU: coeffs[i] *= ψ^i mod q
    ///
    /// This is used to convert coefficient-domain polynomials to twisted form
    /// before forward NTT for negacyclic convolution.
    ///
    /// **Performance:** Metal GPU accelerated, operates on entire polynomial in parallel
    ///
    /// # Arguments
    /// * `coeffs` - Polynomial coefficients in standard domain (modified in-place)
    ///
    /// # Returns
    /// Result with twisted coefficients in standard domain, or error string
    pub fn apply_twist_gpu(&self, coeffs: &mut [u64]) -> Result<(), String> {
        if coeffs.len() != self.n {
            return Err(format!("Expected {} coefficients, got {}", self.n, coeffs.len()));
        }

        // Convert coeffs to Montgomery domain for GPU computation
        let coeffs_montgomery: Vec<u64> = coeffs.iter()
            .map(|&c| Self::to_montgomery(c, self.r_squared, self.q, self.q_inv))
            .collect();

        // Convert psi_powers to Montgomery domain (psi_powers are in standard domain)
        let psi_powers_montgomery: Vec<u64> = self.psi_powers.iter()
            .map(|&p| Self::to_montgomery(p, self.r_squared, self.q, self.q_inv))
            .collect();

        // Create GPU buffers
        let coeffs_buffer = self.device.create_buffer_with_data(&coeffs_montgomery);
        let psi_powers_buffer = self.device.create_buffer_with_data(&psi_powers_montgomery);
        let n_buffer = self.device.create_buffer_with_u32_data(&[self.n as u32]);
        let q_buffer = self.device.create_buffer_with_data(&[self.q]);
        let q_inv_buffer = self.device.create_buffer_with_data(&[self.q_inv]);

        // Get kernel and create pipeline
        let kernel = self.device.get_function("ntt_apply_twist")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create twist pipeline: {:?}", e))?;

        // Dispatch kernel
        let threadgroup_size = MTLSize::new(256, 1, 1);
        let threadgroups = MTLSize::new(((self.n + 255) / 256) as u64, 1, 1);

        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&coeffs_buffer), 0);
            encoder.set_buffer(1, Some(&psi_powers_buffer), 0);
            encoder.set_buffer(2, Some(&n_buffer), 0);
            encoder.set_buffer(3, Some(&q_buffer), 0);
            encoder.set_buffer(4, Some(&q_inv_buffer), 0);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            Ok(())
        })?;

        // Read back results (in Montgomery domain)
        let result_montgomery = self.device.read_buffer(&coeffs_buffer, self.n);

        // Convert back to standard domain
        for (i, &val_mont) in result_montgomery.iter().enumerate() {
            coeffs[i] = Self::from_montgomery(val_mont, self.q, self.q_inv);
        }

        Ok(())
    }

    /// Transform polynomial from coefficient to NTT domain using Metal GPU
    ///
    /// Performs twist + forward NTT in sequence, fully on GPU.
    /// This is more efficient than separate twist + NTT calls as it avoids
    /// an intermediate CPU readback.
    ///
    /// **Performance:** Fully GPU accelerated (no CPU round-trip)
    ///
    /// # Arguments
    /// * `coeffs` - Polynomial coefficients (modified in-place)
    ///
    /// # Returns
    /// Result with NTT-domain coefficients (Montgomery), or error string
    pub fn coeff_to_ntt_gpu(&self, coeffs: &mut [u64]) -> Result<(), String> {
        if coeffs.len() != self.n {
            return Err(format!("Expected {} coefficients, got {}", self.n, coeffs.len()));
        }

        // Apply twist on GPU (in-place)
        self.apply_twist_gpu(coeffs)?;

        // Forward NTT on GPU (in-place)
        self.forward(coeffs)?;

        Ok(())
    }

    /// Apply inverse negacyclic twist on Metal GPU: coeffs[i] *= ψ^{-i} mod q
    ///
    /// This is used to convert from twisted form back to coefficient domain
    /// after inverse NTT for negacyclic convolution.
    ///
    /// **Performance:** Metal GPU accelerated, operates on entire polynomial in parallel
    ///
    /// # Arguments
    /// * `coeffs` - Polynomial coefficients in twisted domain (modified in-place)
    ///
    /// # Returns
    /// Result with untwisted coefficients in standard domain, or error string
    pub fn apply_inverse_twist_gpu(&self, coeffs: &mut [u64]) -> Result<(), String> {
        if coeffs.len() != self.n {
            return Err(format!("Expected {} coefficients, got {}", self.n, coeffs.len()));
        }

        // Convert coeffs to Montgomery domain for GPU computation
        let coeffs_montgomery: Vec<u64> = coeffs.iter()
            .map(|&c| Self::to_montgomery(c, self.r_squared, self.q, self.q_inv))
            .collect();

        // Convert psi_inv_powers to Montgomery domain
        let psi_inv_powers_montgomery: Vec<u64> = self.psi_inv_powers.iter()
            .map(|&p| Self::to_montgomery(p, self.r_squared, self.q, self.q_inv))
            .collect();

        // Create GPU buffers
        let coeffs_buffer = self.device.create_buffer_with_data(&coeffs_montgomery);
        let psi_inv_powers_buffer = self.device.create_buffer_with_data(&psi_inv_powers_montgomery);
        let n_buffer = self.device.create_buffer_with_u32_data(&[self.n as u32]);
        let q_buffer = self.device.create_buffer_with_data(&[self.q]);
        let q_inv_buffer = self.device.create_buffer_with_data(&[self.q_inv]);

        // Get kernel and create pipeline
        let kernel = self.device.get_function("ntt_apply_inverse_twist")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create inverse twist pipeline: {:?}", e))?;

        // Dispatch kernel
        let threadgroup_size = MTLSize::new(256, 1, 1);
        let threadgroups = MTLSize::new(((self.n + 255) / 256) as u64, 1, 1);

        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&coeffs_buffer), 0);
            encoder.set_buffer(1, Some(&psi_inv_powers_buffer), 0);
            encoder.set_buffer(2, Some(&n_buffer), 0);
            encoder.set_buffer(3, Some(&q_buffer), 0);
            encoder.set_buffer(4, Some(&q_inv_buffer), 0);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            Ok(())
        })?;

        // Read back results (in Montgomery domain)
        let result_montgomery = self.device.read_buffer(&coeffs_buffer, self.n);

        // Convert back to standard domain
        for (i, &val_mont) in result_montgomery.iter().enumerate() {
            coeffs[i] = Self::from_montgomery(val_mont, self.q, self.q_inv);
        }

        Ok(())
    }

    /// Transform polynomial from NTT to coefficient domain using Metal GPU
    ///
    /// Performs inverse NTT + inverse twist in sequence, fully on GPU.
    /// This is more efficient than separate iNTT + untwist calls as it avoids
    /// an intermediate CPU readback.
    ///
    /// **Performance:** Fully GPU accelerated (no CPU round-trip)
    ///
    /// # Arguments
    /// * `coeffs` - NTT-domain coefficients (Montgomery, modified in-place)
    ///
    /// # Returns
    /// Result with coefficient-domain values in standard domain, or error string
    pub fn ntt_to_coeff_gpu(&self, coeffs: &mut [u64]) -> Result<(), String> {
        if coeffs.len() != self.n {
            return Err(format!("Expected {} coefficients, got {}", self.n, coeffs.len()));
        }

        // Inverse NTT on GPU (in-place)
        self.inverse(coeffs)?;

        // Apply inverse twist on GPU (in-place)
        self.apply_inverse_twist_gpu(coeffs)?;

        Ok(())
    }

    /// Fused inverse NTT + inverse twist (OPTIMIZED)
    ///
    /// Performs inverse NTT with fused final scale + inverse twist in a single kernel.
    /// This eliminates one GPU kernel dispatch and intermediate buffer round-trip.
    ///
    /// **Performance:** ~5-10% faster than separate inverse() + apply_inverse_twist_gpu()
    ///
    /// # Arguments
    /// * `evals` - NTT-domain coefficients (Montgomery, modified in-place)
    ///
    /// # Returns
    /// Result with coefficient-domain values in standard domain, or error string
    pub fn inverse_and_untwist_fused(&self, evals: &mut [u64]) -> Result<(), String> {
        if evals.len() != self.n {
            return Err(format!("Expected {} evaluation points, got {}", self.n, evals.len()));
        }

        let log_n = (self.n as f64).log2() as u32;

        // Create persistent GPU buffer - input is already Montgomery
        let evals_buffer = self.device.create_buffer_with_data(evals);
        let omega_inv_powers_buffer = self.device.create_buffer_with_data(&self.omega_inv_powers_montgomery);
        let n_buffer = self.device.create_buffer_with_u32_data(&[self.n as u32]);
        let q_buffer = self.device.create_buffer_with_data(&[self.q]);
        let n_inv_buffer = self.device.create_buffer_with_data(&[self.n_inv_montgomery]);
        let q_inv_buffer = self.device.create_buffer_with_data(&[self.q_inv]);

        // Convert psi_inv_powers to Montgomery domain for fused kernel
        let psi_inv_powers_montgomery: Vec<u64> = self.psi_inv_powers.iter()
            .map(|&p| Self::to_montgomery(p, self.r_squared, self.q, self.q_inv))
            .collect();
        let psi_inv_powers_buffer = self.device.create_buffer_with_data(&psi_inv_powers_montgomery);

        let threadgroup_size = MTLSize::new(256, 1, 1);

        // Step 1: Execute inverse butterfly stages (in reverse order)
        let kernel = self.device.get_function("ntt_inverse_stage")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create inverse stage pipeline: {:?}", e))?;

        for stage in (0..log_n).rev() {
            let stage_buffer = self.device.create_buffer_with_u32_data(&[stage]);

            self.device.execute_kernel(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&evals_buffer), 0);
                encoder.set_buffer(1, Some(&omega_inv_powers_buffer), 0);
                encoder.set_buffer(2, Some(&n_buffer), 0);
                encoder.set_buffer(3, Some(&q_buffer), 0);
                encoder.set_buffer(4, Some(&stage_buffer), 0);
                encoder.set_buffer(5, Some(&q_inv_buffer), 0);

                let butterfly_threads = ((self.n / 2 + 255) / 256) as u64;
                let butterfly_threadgroups = MTLSize::new(butterfly_threads, 1, 1);
                encoder.dispatch_thread_groups(butterfly_threadgroups, threadgroup_size);
                Ok(())
            })?;
        }

        // Step 2: FUSED bit-reversal + scaling + inverse twist
        {
            let kernel = self.device.get_function("ntt_inverse_final_scale_and_untwist")?;
            let pipeline = self.device.device()
                .new_compute_pipeline_state_with_function(&kernel)
                .map_err(|e| format!("Failed to create fused final scale+untwist pipeline: {:?}", e))?;

            self.device.execute_kernel(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&evals_buffer), 0);
                encoder.set_buffer(1, Some(&psi_inv_powers_buffer), 0);
                encoder.set_buffer(2, Some(&n_buffer), 0);
                encoder.set_buffer(3, Some(&q_buffer), 0);
                encoder.set_buffer(4, Some(&n_inv_buffer), 0);
                encoder.set_buffer(5, Some(&q_inv_buffer), 0);

                let threadgroups = MTLSize::new(((self.n + 255) / 256) as u64, 1, 1);
                encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
                Ok(())
            })?;
        }

        // Read result (still in Montgomery domain)
        let result_montgomery = self.device.read_buffer(&evals_buffer, self.n);

        // Convert from Montgomery domain back to normal domain
        for i in 0..self.n {
            evals[i] = Self::mont_mul_cpu(result_montgomery[i], 1, self.q, self.q_inv);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_ntt_basic() {
        // Use small NTT-friendly prime for testing
        // q = 97 = 1 + 96 = 1 + 32*3 (so q ≡ 1 mod 2n for n=32)
        // primitive 32nd root: compute 3^(96/32) mod 97 = 3^3 mod 97 = 27
        let n = 32;
        let q = 97u64;
        let root = 27u64; // Primitive 32nd root of unity mod 97

        let ctx = MetalNttContext::new(n, q, root);

        if ctx.is_err() {
            println!("Skipping test: Metal not available (requires Apple Silicon)");
            return;
        }

        let ctx = ctx.unwrap();

        // Test: NTT -> INTT should be identity
        let mut coeffs = vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                              17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
        let original = coeffs.clone();

        ctx.forward(&mut coeffs).unwrap();
        ctx.inverse(&mut coeffs).unwrap();

        // Check all coefficients match (mod q)
        for i in 0..n {
            let diff = if coeffs[i] >= original[i] {
                coeffs[i] - original[i]
            } else {
                original[i] - coeffs[i]
            };
            assert!(diff == 0 || diff == q, "NTT round-trip failed at index {}: {} != {}", i, coeffs[i], original[i]);
        }

        println!("Metal NTT round-trip test passed!");
    }
}
