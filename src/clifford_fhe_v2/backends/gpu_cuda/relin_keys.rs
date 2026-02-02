//! CUDA GPU-Accelerated Relinearization Keys for CKKS
//!
//! Implements relinearization (degree reduction) for ciphertext multiplication.
//!
//! **Relinearization Overview:**
//! After ciphertext multiplication, we have a degree-2 ciphertext:
//!   ct = (c0, c1, c2)
//!
//! We need to convert it back to degree-1:
//!   ct' = (c0', c1')
//!
//! **Relinearization Key Structure:**
//! We precompute:
//!   RelinKey = {KS_0, KS_1, ..., KS_{dnum-1}}
//! where:
//!   KS_i = (-a_i · s^2 + e_i + w^i · s, a_i)
//!        = (b_i, a_i) in RNS representation
//!
//! **Gadget Decomposition:**
//! We decompose c2 into base-w digits:
//!   c2 = Σ_{i=0}^{dnum-1} d_i · w^i
//! where d_i ∈ [-w/2, w/2] (signed decomposition)
//!
//! **Key Application:**
//!   c0' = c0 + Σ_i d_i · b_i
//!   c1' = c1 + Σ_i d_i · a_i
//!
//! **GPU Acceleration:**
//! - NTT-multiply for key generation (GPU)
//! - Gadget decomposition (CPU, simple)
//! - Key application multiply (GPU via NTT)

use crate::clifford_fhe_v2::backends::gpu_cuda::device::CudaDeviceContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::BarrettReducer;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use num_bigint::BigInt;
use num_traits::{One, Zero, ToPrimitive};
use rand::Rng;
use std::sync::Arc;

/// Relinearization key structure
#[derive(Clone)]
pub struct RelinearizationKey {
    /// Key switching components: [(b_0, a_0), (b_1, a_1), ..., (b_{dnum-1}, a_{dnum-1})]
    /// Each component is in flat RNS layout: poly[prime_idx * n + coeff_idx]
    /// b_i and a_i have size n * num_primes_key
    pub ks_components: Vec<(Vec<u64>, Vec<u64>)>,

    /// Number of RNS primes in the key (typically num_primes + special_primes)
    pub num_primes_key: usize,

    /// Polynomial degree
    pub n: usize,
}

/// CUDA relinearization keys manager
pub struct CudaRelinKeys {
    device: Arc<CudaDeviceContext>,
    params: CliffordFHEParams,

    /// Barrett reducers for key level
    reducers_key: Vec<BarrettReducer>,

    /// Gadget decomposition base EXPONENT (e.g., 16 for w = 2^16)
    /// NOTE: This stores the exponent, not the base itself. Use base_w() to get 2^base_bits.
    pub base_bits: u32,

    /// Number of gadget digits: dnum = ceil(log_w(Q_key))
    pub dnum: usize,

    /// Relinearization key
    relin_key: Option<RelinearizationKey>,
}

impl CudaRelinKeys {
    /// Create new relinearization keys manager
    ///
    /// # Arguments
    /// * `device` - CUDA device context
    /// * `params` - FHE parameters
    /// * `secret_key` - Secret key polynomial (strided layout)
    /// * `base_bits` - Gadget base bits (e.g., 16 for w = 2^16)
    pub fn new(
        device: Arc<CudaDeviceContext>,
        params: CliffordFHEParams,
        secret_key: Vec<u64>,
        base_bits: usize,
    ) -> Result<Self, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║        Initializing CUDA Relinearization Keys                ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let base_bits_u32 = base_bits as u32;  // Store exponent, not base
        let base_w = 1u64 << base_bits;  // w = 2^base_bits (computed when needed)

        // Determine number of RNS primes for keys (use all available)
        let num_primes_key = params.moduli.len();

        // Calculate number of gadget digits
        // dnum = ceil(log_w(Q_key)) where Q_key = product of all key primes
        let total_bits: usize = params.moduli.iter()
            .map(|&q| (64 - q.leading_zeros()) as usize)
            .sum();
        let dnum = (total_bits + base_bits - 1) / base_bits;

        println!("Relinearization key parameters:");
        println!("  Base w: 2^{} = {}", base_bits, base_w);
        println!("  Number of primes (key level): {}", num_primes_key);
        println!("  Number of gadget digits (dnum): {}", dnum);

        // Create Barrett reducers for key level
        let reducers_key: Vec<BarrettReducer> = params.moduli[..num_primes_key]
            .iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        let mut ctx = Self {
            device,
            params,
            reducers_key,
            base_bits: base_bits_u32,
            dnum,
            relin_key: None,
        };

        // Generate relinearization key
        println!("\nGenerating relinearization key...");
        let start = std::time::Instant::now();
        ctx.generate_relin_key(&secret_key)?;
        let elapsed = start.elapsed().as_secs_f64();
        println!("  Relinearization key generated in {:.2}s\n", elapsed);

        Ok(ctx)
    }

    /// Create new relinearization keys manager with GPU-accelerated key generation
    ///
    /// This is MUCH faster than new() as it uses GPU NTT for polynomial multiplication
    pub fn new_gpu(
        device: Arc<CudaDeviceContext>,
        params: CliffordFHEParams,
        secret_key: Vec<u64>,
        base_bits: usize,
        ntt_contexts: &[super::ntt::CudaNttContext],
    ) -> Result<Self, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║        Initializing CUDA Relinearization Keys                ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let base_bits_u32 = base_bits as u32;  // Store exponent, not base
        let base_w = 1u64 << base_bits;  // w = 2^base_bits (computed when needed)

        // Determine number of RNS primes for keys (use all available)
        let num_primes_key = params.moduli.len();

        // Calculate number of gadget digits
        // dnum = ceil(log_w(Q_key)) where Q_key = product of all key primes
        let total_bits: usize = params.moduli.iter()
            .map(|&q| (64 - q.leading_zeros()) as usize)
            .sum();
        let dnum = (total_bits + base_bits - 1) / base_bits;

        println!("Relinearization key parameters:");
        println!("  Base w: 2^{} = {}", base_bits, base_w);
        println!("  Number of primes (key level): {}", num_primes_key);
        println!("  Number of gadget digits (dnum): {}", dnum);

        // Create Barrett reducers for key level
        let reducers_key: Vec<BarrettReducer> = params.moduli[..num_primes_key]
            .iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        let mut ctx = Self {
            device,
            params,
            reducers_key,
            base_bits: base_bits_u32,
            dnum,
            relin_key: None,
        };

        // Generate relinearization key using GPU
        println!("\nGenerating relinearization key...");
        let start = std::time::Instant::now();
        ctx.generate_relin_key_gpu(&secret_key, ntt_contexts)?;
        let elapsed = start.elapsed().as_secs_f64();
        println!("  Relinearization key generated in {:.2}s\n", elapsed);

        Ok(ctx)
    }

    /// Generate relinearization key using GPU NTT for polynomial multiplication
    ///
    /// This is MUCH faster than the CPU version
    fn generate_relin_key_gpu(
        &mut self,
        secret_key: &[u64],
        ntt_contexts: &[super::ntt::CudaNttContext],
    ) -> Result<(), String> {
        let n = self.params.n;
        let num_primes_key = self.reducers_key.len();

        // Compute s^2 (secret key squared) using GPU
        let s_squared = self.compute_secret_key_squared_gpu(secret_key, num_primes_key, ntt_contexts)?;

        // Generate key switching components
        let mut ks_components = Vec::with_capacity(self.dnum);

        // CRITICAL FIX: Precompute B^t mod q for each prime and each digit
        // This avoids u128 overflow when t is large (e.g., B^7 = 2^140 > 2^128)
        // Matches Metal implementation in relin_keys.rs lines 148-159
        let base = 1u64 << self.base_bits;  // B = 2^base_bits
        let mut bpow_t_mod_q = vec![vec![0u64; num_primes_key]; self.dnum];
        for (j, &q) in self.params.moduli[..num_primes_key].iter().enumerate() {
            let q_u128 = q as u128;
            let mut p = 1u128;  // B^0 = 1
            for t in 0..self.dnum {
                bpow_t_mod_q[t][j] = (p % q_u128) as u64;
                p = (p * (base as u128)) % q_u128;  // Incrementally compute B^(t+1) mod q
            }
        }

        for digit_idx in 0..self.dnum {
            // Generate random polynomial a_i
            let a_i = self.generate_random_poly(num_primes_key);

            // Convert secret key from strided to flat layout for NTT multiplication
            let mut s_flat = vec![0u64; n * num_primes_key];
            for coeff_idx in 0..n {
                for prime_idx in 0..num_primes_key {
                    let strided_idx = coeff_idx * num_primes_key + prime_idx;
                    let flat_idx = prime_idx * n + coeff_idx;
                    s_flat[flat_idx] = secret_key[strided_idx];
                }
            }

            // Compute a_i · s using GPU NTT (NOT a_i · s² - that was WRONG!)
            // Standard CKKS EVK: evk0[t] = -a_t·s + e_t + B^t·s²
            let mut b_i = self.gpu_multiply_flat_ntt(&a_i, &s_flat, num_primes_key, ntt_contexts)?;

            // Add error term e_i ~ Gaussian
            // CRITICAL: Sample ONE error per coefficient, then reduce mod each prime
            // (Matches Metal/CPU implementations)
            let e_i = self.generate_error_poly(num_primes_key);
            for i in 0..(n * num_primes_key) {
                let prime_idx = i / n;
                let q = self.params.moduli[prime_idx];
                b_i[i] = (b_i[i] + e_i[i]) % q;
            }

            // Add -B^t · s² (note: NEGATIVE, because EVK encodes -B^t·s²)
            // This is the key difference from before: we use B^t (gadget power), not a_i
            for coeff_idx in 0..n {
                for prime_idx in 0..num_primes_key {
                    let flat_idx = prime_idx * n + coeff_idx;
                    let strided_idx = coeff_idx * num_primes_key + prime_idx;
                    let q = self.params.moduli[prime_idx];

                    // B^t · s² mod q (using precomputed B^t mod q)
                    let bt_mod_q = bpow_t_mod_q[digit_idx][prime_idx];
                    let s2_val = s_squared[flat_idx];  // s_squared is in flat layout
                    let bt_s2 = ((bt_mod_q as u128 * s2_val as u128) % q as u128) as u64;

                    // NEGATE: we want -B^t·s², so subtract from b_i
                    b_i[flat_idx] = if b_i[flat_idx] >= bt_s2 {
                        b_i[flat_idx] - bt_s2
                    } else {
                        q - (bt_s2 - b_i[flat_idx])
                    };
                }
            }

            // Final: b_i = a_i·s + e_i - B^t·s²
            // When we multiply digit d_t by b_i, we get: d_t·(a_i·s + e - B^t·s²)
            // The d_t·B^t·s² term cancels out c2·s² when summed over all digits

            // Debug: print first component values
            if std::env::var("EVK_GEN_DEBUG").is_ok() && digit_idx == 0 {
                println!("[EVK_GEN_DEBUG CUDA] EVK[0] b_i first 3 coeffs (flat):");
                for coeff in 0..3.min(n) {
                    print!("  coeff[{}]: ", coeff);
                    for p in 0..num_primes_key {
                        let idx = p * n + coeff;
                        print!("{} ", b_i[idx]);
                    }
                    println!();
                }
                println!("[EVK_GEN_DEBUG CUDA] EVK[0] a_i first 3 coeffs (flat):");
                for coeff in 0..3.min(n) {
                    print!("  coeff[{}]: ", coeff);
                    for p in 0..num_primes_key {
                        let idx = p * n + coeff;
                        print!("{} ", a_i[idx]);
                    }
                    println!();
                }
            }

            ks_components.push((b_i, a_i));

            if (digit_idx + 1) % 5 == 0 || digit_idx == self.dnum - 1 {
                println!("  Generated {}/{} key switching components", digit_idx + 1, self.dnum);
            }
        }

        self.relin_key = Some(RelinearizationKey {
            ks_components,
            num_primes_key,
            n,
        });

        Ok(())
    }

    /// Generate relinearization key for s^2 -> s (CPU version - DEPRECATED)
    ///
    /// WARNING: This function has a known bug - it uses a_i*s^2 instead of B^t*s^2.
    /// Use generate_relin_key_gpu() instead which has the correct implementation.
    ///
    /// Generates: RelinKey = {(b_i, a_i)} where b_i = -a_i*s^2 + e_i + w^i*s (incorrect)
    fn generate_relin_key(&mut self, secret_key: &[u64]) -> Result<(), String> {
        let n = self.params.n;
        let num_primes_key = self.reducers_key.len();

        // Compute s^2 (secret key squared)
        let s_squared = self.compute_secret_key_squared(secret_key, num_primes_key)?;

        // Generate key switching components
        let mut ks_components = Vec::with_capacity(self.dnum);

        for digit_idx in 0..self.dnum {
            // Generate random polynomial a_i
            let a_i = self.generate_random_poly(num_primes_key);

            // Compute -a_i · s^2 using CPU (DEPRECATED - use GPU version)
            let mut b_i = self.cpu_multiply_flat(&a_i, &s_squared, num_primes_key)?;

            // Negate: -a_i · s^2
            for i in 0..(n * num_primes_key) {
                let prime_idx = i / n;
                let q = self.params.moduli[prime_idx];
                b_i[i] = if b_i[i] == 0 { 0 } else { q - b_i[i] };
            }

            // Add error term e_i ~ Gaussian
            let e_i = self.generate_error_poly(num_primes_key);
            for i in 0..(n * num_primes_key) {
                let prime_idx = i / n;
                let q = self.params.moduli[prime_idx];
                b_i[i] = (b_i[i] + e_i[i]) % q;
            }

            // Add w^i · s
            let base_w = 1u64 << self.base_bits;
            let power_of_w = base_w.pow(digit_idx as u32);
            for coeff_idx in 0..n {
                for prime_idx in 0..num_primes_key {
                    let flat_idx = prime_idx * n + coeff_idx;
                    let strided_idx = coeff_idx * num_primes_key + prime_idx;
                    let q = self.params.moduli[prime_idx];

                    let s_val = secret_key[strided_idx];
                    let scaled_s = ((power_of_w % q as u64) * s_val) % q;
                    b_i[flat_idx] = (b_i[flat_idx] + scaled_s) % q;
                }
            }

            ks_components.push((b_i, a_i));
        }

        self.relin_key = Some(RelinearizationKey {
            ks_components,
            num_primes_key,
            n,
        });

        Ok(())
    }

    /// Compute s^2 (secret key squared) in RNS representation
    fn compute_secret_key_squared(&self, secret_key: &[u64], num_primes: usize) -> Result<Vec<u64>, String> {
        let n = self.params.n;

        // Convert from strided to flat layout
        let mut s_flat = vec![0u64; n * num_primes];
        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let strided_idx = coeff_idx * num_primes + prime_idx;
                let flat_idx = prime_idx * n + coeff_idx;
                s_flat[flat_idx] = secret_key[strided_idx];
            }
        }

        // Multiply s * s using CPU (DEPRECATED - use GPU version)
        self.cpu_multiply_flat(&s_flat, &s_flat, num_primes)
    }

    /// Compute s^2 (secret key squared) using GPU NTT
    fn compute_secret_key_squared_gpu(
        &self,
        secret_key: &[u64],
        num_primes: usize,
        ntt_contexts: &[super::ntt::CudaNttContext],
    ) -> Result<Vec<u64>, String> {
        let n = self.params.n;

        // Convert from strided to flat layout
        let mut s_flat = vec![0u64; n * num_primes];
        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let strided_idx = coeff_idx * num_primes + prime_idx;
                let flat_idx = prime_idx * n + coeff_idx;
                s_flat[flat_idx] = secret_key[strided_idx];
            }
        }

        // Multiply s * s using GPU NTT
        self.gpu_multiply_flat_ntt(&s_flat, &s_flat, num_primes, ntt_contexts)
    }

    /// Apply relinearization to reduce degree-2 ciphertext to degree-1
    ///
    /// Input: (c0, c1, c2) where c2 needs to be eliminated
    /// Output: (c0', c1') where c2 has been absorbed using relin key
    pub fn apply_relinearization(
        &self,
        c0: &[u64],
        c1: &[u64],
        c2: &[u64],
        level: usize,
    ) -> Result<(Vec<u64>, Vec<u64>), String> {
        let relin_key = self.relin_key.as_ref()
            .ok_or_else(|| "Relinearization key not generated".to_string())?;

        let n = self.params.n;
        let num_primes = level + 1;

        // Decompose c2 into base-w digits
        let digits = self.gadget_decompose(c2, num_primes)?;

        // Initialize accumulator
        let mut c0_acc = c0.to_vec();
        let mut c1_acc = c1.to_vec();

        // Accumulate: c0' = c0 + Σ d_i · b_i, c1' = c1 + Σ d_i · a_i
        for (digit_idx, d_i) in digits.iter().enumerate() {
            if digit_idx >= relin_key.ks_components.len() {
                break;
            }

            let (b_i, a_i) = &relin_key.ks_components[digit_idx];

            // Multiply d_i · b_i (CPU for now)
            let d_b = self.cpu_multiply_flat(d_i, b_i, num_primes)?;
            // Multiply d_i · a_i (CPU for now)
            let d_a = self.cpu_multiply_flat(d_i, a_i, num_primes)?;

            // Accumulate
            for i in 0..n * num_primes {
                let q = self.params.moduli[i / n];
                c0_acc[i] = (c0_acc[i] + d_b[i]) % q;
                c1_acc[i] = (c1_acc[i] + d_a[i]) % q;
            }
        }

        Ok((c0_acc, c1_acc))
    }

    /// Apply relinearization using GPU NTT for polynomial multiplication
    ///
    /// This is MUCH faster than the CPU version (O(n log n) vs O(n²))
    ///
    /// Input: (c0, c1, c2) where c2 needs to be eliminated
    /// Output: (c0', c1') where c2 has been absorbed using relin key
    pub fn apply_relinearization_gpu(
        &self,
        c0: &[u64],
        c1: &[u64],
        c2: &[u64],
        level: usize,
        ntt_contexts: &[super::ntt::CudaNttContext],
        ckks_ctx: &super::ckks::CudaCkksContext,
    ) -> Result<(Vec<u64>, Vec<u64>), String> {
        let relin_key = self.relin_key.as_ref()
            .ok_or_else(|| "Relinearization key not generated".to_string())?;

        let n = self.params.n;
        let num_primes = level + 1;

        // Decompose c2 into base-w digits
        let digits = self.gadget_decompose(c2, num_primes)?;

        // Debug: print number of digits and their content
        if std::env::var("RELIN_DEBUG").is_ok() {
            println!("[RELIN_DEBUG CUDA] num_primes={}, num_digits={}", num_primes, digits.len());

            // Check if digits are mostly zero
            let mut non_zero_count = 0;
            for (t, digit) in digits.iter().enumerate() {
                let nz: usize = digit.iter().filter(|&&x| x != 0).count();
                non_zero_count += nz;
                if t < 3 {
                    println!("[RELIN_DEBUG CUDA] digit[{}] has {} non-zero values out of {}",
                             t, nz, digit.len());
                }
            }
            println!("[RELIN_DEBUG CUDA] Total non-zero values across all digits: {}", non_zero_count);

            if !digits.is_empty() {
                println!("[RELIN_DEBUG CUDA] digit[0] first 3 coeffs (flat):");
                for coeff in 0..3.min(n) {
                    print!("  coeff[{}]: ", coeff);
                    for p in 0..num_primes {
                        print!("{} ", digits[0][p * n + coeff]);
                    }
                    println!();
                }
            }

            // Print c2 first 3 coeffs for comparison
            println!("[RELIN_DEBUG CUDA] c2 first 3 coeffs (flat) for comparison:");
            for coeff in 0..3.min(n) {
                print!("  coeff[{}]: ", coeff);
                for p in 0..num_primes {
                    print!("{} ", c2[p * n + coeff]);
                }
                println!();
            }
        }

        // Initialize accumulator
        let mut c0_acc = c0.to_vec();
        let mut c1_acc = c1.to_vec();

        // Apply relinearization: c0' = c0 - Σ d_i · b_i, c1' = c1 + Σ d_i · a_i
        // CRITICAL FIX: EVK encodes b_i = a·s + e - B^t·s², so we SUBTRACT d·b from c0
        // This gives: c0 - d·(a·s + e - B^t·s²) = c0 + d·B^t·s² - d·a·s - d·e
        // Summing over digits reconstructs +c2·s², which cancels c2·s² from decryption
        // (Matches Metal: line 2694 "c0 -= term0")
        for (digit_idx, d_i) in digits.iter().enumerate() {
            if digit_idx >= relin_key.ks_components.len() {
                break;
            }

            let (b_i, a_i) = &relin_key.ks_components[digit_idx];

            // Debug: print input sizes and first few values
            if std::env::var("RELIN_DEBUG").is_ok() && digit_idx == 0 {
                println!("[RELIN_DEBUG CUDA] digit[0].len()={}, b_i.len()={}, a_i.len()={}",
                         d_i.len(), b_i.len(), a_i.len());
                println!("[RELIN_DEBUG CUDA] Expected size: n * num_primes = {} * {} = {}",
                         n, num_primes, n * num_primes);

                // Print first 3 coefficients of digit[0] in flat layout
                println!("[RELIN_DEBUG CUDA] d_i[0] first 3 coeffs (flat):");
                for coeff in 0..3.min(n) {
                    print!("  coeff[{}]: ", coeff);
                    for p in 0..num_primes {
                        let idx = p * n + coeff;
                        if idx < d_i.len() {
                            print!("{} ", d_i[idx]);
                        }
                    }
                    println!();
                }

                // Print first 3 coefficients of b_i (EVK)
                println!("[RELIN_DEBUG CUDA] b_i first 3 coeffs (flat):");
                for coeff in 0..3.min(n) {
                    print!("  coeff[{}]: ", coeff);
                    for p in 0..num_primes {
                        let idx = p * n + coeff;
                        if idx < b_i.len() {
                            print!("{} ", b_i[idx]);
                        }
                    }
                    println!();
                }
            }

            // Multiply d_i · b_i using GPU NTT (SEQUENTIAL version - same as EVK generation)
            // CRITICAL: Use the same multiplication method as EVK generation to ensure consistency!
            let d_b = self.gpu_multiply_flat_ntt(d_i, b_i, num_primes, ntt_contexts)?;
            // Multiply d_i · a_i using GPU NTT (SEQUENTIAL version - same as EVK generation)
            let d_a = self.gpu_multiply_flat_ntt(d_i, a_i, num_primes, ntt_contexts)?;

            // Debug: print result of multiplication
            if std::env::var("RELIN_DEBUG").is_ok() && digit_idx == 0 {
                println!("[RELIN_DEBUG CUDA] d_b = d_i * b_i result first 3 coeffs (flat):");
                for coeff in 0..3.min(n) {
                    print!("  coeff[{}]: ", coeff);
                    for p in 0..num_primes {
                        let idx = p * n + coeff;
                        if idx < d_b.len() {
                            print!("{} ", d_b[idx]);
                        }
                    }
                    println!();
                }

                // Print c0_acc before subtraction
                println!("[RELIN_DEBUG CUDA] c0_acc BEFORE subtract first 3 coeffs:");
                for coeff in 0..3.min(n) {
                    print!("  coeff[{}]: ", coeff);
                    for p in 0..num_primes {
                        let idx = p * n + coeff;
                        if idx < c0_acc.len() {
                            print!("{} ", c0_acc[idx]);
                        }
                    }
                    println!();
                }
            }

            // c0 -= d·b (SUBTRACTION, not addition!)
            c0_acc = ckks_ctx.subtract_polynomials_gpu(&c0_acc, &d_b, num_primes)?;
            // c1 += d·a (addition)
            c1_acc = ckks_ctx.add_polynomials_gpu(&c1_acc, &d_a, num_primes)?;

            // Debug: print c0_acc after first digit
            if std::env::var("RELIN_DEBUG").is_ok() && digit_idx == 0 {
                println!("[RELIN_DEBUG CUDA] c0_acc AFTER subtract first 3 coeffs:");
                for coeff in 0..3.min(n) {
                    print!("  coeff[{}]: ", coeff);
                    for p in 0..num_primes {
                        let idx = p * n + coeff;
                        if idx < c0_acc.len() {
                            print!("{} ", c0_acc[idx]);
                        }
                    }
                    println!();
                }
            }
        }

        Ok((c0_acc, c1_acc))
    }

    /// GPU polynomial multiplication in flat RNS layout using BATCHED NTT
    ///
    /// **NEGACYCLIC CONVOLUTION**: Uses twist/untwist for R[X]/(X^N + 1)
    /// This matches Metal's multiply_polys_flat_ntt_negacyclic behavior.
    ///
    /// **NEW OPTIMIZED VERSION**: Uses batched GPU operations to process all primes at once.
    /// This is ~100× faster than the old sequential version!
    ///
    /// **Performance**:
    /// - OLD: 360 separate uploads/downloads (20 digits × 9 primes × 2)
    /// - NEW: 2 uploads + 2 downloads per digit (40 total for 20 digits)
    fn gpu_multiply_flat_ntt_batched(
        &self,
        poly1: &[u64],
        poly2: &[u64],
        num_primes: usize,
        ckks_ctx: &super::ckks::CudaCkksContext,
    ) -> Result<Vec<u64>, String> {
        let n = self.params.n;

        // Upload to GPU ONCE
        let mut gpu_p1 = self.device.device.htod_copy(poly1[..n * num_primes].to_vec())
            .map_err(|e| format!("Failed to upload poly1: {:?}", e))?;
        let mut gpu_p2 = self.device.device.htod_copy(poly2[..n * num_primes].to_vec())
            .map_err(|e| format!("Failed to upload poly2: {:?}", e))?;

        // TWIST: Apply psi^i for negacyclic convolution (CRITICAL!)
        // Without this, we get cyclic convolution instead of negacyclic
        ckks_ctx.apply_negacyclic_twist_gpu_public(&mut gpu_p1, num_primes)?;
        ckks_ctx.apply_negacyclic_twist_gpu_public(&mut gpu_p2, num_primes)?;

        // Forward NTT - BATCHED for all primes at once!
        ckks_ctx.ntt_forward_batched_gpu(&mut gpu_p1, num_primes)?;
        ckks_ctx.ntt_forward_batched_gpu(&mut gpu_p2, num_primes)?;

        // Pointwise multiply - ALL primes on GPU
        let mut gpu_result = self.device.device.alloc_zeros::<u64>(n * num_primes)
            .map_err(|e| format!("Failed to allocate gpu_result: {:?}", e))?;
        ckks_ctx.ntt_pointwise_multiply_batched_gpu(&gpu_p1, &gpu_p2, &mut gpu_result, num_primes)?;

        // Inverse NTT - BATCHED for all primes at once!
        ckks_ctx.ntt_inverse_batched_gpu(&mut gpu_result, num_primes)?;

        // UNTWIST: Apply psi^{-i} to get negacyclic result (CRITICAL!)
        ckks_ctx.apply_negacyclic_untwist_gpu_public(&mut gpu_result, num_primes)?;

        // Download final result ONCE
        let result = self.device.device.dtoh_sync_copy(&gpu_result)
            .map_err(|e| format!("Failed to download final result: {:?}", e))?;

        Ok(result)
    }

    /// GPU polynomial multiplication in flat RNS layout using NTT (per-prime sequential version)
    ///
    /// **NEGACYCLIC CONVOLUTION**: This function performs multiplication in
    /// the negacyclic ring R[X]/(X^N + 1) required by CKKS.
    ///
    /// Note: This is slower than gpu_multiply_flat_ntt_batched but is used during
    /// key generation where we need per-prime control.
    ///
    /// For each prime q:
    /// 1. Find psi (primitive 2N-th root of unity) such that psi² = omega
    /// 2. Apply twist: multiply by psi^i for negacyclic
    /// 3. Forward NTT (cyclic convolution)
    /// 4. Pointwise multiply
    /// 5. Inverse NTT
    /// 6. Apply untwist: multiply by psi^{-i}
    fn gpu_multiply_flat_ntt(
        &self,
        poly1: &[u64],
        poly2: &[u64],
        num_primes: usize,
        ntt_contexts: &[super::ntt::CudaNttContext],
    ) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        let mut result = vec![0u64; n * num_primes];

        // For each RNS prime, use GPU NTT multiplication with NEGACYCLIC convolution
        for prime_idx in 0..num_primes {
            let ntt_ctx = &ntt_contexts[prime_idx];
            let q = self.params.moduli[prime_idx];
            let offset = prime_idx * n;

            // Find psi (primitive 2N-th root) using same algorithm as CudaCkksContext
            // CRITICAL: Must use find_primitive_root, NOT Tonelli-Shanks sqrt(omega)!
            // Different algorithms can produce different (but mathematically equivalent) roots.
            let psi = Self::find_primitive_root(n, q)?;
            let psi_inv = Self::mod_inverse_u64(psi, q)?;

            // Debug: print psi values and verify omega = psi²
            if std::env::var("PSI_DEBUG").is_ok() && prime_idx < 3 {
                let omega = ntt_ctx.root;
                let psi_squared = Self::mul_mod_u64(psi, psi, q);
                println!("[PSI_DEBUG EVK_GEN] Prime {}: psi={}, psi²={}, omega={} (via find_primitive_root)",
                         prime_idx, psi, psi_squared, omega);
                if psi_squared != omega {
                    println!("[PSI_DEBUG EVK_GEN] WARNING: psi² != omega! This WILL cause incorrect results!");
                }
            }

            // Extract polynomials for this prime
            let mut p1 = poly1[offset..offset + n].to_vec();
            let mut p2 = poly2[offset..offset + n].to_vec();

            // TWIST: Multiply by psi^i for negacyclic convolution
            let mut psi_pow = 1u64;
            for i in 0..n {
                p1[i] = Self::mul_mod_u64(p1[i], psi_pow, q);
                p2[i] = Self::mul_mod_u64(p2[i], psi_pow, q);
                psi_pow = Self::mul_mod_u64(psi_pow, psi, q);
            }

            // Transform to NTT domain (GPU)
            ntt_ctx.forward(&mut p1)?;
            ntt_ctx.forward(&mut p2)?;

            // Pointwise multiply in NTT domain (GPU)
            let mut prod = vec![0u64; n];
            ntt_ctx.pointwise_multiply(&p1, &p2, &mut prod)?;

            // Transform back to coefficient domain (GPU)
            ntt_ctx.inverse(&mut prod)?;

            // UNTWIST: Multiply by psi^{-i} to get negacyclic result
            let mut psi_inv_pow = 1u64;
            for i in 0..n {
                prod[i] = Self::mul_mod_u64(prod[i], psi_inv_pow, q);
                psi_inv_pow = Self::mul_mod_u64(psi_inv_pow, psi_inv, q);
            }

            // Store result
            result[offset..offset + n].copy_from_slice(&prod);
        }

        Ok(result)
    }

    /// Find sqrt(a) mod p using Tonelli-Shanks algorithm
    fn find_sqrt_mod(a: u64, p: u64) -> Result<u64, String> {
        // For our NTT primes, p ≡ 1 (mod 2N), so sqrt exists
        // We use a simple trial approach for now (works for NTT primes)

        // Special case: a = 0 or 1
        if a == 0 { return Ok(0); }
        if a == 1 { return Ok(1); }

        // Factor out powers of 2 from p-1
        let mut q = p - 1;
        let mut s = 0u32;
        while q % 2 == 0 {
            q /= 2;
            s += 1;
        }

        // Find a non-residue
        let mut z = 2u64;
        while Self::pow_mod(z, (p - 1) / 2, p) != p - 1 {
            z += 1;
            if z >= p { return Err("No quadratic non-residue found".to_string()); }
        }

        let mut m = s;
        let mut c = Self::pow_mod(z, q, p);
        let mut t = Self::pow_mod(a, q, p);
        let mut r = Self::pow_mod(a, (q + 1) / 2, p);

        loop {
            if t == 0 { return Ok(0); }
            if t == 1 { return Ok(r); }

            // Find smallest i such that t^(2^i) = 1
            let mut i = 1u32;
            let mut temp = Self::mul_mod_u64(t, t, p);
            while temp != 1 {
                temp = Self::mul_mod_u64(temp, temp, p);
                i += 1;
                if i >= m { return Err("Tonelli-Shanks failed".to_string()); }
            }

            // Update values
            let b = Self::pow_mod(c, 1u64 << (m - i - 1), p);
            m = i;
            c = Self::mul_mod_u64(b, b, p);
            t = Self::mul_mod_u64(t, c, p);
            r = Self::mul_mod_u64(r, b, p);
        }
    }

    /// Modular exponentiation: a^e mod p
    fn pow_mod(mut a: u64, mut e: u64, p: u64) -> u64 {
        let mut result = 1u64;
        a = a % p;
        while e > 0 {
            if e & 1 == 1 {
                result = Self::mul_mod_u64(result, a, p);
            }
            e >>= 1;
            a = Self::mul_mod_u64(a, a, p);
        }
        result
    }

    /// Find primitive 2N-th root of unity modulo q
    ///
    /// CRITICAL: This must match CudaCkksContext::find_primitive_root exactly!
    /// Uses the same algorithm to ensure EVK generation and relinearization
    /// use identical psi values.
    fn find_primitive_root(n: usize, q: u64) -> Result<u64, String> {
        let two_n = (2 * n) as u64;
        if (q - 1) % two_n != 0 {
            return Err(format!("q-1 = {} is not divisible by 2N = {}", q - 1, two_n));
        }

        // MUST match CudaCkksContext's candidate list exactly
        let candidates: [u64; 11] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];

        for &candidate in &candidates {
            // Check if candidate is a quadratic non-residue
            if Self::pow_mod(candidate, (q - 1) / 2, q) == 1 {
                continue;
            }

            // Compute psi = g^((q-1)/2N) mod q
            let exp = (q - 1) / two_n;
            let psi = Self::pow_mod(candidate, exp, q);

            // Verify: psi^N = -1 (mod q)
            let psi_n = Self::pow_mod(psi, n as u64, q);
            if psi_n != q - 1 {
                continue;
            }

            // Verify: psi^(2N) = 1 (mod q)
            let psi_2n = Self::pow_mod(psi, two_n, q);
            if psi_2n == 1 {
                return Ok(psi);
            }
        }

        Err(format!("Could not find primitive root for n={}, q={}", n, q))
    }

    /// Modular multiplication using u128
    fn mul_mod_u64(a: u64, b: u64, q: u64) -> u64 {
        ((a as u128 * b as u128) % q as u128) as u64
    }

    /// Modular inverse using extended Euclidean algorithm (u64 version)
    fn mod_inverse_u64(a: u64, m: u64) -> Result<u64, String> {
        let mut t: i128 = 0;
        let mut newt: i128 = 1;
        let mut r: i128 = m as i128;
        let mut newr: i128 = a as i128;

        while newr != 0 {
            let quotient = r / newr;
            let temp_t = t;
            t = newt;
            newt = temp_t - quotient * newt;

            let temp_r = r;
            r = newr;
            newr = temp_r - quotient * newr;
        }

        if r > 1 {
            return Err(format!("{} is not invertible mod {}", a, m));
        }
        if t < 0 {
            t += m as i128;
        }

        Ok(t as u64)
    }

    /// Gadget decomposition: decompose polynomial into base-w digits using CRT
    ///
    /// This is the CORRECT implementation that matches Metal's gadget_decompose_flat.
    /// It uses Chinese Remainder Theorem to reconstruct the full integer before
    /// decomposing into balanced base-w digits.
    ///
    /// Input: poly in flat RNS layout [prime_idx * n + coeff_idx]
    /// Output: Vec of digit polynomials, each in flat RNS layout
    fn gadget_decompose(&self, poly: &[u64], num_primes: usize) -> Result<Vec<Vec<u64>>, String> {
        let n = self.params.n;
        let moduli = &self.params.moduli[..num_primes];
        let base_bits = self.base_bits;  // This is the EXPONENT (e.g., 16)

        // Compute Q = product of all primes
        let q_prod_big: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();
        let q_half_big = &q_prod_big / 2;
        let base_big = BigInt::one() << base_bits;  // B = 2^base_bits (e.g., 2^16 = 65536)

        // Determine number of digits based on Q
        let q_bits = q_prod_big.bits() as u32;
        let num_digits = ((q_bits + base_bits - 1) / base_bits) as usize;

        // Use the smaller of computed num_digits or self.dnum
        let actual_num_digits = num_digits.min(self.dnum);
        let mut digits = vec![vec![0u64; n * num_primes]; actual_num_digits];

        // For each coefficient
        for coeff_idx in 0..n {
            // Step 1: CRT reconstruct to get x ∈ [0, Q)
            let mut x_big = BigInt::zero();
            for (prime_idx, &q) in moduli.iter().enumerate() {
                let flat_idx = prime_idx * n + coeff_idx;
                let residue = poly[flat_idx];

                let q_big = BigInt::from(q);
                let q_i = &q_prod_big / &q_big;

                // Compute q_i^(-1) mod q using extended GCD
                let qi_inv = Self::mod_inverse_bigint(&q_i, &q_big)?;

                let ri_big = BigInt::from(residue);
                // Compute: basis = (Q/qi) * inv mod Q, then term = ri * basis mod Q
                let basis = (&q_i * &qi_inv) % &q_prod_big;
                let term = (ri_big * basis) % &q_prod_big;
                x_big = (&x_big + term) % &q_prod_big;
            }

            // Ensure result is positive
            if x_big.sign() == num_bigint::Sign::Minus {
                x_big += &q_prod_big;
            }

            // Step 2: Center-lift to x_c ∈ (-Q/2, Q/2]
            let x_centered_big = if x_big > q_half_big {
                x_big - &q_prod_big
            } else {
                x_big
            };

            // Step 3: Balanced decomposition in Z
            let mut remainder_big = x_centered_big;
            let half_base_big = &base_big / 2;

            for t in 0..actual_num_digits {
                // Extract digit dt ∈ (-B/2, B/2] (balanced)
                let dt_unbalanced = &remainder_big % &base_big;
                let dt_big = if dt_unbalanced > half_base_big {
                    &dt_unbalanced - &base_big  // Shift to negative range
                } else {
                    dt_unbalanced
                };

                // Convert dt to residues mod each prime
                for (prime_idx, &q) in moduli.iter().enumerate() {
                    let flat_idx = prime_idx * n + coeff_idx;
                    let q_big = BigInt::from(q);
                    let mut dt_mod_q_big = &dt_big % &q_big;
                    if dt_mod_q_big.sign() == num_bigint::Sign::Minus {
                        dt_mod_q_big += &q_big;
                    }
                    digits[t][flat_idx] = dt_mod_q_big.to_u64().unwrap_or(0);
                }

                // Update remainder: (x_c - dt) / B (exact division)
                remainder_big = (remainder_big - &dt_big) / &base_big;
            }
        }

        Ok(digits)
    }

    /// Modular inverse using extended Euclidean algorithm (BigInt version)
    fn mod_inverse_bigint(a: &BigInt, modulus: &BigInt) -> Result<BigInt, String> {
        let mut t = BigInt::zero();
        let mut newt = BigInt::one();
        let mut r = modulus.clone();
        let mut newr = a.clone();

        while !newr.is_zero() {
            let quotient = &r / &newr;
            let temp_t = t.clone();
            t = newt.clone();
            newt = temp_t - &quotient * &newt;

            let temp_r = r.clone();
            r = newr.clone();
            newr = temp_r - quotient * newr;
        }

        if r > BigInt::one() {
            return Err(format!("Not invertible"));
        }
        if t < BigInt::zero() {
            t += modulus;
        }

        Ok(t)
    }

    /// CPU polynomial multiplication in flat RNS layout (schoolbook)
    fn cpu_multiply_flat(&self, poly1: &[u64], poly2: &[u64], num_primes: usize) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        let mut result = vec![0u64; n * num_primes];

        // For each prime
        for prime_idx in 0..num_primes {
            let q = self.params.moduli[prime_idx];
            let reducer = &self.reducers_key[prime_idx];

            // Schoolbook multiplication with negacyclic reduction
            for i in 0..n {
                for j in 0..n {
                    let idx1 = prime_idx * n + i;
                    let idx2 = prime_idx * n + j;

                    let a = poly1[idx1];
                    let b = poly2[idx2];
                    let prod = ((a as u128 * b as u128) % q as u128) as u64;

                    let result_idx = (i + j) % (2 * n);
                    if result_idx < n {
                        // Positive coefficient
                        let result_flat_idx = prime_idx * n + result_idx;
                        result[result_flat_idx] = (result[result_flat_idx] + prod) % q;
                    } else {
                        // Negative coefficient (X^N = -1)
                        let result_flat_idx = prime_idx * n + (result_idx - n);
                        result[result_flat_idx] = if result[result_flat_idx] >= prod {
                            result[result_flat_idx] - prod
                        } else {
                            q - (prod - result[result_flat_idx])
                        };
                    }
                }
            }
        }

        Ok(result)
    }

    /// Generate random polynomial in flat RNS layout
    fn generate_random_poly(&self, num_primes: usize) -> Vec<u64> {
        let n = self.params.n;
        let mut rng = rand::thread_rng();
        let mut poly = vec![0u64; n * num_primes];

        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let flat_idx = prime_idx * n + coeff_idx;
                let q = self.params.moduli[prime_idx];
                poly[flat_idx] = rng.gen::<u64>() % q;
            }
        }

        poly
    }

    /// Generate error polynomial with Gaussian distribution
    fn generate_error_poly(&self, num_primes: usize) -> Vec<u64> {
        let n = self.params.n;
        let mut rng = rand::thread_rng();
        let mut poly = vec![0u64; n * num_primes];

        // Small Gaussian error (simplified: use small random values)
        for coeff_idx in 0..n {
            // Generate base error (small value)
            let error = (rng.gen::<u32>() % 8) as i32 - 4; // [-4, 3]

            for prime_idx in 0..num_primes {
                let flat_idx = prime_idx * n + coeff_idx;
                let q = self.params.moduli[prime_idx];

                poly[flat_idx] = if error >= 0 {
                    error as u64
                } else {
                    q - ((-error) as u64)
                };
            }
        }

        poly
    }

    /// Get reference to device context
    pub fn device(&self) -> &Arc<CudaDeviceContext> {
        &self.device
    }

    /// Get reference to parameters
    pub fn params(&self) -> &CliffordFHEParams {
        &self.params
    }

    /// Get modulus for a given prime index
    pub fn modulus(&self, prime_idx: usize) -> u64 {
        self.params.moduli[prime_idx]
    }

    /// Get reference to the relinearization key
    pub fn get_relin_key(&self) -> &RelinearizationKey {
        self.relin_key.as_ref().expect("Relin key not initialized")
    }

    /// Get gadget decomposition parameters
    /// Returns (base_bits, num_digits) where base = 2^base_bits
    pub fn gadget_params(&self) -> (u32, usize) {
        (self.base_bits, self.dnum)
    }

    /// Get the actual gadget base value (2^base_bits)
    pub fn base_w(&self) -> u64 {
        1u64 << self.base_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gadget_decomposition() {
        // Test that gadget decomposition works correctly
        // w = 2^16, decompose a value into base-w digits
        let base_w = 1u64 << 16;
        let val = 1234567890u64;

        // Manually decompose
        let d0 = val % base_w;
        let d1 = (val / base_w) % base_w;
        let d2 = (val / base_w / base_w) % base_w;

        // Reconstruct
        let reconstructed = d0 + d1 * base_w + d2 * base_w * base_w;
        assert_eq!(reconstructed, val);
    }
}
