//! CUDA GPU-Accelerated Rotation Keys for CKKS
//!
//! Implements key switching for rotation operations using GPU-accelerated NTT.
//!
//! **Key Switching Overview:**
//! After applying Galois automorphism (rotation), we have:
//!   ct = (c0(X^g), c1(X^g))
//!
//! But c1(X^g) is encrypted under s(X^g), not s(X). We need to "switch" it back:
//!   ct' = (c0', c1') where c1' is encrypted under s(X)
//!
//! **Rotation Key Structure:**
//! For each Galois element g, we precompute:
//!   RotKey_g = {KS_0(g), KS_1(g), ..., KS_{dnum-1}(g)}
//! where:
//!   KS_i(g) = (-a_i · s(X^g) + e_i + w^i · s(X^g), a_i)
//!           = (b_i, a_i) in RNS representation
//!
//! **Gadget Decomposition:**
//! We decompose c1(X^g) into base-w digits:
//!   c1(X^g) = Σ_{i=0}^{dnum-1} d_i · w^i
//! where d_i ∈ [-w/2, w/2] (signed decomposition)
//!
//! **Key Application:**
//!   c0' = c0(X^g) + Σ_i d_i · b_i
//!   c1' = Σ_i d_i · a_i
//!
//! **GPU Acceleration:**
//! - NTT-multiply for key generation (GPU)
//! - Gadget decomposition (CPU, simple)
//! - Key application multiply (GPU via NTT)

use crate::clifford_fhe_v2::backends::gpu_cuda::device::CudaDeviceContext;
use crate::clifford_fhe_v2::backends::gpu_cuda::rotation::CudaRotationContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::BarrettReducer;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use rand::Rng;
use std::sync::Arc;

/// Rotation key for a specific Galois element
#[derive(Clone)]
pub struct RotationKey {
    /// Galois element (g = 5^k mod 2N for rotation by k slots)
    pub galois_elt: u64,

    /// Key switching components: [(b_0, a_0), (b_1, a_1), ..., (b_{dnum-1}, a_{dnum-1})]
    /// Each component is in flat RNS layout: poly[prime_idx * n + coeff_idx]
    /// b_i and a_i have size n * num_primes_key
    pub ks_components: Vec<(Vec<u64>, Vec<u64>)>,

    /// Number of RNS primes in the key (typically num_primes + special_primes)
    pub num_primes_key: usize,

    /// Polynomial degree
    pub n: usize,
}

/// CUDA rotation keys manager
pub struct CudaRotationKeys {
    device: Arc<CudaDeviceContext>,
    params: CliffordFHEParams,
    rotation_ctx: Arc<CudaRotationContext>,

    /// Barrett reducers for key level
    reducers_key: Vec<BarrettReducer>,

    /// Gadget decomposition base EXPONENT (e.g., 16 for w = 2^16)
    /// NOTE: This stores the exponent, not the base itself. Use base_w() to get 2^base_bits.
    pub base_bits: u32,

    /// Number of gadget digits: dnum = ceil(log_w(Q_key))
    pub dnum: usize,

    /// Secret key (in coefficient form, strided layout)
    secret_key: Vec<u64>,

    /// Rotation keys for different Galois elements
    /// Key: Galois element g
    /// Value: RotationKey
    keys: std::collections::HashMap<u64, RotationKey>,
}

impl CudaRotationKeys {
    /// Create new rotation keys manager
    ///
    /// # Arguments
    /// * `device` - CUDA device context
    /// * `params` - FHE parameters
    /// * `rotation_ctx` - Rotation context (for Galois elements)
    /// * `secret_key` - Secret key polynomial (strided layout)
    /// * `base_bits` - Gadget base bits (e.g., 16 for w = 2^16)
    pub fn new(
        device: Arc<CudaDeviceContext>,
        params: CliffordFHEParams,
        rotation_ctx: Arc<CudaRotationContext>,
        secret_key: Vec<u64>,
        base_bits: usize,
    ) -> Result<Self, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║        Initializing CUDA Rotation Keys                       ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let n = params.n;
        let num_primes = params.moduli.len();

        // For rotation keys, we use all available primes
        // In full implementation, we'd add special primes here
        let num_primes_key = num_primes;
        let moduli_key = &params.moduli[..num_primes_key];

        // Create Barrett reducers
        let reducers_key: Vec<BarrettReducer> = moduli_key
            .iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        // Compute gadget parameters
        let base_bits_u32 = base_bits as u32;  // Store exponent, not base
        let base_w = 1u64 << base_bits;  // Computed when needed
        let max_prime_bits = moduli_key.iter().map(|&q| (64 - q.leading_zeros()) as usize).max().unwrap_or(60);
        let total_bits = max_prime_bits * num_primes_key;
        let dnum = (total_bits + base_bits - 1) / base_bits;

        println!("Gadget decomposition parameters:");
        println!("  Base w: 2^{} = {}", base_bits, base_w);
        println!("  Number of digits (dnum): {}", dnum);
        println!("  Total modulus bits: {}\n", total_bits);

        Ok(Self {
            device,
            params,
            rotation_ctx,
            reducers_key,
            base_bits: base_bits_u32,
            dnum,
            secret_key,
            keys: std::collections::HashMap::new(),
        })
    }

    /// Generate rotation key for a specific rotation amount
    ///
    /// This generates key switching components for rotating by k slots.
    ///
    /// # Arguments
    /// * `rotation_steps` - Number of slots to rotate (positive or negative)
    /// Generate rotation key using GPU NTT for polynomial multiplication
    ///
    /// This is MUCH faster than the CPU version
    pub fn generate_rotation_key_gpu(
        &mut self,
        rotation_steps: i32,
        ntt_contexts: &[super::ntt::CudaNttContext],
    ) -> Result<(), String> {
        println!("Generating rotation key for rotation by {} slots...", rotation_steps);

        let n = self.params.n;
        let num_primes_key = self.params.moduli.len();

        // Compute Galois element
        let galois_elt = self.compute_galois_element(rotation_steps);
        println!("  Galois element g: {}", galois_elt);

        // Check if already generated
        if self.keys.contains_key(&galois_elt) {
            println!("  (Already cached)\n");
            return Ok(());
        }

        // Apply Galois automorphism to secret key: s(X) → s(X^g)
        let s_galois = self.apply_galois_to_secret_key(galois_elt)?;

        // Generate dnum key switching components
        let mut ks_components = Vec::new();
        let mut rng = rand::thread_rng();

        for digit_idx in 0..self.dnum {
            // Generate random polynomial a_i
            let mut a_i = vec![0u64; n * num_primes_key];
            for prime_idx in 0..num_primes_key {
                let q = self.params.moduli[prime_idx];
                for coeff_idx in 0..n {
                    a_i[prime_idx * n + coeff_idx] = rng.gen::<u64>() % q;
                }
            }

            // Generate small error polynomial e_i (Gaussian, approximate with small uniform)
            let mut e_i = vec![0u64; n * num_primes_key];
            let error_bound = 16u64;  // Small error for security
            for prime_idx in 0..num_primes_key {
                let q = self.params.moduli[prime_idx];
                for coeff_idx in 0..n {
                    let e = (rng.gen::<u64>() % (2 * error_bound + 1)) as i64 - error_bound as i64;
                    e_i[prime_idx * n + coeff_idx] = if e >= 0 {
                        e as u64
                    } else {
                        (q as i64 + e) as u64
                    };
                }
            }

            // Compute b_i = -a_i · s(X^g) + e_i + w^i · s(X^g)
            //             = (w^i - a_i) · s(X^g) + e_i

            let base_w = 1u64 << self.base_bits;
            let w_power = base_w.pow(digit_idx as u32);

            // First compute: (w^i - a_i) mod each prime
            let mut w_minus_a = vec![0u64; n * num_primes_key];
            for prime_idx in 0..num_primes_key {
                let q = self.params.moduli[prime_idx];
                let w_mod_q = w_power % q;
                for coeff_idx in 0..n {
                    let a_val = a_i[prime_idx * n + coeff_idx];
                    w_minus_a[prime_idx * n + coeff_idx] = if w_mod_q >= a_val {
                        w_mod_q - a_val
                    } else {
                        q - (a_val - w_mod_q)
                    };
                }
            }

            // Multiply (w^i - a_i) · s(X^g) using GPU NTT
            let product = self.gpu_multiply_flat_ntt(&w_minus_a, &s_galois, num_primes_key, ntt_contexts)?;

            // Add error: b_i = product + e_i
            let mut b_i = vec![0u64; n * num_primes_key];
            for prime_idx in 0..num_primes_key {
                let q = self.params.moduli[prime_idx];
                for coeff_idx in 0..n {
                    let idx = prime_idx * n + coeff_idx;
                    b_i[idx] = (product[idx] + e_i[idx]) % q;
                }
            }

            ks_components.push((b_i, a_i));

            if (digit_idx + 1) % 5 == 0 || digit_idx == self.dnum - 1 {
                println!("  Generated {}/{} key switching components", digit_idx + 1, self.dnum);
            }
        }

        // Store rotation key
        let rot_key = RotationKey {
            galois_elt,
            ks_components,
            num_primes_key,
            n,
        };

        self.keys.insert(galois_elt, rot_key);
        println!("  Rotation key generated\n");

        Ok(())
    }

    /// Generate rotation key (CPU version - DEPRECATED)
    ///
    /// Use generate_rotation_key_gpu() instead for much better performance
    pub fn generate_rotation_key(&mut self, rotation_steps: i32) -> Result<(), String> {
        println!("Generating rotation key for rotation by {} slots...", rotation_steps);

        let n = self.params.n;
        let num_primes_key = self.params.moduli.len();

        // Compute Galois element
        let galois_elt = self.compute_galois_element(rotation_steps);
        println!("  Galois element g: {}", galois_elt);

        // Check if already generated
        if self.keys.contains_key(&galois_elt) {
            println!("  (Already cached)\n");
            return Ok(());
        }

        // Apply Galois automorphism to secret key: s(X) → s(X^g)
        let s_galois = self.apply_galois_to_secret_key(galois_elt)?;

        // Generate dnum key switching components
        let mut ks_components = Vec::new();
        let mut rng = rand::thread_rng();

        for digit_idx in 0..self.dnum {
            // Generate random polynomial a_i
            let mut a_i = vec![0u64; n * num_primes_key];
            for prime_idx in 0..num_primes_key {
                let q = self.params.moduli[prime_idx];
                for coeff_idx in 0..n {
                    a_i[prime_idx * n + coeff_idx] = rng.gen::<u64>() % q;
                }
            }

            // Generate small error polynomial e_i (Gaussian, approximate with small uniform)
            let mut e_i = vec![0u64; n * num_primes_key];
            let error_bound = 16u64;  // Small error for security
            for prime_idx in 0..num_primes_key {
                let q = self.params.moduli[prime_idx];
                for coeff_idx in 0..n {
                    let e = (rng.gen::<u64>() % (2 * error_bound + 1)) as i64 - error_bound as i64;
                    e_i[prime_idx * n + coeff_idx] = if e >= 0 {
                        e as u64
                    } else {
                        (q as i64 + e) as u64
                    };
                }
            }

            // Compute b_i = -a_i · s(X^g) + e_i + w^i · s(X^g)
            //             = (w^i - a_i) · s(X^g) + e_i

            let base_w = 1u64 << self.base_bits;
            let w_power = base_w.pow(digit_idx as u32);

            // First compute: (w^i - a_i) mod each prime
            let mut w_minus_a = vec![0u64; n * num_primes_key];
            for prime_idx in 0..num_primes_key {
                let q = self.params.moduli[prime_idx];
                let w_mod_q = w_power % q;
                for coeff_idx in 0..n {
                    let a_val = a_i[prime_idx * n + coeff_idx];
                    w_minus_a[prime_idx * n + coeff_idx] = if w_mod_q >= a_val {
                        w_mod_q - a_val
                    } else {
                        q - (a_val - w_mod_q)
                    };
                }
            }

            // Multiply (w^i - a_i) · s(X^g) using CPU (DEPRECATED - use GPU version)
            let product = self.cpu_multiply_flat(&w_minus_a, &s_galois, num_primes_key)?;

            // Add error: b_i = product + e_i
            let mut b_i = vec![0u64; n * num_primes_key];
            for prime_idx in 0..num_primes_key {
                let q = self.params.moduli[prime_idx];
                for coeff_idx in 0..n {
                    let idx = prime_idx * n + coeff_idx;
                    b_i[idx] = (product[idx] + e_i[idx]) % q;
                }
            }

            ks_components.push((b_i, a_i));

            if (digit_idx + 1) % 5 == 0 || digit_idx == self.dnum - 1 {
                println!("  Generated {}/{} key switching components", digit_idx + 1, self.dnum);
            }
        }

        // Store rotation key
        let rot_key = RotationKey {
            galois_elt,
            ks_components,
            num_primes_key,
            n,
        };

        self.keys.insert(galois_elt, rot_key);
        println!("  Rotation key generated\n");

        Ok(())
    }

    /// Compute Galois element for rotation (delegates to rotation context)
    fn compute_galois_element(&self, rotation_steps: i32) -> u64 {
        let n = self.params.n as i64;
        let two_n = 2 * n;

        let k = if rotation_steps >= 0 {
            rotation_steps as i64
        } else {
            let slots = n / 2;
            (slots + rotation_steps as i64) % slots
        };

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

    /// Apply Galois automorphism to secret key
    fn apply_galois_to_secret_key(&self, galois_elt: u64) -> Result<Vec<u64>, String> {
        // Convert secret key from strided to flat layout
        let n = self.params.n;
        let num_primes = self.params.moduli.len();

        let mut sk_flat = vec![0u64; n * num_primes];
        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                sk_flat[prime_idx * n + coeff_idx] = self.secret_key[coeff_idx * num_primes + prime_idx];
            }
        }

        // Compute permutation map
        let perm_map = self.compute_permutation_map(galois_elt);

        // Apply permutation
        let mut sk_galois = vec![0u64; n * num_primes];
        for coeff_idx in 0..n {
            let perm_entry = perm_map[coeff_idx];
            let negate = perm_entry < 0;
            let src_idx = if negate {
                ((-perm_entry) - 1) as usize
            } else {
                perm_entry as usize
            };

            for prime_idx in 0..num_primes {
                let q = self.params.moduli[prime_idx];
                let src_pos = prime_idx * n + src_idx;
                let dst_pos = prime_idx * n + coeff_idx;

                let mut val = sk_flat[src_pos];
                if negate && val != 0 {
                    val = q - val;
                }
                sk_galois[dst_pos] = val;
            }
        }

        Ok(sk_galois)
    }

    /// Compute permutation map (same as in rotation context)
    fn compute_permutation_map(&self, galois_elt: u64) -> Vec<i32> {
        let n = self.params.n;
        let two_n = 2 * n;
        let mut perm = vec![0i32; n];

        for i in 0..n {
            let raw_idx = ((i as u64 * galois_elt) % (two_n as u64)) as usize;

            if raw_idx < n {
                perm[i] = raw_idx as i32;
            } else {
                let mapped_idx = two_n - raw_idx;
                perm[i] = -(mapped_idx as i32) - 1;
            }
        }

        perm
    }

    /// GPU polynomial multiplication in flat RNS layout using NTT
    ///
    /// This is MUCH faster than CPU schoolbook O(n²) - uses O(n log n) NTT
    fn gpu_multiply_flat_ntt(
        &self,
        poly1: &[u64],
        poly2: &[u64],
        num_primes: usize,
        ntt_contexts: &[super::ntt::CudaNttContext],
    ) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        let mut result = vec![0u64; n * num_primes];

        // For each RNS prime, use GPU NTT multiplication
        for prime_idx in 0..num_primes {
            let ntt_ctx = &ntt_contexts[prime_idx];
            let offset = prime_idx * n;

            // Extract polynomials for this prime
            let mut p1 = poly1[offset..offset + n].to_vec();
            let mut p2 = poly2[offset..offset + n].to_vec();

            // Transform to NTT domain (GPU)
            ntt_ctx.forward(&mut p1)?;
            ntt_ctx.forward(&mut p2)?;

            // Pointwise multiply in NTT domain (GPU)
            let mut prod = vec![0u64; n];
            ntt_ctx.pointwise_multiply(&p1, &p2, &mut prod)?;

            // Transform back to coefficient domain (GPU)
            ntt_ctx.inverse(&mut prod)?;

            // Store result
            result[offset..offset + n].copy_from_slice(&prod);
        }

        Ok(result)
    }

    /// Multiply two polynomials using CPU schoolbook (flat RNS layout)
    /// DEPRECATED: Use gpu_multiply_flat_ntt() instead for much better performance
    fn cpu_multiply_flat(&self, a: &[u64], b: &[u64], num_primes: usize) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        let mut result = vec![0u64; n * num_primes];

        for prime_idx in 0..num_primes {
            let q = self.params.moduli[prime_idx];
            let offset = prime_idx * n;

            // Extract polynomials for this prime
            let a_slice = &a[offset..offset + n];
            let b_slice = &b[offset..offset + n];

            // Schoolbook polynomial multiplication in Z[X]
            let mut temp = vec![0u64; 2 * n];
            for i in 0..n {
                for j in 0..n {
                    let prod = ((a_slice[i] as u128) * (b_slice[j] as u128)) % (q as u128);
                    temp[i + j] = (temp[i + j] + prod as u64) % q;
                }
            }

            // Reduce modulo X^N + 1: X^N = -1
            for i in 0..n {
                let high_part = temp[i + n];
                result[offset + i] = if temp[i] >= high_part {
                    temp[i] - high_part
                } else {
                    q - (high_part - temp[i])
                };
            }
        }

        Ok(result)
    }

    /// Apply rotation key using GPU NTT for polynomial multiplication
    ///
    /// This is MUCH faster than the CPU version
    pub fn apply_rotation_key_gpu(
        &self,
        c1_galois: &[u64],
        galois_elt: u64,
        level: usize,
        ntt_contexts: &[super::ntt::CudaNttContext],
    ) -> Result<(Vec<u64>, Vec<u64>), String> {
        // Get rotation key
        let rot_key = self.keys.get(&galois_elt)
            .ok_or_else(|| format!("Rotation key for galois element {} not found", galois_elt))?;

        let n = self.params.n;
        let num_primes = level + 1;

        // Decompose c1(X^g) into base-w digits
        let digits = self.gadget_decompose(c1_galois, num_primes)?;

        // Initialize accumulator
        let mut c0_acc = vec![0u64; n * num_primes];
        let mut c1_acc = vec![0u64; n * num_primes];

        // Accumulate: c0' += Σ d_i · b_i, c1' += Σ d_i · a_i
        for (digit_idx, d_i) in digits.iter().enumerate() {
            let (b_i, a_i) = &rot_key.ks_components[digit_idx];

            // Multiply d_i · b_i using GPU NTT
            let d_b = self.gpu_multiply_flat_ntt(d_i, b_i, num_primes, ntt_contexts)?;
            // Multiply d_i · a_i using GPU NTT
            let d_a = self.gpu_multiply_flat_ntt(d_i, a_i, num_primes, ntt_contexts)?;

            // Accumulate
            for i in 0..n * num_primes {
                let q = self.params.moduli[i / n];
                c0_acc[i] = (c0_acc[i] + d_b[i]) % q;
                c1_acc[i] = (c1_acc[i] + d_a[i]) % q;
            }
        }

        Ok((c0_acc, c1_acc))
    }

    /// Apply rotation key to ciphertext component (CPU version - DEPRECATED)
    ///
    /// Use apply_rotation_key_gpu() instead for much better performance
    /// Given c1(X^g), compute key-switched result using rotation key.
    /// This is used internally during rotation operations.
    pub fn apply_rotation_key(
        &self,
        c1_galois: &[u64],
        galois_elt: u64,
        level: usize,
    ) -> Result<(Vec<u64>, Vec<u64>), String> {
        // Get rotation key
        let rot_key = self.keys.get(&galois_elt)
            .ok_or_else(|| format!("Rotation key for galois element {} not found", galois_elt))?;

        let n = self.params.n;
        let num_primes = level + 1;

        // Decompose c1(X^g) into base-w digits
        let digits = self.gadget_decompose(c1_galois, num_primes)?;

        // Initialize accumulator
        let mut c0_acc = vec![0u64; n * num_primes];
        let mut c1_acc = vec![0u64; n * num_primes];

        // Accumulate: c0' += Σ d_i · b_i, c1' += Σ d_i · a_i
        for (digit_idx, d_i) in digits.iter().enumerate() {
            let (b_i, a_i) = &rot_key.ks_components[digit_idx];

            // Multiply d_i · b_i (DEPRECATED - use GPU version)
            let d_b = self.cpu_multiply_flat(d_i, b_i, num_primes)?;
            // Multiply d_i · a_i (DEPRECATED - use GPU version)
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

    /// Gadget decomposition: decompose polynomial into base-w digits
    ///
    /// Input: poly in flat RNS layout
    /// Output: Vec of digit polynomials, each in flat RNS layout
    fn gadget_decompose(&self, poly: &[u64], num_primes: usize) -> Result<Vec<Vec<u64>>, String> {
        // Placeholder implementation
        // In full version, we'd do proper multi-digit decomposition
        // For now, return a simple single-digit decomposition
        Ok(vec![poly.to_vec()])
    }

    /// Get number of rotation keys generated
    pub fn num_keys(&self) -> usize {
        self.keys.len()
    }

    /// Get reference to rotation context
    pub fn rotation_context(&self) -> &Arc<CudaRotationContext> {
        &self.rotation_ctx
    }

    /// Get modulus for a given prime index
    pub fn modulus(&self, prime_idx: usize) -> u64 {
        self.params.moduli[prime_idx]
    }

    /// Get a rotation key by Galois element
    ///
    /// Returns None if no key exists for the given Galois element.
    /// Use `generate_rotation_key_gpu` to create keys first.
    pub fn get_rotation_key(&self, galois_elt: u64) -> Option<&RotationKey> {
        self.keys.get(&galois_elt)
    }

    /// Get the secret key (for V6 integration)
    pub fn secret_key(&self) -> &[u64] {
        &self.secret_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gadget_parameters() {
        // Test gadget parameter computation
        let base_bits = 16;
        let base_w = 1u64 << base_bits;  // 65536
        let max_prime_bits = 60;
        let num_primes = 3;
        let total_bits = max_prime_bits * num_primes;  // 180
        let dnum = (total_bits + base_bits - 1) / base_bits;  // ceil(180/16) = 12

        assert_eq!(base_w, 65536);
        assert_eq!(dnum, 12);
    }
}
