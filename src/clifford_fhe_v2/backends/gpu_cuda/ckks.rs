// CUDA CKKS Implementation
//
// Provides CKKS homomorphic encryption operations on NVIDIA GPUs using CUDA.
// Based on the Metal GPU implementation but adapted for CUDA/cudarc.

use super::device::CudaDeviceContext;
use super::ntt::CudaNttContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{PublicKey, SecretKey};
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::BarrettReducer;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

/// CUDA CKKS context for FHE operations on NVIDIA GPU
pub struct CudaCkksContext {
    /// CUDA device
    device: Arc<CudaDeviceContext>,

    /// FHE parameters
    params: CliffordFHEParams,

    /// NTT contexts for each RNS prime
    ntt_contexts: Vec<CudaNttContext>,

    /// Barrett reducers for modular reduction (lightweight, keep on CPU)
    reducers: Vec<BarrettReducer>,

    /// Precomputed rescaling constants for GPU-native exact rescale
    /// rescale_inv_table[level][i] = q_{level}^{-1} mod q_i for i < level
    pub rescale_inv_table: Vec<Vec<u64>>,

    /// RNS kernels loaded flag
    rns_kernels_loaded: bool,

    /// GPU-cached twiddles for batched NTT (all primes concatenated)
    /// Layout: [prime_0 twiddles (n elements), prime_1 twiddles, ..., prime_L twiddles]
    /// Total size: num_primes × n × u64
    gpu_twiddles_fwd: Option<CudaSlice<u64>>,

    /// GPU-cached inverse twiddles for batched NTT
    gpu_twiddles_inv: Option<CudaSlice<u64>>,

    /// GPU-cached RNS moduli for batched operations
    /// Layout: [q_0, q_1, ..., q_L]
    gpu_moduli: Option<CudaSlice<u64>>,

    /// Per-prime primitive 2N-th roots (psi) for negacyclic twist/untwist
    /// CRITICAL: These must match the omega values used in NTT contexts (omega = psi^2)
    psi_per_prime: Vec<u64>,

    /// Per-prime psi inverses for negacyclic untwist
    psi_inv_per_prime: Vec<u64>,

    /// GPU-cached psi powers for negacyclic twist (FLAT layout)
    /// Layout: psi_powers[prime_idx * n + coeff_idx] = psi[prime_idx]^coeff_idx
    gpu_psi_powers: Option<CudaSlice<u64>>,

    /// GPU-cached psi inverse powers for negacyclic untwist (FLAT layout)
    /// Layout: psi_inv_powers[prime_idx * n + coeff_idx] = psi[prime_idx]^{-coeff_idx}
    gpu_psi_inv_powers: Option<CudaSlice<u64>>,
}

/// CUDA ciphertext representation
#[derive(Clone, Debug)]
pub struct CudaCiphertext {
    /// First polynomial (c0) in strided RNS layout: c0[coeff_idx * num_primes + prime_idx]
    pub c0: Vec<u64>,

    /// Second polynomial (c1) in strided RNS layout
    pub c1: Vec<u64>,

    /// Ring dimension
    pub n: usize,

    /// Total stride (number of RNS primes in memory)
    pub num_primes: usize,

    /// Current level (number of active primes - 1)
    pub level: usize,

    /// Scaling factor
    pub scale: f64,
}

/// CUDA plaintext representation
pub struct CudaPlaintext {
    /// Polynomial coefficients in strided RNS layout
    pub poly: Vec<u64>,

    /// Ring dimension
    pub n: usize,

    /// Number of RNS primes
    pub num_primes: usize,

    /// Current level in modulus chain
    pub level: usize,

    /// Scaling factor
    pub scale: f64,
}

impl CudaPlaintext {
    /// Encode values at a specific level
    ///
    /// This is useful when you need to create a plaintext at a specific level
    /// to match the level of a ciphertext (e.g., for adding constants).
    ///
    /// # Arguments
    ///
    /// * `values` - Values to encode (complex numbers or real values)
    /// * `scale` - Scaling factor
    /// * `params` - FHE parameters
    /// * `level` - Level in modulus chain (0 to num_primes-1)
    ///
    /// # Returns
    ///
    /// Encoded plaintext at the specified level
    pub fn encode_at_level(
        values: &[f64],
        scale: f64,
        params: &CliffordFHEParams,
        level: usize,
    ) -> Self {
        let n = params.n;
        let num_primes = level + 1;

        // Simplified encoding: just scale and convert to integer mod each prime
        // For division, we mainly use this for constant 2.0, so this is sufficient
        let mut poly = vec![0u64; n * num_primes];

        for (i, &val) in values.iter().enumerate().take(n / 2) {
            let scaled_val = (val * scale).round() as i64;

            for prime_idx in 0..num_primes {
                let q = params.moduli[prime_idx];
                let idx = i * num_primes + prime_idx;

                // Handle negative values
                let val_mod_q = if scaled_val >= 0 {
                    (scaled_val as u64) % q
                } else {
                    let abs_val = (-scaled_val) as u64;
                    q - (abs_val % q)
                };

                poly[idx] = val_mod_q;
            }
        }

        Self {
            poly,
            n,
            num_primes,
            level,
            scale,
        }
    }

    /// Encode values using default level (max level)
    pub fn encode(values: &[f64], scale: f64, params: &CliffordFHEParams) -> Self {
        let max_level = params.moduli.len() - 1;
        Self::encode_at_level(values, scale, params, max_level)
    }
}

impl CudaCkksContext {
    /// Create new CUDA CKKS context
    pub fn new(params: CliffordFHEParams) -> Result<Self, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║           Initializing CUDA CKKS Context                     ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let device = Arc::new(CudaDeviceContext::new()?);

        println!("Creating NTT contexts for {} primes...", params.moduli.len());
        let start = std::time::Instant::now();

        // Create NTT context for each RNS prime
        let mut ntt_contexts = Vec::new();
        let mut psi_per_prime = Vec::new();
        let mut psi_inv_per_prime = Vec::new();

        for (i, &q) in params.moduli.iter().enumerate() {
            // Find primitive 2N-th root (psi) for this prime
            let psi = Self::find_primitive_root(params.n, q)?;

            // CUDA NTT needs primitive N-th root (omega) for cyclic convolution
            // omega = psi^2 (since psi is 2N-th root, psi^2 is N-th root)
            let omega = ((psi as u128 * psi as u128) % q as u128) as u64;

            // Compute psi inverse for untwisting
            let psi_inv = Self::mod_inverse(psi, q)?;

            // DEBUG: Print psi values for comparison with CPU
            if i < 3 {
                println!("  [CUDA] Prime {}: q={}, psi={}, omega={}", i, q, psi, omega);
            }

            let ntt_ctx = CudaNttContext::new(params.n, q, omega)?;
            ntt_contexts.push(ntt_ctx);
            psi_per_prime.push(psi);
            psi_inv_per_prime.push(psi_inv);

            if (i + 1) % 5 == 0 || i + 1 == params.moduli.len() {
                println!("  Created {}/{} NTT contexts", i + 1, params.moduli.len());
            }
        }

        println!("  [CUDA CKKS] NTT contexts created in {:.2}s", start.elapsed().as_secs_f64());

        // Create Barrett reducers for CPU-side operations
        let reducers: Vec<BarrettReducer> = params.moduli.iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        // Precompute rescaling inverse constants
        let rescale_inv_table = Self::precompute_rescale_inv_table(&params.moduli);

        // Load RNS kernels
        println!("Loading RNS CUDA kernels...");
        let kernel_src = include_str!("kernels/rns.cu");
        let ptx = compile_ptx(kernel_src)
            .map_err(|e| format!("Failed to compile RNS CUDA kernel: {:?}", e))?;

        device.device.load_ptx(ptx, "rns_module", &[
            "rns_exact_rescale",
            "rns_exact_rescale_strided",
            "rns_strided_to_flat",
            "rns_flat_to_strided",
            "rns_add",
            "rns_sub",
            "rns_negate",
            "rns_pointwise_multiply_strided",
            "rns_negacyclic_twist",
            "rns_negacyclic_untwist",
        ]).map_err(|e| format!("Failed to load RNS PTX: {:?}", e))?;

        // Precompute and cache twiddles on GPU for batched NTT
        println!("Caching twiddles and moduli on GPU for batched NTT...");
        let cache_start = std::time::Instant::now();

        let n = params.n;
        let num_primes = params.moduli.len();

        // Collect all forward twiddles
        let mut all_twiddles_fwd = Vec::with_capacity(n * num_primes);
        let mut all_twiddles_inv = Vec::with_capacity(n * num_primes);
        let mut all_moduli = Vec::with_capacity(num_primes);

        for ntt_ctx in &ntt_contexts {
            all_twiddles_fwd.extend_from_slice(&ntt_ctx.twiddles);
            all_twiddles_inv.extend_from_slice(&ntt_ctx.twiddles_inv);
            all_moduli.push(ntt_ctx.q);
        }

        // Upload to GPU once
        let gpu_twiddles_fwd = device.device.htod_copy(all_twiddles_fwd)
            .map_err(|e| format!("Failed to cache forward twiddles on GPU: {:?}", e))?;

        let gpu_twiddles_inv = device.device.htod_copy(all_twiddles_inv)
            .map_err(|e| format!("Failed to cache inverse twiddles on GPU: {:?}", e))?;

        let gpu_moduli = device.device.htod_copy(all_moduli)
            .map_err(|e| format!("Failed to cache moduli on GPU: {:?}", e))?;

        // Precompute psi powers for negacyclic twist/untwist (FLAT layout)
        // Layout: psi_powers_flat[prime_idx * n + coeff_idx] = psi[prime_idx]^coeff_idx
        let mut all_psi_powers = Vec::with_capacity(n * num_primes);
        let mut all_psi_inv_powers = Vec::with_capacity(n * num_primes);

        for (prime_idx, &q) in params.moduli.iter().enumerate() {
            let psi = psi_per_prime[prime_idx];
            let psi_inv = psi_inv_per_prime[prime_idx];

            // Compute psi^0, psi^1, psi^2, ..., psi^(n-1)
            let mut psi_power = 1u64;
            let mut psi_inv_power = 1u64;
            for _ in 0..n {
                all_psi_powers.push(psi_power);
                all_psi_inv_powers.push(psi_inv_power);
                psi_power = Self::mul_mod(psi_power, psi, q);
                psi_inv_power = Self::mul_mod(psi_inv_power, psi_inv, q);
            }
        }

        let gpu_psi_powers = device.device.htod_copy(all_psi_powers)
            .map_err(|e| format!("Failed to cache psi powers on GPU: {:?}", e))?;

        let gpu_psi_inv_powers = device.device.htod_copy(all_psi_inv_powers)
            .map_err(|e| format!("Failed to cache psi inverse powers on GPU: {:?}", e))?;

        println!("  ✓ Cached {}KB twiddles, psi powers, and {} moduli on GPU in {:.3}s",
                 (n * num_primes * 8 * 4) / 1024,  // 2 for twiddles, 2 for psi
                 num_primes,
                 cache_start.elapsed().as_secs_f64());

        println!("  [CUDA CKKS] ✓ GPU-only CKKS context ready!\n");

        Ok(Self {
            device,
            params,
            ntt_contexts,
            reducers,
            rescale_inv_table,
            rns_kernels_loaded: true,
            gpu_twiddles_fwd: Some(gpu_twiddles_fwd),
            gpu_twiddles_inv: Some(gpu_twiddles_inv),
            gpu_moduli: Some(gpu_moduli),
            psi_per_prime,
            psi_inv_per_prime,
            gpu_psi_powers: Some(gpu_psi_powers),
            gpu_psi_inv_powers: Some(gpu_psi_inv_powers),
        })
    }

    /// Get reference to FHE parameters
    pub fn params(&self) -> &CliffordFHEParams {
        &self.params
    }

    /// Get reference to NTT contexts
    pub fn ntt_contexts(&self) -> &[CudaNttContext] {
        &self.ntt_contexts
    }

    /// Get reference to rescale inversion table
    pub fn rescale_inv_table(&self) -> &[Vec<u64>] {
        &self.rescale_inv_table
    }

    /// Get reference to psi values per prime (for negacyclic twist)
    pub fn psi_per_prime(&self) -> &[u64] {
        &self.psi_per_prime
    }

    /// Get reference to psi inverse values per prime (for negacyclic untwist)
    pub fn psi_inv_per_prime(&self) -> &[u64] {
        &self.psi_inv_per_prime
    }

    /// Modular multiplication: (a * b) mod q
    /// Uses u128 to avoid overflow
    fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
        ((a as u128 * b as u128) % q as u128) as u64
    }

    /// Encrypt plaintext using public key
    ///
    /// Implements CKKS encryption: (c0, c1) where
    /// - c0 = b*u + e0 + m
    /// - c1 = a*u + e1
    ///
    /// where (a, b) is the public key and u, e0, e1 are random noise
    pub fn encrypt(&self, pt: &CudaPlaintext, pk: &PublicKey) -> Result<CudaCiphertext, String> {
        use rand::{thread_rng, Rng};
        use rand_distr::{Distribution, Normal};

        let n = self.params.n;
        let level = pt.level;
        let num_primes = level + 1;
        let moduli = &self.params.moduli[..num_primes];
        let mut rng = thread_rng();

        // Sample ternary random polynomial u ∈ {-1, 0, 1}^n
        let u_coeffs: Vec<i64> = (0..n)
            .map(|_| {
                let val: f64 = rng.gen();
                if val < 0.33 {
                    -1
                } else if val < 0.66 {
                    0
                } else {
                    1
                }
            })
            .collect();

        // Sample error polynomials e0, e1 from Gaussian distribution
        let normal = Normal::new(0.0, self.params.error_std).unwrap();
        let e0_coeffs: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();
        let e1_coeffs: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();

        // Convert to flat RNS layout (coefficient-interleaved)
        let u_flat = self.coeffs_to_flat_rns(&u_coeffs, moduli);
        let e0_flat = self.coeffs_to_flat_rns(&e0_coeffs, moduli);
        let e1_flat = self.coeffs_to_flat_rns(&e1_coeffs, moduli);

        // Extract pk.a and pk.b as flat RNS
        let pk_a_flat = self.rns_vec_to_flat(pk.a[..n].to_vec(), moduli);
        let pk_b_flat = self.rns_vec_to_flat(pk.b[..n].to_vec(), moduli);

        // Multiply pk.b * u using NTT
        let bu_flat = self.multiply_flat_rns(&pk_b_flat, &u_flat, moduli)?;

        // Multiply pk.a * u using NTT
        let au_flat = self.multiply_flat_rns(&pk_a_flat, &u_flat, moduli)?;

        // c0 = b*u + e0 + m (all flat RNS)
        let mut c0 = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let q = moduli[i % num_primes];
            let sum = (bu_flat[i] as u128 + e0_flat[i] as u128 + pt.poly[i] as u128) % q as u128;
            c0[i] = sum as u64;
        }

        // c1 = a*u + e1 (all flat RNS)
        let mut c1 = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let q = moduli[i % num_primes];
            let sum = (au_flat[i] as u128 + e1_flat[i] as u128) % q as u128;
            c1[i] = sum as u64;
        }

        Ok(CudaCiphertext {
            c0,
            c1,
            n,
            num_primes,
            level,
            scale: pt.scale,
        })
    }

    /// Convert signed coefficients to flat RNS layout
    fn coeffs_to_flat_rns(&self, coeffs: &[i64], moduli: &[u64]) -> Vec<u64> {
        let n = coeffs.len();
        let num_primes = moduli.len();
        let mut flat = vec![0u64; n * num_primes];

        for (i, &coeff) in coeffs.iter().enumerate() {
            for (j, &q) in moduli.iter().enumerate() {
                let val_mod_q = if coeff >= 0 {
                    (coeff as u64) % q
                } else {
                    let abs_val = (-coeff) as u64;
                    q - (abs_val % q)
                };
                flat[i * num_primes + j] = val_mod_q;
            }
        }

        flat
    }

    /// Convert RNS representation vector to flat layout
    fn rns_vec_to_flat(&self, rns_vec: Vec<crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation>, moduli: &[u64]) -> Vec<u64> {
        let n = rns_vec.len();
        let num_primes = moduli.len();
        let mut flat = vec![0u64; n * num_primes];

        for (i, rns_rep) in rns_vec.iter().enumerate() {
            for j in 0..num_primes {
                flat[i * num_primes + j] = rns_rep.values[j];
            }
        }

        flat
    }

    /// Multiply two polynomials in flat RNS layout using NTT
    fn multiply_flat_rns(&self, a: &[u64], b: &[u64], moduli: &[u64]) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        let num_primes = moduli.len();
        let mut result = vec![0u64; n * num_primes];

        // For each RNS prime, perform NEGACYCLIC polynomial multiplication
        // IMPORTANT: CKKS requires negacyclic convolution in R[X]/(X^N + 1)
        // We use CUDA NTT for cyclic convolution, with CPU-side psi twisting for negacyclic
        for (prime_idx, &q) in moduli.iter().enumerate() {
            let ntt_ctx = &self.ntt_contexts[prime_idx];

            // Extract coefficients for this prime (strided layout)
            let mut a_prime = vec![0u64; n];
            let mut b_prime = vec![0u64; n];
            for i in 0..n {
                a_prime[i] = a[i * num_primes + prime_idx];
                b_prime[i] = b[i * num_primes + prime_idx];
            }

            // Use the SAME psi that was used to derive omega for this prime
            // CRITICAL: This ensures omega = psi^2, which makes the negacyclic trick work
            let psi = self.psi_per_prime[prime_idx];
            let psi_inv = self.psi_inv_per_prime[prime_idx];

            // Debug psi value for first prime
            if std::env::var("ENCRYPT_DEBUG").is_ok() && prime_idx == 0 {
                let psi_n = Self::pow_mod(psi, n as u64, q);
                let psi_2n = Self::pow_mod(psi, 2 * n as u64, q);
                println!("[ENCRYPT_DEBUG] prime[0]: psi={}, psi^N={} (q-1={}), psi^(2N)={}",
                    psi, psi_n, q - 1, psi_2n);
            }

            // TWIST: Multiply by psi^i to convert to negacyclic
            Self::apply_psi_powers(&mut a_prime, psi, q);
            Self::apply_psi_powers(&mut b_prime, psi, q);

            if std::env::var("ENCRYPT_DEBUG").is_ok() && prime_idx == 0 {
                println!("[ENCRYPT_DEBUG] After twist: a_prime[0]={}, b_prime[0]={}", a_prime[0], b_prime[0]);
            }

            // Forward NTT (cyclic)
            ntt_ctx.forward(&mut a_prime)?;
            ntt_ctx.forward(&mut b_prime)?;

            // Pointwise multiplication in NTT domain
            let mut product = vec![0u64; n];
            for i in 0..n {
                product[i] = ((a_prime[i] as u128 * b_prime[i] as u128) % q as u128) as u64;
            }

            // Inverse NTT (cyclic)
            ntt_ctx.inverse(&mut product)?;

            // UNTWIST: Multiply by psi^{-i} to get final negacyclic result
            Self::apply_psi_powers(&mut product, psi_inv, q);

            // Store back to strided layout
            for i in 0..n {
                result[i * num_primes + prime_idx] = product[i];
            }
        }

        Ok(result)
    }

    /// Compute psi (primitive 2N-th root of unity) from NTT root
    /// For negacyclic NTT: psi^(2N) = 1 and psi^N = -1 (mod q)
    fn compute_psi(n: usize, q: u64) -> Result<u64, String> {
        // Find a primitive 2N-th root of unity
        // For NTT-friendly primes, q ≡ 1 (mod 2N), so such roots exist
        let two_n = 2 * n as u64;

        if (q - 1) % two_n != 0 {
            return Err(format!("q-1 = {} not divisible by 2N = {}", q - 1, two_n));
        }

        // Try small generators to find primitive 2N-th root
        for g in 2..100u64 {
            let psi = Self::pow_mod(g, (q - 1) / two_n, q);

            // Verify: psi^(2N) = 1 and psi^N = -1 (mod q)
            let psi_2n = Self::pow_mod(psi, two_n, q);
            let psi_n = Self::pow_mod(psi, n as u64, q);

            if psi_2n == 1 && psi_n == q - 1 {
                return Ok(psi);
            }
        }

        Err(format!("Could not find primitive 2N-th root for n={}, q={}", n, q))
    }

    /// Apply psi powers: multiply poly[i] by psi^i for i=0..n
    fn apply_psi_powers(poly: &mut [u64], psi: u64, q: u64) {
        let mut psi_pow = 1u64;
        for i in 0..poly.len() {
            poly[i] = ((poly[i] as u128 * psi_pow as u128) % q as u128) as u64;
            psi_pow = ((psi_pow as u128 * psi as u128) % q as u128) as u64;
        }
    }

    /// Modular exponentiation: base^exp mod m
    fn pow_mod(base: u64, exp: u64, m: u64) -> u64 {
        let mut result = 1u64;
        let mut base = base % m;
        let mut exp = exp;

        while exp > 0 {
            if exp % 2 == 1 {
                result = ((result as u128 * base as u128) % m as u128) as u64;
            }
            base = ((base as u128 * base as u128) % m as u128) as u64;
            exp /= 2;
        }

        result
    }

    /// Modular inverse using extended Euclidean algorithm
    fn mod_inverse(a: u64, m: u64) -> Result<u64, String> {
        let (mut t, mut new_t) = (0i128, 1i128);
        let (mut r, mut new_r) = (m as i128, a as i128);

        while new_r != 0 {
            let quotient = r / new_r;
            (t, new_t) = (new_t, t - quotient * new_t);
            (r, new_r) = (new_r, r - quotient * new_r);
        }

        if r > 1 {
            return Err(format!("{} is not invertible mod {}", a, m));
        }
        if t < 0 {
            t += m as i128;
        }

        Ok(t as u64)
    }

    /// Find primitive 2N-th root of unity modulo q
    ///
    /// CRITICAL: This must match the CPU implementation in ntt.rs exactly,
    /// because keys are generated using CPU NTT and encryption/decryption
    /// use CUDA NTT. If different psi values are used, the negacyclic
    /// convolution results will be incompatible!
    fn find_primitive_root(n: usize, q: u64) -> Result<u64, String> {
        // For CKKS, we need a 2N-th primitive root
        // q - 1 must be divisible by 2N
        let two_n = (2 * n) as u64;
        if (q - 1) % two_n != 0 {
            return Err(format!("q-1 = {} is not divisible by 2N = {}", q - 1, two_n));
        }

        // MUST match CPU's candidate list exactly: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        // CPU's find_primitive_root in ntt.rs uses these candidates
        let candidates: [u64; 11] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];

        for &candidate in &candidates {
            // Check if candidate is a quadratic non-residue (same as CPU)
            // If g^((q-1)/2) == 1, it's NOT a generator - skip it
            if Self::mod_exp(candidate, (q - 1) / 2, q) == 1 {
                continue;
            }

            // Compute psi = g^((q-1)/2N) mod q
            let exp = (q - 1) / two_n;
            let psi = Self::mod_exp(candidate, exp, q);

            // Verify: psi^N = -1 (mod q)
            let psi_n = Self::mod_exp(psi, n as u64, q);
            if psi_n != q - 1 {
                continue;
            }

            // Verify: psi^(2N) = 1 (mod q)
            let psi_2n = Self::mod_exp(psi, two_n, q);
            if psi_2n == 1 {
                return Ok(psi);
            }
        }

        // Fallback: exhaustive search (matches CPU fallback)
        for candidate in 2..q {
            if Self::mod_exp(candidate, (q - 1) / 2, q) == 1 {
                continue; // Not a quadratic non-residue
            }

            let exp = (q - 1) / two_n;
            let psi = Self::mod_exp(candidate, exp, q);

            let psi_n = Self::mod_exp(psi, n as u64, q);
            if psi_n != q - 1 {
                continue;
            }

            let psi_2n = Self::mod_exp(psi, two_n, q);
            if psi_2n == 1 {
                return Ok(psi);
            }
        }

        Err(format!("Could not find primitive root for n={}, q={}", n, q))
    }

    /// Modular exponentiation: base^exp mod m
    fn mod_exp(base: u64, mut exp: u64, m: u64) -> u64 {
        let mut result = 1u64;
        let mut base = base % m;

        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % m as u128) as u64;
            }
            base = ((base as u128 * base as u128) % m as u128) as u64;
            exp >>= 1;
        }

        result
    }

    /// Precompute rescaling inverse table
    fn precompute_rescale_inv_table(moduli: &[u64]) -> Vec<Vec<u64>> {
        let mut table = Vec::new();

        for level in 1..moduli.len() {
            let q_last = moduli[level];
            let mut inv_row = Vec::new();

            for i in 0..level {
                let q_i = moduli[i];
                // Compute q_last^{-1} mod q_i
                let inv = Self::mod_inverse_static(q_last, q_i)
                    .expect("Modular inverse must exist");
                inv_row.push(inv);
            }

            table.push(inv_row);
        }

        table
    }

    /// Modular inverse using extended Euclidean algorithm
    fn mod_inverse_static(a: u64, m: u64) -> Option<u64> {
        let (mut t, mut new_t) = (0i128, 1i128);
        let (mut r, mut new_r) = (m as i128, a as i128);

        while new_r != 0 {
            let quotient = r / new_r;
            (t, new_t) = (new_t, t - quotient * new_t);
            (r, new_r) = (new_r, r - quotient * new_r);
        }

        if r > 1 {
            return None;
        }
        if t < 0 {
            t += m as i128;
        }

        Some(t as u64)
    }

    /// GPU-Native Exact Rescaling using CUDA kernel
    ///
    /// Rescale ciphertext by dropping last prime using exact DRLMQ with centered rounding.
    /// Uses Russian peasant multiplication for 128-bit modular arithmetic.
    pub fn exact_rescale_gpu(&self, poly_in: &[u64], level: usize) -> Result<Vec<u64>, String> {
        if level == 0 {
            return Err("Cannot rescale at level 0".to_string());
        }

        let n = self.params.n;
        let moduli = &self.params.moduli[..=level];
        let num_primes_in = moduli.len();
        let num_primes_out = num_primes_in - 1;

        assert_eq!(poly_in.len(), n * num_primes_in, "Input size mismatch");

        // Convert from strided layout to flat RNS layout for GPU (parallelized)
        // Strided: poly_in[coeff_idx * num_primes_in + prime_idx]
        // Flat:    poly_flat[prime_idx * n + coeff_idx]
        use rayon::prelude::*;
        let poly_flat: Vec<u64> = (0..num_primes_in)
            .into_par_iter()
            .flat_map(|prime_idx| {
                (0..n)
                    .map(|coeff_idx| poly_in[coeff_idx * num_primes_in + prime_idx])
                    .collect::<Vec<_>>()
            })
            .collect();

        // Get precomputed constants
        // Table is indexed from level-1 (since level 0 doesn't rescale)
        let qlast_inv = &self.rescale_inv_table[level - 1];
        assert_eq!(qlast_inv.len(), num_primes_out, "qlast_inv table size mismatch");

        // Copy to GPU
        let gpu_input = self.device.device.htod_copy(poly_flat)
            .map_err(|e| format!("Failed to copy input to GPU: {:?}", e))?;

        let mut gpu_output = self.device.device.alloc_zeros::<u64>(n * num_primes_out)
            .map_err(|e| format!("Failed to allocate output: {:?}", e))?;

        let gpu_moduli = self.device.device.htod_copy(moduli.to_vec())
            .map_err(|e| format!("Failed to copy moduli: {:?}", e))?;

        let gpu_qtop_inv = self.device.device.htod_copy(qlast_inv.clone())
            .map_err(|e| format!("Failed to copy qtop_inv: {:?}", e))?;

        // Get kernel function
        let func = self.device.device.get_func("rns_module", "rns_exact_rescale")
            .ok_or("Failed to get rns_exact_rescale function")?;

        // Launch kernel
        let config = self.device.get_launch_config(n);
        unsafe {
            func.launch(config, (
                &gpu_input,
                &mut gpu_output,
                &gpu_moduli,
                &gpu_qtop_inv,
                n as u32,
                num_primes_in as u32,
                num_primes_out as u32,
            )).map_err(|e| format!("Rescale kernel launch failed: {:?}", e))?;
        }

        // Copy result back
        let result_flat = self.device.device.dtoh_sync_copy(&gpu_output)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        // Convert back to strided layout
        let mut result = vec![0u64; n * num_primes_out];
        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes_out {
                result[coeff_idx * num_primes_out + prime_idx] = result_flat[prime_idx * n + coeff_idx];
            }
        }

        Ok(result)
    }

    /// Rescale polynomial in strided RNS layout (GPU-optimized, NO layout conversion!)
    ///
    /// This is MUCH faster than exact_rescale_gpu because it avoids expensive CPU layout conversions.
    /// Use this for V3 bootstrap operations where ciphertexts are in strided format.
    ///
    /// Input and output are in strided layout: poly[coeff_idx * num_primes + prime_idx]
    pub fn exact_rescale_gpu_strided(&self, poly_in: &[u64], level: usize) -> Result<Vec<u64>, String> {
        if level == 0 {
            return Err("Cannot rescale at level 0".to_string());
        }

        let n = self.params.n;
        let moduli = &self.params.moduli[..=level];
        let num_primes_in = moduli.len();
        let num_primes_out = num_primes_in - 1;

        assert_eq!(poly_in.len(), n * num_primes_in, "Input size mismatch");

        // NO LAYOUT CONVERSION NEEDED - input is already strided, output will be strided!

        // Get precomputed constants
        let qlast_inv = &self.rescale_inv_table[level - 1];
        assert_eq!(qlast_inv.len(), num_primes_out, "qlast_inv table size mismatch");

        // Copy to GPU in strided format
        let gpu_input = self.device.device.htod_copy(poly_in.to_vec())
            .map_err(|e| format!("Failed to copy input to GPU: {:?}", e))?;

        let mut gpu_output = self.device.device.alloc_zeros::<u64>(n * num_primes_out)
            .map_err(|e| format!("Failed to allocate output: {:?}", e))?;

        let gpu_moduli = self.device.device.htod_copy(moduli.to_vec())
            .map_err(|e| format!("Failed to copy moduli: {:?}", e))?;

        let gpu_qtop_inv = self.device.device.htod_copy(qlast_inv.clone())
            .map_err(|e| format!("Failed to copy qtop_inv: {:?}", e))?;

        // Get strided kernel function
        let func = self.device.device.get_func("rns_module", "rns_exact_rescale_strided")
            .ok_or("Failed to get rns_exact_rescale_strided function")?;

        // Launch kernel
        let config = self.device.get_launch_config(n);
        unsafe {
            func.launch(config, (
                &gpu_input,
                &mut gpu_output,
                &gpu_moduli,
                &gpu_qtop_inv,
                n as u32,
                num_primes_in as u32,
                num_primes_out as u32,
            )).map_err(|e| format!("Rescale kernel launch failed: {:?}", e))?;
        }

        // Copy result back - already in strided format!
        let result = self.device.device.dtoh_sync_copy(&gpu_output)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        Ok(result)
    }

    /// Rescale polynomial in flat RNS layout (for golden compare testing).
    /// Input and output are in flat layout: poly[prime_idx * n + coeff_idx]
    pub fn exact_rescale_gpu_flat(&self, poly_in: &[u64], level: usize) -> Result<Vec<u64>, String> {
        if level == 0 {
            return Err("Cannot rescale at level 0".to_string());
        }

        let n = self.params.n;
        let moduli = &self.params.moduli[..=level];
        let num_primes_in = moduli.len();
        let num_primes_out = num_primes_in - 1;

        assert_eq!(poly_in.len(), n * num_primes_in, "Input size mismatch");

        // Input is already in flat RNS layout - no conversion needed

        // Get precomputed constants
        // Table is indexed from level-1 (since level 0 doesn't rescale)
        let qlast_inv = &self.rescale_inv_table[level - 1];
        assert_eq!(qlast_inv.len(), num_primes_out, "qlast_inv table size mismatch");

        // Copy to GPU
        let gpu_input = self.device.device.htod_copy(poly_in.to_vec())
            .map_err(|e| format!("Failed to copy input to GPU: {:?}", e))?;

        let mut gpu_output = self.device.device.alloc_zeros::<u64>(n * num_primes_out)
            .map_err(|e| format!("Failed to allocate output: {:?}", e))?;

        let gpu_moduli = self.device.device.htod_copy(moduli.to_vec())
            .map_err(|e| format!("Failed to copy moduli: {:?}", e))?;

        let gpu_qtop_inv = self.device.device.htod_copy(qlast_inv.clone())
            .map_err(|e| format!("Failed to copy qtop_inv: {:?}", e))?;

        // Get kernel function
        let func = self.device.device.get_func("rns_module", "rns_exact_rescale")
            .ok_or("Failed to get rns_exact_rescale function")?;

        // Launch kernel
        let config = self.device.get_launch_config(n);
        unsafe {
            func.launch(config, (
                &gpu_input,
                &mut gpu_output,
                &gpu_moduli,
                &gpu_qtop_inv,
                n as u32,
                num_primes_in as u32,
                num_primes_out as u32,
            )).map_err(|e| format!("Rescale kernel launch failed: {:?}", e))?;
        }

        // Copy result back (already in flat layout)
        let result = self.device.device.dtoh_sync_copy(&gpu_output)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        Ok(result)
    }

    /// Encode floating-point values to polynomial (CPU operation)
    ///
    /// Uses the same Galois orbit-based canonical embedding as Metal backend and
    /// the decode function, ensuring encode/decode are inverses.
    pub fn encode(&self, values: &[f64], scale: f64, level: usize) -> Result<CudaPlaintext, String> {
        use std::f64::consts::PI;

        let n = self.params.n;
        let slots = n / 2;
        if values.len() > slots {
            return Err(format!("Too many values: {} (max {})", values.len(), slots));
        }

        let m = 2 * n; // Cyclotomic index M = 2N
        let g = 5; // Generator for power-of-two cyclotomics

        // Use Galois orbit order (same as decode)
        let e = Self::orbit_order(n, g);

        // Pad values to full slot count
        let mut slot_vals = vec![0.0; slots];
        for (i, &val) in values.iter().enumerate() {
            slot_vals[i] = val;
        }

        // Inverse canonical embedding using Galois orbit
        // Formula: c[j] = (2/N) * Σ_{t=0}^{N/2-1} z[t] * cos(2π * e[t] * j / M)
        let mut coeffs_float = vec![0.0; n];

        for j in 0..n {
            let mut sum = 0.0;

            for t in 0..slots {
                // w_t(j) = exp(-2πi * e[t] * j / M)
                let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
                let cos_val = angle.cos();

                // For real slots: contribution is 2 * z[t] * cos(angle)
                sum += slot_vals[t] * cos_val;
            }

            // Normalize by 2/N
            coeffs_float[j] = (2.0 / n as f64) * sum;
        }

        // Scale and round to integers, then reduce mod each prime
        let num_primes = level + 1;
        let mut poly = vec![0u64; n * num_primes];

        for coeff_idx in 0..n {
            let val = (coeffs_float[coeff_idx] * scale).round();
            for prime_idx in 0..num_primes {
                let q = self.params.moduli[prime_idx];
                let val_mod = if val >= 0.0 {
                    (val as u64) % q
                } else {
                    let abs_val = (-val) as u64;
                    q - (abs_val % q)
                };
                poly[coeff_idx * num_primes + prime_idx] = val_mod;
            }
        }

        Ok(CudaPlaintext {
            poly,
            n,
            num_primes,
            level: num_primes - 1,
            scale,
        })
    }

    /// Add two ciphertexts (CPU operation, simple)
    pub fn add(&self, ct1: &CudaCiphertext, ct2: &CudaCiphertext) -> Result<CudaCiphertext, String> {
        if ct1.level != ct2.level {
            return Err("Ciphertexts must be at same level".to_string());
        }

        // Use the actual num_primes from the input ciphertexts (not params.moduli.len())
        let num_active_primes = ct1.num_primes;
        let mut c0 = vec![0u64; self.params.n * num_active_primes];
        let mut c1 = vec![0u64; self.params.n * num_active_primes];

        for coeff_idx in 0..self.params.n {
            for prime_idx in 0..num_active_primes {
                let q = self.params.moduli[prime_idx];
                // Use num_active_primes (actual stride) instead of params.moduli.len()
                let idx = coeff_idx * num_active_primes + prime_idx;

                let sum0 = ct1.c0[idx] + ct2.c0[idx];
                c0[idx] = if sum0 >= q { sum0 - q } else { sum0 };

                let sum1 = ct1.c1[idx] + ct2.c1[idx];
                c1[idx] = if sum1 >= q { sum1 - q } else { sum1 };
            }
        }

        Ok(CudaCiphertext {
            c0,
            c1,
            n: self.params.n,
            num_primes: num_active_primes,
            level: ct1.level,
            scale: ct1.scale,  // Assuming same scale
        })
    }

    /// Get device
    pub fn device(&self) -> &Arc<CudaDeviceContext> {
        &self.device
    }

    /// Multiply two ciphertexts using GPU NTT-based polynomial multiplication
    ///
    /// Computes tensor product: (c0, c1) × (d0, d1) = (c0×d0, c0×d1 + c1×d0, c1×d1)
    ///
    /// Returns (c0_result, c1_result, c2_result) in flat RNS layout
    /// Caller is responsible for relinearization to reduce back to size-2 ciphertext
    pub fn multiply_ciphertexts_tensored(
        &self,
        ct1: &CudaCiphertext,
        ct2: &CudaCiphertext,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>), String> {
        // Use GPU-resident pipeline and download results only at the end
        let (gpu_c0, gpu_c1, gpu_c2) = self.multiply_ciphertexts_tensored_gpu(ct1, ct2)?;

        // Download results from GPU (3 transfers instead of 12!)
        let c0_result = self.device.device.dtoh_sync_copy(&gpu_c0)
            .map_err(|e| format!("Failed to download c0: {:?}", e))?;
        let c1_result = self.device.device.dtoh_sync_copy(&gpu_c1)
            .map_err(|e| format!("Failed to download c1: {:?}", e))?;
        let c2_result = self.device.device.dtoh_sync_copy(&gpu_c2)
            .map_err(|e| format!("Failed to download c2: {:?}", e))?;

        Ok((c0_result, c1_result, c2_result))
    }

    pub fn multiply_ciphertexts_tensored_OLD(
        &self,
        ct1: &CudaCiphertext,
        ct2: &CudaCiphertext,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>), String> {
        if ct1.level != ct2.level {
            return Err(format!("Level mismatch: {} vs {}", ct1.level, ct2.level));
        }
        if ct1.n != ct2.n {
            return Err(format!("Ring dimension mismatch: {} vs {}", ct1.n, ct2.n));
        }

        let n = ct1.n;
        let num_active_primes = ct1.level + 1;

        // Convert from strided to flat layout for NTT operations
        let c0_flat = self.strided_to_flat(&ct1.c0, n, ct1.num_primes, num_active_primes);
        let c1_flat = self.strided_to_flat(&ct1.c1, n, ct1.num_primes, num_active_primes);
        let d0_flat = self.strided_to_flat(&ct2.c0, n, ct2.num_primes, num_active_primes);
        let d1_flat = self.strided_to_flat(&ct2.c1, n, ct2.num_primes, num_active_primes);

        // ============================================================
        // BATCHED NTT OPERATIONS - Process all primes in parallel!
        // ============================================================
        // Old approach: 240 kernel launches per multiplication
        //   - 4 forward NTTs × 20 primes = 80 launches
        //   - 4 pointwise ops × 20 primes = 80 launches
        //   - 4 inverse NTTs × 20 primes = 80 launches
        //
        // New approach: ~13 kernel launches per multiplication (20× reduction!)
        //   - 4 batched forward NTTs = 4 launches
        //   - 4 batched pointwise ops = 4 launches
        //   - 4 batched inverse NTTs = 4 launches
        //   - 1 CPU addition operation
        // ============================================================

        // Make mutable copies for in-place NTT
        let mut c0_flat = c0_flat;
        let mut c1_flat = c1_flat;
        let mut d0_flat = d0_flat;
        let mut d1_flat = d1_flat;

        // Step 1: Forward NTT - ALL primes at once (4 launches instead of 80)
        self.ntt_forward_batched(&mut c0_flat, num_active_primes)?;  // 1 launch
        self.ntt_forward_batched(&mut c1_flat, num_active_primes)?;  // 1 launch
        self.ntt_forward_batched(&mut d0_flat, num_active_primes)?;  // 1 launch
        self.ntt_forward_batched(&mut d1_flat, num_active_primes)?;  // 1 launch

        // Step 2: Pointwise multiply - ALL primes at once (4 launches instead of 80)
        let mut c0_result = vec![0u64; n * num_active_primes];
        let mut c1_part1 = vec![0u64; n * num_active_primes];
        let mut c1_part2 = vec![0u64; n * num_active_primes];
        let mut c2_result = vec![0u64; n * num_active_primes];

        self.ntt_pointwise_multiply_batched(&c0_flat, &d0_flat, &mut c0_result, num_active_primes)?;  // 1 launch
        self.ntt_pointwise_multiply_batched(&c0_flat, &d1_flat, &mut c1_part1, num_active_primes)?;   // 1 launch
        self.ntt_pointwise_multiply_batched(&c1_flat, &d0_flat, &mut c1_part2, num_active_primes)?;   // 1 launch
        self.ntt_pointwise_multiply_batched(&c1_flat, &d1_flat, &mut c2_result, num_active_primes)?;  // 1 launch

        // Step 3: Inverse NTT - ALL primes at once (4 launches instead of 80)
        self.ntt_inverse_batched(&mut c0_result, num_active_primes)?;  // 1 launch
        self.ntt_inverse_batched(&mut c1_part1, num_active_primes)?;   // 1 launch
        self.ntt_inverse_batched(&mut c1_part2, num_active_primes)?;   // 1 launch
        self.ntt_inverse_batched(&mut c2_result, num_active_primes)?;  // 1 launch

        // Step 4: Add c1_part1 + c1_part2 for final c1 (CPU operation, could GPU-accelerate later)
        let mut c1_result = vec![0u64; n * num_active_primes];
        for prime_idx in 0..num_active_primes {
            let offset = prime_idx * n;
            let q = self.params.moduli[prime_idx];
            for i in 0..n {
                let sum = (c1_part1[offset + i] as u128 + c1_part2[offset + i] as u128) % q as u128;
                c1_result[offset + i] = sum as u64;
            }
        }

        // Total: 13 kernel launches (4 forward + 4 pointwise + 4 inverse + 1 CPU op)
        // vs 240 kernel launches in sequential version
        // Result: 20x reduction in kernel launch overhead

        Ok((c0_result, c1_result, c2_result))
    }

    /// GPU-RESIDENT multiplication - ALL operations stay on GPU!
    ///
    /// This version eliminates ALL CPU↔GPU data copying by:
    /// 1. Converting strided→flat on GPU
    /// 2. Performing all NTT operations on GPU
    /// 3. Returning GPU-resident results (CudaSlice)
    ///
    /// Expected savings: ~2-3 seconds from eliminating 576MB of PCIe transfers
    pub fn multiply_ciphertexts_tensored_gpu(
        &self,
        ct1: &CudaCiphertext,
        ct2: &CudaCiphertext,
    ) -> Result<(CudaSlice<u64>, CudaSlice<u64>, CudaSlice<u64>), String> {
        if ct1.level != ct2.level {
            return Err(format!("Level mismatch: {} vs {}", ct1.level, ct2.level));
        }
        if ct1.n != ct2.n {
            return Err(format!("Ring dimension mismatch: {} vs {}", ct1.n, ct2.n));
        }

        let n = ct1.n;
        let num_active_primes = ct1.level + 1;

        // Step 1: Convert strided→flat on CPU (extracts only active primes) AND upload to GPU
        // Note: strided_to_flat() does important work - it extracts only num_active_primes
        // from the full ct.c0/c1 arrays that contain ct.num_primes (30 total)
        let c0_flat = self.strided_to_flat(&ct1.c0, n, ct1.num_primes, num_active_primes);
        let c1_flat = self.strided_to_flat(&ct1.c1, n, ct1.num_primes, num_active_primes);
        let d0_flat = self.strided_to_flat(&ct2.c0, n, ct2.num_primes, num_active_primes);
        let d1_flat = self.strided_to_flat(&ct2.c1, n, ct2.num_primes, num_active_primes);

        // Upload to GPU ONCE (only active primes, not all 30!)
        let mut gpu_c0 = self.device.device.htod_copy(c0_flat)
            .map_err(|e| format!("Failed to upload c0: {:?}", e))?;
        let mut gpu_c1 = self.device.device.htod_copy(c1_flat)
            .map_err(|e| format!("Failed to upload c1: {:?}", e))?;
        let mut gpu_d0 = self.device.device.htod_copy(d0_flat)
            .map_err(|e| format!("Failed to upload d0: {:?}", e))?;
        let mut gpu_d1 = self.device.device.htod_copy(d1_flat)
            .map_err(|e| format!("Failed to upload d1: {:?}", e))?;

        // Step 2a: TWIST - apply psi^i for negacyclic convolution
        // CRITICAL: Without this, we get cyclic convolution instead of negacyclic!
        self.apply_negacyclic_twist_gpu(&mut gpu_c0, num_active_primes)?;
        self.apply_negacyclic_twist_gpu(&mut gpu_c1, num_active_primes)?;
        self.apply_negacyclic_twist_gpu(&mut gpu_d0, num_active_primes)?;
        self.apply_negacyclic_twist_gpu(&mut gpu_d1, num_active_primes)?;

        // Step 2b: Forward NTT - ALL on GPU!
        self.ntt_forward_batched_gpu(&mut gpu_c0, num_active_primes)?;
        self.ntt_forward_batched_gpu(&mut gpu_c1, num_active_primes)?;
        self.ntt_forward_batched_gpu(&mut gpu_d0, num_active_primes)?;
        self.ntt_forward_batched_gpu(&mut gpu_d1, num_active_primes)?;

        // Step 3: Pointwise multiply - ALL on GPU!
        let mut gpu_c0_result = self.device.device.alloc_zeros::<u64>(n * num_active_primes)
            .map_err(|e| format!("Failed to allocate gpu_c0_result: {:?}", e))?;
        let mut gpu_c1_part1 = self.device.device.alloc_zeros::<u64>(n * num_active_primes)
            .map_err(|e| format!("Failed to allocate gpu_c1_part1: {:?}", e))?;
        let mut gpu_c1_part2 = self.device.device.alloc_zeros::<u64>(n * num_active_primes)
            .map_err(|e| format!("Failed to allocate gpu_c1_part2: {:?}", e))?;
        let mut gpu_c2_result = self.device.device.alloc_zeros::<u64>(n * num_active_primes)
            .map_err(|e| format!("Failed to allocate gpu_c2_result: {:?}", e))?;

        self.ntt_pointwise_multiply_batched_gpu(&gpu_c0, &gpu_d0, &mut gpu_c0_result, num_active_primes)?;
        self.ntt_pointwise_multiply_batched_gpu(&gpu_c0, &gpu_d1, &mut gpu_c1_part1, num_active_primes)?;
        self.ntt_pointwise_multiply_batched_gpu(&gpu_c1, &gpu_d0, &mut gpu_c1_part2, num_active_primes)?;
        self.ntt_pointwise_multiply_batched_gpu(&gpu_c1, &gpu_d1, &mut gpu_c2_result, num_active_primes)?;

        // Step 4a: Inverse NTT - ALL on GPU!
        self.ntt_inverse_batched_gpu(&mut gpu_c0_result, num_active_primes)?;
        self.ntt_inverse_batched_gpu(&mut gpu_c1_part1, num_active_primes)?;
        self.ntt_inverse_batched_gpu(&mut gpu_c1_part2, num_active_primes)?;
        self.ntt_inverse_batched_gpu(&mut gpu_c2_result, num_active_primes)?;

        // Step 4b: UNTWIST - apply psi^{-i} to convert cyclic→negacyclic result
        // CRITICAL: Without this, the result is in the wrong polynomial ring!
        self.apply_negacyclic_untwist_gpu(&mut gpu_c0_result, num_active_primes)?;
        self.apply_negacyclic_untwist_gpu(&mut gpu_c1_part1, num_active_primes)?;
        self.apply_negacyclic_untwist_gpu(&mut gpu_c1_part2, num_active_primes)?;
        self.apply_negacyclic_untwist_gpu(&mut gpu_c2_result, num_active_primes)?;

        // Step 5: Add c1_part1 + c1_part2 on GPU using rns_add kernel
        let mut gpu_c1_result = self.device.device.alloc_zeros::<u64>(n * num_active_primes)
            .map_err(|e| format!("Failed to allocate gpu_c1_result: {:?}", e))?;

        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        let func_add = self.device.device.get_func("rns_module", "rns_add")
            .ok_or("Failed to get rns_add function")?;

        let threads_per_block = 256;
        let total_elements = n * num_active_primes;
        let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func_add.launch(cfg, (
                &gpu_c1_part1,
                &gpu_c1_part2,
                &mut gpu_c1_result,
                gpu_moduli,
                n as u32,
                num_active_primes as u32,
            ))
            .map_err(|e| format!("GPU addition failed: {:?}", e))?;
        }

        // Return GPU-resident results - NO final download!
        Ok((gpu_c0_result, gpu_c1_result, gpu_c2_result))
    }

    /// Helper: Convert from strided to flat RNS layout (GPU-accelerated!)
    ///
    /// This replaces ~650k CPU operations with a single GPU kernel call.
    /// CRITICAL for BSGS performance - called 4× per multiplication.
    pub fn strided_to_flat(&self, data: &[u64], n: usize, stride: usize, num_primes: usize) -> Vec<u64> {
        use cudarc::driver::LaunchAsync;

        let total_elements = n * num_primes;

        // Copy to GPU
        let gpu_input = self.device.device.htod_copy(data.to_vec())
            .expect("Failed to copy to GPU");

        let mut gpu_output = self.device.device.alloc_zeros::<u64>(total_elements)
            .expect("Failed to allocate GPU memory");

        // Get kernel
        let func = self.device.device.get_func("rns_module", "rns_strided_to_flat")
            .expect("Failed to get rns_strided_to_flat kernel");

        // Launch kernel
        let threads_per_block = 256;
        let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (&gpu_input, &mut gpu_output, n as u32, stride as u32, num_primes as u32))
                .expect("Failed to launch rns_strided_to_flat kernel");
        }

        // Copy result back
        self.device.device.dtoh_sync_copy(&gpu_output)
            .expect("Failed to copy from GPU")
    }

    /// Helper: Convert from flat to strided RNS layout (GPU-accelerated!)
    ///
    /// This is the inverse of strided_to_flat().
    /// Uses GPU kernel for parallel conversion - critical for rotation operations.
    ///
    /// Flat:    poly_in[prime_idx * n + coeff_idx]
    /// Strided: poly_out[coeff_idx * stride + prime_idx]
    pub fn flat_to_strided(&self, data: &[u64], n: usize, stride: usize, num_primes: usize) -> Vec<u64> {
        use cudarc::driver::LaunchAsync;

        let total_elements = n * num_primes;

        // Copy to GPU
        let gpu_input = self.device.device.htod_copy(data.to_vec())
            .expect("Failed to copy to GPU");

        let mut gpu_output = self.device.device.alloc_zeros::<u64>(n * stride)
            .expect("Failed to allocate GPU memory");

        // Get kernel
        let func = self.device.device.get_func("rns_module", "rns_flat_to_strided")
            .expect("Failed to get rns_flat_to_strided kernel");

        // Launch kernel
        let threads_per_block = 256;
        let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (&gpu_input, &mut gpu_output, n as u32, stride as u32, num_primes as u32))
                .expect("Failed to launch rns_flat_to_strided kernel");
        }

        // Copy result back
        self.device.device.dtoh_sync_copy(&gpu_output)
            .expect("Failed to copy from GPU")
    }

    /// Batched Forward NTT - Process all primes in parallel
    ///
    /// This replaces sequential per-prime NTT with a single batched operation,
    /// dramatically reducing kernel launch overhead (20× reduction in launches).
    ///
    /// # Arguments
    /// * `data` - Flat RNS layout: [num_primes * n] = all primes concatenated
    /// * `num_primes` - Number of RNS primes to process
    ///
    /// # Layout
    /// Input/Output: data[prime_idx * n + coeff_idx]
    fn ntt_forward_batched(&self, data: &mut [u64], num_primes: usize) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let log_n = (n as f64).log2() as usize;

        if data.len() != n * num_primes {
            return Err(format!("Expected {} elements, got {}", n * num_primes, data.len()));
        }

        // Use GPU-cached twiddles and moduli (uploaded once during initialization)
        let gpu_twiddles = self.gpu_twiddles_fwd.as_ref()
            .ok_or("GPU twiddles not initialized")?;
        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        // Copy input data to GPU
        let mut gpu_data = self.device.device.htod_copy(data.to_vec())
            .map_err(|e| format!("Failed to copy data to GPU: {:?}", e))?;

        // Batched bit-reversal permutation (single kernel launch for all primes!)
        let func_bit_reverse = self.ntt_contexts[0].device.device.get_func("ntt_module", "bit_reverse_permutation_batched")
            .ok_or("Failed to get bit_reverse_permutation_batched function")?;

        let threads_per_block = 256;
        let num_blocks = (n + threads_per_block - 1) / threads_per_block;  // Use n, not n/2!

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, num_primes as u32, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func_bit_reverse.launch(cfg, (&mut gpu_data, n as u32, log_n as u32, num_primes as u32))
                .map_err(|e| format!("Batched bit-reversal failed: {:?}", e))?;
        }

        // CRITICAL: Synchronize after bit-reversal before starting NTT stages
        self.device.device.synchronize()
            .map_err(|e| format!("Sync after bit-reversal failed: {:?}", e))?;

        // Batched NTT stages
        let mut m = 1usize;
        for stage in 0..log_n {
            let func_ntt = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_forward_batched")
                .ok_or("Failed to get ntt_forward_batched function")?;

            // 2D grid: (butterfly_blocks, num_primes)
            let threads_per_block = 256;
            let num_butterfly_blocks = (n / 2 + threads_per_block - 1) / threads_per_block;

            let cfg = LaunchConfig {
                grid_dim: (num_butterfly_blocks as u32, num_primes as u32, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func_ntt.launch(cfg, (
                    &mut gpu_data,
                    gpu_twiddles,
                    gpu_moduli,
                    n as u32,
                    num_primes as u32,
                    stage as u32,
                    m as u32,
                ))
                .map_err(|e| format!("Batched NTT stage {} failed: {:?}", stage, e))?;
            }

            // CRITICAL: Synchronize after each stage
            self.device.device.synchronize()
                .map_err(|e| format!("Sync after forward NTT stage {} failed: {:?}", stage, e))?;

            m *= 2;
        }

        // Copy result back
        let result = self.device.device.dtoh_sync_copy(&gpu_data)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        data.copy_from_slice(&result);
        Ok(())
    }

    /// Batched Inverse NTT - Process all primes in parallel
    ///
    /// # Arguments
    /// * `data` - Flat RNS layout: [num_primes * n]
    /// * `num_primes` - Number of RNS primes to process
    fn ntt_inverse_batched(&self, data: &mut [u64], num_primes: usize) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let log_n = (n as f64).log2() as usize;

        if data.len() != n * num_primes {
            return Err(format!("Expected {} elements, got {}", n * num_primes, data.len()));
        }

        // Use GPU-cached inverse twiddles and moduli
        let gpu_twiddles_inv = self.gpu_twiddles_inv.as_ref()
            .ok_or("GPU inverse twiddles not initialized")?;
        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        // Copy input data to GPU
        let mut gpu_data = self.device.device.htod_copy(data.to_vec())
            .map_err(|e| format!("Failed to copy data to GPU: {:?}", e))?;

        // Batched inverse NTT stages
        let mut m = n / 2;
        for stage in (0..log_n).rev() {
            let func_ntt_inv = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_inverse_batched")
                .ok_or("Failed to get ntt_inverse_batched function")?;

            // 2D grid: (butterfly_blocks, num_primes)
            let threads_per_block = 256;
            let num_butterfly_blocks = (n / 2 + threads_per_block - 1) / threads_per_block;

            let cfg = LaunchConfig {
                grid_dim: (num_butterfly_blocks as u32, num_primes as u32, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func_ntt_inv.launch(cfg, (
                    &mut gpu_data,
                    gpu_twiddles_inv,
                    gpu_moduli,
                    n as u32,
                    num_primes as u32,
                    stage as u32,
                    m as u32,
                ))
                .map_err(|e| format!("Batched inverse NTT stage {} failed: {:?}", stage, e))?;
            }

            // CRITICAL: Synchronize after each stage
            self.device.device.synchronize()
                .map_err(|e| format!("Sync after inverse NTT stage {} failed: {:?}", stage, e))?;

            m /= 2;
        }

        // Step 2: Bit-reversal + scaling by n^(-1) at the END
        // This is CRITICAL - the DIF algorithm produces output in bit-reversed order
        // Upload n_inv values for all primes
        let n_inv_values: Vec<u64> = (0..num_primes)
            .map(|i| self.ntt_contexts[i].n_inv)
            .collect();
        let gpu_n_inv = self.device.device.htod_copy(n_inv_values)
            .map_err(|e| format!("Failed to upload n_inv values: {:?}", e))?;

        let func_final = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_inverse_final_batched")
            .ok_or("Failed to get ntt_inverse_final_batched function")?;

        let threads_per_block = 256;
        let num_blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, num_primes as u32, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func_final.launch(cfg, (
                &mut gpu_data,
                &gpu_n_inv,
                gpu_moduli,
                n as u32,
                num_primes as u32,
                log_n as u32,
            ))
            .map_err(|e| format!("Inverse NTT final step failed: {:?}", e))?;
        }

        // Copy result back
        let result = self.device.device.dtoh_sync_copy(&gpu_data)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        data.copy_from_slice(&result);
        Ok(())
    }

    /// Batched Pointwise Multiplication - Process all primes in parallel
    ///
    /// Computes c = a ⊙ b (element-wise) for all RNS primes simultaneously.
    ///
    /// # Arguments
    /// * `a` - First operand in flat RNS layout: [num_primes * n]
    /// * `b` - Second operand in flat RNS layout: [num_primes * n]
    /// * `result` - Output in flat RNS layout: [num_primes * n]
    /// * `num_primes` - Number of RNS primes
    fn ntt_pointwise_multiply_batched(
        &self,
        a: &[u64],
        b: &[u64],
        result: &mut [u64],
        num_primes: usize,
    ) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let total_elements = n * num_primes;

        if a.len() != total_elements || b.len() != total_elements || result.len() != total_elements {
            return Err(format!("All arrays must have length {}", total_elements));
        }

        // Use GPU-cached moduli
        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        // Copy input arrays to GPU
        let gpu_a = self.device.device.htod_copy(a.to_vec())
            .map_err(|e| format!("Failed to copy a to GPU: {:?}", e))?;

        let gpu_b = self.device.device.htod_copy(b.to_vec())
            .map_err(|e| format!("Failed to copy b to GPU: {:?}", e))?;

        let mut gpu_c = self.device.device.alloc_zeros::<u64>(total_elements)
            .map_err(|e| format!("Failed to allocate result: {:?}", e))?;

        // Launch batched pointwise multiply kernel
        let func = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_pointwise_multiply_batched")
            .ok_or("Failed to get ntt_pointwise_multiply_batched function")?;

        // 2D grid: (coeff_blocks, num_primes)
        let threads_per_block = 256;
        let num_coeff_blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_coeff_blocks as u32, num_primes as u32, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &gpu_a,
                &gpu_b,
                &mut gpu_c,
                gpu_moduli,
                n as u32,
                num_primes as u32,
            ))
            .map_err(|e| format!("Batched pointwise multiply failed: {:?}", e))?;
        }

        // Copy result back
        let res = self.device.device.dtoh_sync_copy(&gpu_c)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        result.copy_from_slice(&res);
        Ok(())
    }

    // ============================================================
    // GPU-RESIDENT BATCHED NTT OPERATIONS
    // These methods work directly on GPU memory (CudaSlice)
    // to eliminate CPU↔GPU data copying overhead
    // ============================================================

    /// GPU-resident batched forward NTT
    ///
    /// Works directly on GPU memory - no CPU↔GPU copies!
    ///
    /// # Arguments
    /// * `gpu_data` - Mutable GPU slice in flat RNS layout [num_primes * n]
    /// * `num_primes` - Number of RNS primes to process
    pub fn ntt_forward_batched_gpu(&self, gpu_data: &mut CudaSlice<u64>, num_primes: usize) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let log_n = (n as f64).log2() as usize;

        if gpu_data.len() != n * num_primes {
            return Err(format!("Expected {} elements, got {}", n * num_primes, gpu_data.len()));
        }

        // Use GPU-cached twiddles and moduli
        let gpu_twiddles = self.gpu_twiddles_fwd.as_ref()
            .ok_or("GPU twiddles not initialized")?;
        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        // Batched bit-reversal permutation (single kernel launch for all primes!)
        let func_bit_reverse = self.ntt_contexts[0].device.device.get_func("ntt_module", "bit_reverse_permutation_batched")
            .ok_or("Failed to get bit_reverse_permutation_batched function")?;

        let threads_per_block = 256;
        let num_blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, num_primes as u32, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func_bit_reverse.launch(cfg, (&mut *gpu_data, n as u32, log_n as u32, num_primes as u32))
                .map_err(|e| format!("Batched bit-reversal failed: {:?}", e))?;
        }

        // CRITICAL: Synchronize after bit-reversal before starting NTT stages
        self.device.device.synchronize()
            .map_err(|e| format!("Sync after bit-reversal failed: {:?}", e))?;

        // Batched NTT stages
        let mut m = 1usize;
        for stage in 0..log_n {
            let func_ntt = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_forward_batched")
                .ok_or("Failed to get ntt_forward_batched function")?;

            let threads_per_block = 256;
            let num_butterfly_blocks = (n / 2 + threads_per_block - 1) / threads_per_block;

            let cfg = LaunchConfig {
                grid_dim: (num_butterfly_blocks as u32, num_primes as u32, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func_ntt.launch(cfg, (
                    &mut *gpu_data,
                    gpu_twiddles,
                    gpu_moduli,
                    n as u32,
                    num_primes as u32,
                    stage as u32,
                    m as u32,
                ))
                .map_err(|e| format!("Batched NTT stage {} failed: {:?}", stage, e))?;
            }

            // CRITICAL: Synchronize after each stage to ensure all threads complete
            // before the next stage reads/writes the same memory locations
            self.device.device.synchronize()
                .map_err(|e| format!("Sync after forward NTT stage {} failed: {:?}", stage, e))?;

            m *= 2;
        }

        Ok(())
    }

    /// GPU-resident batched inverse NTT (Gentleman-Sande DIF algorithm)
    ///
    /// Uses the same algorithm as the working single-prime ntt_inverse:
    /// 1. NO bit-reversal at the start
    /// 2. Butterfly stages in REVERSE order (log_n-1 down to 0)
    /// 3. Gentleman-Sande DIF butterfly: (u, v) -> (u + v, (u - v) * w)
    /// 4. Bit-reversal + scaling by n^(-1) at the END
    pub fn ntt_inverse_batched_gpu(&self, gpu_data: &mut CudaSlice<u64>, num_primes: usize) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let log_n = (n as f64).log2() as usize;

        if gpu_data.len() != n * num_primes {
            return Err(format!("Expected {} elements, got {}", n * num_primes, gpu_data.len()));
        }

        let gpu_twiddles_inv = self.gpu_twiddles_inv.as_ref()
            .ok_or("GPU inverse twiddles not initialized")?;
        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        // Step 1: Batched inverse NTT butterfly stages in REVERSE order (log_n-1 down to 0)
        for stage in (0..log_n).rev() {
            let func_ntt_inv = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_inverse_batched")
                .ok_or("Failed to get ntt_inverse_batched function")?;

            let m = 1usize << (stage + 1);  // m = 2^(stage+1), matching single-prime version
            let threads_per_block = 256;
            let num_butterfly_blocks = (n / 2 + threads_per_block - 1) / threads_per_block;

            let cfg = LaunchConfig {
                grid_dim: (num_butterfly_blocks as u32, num_primes as u32, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func_ntt_inv.launch(cfg, (
                    &mut *gpu_data,
                    gpu_twiddles_inv,
                    gpu_moduli,
                    n as u32,
                    num_primes as u32,
                    stage as u32,
                    m as u32,
                ))
                .map_err(|e| format!("Batched inverse NTT stage {} failed: {:?}", stage, e))?;
            }

            // CRITICAL: Synchronize after each stage to ensure all threads complete
            // before the next stage reads/writes the same memory locations
            self.device.device.synchronize()
                .map_err(|e| format!("Sync after inverse NTT stage {} failed: {:?}", stage, e))?;
        }

        // Step 2: Bit-reversal + scaling by n^(-1) at the END
        // Upload n_inv values for all primes
        let n_inv_values: Vec<u64> = (0..num_primes)
            .map(|i| self.ntt_contexts[i].n_inv)
            .collect();
        let gpu_n_inv = self.device.device.htod_copy(n_inv_values)
            .map_err(|e| format!("Failed to upload n_inv values: {:?}", e))?;

        let func_final = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_inverse_final_batched")
            .ok_or("Failed to get ntt_inverse_final_batched function")?;

        let threads_per_block = 256;
        let num_blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, num_primes as u32, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func_final.launch(cfg, (
                &mut *gpu_data,
                &gpu_n_inv,
                gpu_moduli,
                n as u32,
                num_primes as u32,
                log_n as u32,
            ))
            .map_err(|e| format!("Batched inverse NTT final step failed: {:?}", e))?;
        }

        Ok(())
    }

    /// GPU-resident batched pointwise multiplication
    pub fn ntt_pointwise_multiply_batched_gpu(
        &self,
        gpu_a: &CudaSlice<u64>,
        gpu_b: &CudaSlice<u64>,
        gpu_result: &mut CudaSlice<u64>,
        num_primes: usize,
    ) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let total_elements = n * num_primes;

        if gpu_a.len() != total_elements || gpu_b.len() != total_elements || gpu_result.len() != total_elements {
            return Err(format!("All arrays must have length {}", total_elements));
        }

        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        let func = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_pointwise_multiply_batched")
            .ok_or("Failed to get ntt_pointwise_multiply_batched function")?;

        let threads_per_block = 256;
        let num_coeff_blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_coeff_blocks as u32, num_primes as u32, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                gpu_a,
                gpu_b,
                gpu_result,
                gpu_moduli,
                n as u32,
                num_primes as u32,
            ))
            .map_err(|e| format!("Batched pointwise multiply failed: {:?}", e))?;
        }

        Ok(())
    }

    /// Apply negacyclic TWIST on GPU: multiply each coefficient by psi^i
    ///
    /// This converts polynomial multiplication in the standard ring to
    /// negacyclic convolution in R[X]/(X^N + 1), which is required for CKKS.
    ///
    /// # Arguments
    /// * `gpu_data` - Polynomial in FLAT layout: data[prime_idx * n + coeff_idx]
    /// * `num_primes` - Number of active RNS primes
    fn apply_negacyclic_twist_gpu(
        &self,
        gpu_data: &mut CudaSlice<u64>,
        num_primes: usize,
    ) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let total_elements = n * num_primes;

        if gpu_data.len() != total_elements {
            return Err(format!("Expected {} elements, got {}", total_elements, gpu_data.len()));
        }

        let gpu_psi_powers = self.gpu_psi_powers.as_ref()
            .ok_or("GPU psi powers not initialized")?;
        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        let func = self.device.device.get_func("rns_module", "rns_negacyclic_twist")
            .ok_or("Failed to get rns_negacyclic_twist function")?;

        let threads_per_block = 256;
        let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                gpu_data,
                gpu_psi_powers,
                gpu_moduli,
                n as u32,
                num_primes as u32,
            ))
            .map_err(|e| format!("Negacyclic twist failed: {:?}", e))?;
        }

        Ok(())
    }

    /// Apply negacyclic UNTWIST on GPU: multiply each coefficient by psi^{-i}
    ///
    /// This is the inverse of the twist operation, applied after inverse NTT
    /// to get the final negacyclic convolution result.
    ///
    /// # Arguments
    /// * `gpu_data` - Polynomial in FLAT layout: data[prime_idx * n + coeff_idx]
    /// * `num_primes` - Number of active RNS primes
    fn apply_negacyclic_untwist_gpu(
        &self,
        gpu_data: &mut CudaSlice<u64>,
        num_primes: usize,
    ) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let total_elements = n * num_primes;

        if gpu_data.len() != total_elements {
            return Err(format!("Expected {} elements, got {}", total_elements, gpu_data.len()));
        }

        let gpu_psi_inv_powers = self.gpu_psi_inv_powers.as_ref()
            .ok_or("GPU psi inverse powers not initialized")?;
        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        let func = self.device.device.get_func("rns_module", "rns_negacyclic_untwist")
            .ok_or("Failed to get rns_negacyclic_untwist function")?;

        let threads_per_block = 256;
        let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                gpu_data,
                gpu_psi_inv_powers,
                gpu_moduli,
                n as u32,
                num_primes as u32,
            ))
            .map_err(|e| format!("Negacyclic untwist failed: {:?}", e))?;
        }

        Ok(())
    }

    /// Public wrapper for apply_negacyclic_twist_gpu
    /// Used by relinearization to apply twist on GPU-resident data
    pub fn apply_negacyclic_twist_gpu_public(
        &self,
        gpu_data: &mut CudaSlice<u64>,
        num_primes: usize,
    ) -> Result<(), String> {
        self.apply_negacyclic_twist_gpu(gpu_data, num_primes)
    }

    /// Public wrapper for apply_negacyclic_untwist_gpu
    /// Used by relinearization to apply untwist on GPU-resident data
    pub fn apply_negacyclic_untwist_gpu_public(
        &self,
        gpu_data: &mut CudaSlice<u64>,
        num_primes: usize,
    ) -> Result<(), String> {
        self.apply_negacyclic_untwist_gpu(gpu_data, num_primes)
    }

    /// Convert strided layout to flat layout on GPU
    ///
    /// Strided: poly_in[coeff_idx * stride + prime_idx]
    /// Flat:    poly_out[prime_idx * n + coeff_idx]
    ///
    /// This eliminates CPU↔GPU transfers compared to the old strided_to_flat()
    /// which would download, convert on CPU, then upload again.
    fn strided_to_flat_gpu(
        &self,
        gpu_strided: &CudaSlice<u64>,
        gpu_flat: &mut CudaSlice<u64>,
        n: usize,
        stride: usize,
        num_primes: usize,
    ) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let func = self.device.device
            .get_func("rns_module", "rns_strided_to_flat")
            .ok_or("Failed to get rns_strided_to_flat kernel")?;

        let total_elements = n * num_primes;
        let threads_per_block = 256;
        let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (gpu_strided, gpu_flat, n as u32, stride as u32, num_primes as u32),
            )
            .map_err(|e| format!("strided_to_flat GPU kernel failed: {:?}", e))?;
        }

        Ok(())
    }

    /// Add two RNS polynomials using GPU kernel (flat layout)
    ///
    /// Computes c[i] = (a[i] + b[i]) % q for each RNS limb
    ///
    /// # Arguments
    /// * `a` - First polynomial in flat RNS layout [n × num_primes]
    /// * `b` - Second polynomial in flat RNS layout [n × num_primes]
    /// * `num_primes` - Number of RNS primes to use
    ///
    /// # Returns
    /// Result polynomial in flat RNS layout [n × num_primes]
    pub fn add_polynomials_gpu(&self, a: &[u64], b: &[u64], num_primes: usize) -> Result<Vec<u64>, String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let total_elements = n * num_primes;

        if a.len() < total_elements || b.len() < total_elements {
            return Err(format!(
                "Input polynomials too small: expected {}, got {} and {}",
                total_elements, a.len(), b.len()
            ));
        }

        // Copy inputs to GPU
        let a_gpu = self.device.device.htod_copy(a[..total_elements].to_vec())
            .map_err(|e| format!("Failed to copy a to GPU: {:?}", e))?;
        let b_gpu = self.device.device.htod_copy(b[..total_elements].to_vec())
            .map_err(|e| format!("Failed to copy b to GPU: {:?}", e))?;

        // Allocate output on GPU
        let c_gpu = self.device.device.alloc_zeros::<u64>(total_elements)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;

        // Copy moduli to GPU
        let moduli_gpu = self.device.device.htod_copy(self.params.moduli.clone())
            .map_err(|e| format!("Failed to copy moduli to GPU: {:?}", e))?;

        // Get kernel function
        let func = self.device.device.get_func("rns_module", "rns_add")
            .ok_or_else(|| "rns_add kernel not found".to_string())?;

        // Launch configuration
        let threads_per_block = 256;
        let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel: rns_add(a, b, c, moduli, n, num_primes)
        unsafe {
            func.launch(
                cfg,
                (
                    &a_gpu,
                    &b_gpu,
                    &c_gpu,
                    &moduli_gpu,
                    n as u32,
                    num_primes as u32,
                ),
            ).map_err(|e| format!("Failed to launch rns_add kernel: {:?}", e))?;
        }

        // Copy result back to CPU
        let result = self.device.device.dtoh_sync_copy(&c_gpu)
            .map_err(|e| format!("Failed to copy result from GPU: {:?}", e))?;

        Ok(result)
    }

    /// Subtract two polynomials in RNS representation using GPU
    ///
    /// Computes c[i] = (a[i] - b[i]) % q for each RNS limb
    ///
    /// Uses GPU rns_sub kernel for parallel computation.
    pub fn subtract_polynomials_gpu(&self, a: &[u64], b: &[u64], num_primes: usize) -> Result<Vec<u64>, String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let total_elements = n * num_primes;

        if a.len() < total_elements || b.len() < total_elements {
            return Err(format!(
                "Input polynomials too small: expected {}, got {} and {}",
                total_elements, a.len(), b.len()
            ));
        }

        // Copy inputs to GPU
        let a_gpu = self.device.device.htod_copy(a[..total_elements].to_vec())
            .map_err(|e| format!("Failed to copy a to GPU: {:?}", e))?;
        let b_gpu = self.device.device.htod_copy(b[..total_elements].to_vec())
            .map_err(|e| format!("Failed to copy b to GPU: {:?}", e))?;

        // Allocate output on GPU
        let c_gpu = self.device.device.alloc_zeros::<u64>(total_elements)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;

        // Use cached moduli
        let moduli_gpu = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not cached")?;

        // Get kernel function
        let func = self.device.device.get_func("rns_module", "rns_sub")
            .ok_or_else(|| "rns_sub kernel not found".to_string())?;

        // Launch configuration
        let threads_per_block = 256;
        let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            func.launch(
                cfg,
                (
                    &a_gpu,
                    &b_gpu,
                    &c_gpu,
                    moduli_gpu,
                    n as u32,
                    num_primes as u32,
                ),
            )
            .map_err(|e| format!("Failed to launch rns_sub kernel: {:?}", e))?;
        }

        // Copy result back to CPU
        let result = self.device.device.dtoh_sync_copy(&c_gpu)
            .map_err(|e| format!("Failed to copy result from GPU: {:?}", e))?;

        Ok(result)
    }

    /// Pointwise multiply two polynomials in strided RNS layout on GPU
    ///
    /// Uses GPU rns_pointwise_multiply_strided kernel with 128-bit modular multiplication.
    pub fn pointwise_multiply_polynomials_gpu_strided(
        &self,
        a: &[u64],
        b: &[u64],
        stride: usize,
        num_primes: usize,
    ) -> Result<Vec<u64>, String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;

        if a.len() < n * stride || b.len() < n * stride {
            return Err(format!(
                "Input polynomials too small: expected {}, got {} and {}",
                n * stride, a.len(), b.len()
            ));
        }

        // Copy inputs to GPU
        let a_gpu = self.device.device.htod_copy(a[..n * stride].to_vec())
            .map_err(|e| format!("Failed to copy a to GPU: {:?}", e))?;
        let b_gpu = self.device.device.htod_copy(b[..n * stride].to_vec())
            .map_err(|e| format!("Failed to copy b to GPU: {:?}", e))?;

        // Allocate output on GPU
        let c_gpu = self.device.device.alloc_zeros::<u64>(n * stride)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;

        // Use cached moduli
        let moduli_gpu = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not cached")?;

        // Get kernel function
        let func = self.device.device.get_func("rns_module", "rns_pointwise_multiply_strided")
            .ok_or_else(|| "rns_pointwise_multiply_strided kernel not found".to_string())?;

        // Launch configuration (one thread per coefficient)
        let threads_per_block = 256;
        let num_blocks = (n + threads_per_block - 1) / threads_per_block;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            func.launch(
                cfg,
                (
                    &a_gpu,
                    &b_gpu,
                    &c_gpu,
                    moduli_gpu,
                    n as u32,
                    stride as u32,
                    num_primes as u32,
                ),
            )
            .map_err(|e| format!("Failed to launch rns_pointwise_multiply_strided kernel: {:?}", e))?;
        }

        // Copy result back to CPU
        let result = self.device.device.dtoh_sync_copy(&c_gpu)
            .map_err(|e| format!("Failed to copy result from GPU: {:?}", e))?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires NVIDIA GPU
    fn test_cuda_ckks_context() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CudaCkksContext::new(params).expect("Failed to create CUDA CKKS context");

        assert_eq!(ctx.ntt_contexts.len(), ctx.params.moduli.len());
        println!("CUDA CKKS context created successfully");
    }

    #[test]
    #[ignore] // Requires NVIDIA GPU
    fn test_cuda_rescale() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CudaCkksContext::new(params.clone()).expect("Failed to create context");

        // Create test polynomial at level 2
        let n = params.n;
        let level = 2;
        let num_primes = level + 1;

        let mut poly = vec![0u64; n * num_primes];
        for i in 0..n {
            for j in 0..num_primes {
                poly[i * num_primes + j] = (i * 1000 + j) as u64 % params.moduli[j];
            }
        }

        // Rescale
        let result = ctx.exact_rescale_gpu(&poly, level).expect("Rescale failed");

        // Check result has correct size
        let expected_size = n * (num_primes - 1);
        assert_eq!(result.len(), expected_size);

        println!("CUDA rescale test passed");
    }
}

// Add missing decrypt, decode, and rescale_to_next methods
impl CudaCkksContext {
    /// Decrypt ciphertext using secret key
    ///
    /// Computes m = c0 + c1*s where s is the secret key
    pub fn decrypt(
        &self,
        ct: &CudaCiphertext,
        sk: &SecretKey,
    ) -> Result<CudaPlaintext, String> {
        let n = ct.n;
        let level = ct.level;
        let moduli = &self.params.moduli[..=level];
        let num_primes = moduli.len();

        // Convert secret key to strided layout at ciphertext's level
        let sk_strided = Self::rns_vec_to_flat_at_level(&sk.coeffs, level, num_primes);

        // Extract c1 in strided layout (truncate to active primes if needed)
        let c1_strided: Vec<u64> = if ct.num_primes == num_primes {
            ct.c1.clone()
        } else {
            // Truncate to only active primes
            let mut c1_active = vec![0u64; n * num_primes];
            for coeff_idx in 0..n {
                for prime_idx in 0..num_primes {
                    c1_active[coeff_idx * num_primes + prime_idx] =
                        ct.c1[coeff_idx * ct.num_primes + prime_idx];
                }
            }
            c1_active
        };

        // Compute c1 * s using NTT-based multiplication (both in strided layout)
        let c1s = self.multiply_polys_ntt(&c1_strided, &sk_strided, num_primes)?;

        // Extract c0 in strided layout
        let c0_strided: Vec<u64> = if ct.num_primes == num_primes {
            ct.c0.clone()
        } else {
            // Truncate to only active primes
            let mut c0_active = vec![0u64; n * num_primes];
            for coeff_idx in 0..n {
                for prime_idx in 0..num_primes {
                    c0_active[coeff_idx * num_primes + prime_idx] =
                        ct.c0[coeff_idx * ct.num_primes + prime_idx];
                }
            }
            c0_active
        };

        // m = c0 + c1*s (in strided layout)
        let mut m_strided = vec![0u64; n * num_primes];
        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let q = moduli[prime_idx];
                let idx = coeff_idx * num_primes + prime_idx;
                m_strided[idx] = ((c0_strided[idx] as u128 + c1s[idx] as u128) % q as u128) as u64;
            }
        }

        // Debug: print first coefficient
        if std::env::var("DECRYPT_DEBUG").is_ok() {
            println!("[DECRYPT_DEBUG CUDA] c0[0] prime 0: {}", c0_strided[0]);
            println!("[DECRYPT_DEBUG CUDA] c1[0] prime 0: {}", c1_strided[0]);
            println!("[DECRYPT_DEBUG CUDA] sk[0] prime 0: {}", sk_strided[0]);
            println!("[DECRYPT_DEBUG CUDA] c1s[0] prime 0: {}", c1s[0]);
            println!("[DECRYPT_DEBUG CUDA] m[0] prime 0: {}", m_strided[0]);
        }

        Ok(CudaPlaintext {
            poly: m_strided,
            n,
            num_primes,
            level,
            scale: ct.scale,
        })
    }

    /// Decode plaintext to recover encrypted values
    ///
    /// Extracts the real values from the plaintext polynomial coefficients
    pub fn decode(&self, pt: &CudaPlaintext) -> Result<Vec<f64>, String> {
        let n = pt.n;
        let num_primes = pt.num_primes;

        // Step 1: Reconstruct coefficients from RNS using first prime
        let q0 = self.params.moduli[0];
        let mut coeffs_i64 = vec![0i64; n];

        for i in 0..n {
            let val = pt.poly[i * num_primes]; // Use first RNS component

            // Convert from mod q to signed integer (centered representation)
            let half_q0 = q0 / 2;
            coeffs_i64[i] = if val > half_q0 {
                -((q0 - val) as i64)
            } else {
                val as i64
            };
        }

        // Step 2: Canonical embedding decode
        let slots = Self::canonical_embed_decode_real(&coeffs_i64, pt.scale, n);

        Ok(slots)
    }

    /// Public test wrapper for multiply_polys_ntt
    pub fn test_multiply_polys_ntt(&self, a: &[u64], b: &[u64], num_primes: usize) -> Result<Vec<u64>, String> {
        self.multiply_polys_ntt(a, b, num_primes)
    }

    // ============================================================
    // TEST WRAPPERS for debugging batched NTT operations
    // These expose the internal batched operations for comparison testing
    // ============================================================

    /// Test wrapper: Apply negacyclic twist to flat layout data
    pub fn apply_negacyclic_twist_flat(&self, data: &mut [u64], num_primes: usize) -> Result<(), String> {
        let n = self.params.n;
        let total_elements = n * num_primes;

        if data.len() != total_elements {
            return Err(format!("Expected {} elements, got {}", total_elements, data.len()));
        }

        // Upload to GPU
        let mut gpu_data = self.device.device.htod_copy(data.to_vec())
            .map_err(|e| format!("Failed to upload data: {:?}", e))?;

        // Apply twist
        self.apply_negacyclic_twist_gpu(&mut gpu_data, num_primes)?;

        // Download result
        let result = self.device.device.dtoh_sync_copy(&gpu_data)
            .map_err(|e| format!("Failed to download data: {:?}", e))?;

        data.copy_from_slice(&result);
        Ok(())
    }

    /// Test wrapper: Apply negacyclic untwist to flat layout data
    pub fn apply_negacyclic_untwist_flat(&self, data: &mut [u64], num_primes: usize) -> Result<(), String> {
        let n = self.params.n;
        let total_elements = n * num_primes;

        if data.len() != total_elements {
            return Err(format!("Expected {} elements, got {}", total_elements, data.len()));
        }

        // Upload to GPU
        let mut gpu_data = self.device.device.htod_copy(data.to_vec())
            .map_err(|e| format!("Failed to upload data: {:?}", e))?;

        // Apply untwist
        self.apply_negacyclic_untwist_gpu(&mut gpu_data, num_primes)?;

        // Download result
        let result = self.device.device.dtoh_sync_copy(&gpu_data)
            .map_err(|e| format!("Failed to download data: {:?}", e))?;

        data.copy_from_slice(&result);
        Ok(())
    }

    /// Test wrapper: Batched forward NTT on flat layout data
    pub fn ntt_forward_batched_flat(&self, data: &mut [u64], num_primes: usize) -> Result<(), String> {
        self.ntt_forward_batched(data, num_primes)
    }

    /// Test wrapper: Batched inverse NTT on flat layout data
    pub fn ntt_inverse_batched_flat(&self, data: &mut [u64], num_primes: usize) -> Result<(), String> {
        self.ntt_inverse_batched(data, num_primes)
    }

    /// Test wrapper: Batched pointwise multiply on flat layout data
    pub fn ntt_pointwise_multiply_batched_flat(
        &self,
        a: &[u64],
        b: &[u64],
        result: &mut [u64],
        num_primes: usize,
    ) -> Result<(), String> {
        self.ntt_pointwise_multiply_batched(a, b, result, num_primes)
    }

    /// NTT-based polynomial multiplication for negacyclic ring
    ///
    /// CRITICAL: Uses the SAME psi values stored in self.psi_per_prime that were
    /// used during encryption. This ensures consistent twist/untwist between
    /// encrypt (multiply_flat_rns) and decrypt (this function).
    ///
    /// Previously this function created new CPU NTT contexts which could find
    /// DIFFERENT primitive roots, causing encryption/decryption to fail!
    fn multiply_polys_ntt(&self, a: &[u64], b: &[u64], num_primes: usize) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        let mut result = vec![0u64; n * num_primes];

        // For each RNS prime, perform negacyclic multiplication using the SAME
        // psi values that were used during encryption
        for prime_idx in 0..num_primes {
            let q = self.params.moduli[prime_idx];
            let ntt_ctx = &self.ntt_contexts[prime_idx];

            // Use the SAME psi that was stored during context creation
            // CRITICAL: This must match what multiply_flat_rns uses in encrypt!
            let psi = self.psi_per_prime[prime_idx];
            let psi_inv = self.psi_inv_per_prime[prime_idx];

            // Extract polynomials for this prime (strided layout)
            let mut p1 = vec![0u64; n];
            let mut p2 = vec![0u64; n];
            for coeff_idx in 0..n {
                p1[coeff_idx] = a[coeff_idx * num_primes + prime_idx];
                p2[coeff_idx] = b[coeff_idx * num_primes + prime_idx];
            }

            // Debug first prime with more detail
            if std::env::var("NTT_DEBUG").is_ok() && prime_idx == 0 {
                println!("[NTT_DEBUG] modulus q={}", q);
                println!("[NTT_DEBUG] n={}", n);
                println!("[NTT_DEBUG] psi={}, psi_inv={}", psi, psi_inv);
                println!("[NTT_DEBUG] Before multiply: p1[0]={}, p1[1]={}, p2[0]={}, p2[1]={}",
                    p1[0], p1[1], p2[0], p2[1]);

                // Count non-zero coefficients in p2 (sk)
                let nonzero_count = p2.iter().filter(|&&x| x != 0).count();
                println!("[NTT_DEBUG] p2 (sk) non-zero coeffs: {} / {}", nonzero_count, n);

                // For ternary secret key, check how many are 1, q-1 (=-1), or other
                let ones = p2.iter().filter(|&&x| x == 1).count();
                let neg_ones = p2.iter().filter(|&&x| x == q - 1).count();
                let zeros = p2.iter().filter(|&&x| x == 0).count();
                let others = n - ones - neg_ones - zeros;
                println!("[NTT_DEBUG] p2 distribution: {} ones, {} neg-ones (q-1), {} zeros, {} others",
                    ones, neg_ones, zeros, others);
            }

            // TWIST: Multiply by psi^i to convert to negacyclic (same as multiply_flat_rns)
            Self::apply_psi_powers(&mut p1, psi, q);
            Self::apply_psi_powers(&mut p2, psi, q);

            // Forward NTT (cyclic)
            ntt_ctx.forward(&mut p1)?;
            ntt_ctx.forward(&mut p2)?;

            // Pointwise multiplication in NTT domain
            let mut product = vec![0u64; n];
            for i in 0..n {
                product[i] = ((p1[i] as u128 * p2[i] as u128) % q as u128) as u64;
            }

            // Inverse NTT (cyclic)
            ntt_ctx.inverse(&mut product)?;

            // UNTWIST: Multiply by psi^{-i} to get final negacyclic result
            Self::apply_psi_powers(&mut product, psi_inv, q);

            if std::env::var("NTT_DEBUG").is_ok() && prime_idx == 0 {
                println!("[NTT_DEBUG] After multiply: product[0]={}, product[1]={}", product[0], product[1]);
            }

            // Store result back in strided layout
            for coeff_idx in 0..n {
                result[coeff_idx * num_primes + prime_idx] = product[coeff_idx];
            }
        }

        Ok(result)
    }

    /// CPU polynomial multiplication for negacyclic ring (DEPRECATED - use multiply_polys_ntt)
    #[allow(dead_code)]
    fn multiply_polys_cpu_negacyclic(&self, a: &[u64], b: &[u64], moduli: &[u64]) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        let num_primes = moduli.len();
        let mut result = vec![0u64; n * num_primes];

        // For each prime
        for prime_idx in 0..num_primes {
            let q = moduli[prime_idx];

            // Schoolbook multiplication with negacyclic reduction
            for i in 0..n {
                for j in 0..n {
                    let a_val = a[i * num_primes + prime_idx];
                    let b_val = b[j * num_primes + prime_idx];
                    let prod = ((a_val as u128 * b_val as u128) % q as u128) as u64;

                    let k = (i + j) % (2 * n);
                    if k < n {
                        // Positive coefficient
                        let result_idx = k * num_primes + prime_idx;
                        result[result_idx] = ((result[result_idx] as u128 + prod as u128) % q as u128) as u64;
                    } else {
                        // Negative coefficient (x^n = -1)
                        let result_idx = (k - n) * num_primes + prime_idx;
                        result[result_idx] = if result[result_idx] >= prod {
                            result[result_idx] - prod
                        } else {
                            q - (prod - result[result_idx])
                        };
                    }
                }
            }
        }

        Ok(result)
    }

    /// Convert RNS vector to flat layout at specific level
    fn rns_vec_to_flat_at_level(
        rns_vec: &[crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation],
        level: usize,
        num_primes: usize,
    ) -> Vec<u64> {
        let n = rns_vec.len();
        let mut flat = vec![0u64; n * num_primes];

        for i in 0..n {
            for j in 0..num_primes {
                flat[i * num_primes + j] = rns_vec[i].values[j];
            }
        }

        flat
    }

    /// Canonical embedding decode (real values only)
    ///
    /// Performs forward canonical embedding to recover slot values from polynomial coefficients.
    /// Uses the same Galois orbit ordering as encode(), ensuring they are inverses.
    fn canonical_embed_decode_real(coeffs: &[i64], scale: f64, n: usize) -> Vec<f64> {
        use std::f64::consts::PI;

        let m = 2 * n; // M = 2N
        let num_slots = n / 2;
        let g = 5; // Generator

        // Compute Galois orbit order
        let e = Self::orbit_order(n, g);

        // Convert to floating point (with scale normalization)
        let coeffs_float: Vec<f64> = coeffs.iter().map(|&c| c as f64 / scale).collect();

        // Forward canonical embedding: evaluate polynomial at ζ_M^{e[t]} for t = 0..N/2-1
        // Formula: y_t = Σ_{j=0}^{N-1} c[j] * exp(+2πi * e[t] * j / M)
        // For real results, take real part
        let mut slots = vec![0.0; num_slots];

        for t in 0..num_slots {
            let mut sum_real = 0.0;
            for j in 0..n {
                // w_t(j) = exp(+2πi * e[t] * j / M)  (note: positive angle for decode)
                let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
                let cos_val = angle.cos();
                sum_real += coeffs_float[j] * cos_val;
            }
            slots[t] = sum_real;
        }

        slots
    }

    /// Compute Galois orbit order for canonical embedding
    fn orbit_order(n: usize, g: usize) -> Vec<usize> {
        let m = 2 * n; // M = 2N
        let num_slots = n / 2; // N/2 slots

        let mut e = vec![0usize; num_slots];
        let mut cur = 1usize;

        for t in 0..num_slots {
            e[t] = cur; // odd exponent in [1..2N-1]
            cur = (cur * g) % m;
        }

        e
    }
}

impl CudaCiphertext {
    /// Rescale ciphertext to next level (drop one prime from modulus chain)
    ///
    /// This is essential for CKKS to manage noise growth
    pub fn rescale_to_next(&self, ctx: &CudaCkksContext) -> Result<Self, String> {
        if self.level == 0 {
            return Err("Cannot rescale at level 0".to_string());
        }

        let n = self.n;
        let moduli_before = &ctx.params().moduli[..=self.level];
        let q_last = moduli_before[moduli_before.len() - 1];
        let new_level = self.level - 1;
        let num_primes_after = new_level + 1;

        // Convert to flat layout
        let c0_flat = ctx.strided_to_flat(&self.c0, n, self.num_primes, self.num_primes);
        let c1_flat = ctx.strided_to_flat(&self.c1, n, self.num_primes, self.num_primes);

        // Rescale using GPU
        let c0_rescaled = ctx.exact_rescale_gpu_flat(&c0_flat, self.level)?;
        let c1_rescaled = ctx.exact_rescale_gpu_flat(&c1_flat, self.level)?;

        // Convert back to strided layout
        let c0_strided = ctx.flat_to_strided(&c0_rescaled, n, num_primes_after, num_primes_after);
        let c1_strided = ctx.flat_to_strided(&c1_rescaled, n, num_primes_after, num_primes_after);

        // New scale after dividing by q_last
        let new_scale = self.scale / q_last as f64;

        Ok(Self {
            c0: c0_strided,
            c1: c1_strided,
            n,
            num_primes: num_primes_after,
            level: new_level,
            scale: new_scale,
        })
    }

    /// Mod-switch ciphertext to a lower level WITHOUT dividing by the dropped primes
    ///
    /// This is different from rescale:
    /// - **Rescale**: Divides coefficients by dropped prime, reduces scale
    /// - **Mod-switch**: Just truncates RNS representation, keeps scale the same
    ///
    /// Use mod-switch for level alignment before operations.
    /// Use rescale after multiplication to manage noise growth.
    ///
    /// # Arguments
    /// * `target_level` - Target level (must be < current level)
    ///
    /// # Returns
    /// Ciphertext at the target level with same scale
    pub fn mod_switch_to_level(&self, target_level: usize) -> Self {
        if target_level == self.level {
            return self.clone();
        }

        assert!(
            target_level < self.level,
            "Target level {} must be less than current level {}",
            target_level,
            self.level
        );

        let n = self.n;
        let old_num_primes = self.num_primes;
        let new_num_primes = target_level + 1;

        // Truncate c0: keep only first (target_level + 1) primes for each coefficient
        // CUDA uses strided layout: c[coeff_idx * num_primes + prime_idx]
        let mut new_c0 = vec![0u64; n * new_num_primes];
        for coeff_idx in 0..n {
            for prime_idx in 0..new_num_primes {
                let old_idx = coeff_idx * old_num_primes + prime_idx;
                let new_idx = coeff_idx * new_num_primes + prime_idx;
                new_c0[new_idx] = self.c0[old_idx];
            }
        }

        // Truncate c1: keep only first (target_level + 1) primes for each coefficient
        let mut new_c1 = vec![0u64; n * new_num_primes];
        for coeff_idx in 0..n {
            for prime_idx in 0..new_num_primes {
                let old_idx = coeff_idx * old_num_primes + prime_idx;
                let new_idx = coeff_idx * new_num_primes + prime_idx;
                new_c1[new_idx] = self.c1[old_idx];
            }
        }

        Self {
            c0: new_c0,
            c1: new_c1,
            n,
            num_primes: new_num_primes,
            level: target_level,
            scale: self.scale,  // Scale stays the same for mod_switch
        }
    }
}
