//! V2 Key Generation with NTT Optimization
//!
//! **Optimizations over V1:**
//! - Uses NTT for fast polynomial multiplication during key generation
//! - Barrett reduction for modular arithmetic
//! - Precomputed NTT contexts for efficiency
//!
//! **Performance Target:** 5× faster key generation vs V1

use crate::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::{BarrettReducer, RnsRepresentation};
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use rand::distributions::Distribution;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::Normal;
use rayon::prelude::*;
use std::cell::RefCell;

// Thread-local buffers and RNG for key generation
thread_local! {
    static KEYGEN_BUFFERS: RefCell<KeygenBuffers> = RefCell::new(KeygenBuffers::new());
    static THREAD_RNG: RefCell<ChaCha20Rng> = RefCell::new({
        // Create thread-specific seed using thread ID
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let thread_id = COUNTER.fetch_add(1, Ordering::Relaxed);

        // Mix with system entropy for security
        use rand::thread_rng as sys_rng;
        use rand::RngCore;
        let mut seed = [0u8; 32];
        sys_rng().fill_bytes(&mut seed[0..24]);
        seed[24..32].copy_from_slice(&thread_id.to_le_bytes());

        ChaCha20Rng::from_seed(seed)
    });
}

/// Preallocated buffers for key generation operations
struct KeygenBuffers {
    /// Temporary buffer for polynomial coefficients (per-prime)
    tmp_poly_a: Vec<u64>,
    tmp_poly_b: Vec<u64>,
    tmp_result: Vec<u64>,
    /// Size parameters (to detect when resize is needed)
    current_n: usize,
}

impl KeygenBuffers {
    fn new() -> Self {
        Self {
            tmp_poly_a: Vec::new(),
            tmp_poly_b: Vec::new(),
            tmp_result: Vec::new(),
            current_n: 0,
        }
    }

    fn ensure_capacity(&mut self, n: usize) {
        if self.current_n != n {
            self.tmp_poly_a.resize(n, 0);
            self.tmp_poly_b.resize(n, 0);
            self.tmp_result.resize(n, 0);
            self.current_n = n;
        }
    }
}

/// V2 Secret Key
#[derive(Clone, Debug)]
pub struct SecretKey {
    /// Secret polynomial coefficients in RNS form
    /// Ternary polynomial: coefficients in {-1, 0, 1}
    pub coeffs: Vec<RnsRepresentation>,

    /// Ring dimension
    pub n: usize,

    /// Current level
    pub level: usize,
}

impl SecretKey {
    /// Create new secret key
    pub fn new(coeffs: Vec<RnsRepresentation>, level: usize) -> Self {
        let n = coeffs.len();
        Self { coeffs, n, level }
    }

    /// Sample ternary secret key: coefficients in {-1, 0, 1}
    /// Uses fast thread-local ChaCha20 RNG for performance
    pub fn sample_ternary(params: &CliffordFHEParams) -> Self {
        let n = params.n;
        let level = params.max_level();
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();

        let coeffs = THREAD_RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let mut coeffs = Vec::with_capacity(n);

            for _ in 0..n {
                // Sample from {-1, 0, 1} with equal probability
                let val: f64 = rng.gen();
                let ternary = if val < 0.33 {
                    -1i64
                } else if val < 0.66 {
                    0i64
                } else {
                    1i64
                };

                // Convert to RNS representation
                let rns_values: Vec<u64> = moduli
                    .iter()
                    .map(|&q| {
                        if ternary >= 0 {
                            (ternary as u64) % q
                        } else {
                            // -1 mod q = q - 1
                            q - 1
                        }
                    })
                    .collect();

                coeffs.push(RnsRepresentation::new(rns_values, moduli.clone()));
            }

            coeffs
        });

        Self::new(coeffs, level)
    }
}

/// V2 Public Key
#[derive(Clone, Debug)]
pub struct PublicKey {
    /// First component (uniform random)
    pub a: Vec<RnsRepresentation>,

    /// Second component: b = -a*s - e
    pub b: Vec<RnsRepresentation>,

    /// Ring dimension
    pub n: usize,

    /// Current level
    pub level: usize,
}

impl PublicKey {
    /// Create new public key
    pub fn new(a: Vec<RnsRepresentation>, b: Vec<RnsRepresentation>, level: usize) -> Self {
        let n = a.len();
        assert_eq!(b.len(), n, "a and b must have same length");

        Self { a, b, n, level }
    }
}

/// V2 Evaluation Key (for relinearization)
#[derive(Clone, Debug)]
pub struct EvaluationKey {
    /// Digit width for gadget decomposition (e.g., w=20 means base B=2^20)
    pub base_w: u32,

    /// evk0[t] encrypts B^t · s^2 (one per digit)
    pub evk0: Vec<Vec<RnsRepresentation>>,

    /// evk1[t] = uniform randomness for evk0[t]
    pub evk1: Vec<Vec<RnsRepresentation>>,

    /// Ring dimension
    pub n: usize,

    /// Current level
    pub level: usize,
}

impl EvaluationKey {
    /// Create new evaluation key
    pub fn new(
        base_w: u32,
        evk0: Vec<Vec<RnsRepresentation>>,
        evk1: Vec<Vec<RnsRepresentation>>,
        level: usize,
    ) -> Self {
        let n = evk0[0].len();

        Self {
            base_w,
            evk0,
            evk1,
            n,
            level,
        }
    }
}

/// V2 Key Context with precomputed NTT transforms
#[derive(Debug)]
pub struct KeyContext {
    /// Parameters
    pub params: CliffordFHEParams,

    /// NTT contexts for each prime
    pub ntt_contexts: Vec<NttContext>,

    /// Barrett reducers for each prime
    pub reducers: Vec<BarrettReducer>,
}

impl KeyContext {
    /// Create new key context
    /// Parallelizes NTT context creation for faster initialization
    pub fn new(params: CliffordFHEParams) -> Self {
        let moduli = params.moduli.clone();

        println!("  [DEBUG] Creating NTT contexts for {} primes (parallelized)...", moduli.len());
        let start = std::time::Instant::now();

        // Parallelize NTT context creation (expensive: O(N log N) per prime)
        let ntt_contexts: Vec<NttContext> = moduli
            .par_iter()
            .map(|&q| NttContext::new(params.n, q))
            .collect();

        // DEBUG: Print CPU psi values for comparison with CUDA
        for (i, ctx) in ntt_contexts.iter().enumerate() {
            if i < 3 {
                println!("  [CPU] Prime {}: q={}, psi={}, omega={}", i, ctx.q, ctx.psi, ctx.omega);
            }
        }

        println!("  [DEBUG] NTT contexts created in {:.2}s", start.elapsed().as_secs_f64());

        // Barrett reducers are cheap, can stay sequential
        let reducers: Vec<BarrettReducer> = moduli
            .iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        Self {
            params,
            ntt_contexts,
            reducers,
        }
    }

    /// Generate secret, public, and evaluation keys
    ///
    /// # Returns
    /// (PublicKey, SecretKey, EvaluationKey)
    pub fn keygen(&self) -> (PublicKey, SecretKey, EvaluationKey) {
        use std::time::Instant;

        let n = self.params.n;
        let level = self.params.max_level();
        let moduli: Vec<u64> = self.params.moduli[..=level].to_vec();

        println!("  [DEBUG] Starting keygen for N={}, {} primes", n, moduli.len());

        // 1. Sample ternary secret key s ∈ {-1, 0, 1}^N
        let start = Instant::now();
        let sk = SecretKey::sample_ternary(&self.params);
        println!("  [DEBUG] Step 1/5: Secret key sampled in {:.2}s", start.elapsed().as_secs_f64());

        // 2. Sample uniform random polynomial a
        let start = Instant::now();
        let a = self.sample_uniform(&moduli);
        println!("  [DEBUG] Step 2/5: Uniform poly sampled in {:.2}s", start.elapsed().as_secs_f64());

        // 3. Sample error polynomial e from Gaussian distribution
        let start = Instant::now();
        let e = self.sample_error(&moduli);
        println!("  [DEBUG] Step 3/5: Error poly sampled in {:.2}s", start.elapsed().as_secs_f64());

        // 4. Compute b = -a*s + e using NTT multiplication
        let start = Instant::now();
        let a_times_s = self.multiply_polynomials(&a, &sk.coeffs, &moduli);
        let neg_a_times_s = self.negate_polynomial(&a_times_s, &moduli);
        let b = self.add_polynomials(&neg_a_times_s, &e);
        println!("  [DEBUG] Step 4/5: Public key computed in {:.2}s", start.elapsed().as_secs_f64());

        let pk = PublicKey::new(a, b, level);

        // 5. Generate evaluation key for relinearization
        println!("  [DEBUG] Step 5/5: Starting evaluation key generation...");
        let start = Instant::now();
        let evk = self.generate_evaluation_key(&sk, &moduli);
        println!("  [DEBUG] Step 5/5: Evaluation key generated in {:.2}s", start.elapsed().as_secs_f64());

        (pk, sk, evk)
    }

    /// Sample uniform random polynomial from R_q
    /// Uses fast thread-local ChaCha20 RNG for performance
    fn sample_uniform(&self, moduli: &[u64]) -> Vec<RnsRepresentation> {
        let n = self.params.n;

        THREAD_RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            (0..n)
                .map(|_| {
                    let values: Vec<u64> = moduli.iter().map(|&q| rng.gen_range(0..q)).collect();
                    RnsRepresentation::new(values, moduli.to_vec())
                })
                .collect()
        })
    }

    /// Sample error polynomial from Gaussian distribution χ
    /// Uses fast thread-local ChaCha20 RNG for performance
    fn sample_error(&self, moduli: &[u64]) -> Vec<RnsRepresentation> {
        let n = self.params.n;
        let error_std = self.params.error_std;

        let normal = Normal::new(0.0, error_std).expect("Invalid normal distribution parameters");

        THREAD_RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            (0..n)
                .map(|_| {
                    // Sample from Gaussian
                    let error_val = normal.sample(&mut *rng).round() as i64;

                    // Convert to RNS representation
                    let values: Vec<u64> = moduli
                        .iter()
                        .map(|&q| {
                            if error_val >= 0 {
                                (error_val as u64) % q
                            } else {
                                let abs_val = (-error_val) as u64;
                                let remainder = abs_val % q;
                                if remainder == 0 {
                                    0
                                } else {
                                    q - remainder
                                }
                            }
                        })
                        .collect();

                    RnsRepresentation::new(values, moduli.to_vec())
                })
                .collect()
        })
    }

    /// Multiply two polynomials using NTT (negacyclic convolution mod x^n + 1)
    ///
    /// Uses V2's fixed NTT implementation (twisted NTT for negacyclic convolution).
    /// **OPTIMIZED**:
    /// - Uses precomputed NTT contexts from self.ntt_contexts
    /// - Uses thread-local buffers to avoid allocations
    /// - Sequential over primes (parallelism at outer digit level)
    fn multiply_polynomials(
        &self,
        a: &[RnsRepresentation],
        b: &[RnsRepresentation],
        moduli: &[u64],
    ) -> Vec<RnsRepresentation> {
        let n = a.len();
        assert_eq!(b.len(), n);

        // Preallocate result storage
        let mut products_per_prime: Vec<Vec<u64>> = vec![vec![0u64; n]; moduli.len()];

        // Process each prime using thread-local buffers
        KEYGEN_BUFFERS.with(|bufs| {
            let mut bufs = bufs.borrow_mut();
            bufs.ensure_capacity(n);

            for (prime_idx, &_q) in moduli.iter().enumerate() {
                // Use precomputed NTT context (MAJOR SPEEDUP!)
                let ntt_ctx = &self.ntt_contexts[prime_idx];
                let q = moduli[prime_idx] as i64;

                // Extract coefficients into reusable buffers
                for i in 0..n {
                    let val_a = a[i].values[prime_idx] as i64;
                    bufs.tmp_poly_a[i] = ((val_a % q) + q) as u64 % q as u64;

                    let val_b = b[i].values[prime_idx] as i64;
                    bufs.tmp_poly_b[i] = ((val_b % q) + q) as u64 % q as u64;
                }

                // Multiply using precomputed NTT context
                let product = ntt_ctx.multiply_polynomials(&bufs.tmp_poly_a, &bufs.tmp_poly_b);

                // Copy result
                products_per_prime[prime_idx].copy_from_slice(&product);
            }
        });

        // Combine results into RNS representation
        (0..n)
            .map(|i| {
                let values: Vec<u64> = products_per_prime.iter().map(|prod| prod[i]).collect();
                RnsRepresentation::new(values, moduli.to_vec())
            })
            .collect()
    }

    /// Negate polynomial (compute -a mod q for each coefficient)
    fn negate_polynomial(
        &self,
        a: &[RnsRepresentation],
        moduli: &[u64],
    ) -> Vec<RnsRepresentation> {
        a.iter()
            .map(|rns| {
                let negated_values: Vec<u64> = rns
                    .values
                    .iter()
                    .zip(moduli)
                    .map(|(&val, &q)| if val == 0 { 0 } else { q - val })
                    .collect();
                RnsRepresentation::new(negated_values, moduli.to_vec())
            })
            .collect()
    }

    /// Add two polynomials
    fn add_polynomials(
        &self,
        a: &[RnsRepresentation],
        b: &[RnsRepresentation],
    ) -> Vec<RnsRepresentation> {
        a.iter().zip(b).map(|(x, y)| x.add(y)).collect()
    }

    /// Subtract two polynomials
    fn subtract_polynomials(
        &self,
        a: &[RnsRepresentation],
        b: &[RnsRepresentation],
    ) -> Vec<RnsRepresentation> {
        a.iter().zip(b).map(|(x, y)| x.sub(y)).collect()
    }

    /// Generate evaluation key for relinearization
    fn generate_evaluation_key(&self, sk: &SecretKey, moduli: &[u64]) -> EvaluationKey {
        let base_w = 20u32; // Use base 2^20 for gadget decomposition
        let n = self.params.n;

        // Compute s^2 using NTT
        let s_squared = self.multiply_polynomials(&sk.coeffs, &sk.coeffs, moduli);

        // Determine number of digits needed dynamically based on actual moduli
        // Compute Q = product of all primes, then count bits
        use num_bigint::BigInt;
        let q_prod_big: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();
        let q_bits = q_prod_big.bits() as u32;
        let num_digits = ((q_bits + base_w - 1) / base_w) as usize;

        // Sanity check: for 3 primes of ~60 bits, Q ~ 2^180, so num_digits ~ 9
        // For reference: (60+40+40) bits = 140 bits → 7 digits, 180 bits → 9 digits

        let mut evk0 = Vec::with_capacity(num_digits);
        let mut evk1 = Vec::with_capacity(num_digits);

        // Precompute B^t mod q for each prime and each digit (to avoid overflow)
        let base = 1u64 << base_w;
        let mut bpow_t_mod_q = vec![vec![0u64; moduli.len()]; num_digits];
        for (j, &q) in moduli.iter().enumerate() {
            let q_u128 = q as u128;
            let mut p = 1u128;
            for t in 0..num_digits {
                bpow_t_mod_q[t][j] = (p % q_u128) as u64;
                p = (p * (base as u128)) % q_u128;
            }
        }

        // Parallelize evaluation key generation for each digit
        let evk_pairs: Vec<(Vec<RnsRepresentation>, Vec<RnsRepresentation>)> = (0..num_digits)
            .into_par_iter()
            .map(|t| {
                // Compute B^t * s^2 using precomputed B^t mod q for each prime
                let bt_s2: Vec<RnsRepresentation> = s_squared
                    .iter()
                    .map(|rns| {
                        let values: Vec<u64> = rns.values.iter().enumerate()
                            .map(|(j, &val)| {
                                let q = moduli[j];
                                let bt_mod_q = bpow_t_mod_q[t][j];
                                // Compute val * B^t mod q
                                ((val as u128) * (bt_mod_q as u128) % (q as u128)) as u64
                            })
                            .collect();
                        RnsRepresentation::new(values, moduli.to_vec())
                    })
                    .collect();

                // Sample uniform a_t
                let a_t = self.sample_uniform(moduli);

                // Sample error e_t
                let e_t = self.sample_error(moduli);

                // Compute evk0[t] = -B^t*s^2 + a_t*s + e_t
                // This ensures: evk0[t] - evk1[t]*s = -B^t*s^2 + e_t
                let a_t_times_s = self.multiply_polynomials(&a_t, &sk.coeffs, moduli);
                let neg_bt_s2 = self.negate_polynomial(&bt_s2, moduli);
                let temp = self.add_polynomials(&neg_bt_s2, &a_t_times_s);
                let b_t: Vec<RnsRepresentation> = temp
                    .iter()
                    .zip(&e_t)
                    .map(|(x, y)| x.add(y))
                    .collect();

                (b_t, a_t)
            })
            .collect();

        // Unpack the parallel results
        for (b_t, a_t) in evk_pairs {
            evk0.push(b_t);
            evk1.push(a_t);
        }

        EvaluationKey::new(base_w, evk0, evk1, sk.level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_key_sampling() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let sk = SecretKey::sample_ternary(&params);

        assert_eq!(sk.n, params.n);
        assert_eq!(sk.level, params.max_level());
        assert_eq!(sk.coeffs.len(), params.n);

        // Check all coefficients are ternary (in {0, 1, q-1} for each prime)
        for rns in &sk.coeffs {
            for (i, &val) in rns.values.iter().enumerate() {
                let q = rns.moduli[i];
                assert!(
                    val == 0 || val == 1 || val == q - 1,
                    "Non-ternary coefficient: {} (q={})",
                    val,
                    q
                );
            }
        }
    }

    #[test]
    fn test_key_context_creation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = KeyContext::new(params.clone());

        assert_eq!(ctx.ntt_contexts.len(), params.moduli.len());
        assert_eq!(ctx.reducers.len(), params.moduli.len());
    }

    #[test]
    fn test_keygen() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = KeyContext::new(params.clone());

        let (pk, sk, evk) = ctx.keygen();

        // Check dimensions
        assert_eq!(pk.n, params.n);
        assert_eq!(sk.n, params.n);
        assert_eq!(evk.n, params.n);

        // Check levels
        assert_eq!(pk.level, params.max_level());
        assert_eq!(sk.level, params.max_level());
        assert_eq!(evk.level, params.max_level());

        // Check public key has two components
        assert_eq!(pk.a.len(), params.n);
        assert_eq!(pk.b.len(), params.n);

        // Check evaluation key has correct number of digits
        assert!(evk.evk0.len() > 0);
        assert_eq!(evk.evk0.len(), evk.evk1.len());
    }

    #[test]
    fn test_polynomial_multiplication() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = KeyContext::new(params.clone());
        let moduli: Vec<u64> = params.moduli[..=params.max_level()].to_vec();

        // Create simple polynomials for testing
        let mut a = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
        a[0] = RnsRepresentation::from_u64(1, &moduli);

        let mut b = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
        b[0] = RnsRepresentation::from_u64(1, &moduli);

        // Test that multiplication works (a*b should give some result)
        let result = ctx.multiply_polynomials(&a, &b, &moduli);

        // Just verify we got a result with correct dimensions
        assert_eq!(result.len(), params.n);

        // Test commutativity: a*b should equal b*a
        let result_ba = ctx.multiply_polynomials(&b, &a, &moduli);
        for i in 0..params.n {
            assert_eq!(result[i].values[0], result_ba[i].values[0]);
        }
    }

    #[test]
    fn test_negate_polynomial() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = KeyContext::new(params.clone());
        let moduli: Vec<u64> = params.moduli[..=params.max_level()].to_vec();

        let a = vec![RnsRepresentation::from_u64(5, &moduli); params.n];
        let neg_a = ctx.negate_polynomial(&a, &moduli);

        // For each prime q: -5 mod q = q - 5
        for (i, rns) in neg_a.iter().enumerate() {
            for (j, &val) in rns.values.iter().enumerate() {
                let q = moduli[j];
                assert_eq!(val, q - 5, "Negation failed at index {}, prime {}", i, j);
            }
        }
    }
}
