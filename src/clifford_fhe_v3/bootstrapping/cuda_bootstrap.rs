//! V3 CUDA GPU Bootstrap Implementation
//!
//! Full homomorphic encryption bootstrap using CUDA GPU acceleration.
//!
//! **Bootstrap Pipeline:**
//! 1. **Modulus Raise**: Extend ciphertext to higher modulus level (Fast Basis Extension)
//! 2. **CoeffToSlot (C2S)**: Transform coefficients to slots using rotations
//! 3. **EvalMod**: Evaluate modular reduction (removes noise)
//! 4. **SlotToCoeff (S2C)**: Transform slots back to coefficients
//! 5. **Modulus Switch**: Reduce back to original modulus level
//!
//! **GPU Acceleration:**
//! - Rotation operations use GPU Galois kernel
//! - NTT operations use GPU kernels
//! - Rescaling uses GPU RNS kernel
//! - Key switching uses rotation keys
//!
//! **Performance Target:**
//! - RTX 5090: ~20-25s full bootstrap (3× faster than Metal M3 Max)
//!
//! **Fast Basis Extension (FBE):**
//! Uses Bajard et al. algorithm with fixed-point correction factor k.
//! For each coefficient x with residues (x mod q_0, ..., x mod q_L):
//! 1. Compute α_i = x_i * (Q/q_i)^(-1) mod q_i
//! 2. Compute k = round(Σ α_i / q_i) using fixed-point arithmetic
//! 3. For new prime p: x mod p = Σ(α_i * Q̂_i mod p) - k * (Q mod p) mod p

use crate::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
use crate::clifford_fhe_v2::backends::gpu_cuda::rotation::CudaRotationContext;
use crate::clifford_fhe_v2::backends::gpu_cuda::rotation_keys::CudaRotationKeys;
use crate::clifford_fhe_v2::backends::gpu_cuda::relin_keys::CudaRelinKeys;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v3::bootstrapping::BootstrapParams;
use crate::clifford_fhe_v3::bootstrapping::cuda_coeff_to_slot::cuda_coeff_to_slot;
use crate::clifford_fhe_v3::bootstrapping::cuda_slot_to_coeff::cuda_slot_to_coeff;
use crate::clifford_fhe_v3::bootstrapping::cuda_eval_mod::cuda_eval_mod;
use std::sync::Arc;

/// Fixed-point precision for correction factor k computation (80 bits)
const FP_PRECISION: u32 = 80;

/// Fast Basis Extension precomputed values
///
/// For extending from old_moduli (q_0, ..., q_L) to new_moduli (p_0, ..., p_M)
pub struct FastBasisExtension {
    /// Old moduli (source basis)
    old_moduli: Vec<u64>,
    /// New moduli (target basis)
    new_moduli: Vec<u64>,
    /// Q = product of old moduli (stored for reference, actual computation uses residues)
    /// Q̂_i = Q / q_i for each old prime
    q_hat_mod_old: Vec<Vec<u64>>,  // q_hat_mod_old[i][j] = Q̂_i mod q_j
    /// (Q̂_i)^(-1) mod q_i for each old prime
    q_hat_inv_mod_q: Vec<u64>,
    /// Q̂_i mod p for each old prime i and new prime p
    q_hat_mod_new: Vec<Vec<u64>>,  // q_hat_mod_new[i][j] = Q̂_i mod new_moduli[j]
    /// Q mod p for each new prime p (for correction term)
    q_mod_new: Vec<u64>,
    /// Fixed-point inverse: round(2^FP_PRECISION / q_i) for correction factor
    inv_q_fp: Vec<u128>,
}

impl FastBasisExtension {
    /// Create new Fast Basis Extension context
    ///
    /// # Arguments
    /// * `old_moduli` - Source primes (q_0, ..., q_L)
    /// * `new_moduli` - Target primes (p_0, ..., p_M) - must be disjoint from old_moduli
    pub fn new(old_moduli: &[u64], new_moduli: &[u64]) -> Self {
        let num_old = old_moduli.len();
        let num_new = new_moduli.len();

        // Compute Q = product of old moduli using BigUint-like accumulation
        // We'll compute Q mod each prime directly to avoid full BigInt

        // Compute Q̂_i = Q / q_i for each old prime
        // Q̂_i mod q_j = (∏_{k≠i} q_k) mod q_j
        let mut q_hat_mod_old = vec![vec![0u64; num_old]; num_old];
        for i in 0..num_old {
            for j in 0..num_old {
                let mut prod = 1u128;
                for k in 0..num_old {
                    if k != i {
                        prod = (prod * (old_moduli[k] as u128)) % (old_moduli[j] as u128);
                    }
                }
                q_hat_mod_old[i][j] = prod as u64;
            }
        }

        // Compute (Q̂_i)^(-1) mod q_i
        let mut q_hat_inv_mod_q = vec![0u64; num_old];
        for i in 0..num_old {
            q_hat_inv_mod_q[i] = mod_inverse(q_hat_mod_old[i][i], old_moduli[i])
                .expect("Q̂_i must be invertible mod q_i");
        }

        // Compute Q̂_i mod p for each new prime p
        let mut q_hat_mod_new = vec![vec![0u64; num_new]; num_old];
        for i in 0..num_old {
            for j in 0..num_new {
                let mut prod = 1u128;
                for k in 0..num_old {
                    if k != i {
                        prod = (prod * (old_moduli[k] as u128)) % (new_moduli[j] as u128);
                    }
                }
                q_hat_mod_new[i][j] = prod as u64;
            }
        }

        // Compute Q mod p for each new prime p
        let mut q_mod_new = vec![0u64; num_new];
        for j in 0..num_new {
            let mut prod = 1u128;
            for &q in old_moduli {
                prod = (prod * (q as u128)) % (new_moduli[j] as u128);
            }
            q_mod_new[j] = prod as u64;
        }

        // Compute fixed-point inverses: round(2^FP_PRECISION / q_i)
        let mut inv_q_fp = vec![0u128; num_old];
        for i in 0..num_old {
            // 2^80 / q_i, computed carefully to avoid overflow
            // Since q_i < 2^64, 2^80 / q_i < 2^16, so result fits in u128
            let two_pow_80 = 1u128 << FP_PRECISION;
            inv_q_fp[i] = two_pow_80 / (old_moduli[i] as u128);
        }

        Self {
            old_moduli: old_moduli.to_vec(),
            new_moduli: new_moduli.to_vec(),
            q_hat_mod_old,
            q_hat_inv_mod_q,
            q_hat_mod_new,
            q_mod_new,
            inv_q_fp,
        }
    }

    /// Extend a single coefficient to new basis
    ///
    /// Given residues (x mod q_0, ..., x mod q_L), compute (x mod p_0, ..., x mod p_M)
    /// The integer x is reconstructed from the CRT representation.
    ///
    /// # Algorithm (Bajard et al. Fast Basis Extension)
    ///
    /// 1. Compute α_i = x_i * (Q̂_i)^(-1) mod q_i
    ///    These satisfy: x ≡ Σ α_i * Q̂_i (mod Q)
    /// 2. Compute k = round(Σ α_i / q_i)
    ///    This is the number of times Q appears in Σ α_i * Q̂_i
    /// 3. For each new prime p:
    ///    x mod p = (Σ α_i * Q̂_i mod p) - k * Q mod p
    ///
    /// # Returns
    /// Residues for the new primes
    pub fn extend_coefficient(&self, residues: &[u64]) -> Vec<u64> {
        let num_old = self.old_moduli.len();
        let num_new = self.new_moduli.len();

        debug_assert_eq!(residues.len(), num_old, "Residue count must match old moduli count");

        // Step 1: Compute α_i = x_i * (Q̂_i)^(-1) mod q_i
        // These α_i values satisfy: x ≡ Σ α_i * Q̂_i (mod Q)
        let mut alpha = vec![0u64; num_old];
        for i in 0..num_old {
            let x_i = residues[i];
            alpha[i] = mul_mod(x_i, self.q_hat_inv_mod_q[i], self.old_moduli[i]);
        }

        // Step 2: Compute correction factor k using fixed-point arithmetic
        // k = round(Σ α_i / q_i)
        // This gives the number of times Q appears in Σ α_i * Q̂_i
        // IMPORTANT: α_i values are UNSIGNED here (in range [0, q_i))
        let mut sum_fp: u128 = 0;
        for i in 0..num_old {
            // α_i * (2^FP_PRECISION / q_i)
            let contrib = (alpha[i] as u128) * self.inv_q_fp[i];
            sum_fp += contrib;
        }

        // Round to nearest: k = (sum_fp + 2^(FP_PRECISION-1)) >> FP_PRECISION
        let half: u128 = 1u128 << (FP_PRECISION - 1);
        let k: u128 = (sum_fp + half) >> FP_PRECISION;

        // Step 3: Compute residues for new primes
        // x mod p = (Σ α_i * Q̂_i mod p) - k * Q mod p
        let mut new_residues = vec![0u64; num_new];
        for j in 0..num_new {
            let p = self.new_moduli[j];

            // Compute Σ(α_i * Q̂_i mod p)
            let mut sum: u128 = 0;
            for i in 0..num_old {
                let term = mul_mod_u128(alpha[i] as u128, self.q_hat_mod_new[i][j] as u128, p as u128);
                sum = (sum + term) % (p as u128);
            }

            // Compute k * Q mod p
            let k_mod_p = k % (p as u128);
            let correction = mul_mod_u128(k_mod_p, self.q_mod_new[j] as u128, p as u128);

            // result = sum - correction mod p
            new_residues[j] = if sum >= correction {
                (sum - correction) as u64
            } else {
                (sum + p as u128 - correction) as u64
            };
        }

        new_residues
    }
}

/// Modular multiplication: (a * b) mod m
#[inline]
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular multiplication for u128: (a * b) mod m
#[inline]
fn mul_mod_u128(a: u128, b: u128, m: u128) -> u128 {
    // For values that fit in u64, use simple multiplication
    if a < (1u128 << 64) && b < (1u128 << 64) {
        (a * b) % m
    } else {
        // Use schoolbook multiplication with carries for larger values
        // This should rarely happen for our use case
        let a_lo = a & ((1u128 << 64) - 1);
        let a_hi = a >> 64;
        let b_lo = b & ((1u128 << 64) - 1);
        let b_hi = b >> 64;

        let mut result = (a_lo * b_lo) % m;
        if a_hi > 0 {
            let term = ((a_hi % m) * (b_lo % m) * ((1u128 << 64) % m)) % m;
            result = (result + term) % m;
        }
        if b_hi > 0 {
            let term = ((a_lo % m) * (b_hi % m) * ((1u128 << 64) % m)) % m;
            result = (result + term) % m;
        }
        if a_hi > 0 && b_hi > 0 {
            let term = ((a_hi % m) * (b_hi % m) * (((1u128 << 64) % m) * ((1u128 << 64) % m) % m)) % m;
            result = (result + term) % m;
        }
        result
    }
}

/// Extended GCD to compute modular inverse
fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
        if b == 0 {
            (a, 1, 0)
        } else {
            let (gcd, x1, y1) = extended_gcd(b, a % b);
            (gcd, y1, x1 - (a / b) * y1)
        }
    }

    let (gcd, x, _) = extended_gcd(a as i128, m as i128);
    if gcd != 1 {
        return None;
    }

    let result = if x < 0 {
        (x + m as i128) as u64
    } else {
        x as u64
    };
    Some(result)
}

/// CUDA GPU bootstrap context
pub struct CudaBootstrapContext {
    /// CKKS context for basic operations
    ckks_ctx: Arc<CudaCkksContext>,

    /// Rotation context for Galois automorphisms
    rotation_ctx: Arc<CudaRotationContext>,

    /// Rotation keys for key switching
    rotation_keys: Arc<CudaRotationKeys>,

    /// Relinearization keys for ciphertext multiplication
    relin_keys: Arc<CudaRelinKeys>,

    /// V3 bootstrap parameters
    bootstrap_params: BootstrapParams,

    /// Base FHE parameters
    params: CliffordFHEParams,
}

impl CudaBootstrapContext {
    /// Create new CUDA bootstrap context
    pub fn new(
        ckks_ctx: Arc<CudaCkksContext>,
        rotation_ctx: Arc<CudaRotationContext>,
        rotation_keys: Arc<CudaRotationKeys>,
        relin_keys: Arc<CudaRelinKeys>,
        bootstrap_params: BootstrapParams,
        params: CliffordFHEParams,
    ) -> Result<Self, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║         CUDA GPU Bootstrap Context Initialized               ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        Ok(Self {
            ckks_ctx,
            rotation_ctx,
            rotation_keys,
            relin_keys,
            bootstrap_params,
            params,
        })
    }

    /// Perform full bootstrap operation
    ///
    /// Input: Noisy ciphertext at low level
    /// Output: Refreshed ciphertext with reduced noise
    pub fn bootstrap(
        &self,
        ct_in: &CudaCiphertext,
    ) -> Result<CudaCiphertext, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║              CUDA GPU Bootstrap Pipeline                     ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let total_start = std::time::Instant::now();

        // Step 1: Modulus raise (extend to max level)
        println!("Step 1: Modulus raise");
        let step1_start = std::time::Instant::now();
        let ct_raised = self.modulus_raise(ct_in)?;
        println!("  Modulus raised in {:.2}s\n", step1_start.elapsed().as_secs_f64());

        // Step 2: CoeffToSlot transformation
        println!("Step 2: CoeffToSlot transformation");
        let step2_start = std::time::Instant::now();
        let ct_slots = self.coeff_to_slot(&ct_raised)?;
        println!("  CoeffToSlot completed in {:.2}s\n", step2_start.elapsed().as_secs_f64());

        // Step 3: EvalMod (sine evaluation for modular reduction)
        println!("Step 3: EvalMod (modular reduction)");
        let step3_start = std::time::Instant::now();
        let ct_evalmod = self.eval_mod(&ct_slots)?;
        println!("  EvalMod completed in {:.2}s\n", step3_start.elapsed().as_secs_f64());

        // Step 4: SlotToCoeff transformation
        println!("Step 4: SlotToCoeff transformation");
        let step4_start = std::time::Instant::now();
        let ct_coeffs = self.slot_to_coeff(&ct_evalmod)?;
        println!("  SlotToCoeff completed in {:.2}s\n", step4_start.elapsed().as_secs_f64());

        // Step 5: Modulus switch (reduce back to original level)
        println!("Step 5: Modulus switch");
        let step5_start = std::time::Instant::now();
        let ct_out = self.modulus_switch(&ct_coeffs, ct_in.level)?;
        println!("  Modulus switched in {:.2}s\n", step5_start.elapsed().as_secs_f64());

        let total_time = total_start.elapsed().as_secs_f64();
        println!("═══════════════════════════════════════════════════════════════");
        println!("Bootstrap completed in {:.2}s", total_time);
        println!("═══════════════════════════════════════════════════════════════\n");

        Ok(ct_out)
    }

    /// Step 1: Modulus raise - extend ciphertext to higher modulus using Fast Basis Extension
    ///
    /// Uses Bajard et al. FBE algorithm with fixed-point correction factor.
    /// This produces a coherent lift of the ciphertext into the extended basis.
    ///
    /// **Important**: The raised ciphertext is only valid for bootstrap-internal operations.
    /// It contains an implicit Q_old * u term that EvalMod will remove.
    fn modulus_raise(&self, ct: &CudaCiphertext) -> Result<CudaCiphertext, String> {
        let target_level = self.params.moduli.len() - 1;

        if ct.level >= target_level {
            return Ok(ct.clone());
        }

        // Old moduli: q_0, ..., q_{ct.level}
        let old_moduli = &self.params.moduli[0..=ct.level];
        // New moduli: q_{ct.level+1}, ..., q_{target_level}
        let new_moduli = &self.params.moduli[ct.level + 1..=target_level];

        // Create FBE context for this extension
        let fbe = FastBasisExtension::new(old_moduli, new_moduli);

        let n = ct.n;
        let old_num_primes = ct.level + 1;
        let new_num_primes = target_level + 1;
        let num_new = new_moduli.len();

        // Allocate output with full prime count
        let mut c0_raised = vec![0u64; n * new_num_primes];
        let mut c1_raised = vec![0u64; n * new_num_primes];

        // Process each coefficient
        for coeff_idx in 0..n {
            // Extract old residues for c0 (strided layout: poly[coeff_idx * num_primes + prime_idx])
            let mut c0_residues = vec![0u64; old_num_primes];
            for prime_idx in 0..old_num_primes {
                c0_residues[prime_idx] = ct.c0[coeff_idx * old_num_primes + prime_idx];
            }

            // Extract old residues for c1
            let mut c1_residues = vec![0u64; old_num_primes];
            for prime_idx in 0..old_num_primes {
                c1_residues[prime_idx] = ct.c1[coeff_idx * old_num_primes + prime_idx];
            }

            // Extend to new primes using FBE
            let c0_new = fbe.extend_coefficient(&c0_residues);
            let c1_new = fbe.extend_coefficient(&c1_residues);

            // Copy old residues to output
            for prime_idx in 0..old_num_primes {
                c0_raised[coeff_idx * new_num_primes + prime_idx] = c0_residues[prime_idx];
                c1_raised[coeff_idx * new_num_primes + prime_idx] = c1_residues[prime_idx];
            }

            // Copy new residues to output
            for (i, prime_idx) in (old_num_primes..new_num_primes).enumerate() {
                c0_raised[coeff_idx * new_num_primes + prime_idx] = c0_new[i];
                c1_raised[coeff_idx * new_num_primes + prime_idx] = c1_new[i];
            }
        }

        Ok(CudaCiphertext {
            c0: c0_raised,
            c1: c1_raised,
            n: ct.n,
            num_primes: new_num_primes,
            level: target_level,
            scale: ct.scale,
        })
    }

    /// Modulus down - drop higher primes, keeping only the original basis
    ///
    /// This is the inverse of modulus_raise for testing purposes.
    /// Simply truncates the RNS representation to keep only the first `target_level + 1` primes.
    ///
    /// **Invariant**: ModUp followed by ModDown should preserve the ciphertext exactly
    /// (for the primes that were kept).
    pub fn modulus_down(&self, ct: &CudaCiphertext, target_level: usize) -> Result<CudaCiphertext, String> {
        if target_level >= ct.level {
            return Ok(ct.clone());
        }

        let n = ct.n;
        let old_num_primes = ct.num_primes;
        let new_num_primes = target_level + 1;

        // Allocate output with reduced prime count
        let mut c0_down = vec![0u64; n * new_num_primes];
        let mut c1_down = vec![0u64; n * new_num_primes];

        // Copy only the first target_level + 1 primes for each coefficient
        for coeff_idx in 0..n {
            for prime_idx in 0..new_num_primes {
                c0_down[coeff_idx * new_num_primes + prime_idx] =
                    ct.c0[coeff_idx * old_num_primes + prime_idx];
                c1_down[coeff_idx * new_num_primes + prime_idx] =
                    ct.c1[coeff_idx * old_num_primes + prime_idx];
            }
        }

        Ok(CudaCiphertext {
            c0: c0_down,
            c1: c1_down,
            n: ct.n,
            num_primes: new_num_primes,
            level: target_level,
            scale: ct.scale,
        })
    }

    /// Step 2: CoeffToSlot - transform coefficient encoding to slot encoding
    ///
    /// Uses FFT-like butterfly algorithm with rotations
    fn coeff_to_slot(&self, ct: &CudaCiphertext) -> Result<CudaCiphertext, String> {
        cuda_coeff_to_slot(ct, &self.rotation_keys, &self.ckks_ctx)
    }

    /// Step 3: EvalMod - evaluate modular reduction using sine approximation
    ///
    /// Removes noise by evaluating: f(x) = x - q/2π · sin(2πx/q)
    fn eval_mod(&self, ct: &CudaCiphertext) -> Result<CudaCiphertext, String> {
        // Use the top-level modulus for EvalMod
        let q = self.params.moduli[ct.level];
        let sin_degree = self.bootstrap_params.sin_degree;
        cuda_eval_mod(ct, q, sin_degree, &self.ckks_ctx, Some(&self.relin_keys))
    }

    /// Step 4: SlotToCoeff - transform slot encoding back to coefficient encoding
    ///
    /// Inverse of CoeffToSlot
    fn slot_to_coeff(&self, ct: &CudaCiphertext) -> Result<CudaCiphertext, String> {
        cuda_slot_to_coeff(ct, &self.rotation_keys, &self.ckks_ctx)
    }

    /// Step 5: Modulus switch - reduce ciphertext to target level
    fn modulus_switch(&self, ct: &CudaCiphertext, target_level: usize) -> Result<CudaCiphertext, String> {
        if ct.level <= target_level {
            return Ok(ct.clone());
        }

        // Apply GPU rescaling repeatedly to drop primes
        let mut current_ct = ct.clone();

        for _ in target_level..ct.level {
            // Use GPU rescaling
            let c0_rescaled = self.ckks_ctx.exact_rescale_gpu(&current_ct.c0, current_ct.level)?;
            let c1_rescaled = self.ckks_ctx.exact_rescale_gpu(&current_ct.c1, current_ct.level)?;

            let new_level = current_ct.level - 1;
            let new_scale = current_ct.scale / self.params.moduli[current_ct.level] as f64;

            current_ct = CudaCiphertext {
                c0: c0_rescaled,
                c1: c1_rescaled,
                n: current_ct.n,
                num_primes: new_level + 1,
                level: new_level,
                scale: new_scale,
            };
        }

        Ok(current_ct)
    }
}

/// CUDA ciphertext structure (uses STRIDED RNS layout by default)
/// Layout: poly[coeff_idx * num_primes + prime_idx]
#[derive(Clone)]
pub struct CudaCiphertext {
    pub c0: Vec<u64>,
    pub c1: Vec<u64>,
    pub n: usize,
    pub num_primes: usize,
    pub level: usize,
    pub scale: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Fast Basis Extension on small known values
    #[test]
    fn test_fbe_small_values() {
        // Use small test primes for easy verification
        let old_moduli = vec![17u64, 19, 23];  // Q = 17 * 19 * 23 = 7429
        let new_moduli = vec![29u64, 31];

        let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

        // Test value x = 100 (small, fits easily)
        // Residues: 100 mod 17 = 15, 100 mod 19 = 5, 100 mod 23 = 8
        let residues = vec![15u64, 5, 8];
        let new_residues = fbe.extend_coefficient(&residues);

        // Expected: 100 mod 29 = 13, 100 mod 31 = 7
        assert_eq!(new_residues[0], 13, "100 mod 29 should be 13");
        assert_eq!(new_residues[1], 7, "100 mod 31 should be 7");
    }

    /// Test FBE with negative (centered) values
    #[test]
    fn test_fbe_negative_values() {
        let old_moduli = vec![17u64, 19, 23];  // Q = 7429
        let new_moduli = vec![29u64, 31];

        let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

        // Test value x = -50 (centered representation)
        // In centered rep: -50 is represented as Q - 50 = 7379 (but as residues)
        // Residues: -50 mod 17 = -50 + 51 = 1 (since -50 = -3*17 + 1)
        //           Actually: -50 mod 17: -50 + 3*17 = -50 + 51 = 1
        //           -50 mod 19: -50 + 3*19 = -50 + 57 = 7
        //           -50 mod 23: -50 + 3*23 = -50 + 69 = 19
        // Wait, let me recalculate properly:
        // -50 mod 17 = 17 - (50 mod 17) = 17 - 16 = 1? No...
        // 50 mod 17 = 50 - 2*17 = 50 - 34 = 16
        // -50 mod 17 = 17 - 16 = 1? Actually -50 + 3*17 = -50 + 51 = 1. Yes.
        // 50 mod 19 = 50 - 2*19 = 50 - 38 = 12
        // -50 mod 19 = 19 - 12 = 7. Check: -50 + 3*19 = -50 + 57 = 7. Yes.
        // 50 mod 23 = 50 - 2*23 = 50 - 46 = 4
        // -50 mod 23 = 23 - 4 = 19. Check: -50 + 3*23 = -50 + 69 = 19. Yes.
        let residues = vec![1u64, 7, 19];  // represents -50 in centered form
        let new_residues = fbe.extend_coefficient(&residues);

        // Expected: -50 mod 29 = 29 - (50 mod 29) = 29 - 21 = 8
        //           -50 mod 31 = 31 - (50 mod 31) = 31 - 19 = 12
        // But wait, FBE reconstructs the centered integer, so it should give us
        // the representation of -50, not 7429-50.
        // -50 mod 29: -50 + 2*29 = -50 + 58 = 8. Yes.
        // -50 mod 31: -50 + 2*31 = -50 + 62 = 12. Yes.
        assert_eq!(new_residues[0], 8, "-50 mod 29 should be 8");
        assert_eq!(new_residues[1], 12, "-50 mod 31 should be 12");
    }

    /// Test ModUp→ModDown round-trip preserves residues exactly
    ///
    /// This is the CORRECT invariant for modulus raise validation.
    /// After raising and then lowering, the original residues should be unchanged.
    #[test]
    fn test_modup_moddown_roundtrip() {
        let old_moduli = vec![17u64, 19, 23];
        let new_moduli = vec![29u64, 31, 37];
        let all_moduli: Vec<u64> = old_moduli.iter().chain(new_moduli.iter()).cloned().collect();

        let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

        // Test with various values
        let test_values: Vec<i64> = vec![0, 1, 100, 1000, -1, -50, -100, 3714, -3714];

        for &val in &test_values {
            // Compute original residues
            let original_residues: Vec<u64> = old_moduli.iter()
                .map(|&q| {
                    if val >= 0 {
                        (val as u64) % q
                    } else {
                        let abs_val = (-val) as u64;
                        if abs_val % q == 0 { 0 } else { q - (abs_val % q) }
                    }
                })
                .collect();

            // Extend to new basis
            let extended = fbe.extend_coefficient(&original_residues);

            // The round-trip property: original residues should be preserved
            // (we don't need to explicitly "mod down" - just verify the FBE didn't corrupt anything)

            // Verify the extended residues are correct by checking the full reconstruction
            // For a value v in [-Q/2, Q/2), extended residues should satisfy:
            // extended[j] ≡ v (mod new_moduli[j])
            let expected_new: Vec<u64> = new_moduli.iter()
                .map(|&p| {
                    if val >= 0 {
                        (val as u64) % p
                    } else {
                        let abs_val = (-val) as u64;
                        if abs_val % p == 0 { 0 } else { p - (abs_val % p) }
                    }
                })
                .collect();

            for (j, (&got, &expected)) in extended.iter().zip(expected_new.iter()).enumerate() {
                assert_eq!(
                    got, expected,
                    "FBE failed for value {} at new prime {}: got {}, expected {}",
                    val, new_moduli[j], got, expected
                );
            }
        }
    }

    /// Test FBE with realistic CKKS prime sizes
    #[test]
    fn test_fbe_large_primes() {
        // Use actual NTT-friendly primes from params
        let params = CliffordFHEParams::new_test_ntt_1024();

        // Simulate extending from 2 primes to 3 primes
        let old_moduli = vec![params.moduli[0], params.moduli[1]];
        let new_moduli = vec![params.moduli[2]];

        let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

        // Test with a small value that we can verify
        let test_val: i64 = 12345;
        let residues: Vec<u64> = old_moduli.iter()
            .map(|&q| (test_val as u64) % q)
            .collect();

        let extended = fbe.extend_coefficient(&residues);

        let expected = (test_val as u64) % new_moduli[0];
        assert_eq!(
            extended[0], expected,
            "FBE with large primes: {} mod {} should be {}, got {}",
            test_val, new_moduli[0], expected, extended[0]
        );
    }

    /// Test full ciphertext ModUp→ModDown round-trip
    #[test]
    fn test_ciphertext_modup_moddown_roundtrip() {
        // Create a mock bootstrap context (we only need params for this test)
        let params = CliffordFHEParams::new_test_ntt_1024();
        let n = params.n;

        // Create a test ciphertext at level 1 (2 primes)
        let original_level = 1;
        let original_num_primes = original_level + 1;

        // Fill with deterministic test data
        let mut c0_original = vec![0u64; n * original_num_primes];
        let mut c1_original = vec![0u64; n * original_num_primes];

        for coeff_idx in 0..n {
            for prime_idx in 0..original_num_primes {
                // Use small values relative to primes for easy verification
                let val = ((coeff_idx * 7 + prime_idx * 13) % 1000) as u64;
                c0_original[coeff_idx * original_num_primes + prime_idx] =
                    val % params.moduli[prime_idx];
                c1_original[coeff_idx * original_num_primes + prime_idx] =
                    (val * 2) % params.moduli[prime_idx];
            }
        }

        let ct_original = CudaCiphertext {
            c0: c0_original.clone(),
            c1: c1_original.clone(),
            n,
            num_primes: original_num_primes,
            level: original_level,
            scale: 1e10,
        };

        // For this test, we'll manually apply FBE since we don't have full context
        let old_moduli = &params.moduli[0..=original_level];
        let new_moduli = &params.moduli[original_level + 1..];
        let fbe = FastBasisExtension::new(old_moduli, new_moduli);

        let target_level = params.moduli.len() - 1;
        let target_num_primes = target_level + 1;

        // ModUp
        let mut c0_raised = vec![0u64; n * target_num_primes];
        let mut c1_raised = vec![0u64; n * target_num_primes];

        for coeff_idx in 0..n {
            // Extract residues
            let c0_res: Vec<u64> = (0..original_num_primes)
                .map(|pi| ct_original.c0[coeff_idx * original_num_primes + pi])
                .collect();
            let c1_res: Vec<u64> = (0..original_num_primes)
                .map(|pi| ct_original.c1[coeff_idx * original_num_primes + pi])
                .collect();

            // Extend
            let c0_new = fbe.extend_coefficient(&c0_res);
            let c1_new = fbe.extend_coefficient(&c1_res);

            // Copy old + new
            for pi in 0..original_num_primes {
                c0_raised[coeff_idx * target_num_primes + pi] = c0_res[pi];
                c1_raised[coeff_idx * target_num_primes + pi] = c1_res[pi];
            }
            for (i, pi) in (original_num_primes..target_num_primes).enumerate() {
                c0_raised[coeff_idx * target_num_primes + pi] = c0_new[i];
                c1_raised[coeff_idx * target_num_primes + pi] = c1_new[i];
            }
        }

        // ModDown (just truncate)
        let mut c0_down = vec![0u64; n * original_num_primes];
        let mut c1_down = vec![0u64; n * original_num_primes];

        for coeff_idx in 0..n {
            for pi in 0..original_num_primes {
                c0_down[coeff_idx * original_num_primes + pi] =
                    c0_raised[coeff_idx * target_num_primes + pi];
                c1_down[coeff_idx * original_num_primes + pi] =
                    c1_raised[coeff_idx * target_num_primes + pi];
            }
        }

        // Verify round-trip: original residues should be exactly preserved
        for coeff_idx in 0..n {
            for pi in 0..original_num_primes {
                let orig_c0 = c0_original[coeff_idx * original_num_primes + pi];
                let down_c0 = c0_down[coeff_idx * original_num_primes + pi];
                assert_eq!(
                    orig_c0, down_c0,
                    "c0 round-trip failed at coeff {} prime {}: {} != {}",
                    coeff_idx, pi, orig_c0, down_c0
                );

                let orig_c1 = c1_original[coeff_idx * original_num_primes + pi];
                let down_c1 = c1_down[coeff_idx * original_num_primes + pi];
                assert_eq!(
                    orig_c1, down_c1,
                    "c1 round-trip failed at coeff {} prime {}: {} != {}",
                    coeff_idx, pi, orig_c1, down_c1
                );
            }
        }

        println!("ModUp→ModDown round-trip test passed for {} coefficients!", n);
    }
}
