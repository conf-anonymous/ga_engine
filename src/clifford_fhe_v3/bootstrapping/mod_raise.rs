//! Modulus Raising using Fast Basis Extension
//!
//! Raises ciphertext to higher modulus level to create working room for bootstrap.
//!
//! This uses the Bajard et al. Fast Basis Extension (FBE) algorithm with
//! fixed-point correction factor computation for numerical stability.
//!
//! # Algorithm
//!
//! For each coefficient x with residues (x mod q_0, ..., x mod q_L):
//! 1. Compute α_i = x_i * (Q/q_i)^(-1) mod q_i
//! 2. Compute k = round(Σ α_i / q_i) using fixed-point arithmetic
//! 3. For new prime p: x mod p = Σ(α_i * Q̂_i mod p) - k * (Q mod p) mod p

use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

/// Fixed-point precision for correction factor k computation (80 bits)
const FP_PRECISION: u32 = 80;

/// Fast Basis Extension precomputed values
struct FastBasisExtension {
    old_moduli: Vec<u64>,
    new_moduli: Vec<u64>,
    q_hat_inv_mod_q: Vec<u64>,
    q_hat_mod_new: Vec<Vec<u64>>,
    q_mod_new: Vec<u64>,
    inv_q_fp: Vec<u128>,
}

impl FastBasisExtension {
    /// Create new Fast Basis Extension context
    fn new(old_moduli: &[u64], new_moduli: &[u64]) -> Self {
        let num_old = old_moduli.len();
        let num_new = new_moduli.len();

        // Compute Q̂_i mod q_i and its inverse
        let mut q_hat_mod_q = vec![0u64; num_old];
        for i in 0..num_old {
            let mut prod = 1u128;
            for k in 0..num_old {
                if k != i {
                    prod = (prod * (old_moduli[k] as u128)) % (old_moduli[i] as u128);
                }
            }
            q_hat_mod_q[i] = prod as u64;
        }

        let mut q_hat_inv_mod_q = vec![0u64; num_old];
        for i in 0..num_old {
            q_hat_inv_mod_q[i] = mod_inverse(q_hat_mod_q[i], old_moduli[i])
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
            let two_pow_80 = 1u128 << FP_PRECISION;
            inv_q_fp[i] = two_pow_80 / (old_moduli[i] as u128);
        }

        Self {
            old_moduli: old_moduli.to_vec(),
            new_moduli: new_moduli.to_vec(),
            q_hat_inv_mod_q,
            q_hat_mod_new,
            q_mod_new,
            inv_q_fp,
        }
    }

    /// Extend a single coefficient to new basis
    fn extend_coefficient(&self, residues: &[u64]) -> Vec<u64> {
        let num_old = self.old_moduli.len();
        let num_new = self.new_moduli.len();

        // Step 1: Compute α_i = x_i * (Q̂_i)^(-1) mod q_i
        let mut alpha = vec![0u64; num_old];
        for i in 0..num_old {
            alpha[i] = mul_mod(residues[i], self.q_hat_inv_mod_q[i], self.old_moduli[i]);
        }

        // Step 2: Compute correction factor k using fixed-point arithmetic
        // k = round(Σ α_i / q_i)
        let mut sum_fp: u128 = 0;
        for i in 0..num_old {
            let contrib = (alpha[i] as u128) * self.inv_q_fp[i];
            sum_fp += contrib;
        }

        let half: u128 = 1u128 << (FP_PRECISION - 1);
        let k: u128 = (sum_fp + half) >> FP_PRECISION;

        // Step 3: Compute residues for new primes
        let mut new_residues = vec![0u64; num_new];
        for j in 0..num_new {
            let p = self.new_moduli[j];

            let mut sum: u128 = 0;
            for i in 0..num_old {
                let term = ((alpha[i] as u128) * (self.q_hat_mod_new[i][j] as u128)) % (p as u128);
                sum = (sum + term) % (p as u128);
            }

            let k_mod_p = k % (p as u128);
            let correction = (k_mod_p * (self.q_mod_new[j] as u128)) % (p as u128);

            new_residues[j] = if sum >= correction {
                (sum - correction) as u64
            } else {
                (sum + p as u128 - correction) as u64
            };
        }

        new_residues
    }
}

#[inline]
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

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

/// Raise ciphertext modulus to higher level using Fast Basis Extension
///
/// This extends the RNS representation to include additional primes,
/// creating "working room" for bootstrap computation.
///
/// # Arguments
///
/// * `ct` - Input ciphertext with current moduli
/// * `target_moduli` - Target moduli (must include current moduli as prefix)
///
/// # Returns
///
/// Ciphertext with raised modulus
///
/// # Errors
///
/// Returns error if target moduli count is not larger than current,
/// or if target moduli don't include current moduli as prefix.
pub fn mod_raise(
    ct: &Ciphertext,
    target_moduli: &[u64],
) -> Result<Ciphertext, String> {
    let current_moduli = &ct.c0[0].moduli;
    let n = ct.c0.len();

    if target_moduli.len() <= current_moduli.len() {
        return Err(format!(
            "Target moduli count ({}) must be larger than current ({})",
            target_moduli.len(),
            current_moduli.len()
        ));
    }

    // Verify that target moduli include current moduli (as prefix)
    for i in 0..current_moduli.len() {
        if current_moduli[i] != target_moduli[i] {
            return Err(format!(
                "Target moduli must include current moduli as prefix. Mismatch at index {}: {} != {}",
                i, current_moduli[i], target_moduli[i]
            ));
        }
    }

    // Create FBE context
    let old_moduli = current_moduli;
    let new_moduli = &target_moduli[current_moduli.len()..];
    let fbe = FastBasisExtension::new(old_moduli, new_moduli);

    // Scale c0 to higher modulus
    let mut c0_raised = Vec::with_capacity(n);
    for rns in &ct.c0 {
        c0_raised.push(extend_rns(&rns, &fbe, target_moduli)?);
    }

    // Scale c1 to higher modulus
    let mut c1_raised = Vec::with_capacity(n);
    for rns in &ct.c1 {
        c1_raised.push(extend_rns(&rns, &fbe, target_moduli)?);
    }

    Ok(Ciphertext {
        c0: c0_raised,
        c1: c1_raised,
        level: ct.level,
        scale: ct.scale,
        n: ct.n,
    })
}

/// Extend a single RNS representation to higher modulus using FBE
fn extend_rns(
    rns: &RnsRepresentation,
    fbe: &FastBasisExtension,
    target_moduli: &[u64],
) -> Result<RnsRepresentation, String> {
    let num_old = rns.values.len();
    let num_new = target_moduli.len() - num_old;

    // Copy existing residues
    let mut new_residues = rns.values.clone();

    // Extend to new primes using FBE
    let extended = fbe.extend_coefficient(&rns.values);
    new_residues.extend(extended);

    Ok(RnsRepresentation {
        values: new_residues,
        moduli: target_moduli.to_vec(),
    })
}

/// Modulus down - drop higher primes, keeping only the original basis
///
/// This is the inverse of mod_raise for testing purposes.
pub fn mod_down(
    ct: &Ciphertext,
    target_moduli: &[u64],
) -> Result<Ciphertext, String> {
    let current_moduli = &ct.c0[0].moduli;
    let n = ct.c0.len();

    if target_moduli.len() >= current_moduli.len() {
        return Err("Target moduli must be smaller than current".to_string());
    }

    // Simply truncate the RNS representations
    let mut c0_down = Vec::with_capacity(n);
    for rns in &ct.c0 {
        c0_down.push(RnsRepresentation {
            values: rns.values[..target_moduli.len()].to_vec(),
            moduli: target_moduli.to_vec(),
        });
    }

    let mut c1_down = Vec::with_capacity(n);
    for rns in &ct.c1 {
        c1_down.push(RnsRepresentation {
            values: rns.values[..target_moduli.len()].to_vec(),
            moduli: target_moduli.to_vec(),
        });
    }

    Ok(Ciphertext {
        c0: c0_down,
        c1: c1_down,
        level: target_moduli.len() - 1,
        scale: ct.scale,
        n: ct.n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::params::CliffordFHEParams;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;

    #[test]
    fn test_fbe_small_values() {
        let old_moduli = vec![17u64, 19, 23];
        let new_moduli = vec![29u64, 31];

        let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

        // Test x = 100
        let residues = vec![15u64, 5, 8];  // 100 mod 17, 19, 23
        let new_residues = fbe.extend_coefficient(&residues);

        assert_eq!(new_residues[0], 13, "100 mod 29");
        assert_eq!(new_residues[1], 7, "100 mod 31");
    }

    #[test]
    fn test_fbe_negative_values() {
        let old_moduli = vec![17u64, 19, 23];
        let new_moduli = vec![29u64, 31];

        let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

        // Test x = -50 (represented as residues)
        let residues = vec![1u64, 7, 19];  // -50 mod 17, 19, 23
        let new_residues = fbe.extend_coefficient(&residues);

        assert_eq!(new_residues[0], 8, "-50 mod 29");
        assert_eq!(new_residues[1], 12, "-50 mod 31");
    }

    #[test]
    fn test_modup_moddown_roundtrip() {
        let old_moduli = vec![17u64, 19, 23];
        let new_moduli = vec![29u64, 31, 37];

        let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

        // Test all values in [-Q/2, Q/2)
        let q = 17 * 19 * 23;  // 7429
        for val in -(q/2)..=(q/2) {
            let residues: Vec<u64> = old_moduli.iter()
                .map(|&m| {
                    if val >= 0 {
                        (val as u64) % m
                    } else {
                        let abs_val = (-val) as u64;
                        if abs_val % m == 0 { 0 } else { m - (abs_val % m) }
                    }
                })
                .collect();

            let extended = fbe.extend_coefficient(&residues);

            for (j, &got) in extended.iter().enumerate() {
                let p = new_moduli[j];
                let expected = if val >= 0 {
                    (val as u64) % p
                } else {
                    let abs_val = (-val) as u64;
                    if abs_val % p == 0 { 0 } else { p - (abs_val % p) }
                };

                assert_eq!(got, expected, "Failed for val={} at prime {}", val, p);
            }
        }
    }

    #[test]
    fn test_mod_raise_preserves_plaintext() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (public_key, secret_key, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;
        let plaintext_values = vec![1.0, 2.0, 3.0, 4.0];
        let plaintext = Plaintext::encode(&plaintext_values, params.scale, &params);
        let ct = ckks_ctx.encrypt(&plaintext, &public_key);

        // Use additional NTT-friendly primes
        let target_moduli = vec![
            params.moduli[0],
            params.moduli[1],
            params.moduli[2],
            1152921504606584777,
            1152921504606584833,
        ];

        let ct_raised = mod_raise(&ct, &target_moduli).unwrap();
        assert_eq!(ct_raised.c0[0].moduli.len(), 5);

        // Mod down and decrypt
        let original_moduli = &params.moduli[..=ct.level];
        let ct_lowered = mod_down(&ct_raised, original_moduli).unwrap();

        let decrypted_pt = ckks_ctx.decrypt(&ct_lowered, &secret_key);
        let decrypted = decrypted_pt.decode(&params);

        // Verify round-trip preserves plaintext
        for i in 0..plaintext_values.len().min(decrypted.len()) {
            let error = (plaintext_values[i] - decrypted[i]).abs();
            println!("plaintext[{}] = {:.6}, decrypted[{}] = {:.6}, error = {:.2e}",
                     i, plaintext_values[i], i, decrypted[i], error);
            assert!(error < 1e-6, "ModRaise→ModDown changed plaintext at index {}", i);
        }
    }

    #[test]
    fn test_mod_raise_requires_larger_moduli() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (public_key, _, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;
        let plaintext_values = vec![1.0, 2.0];
        let plaintext = Plaintext::encode(&plaintext_values, params.scale, &params);
        let ct = ckks_ctx.encrypt(&plaintext, &public_key);

        let result = mod_raise(&ct, &params.moduli);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be larger"));
    }

    #[test]
    fn test_mod_raise_requires_prefix_match() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (public_key, _, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;
        let plaintext_values = vec![1.0, 2.0];
        let plaintext = Plaintext::encode(&plaintext_values, params.scale, &params);
        let ct = ckks_ctx.encrypt(&plaintext, &public_key);

        let target_moduli = vec![
            1152921504606584777,
            1152921504606584833,
            params.moduli[0],
            params.moduli[1],
        ];

        let result = mod_raise(&ct, &target_moduli);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must include current moduli as prefix"));
    }
}
