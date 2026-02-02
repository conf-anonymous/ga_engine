//! Test Fast Basis Extension implementation
//!
//! These tests verify that the FBE algorithm correctly extends RNS representations.

/// Fixed-point precision for correction factor k computation (80 bits)
const FP_PRECISION: u32 = 80;

/// Fast Basis Extension precomputed values (copied for standalone testing)
pub struct FastBasisExtension {
    old_moduli: Vec<u64>,
    new_moduli: Vec<u64>,
    q_hat_mod_old: Vec<Vec<u64>>,
    q_hat_inv_mod_q: Vec<u64>,
    q_hat_mod_new: Vec<Vec<u64>>,
    q_mod_new: Vec<u64>,
    inv_q_fp: Vec<u128>,
}

impl FastBasisExtension {
    pub fn new(old_moduli: &[u64], new_moduli: &[u64]) -> Self {
        let num_old = old_moduli.len();
        let num_new = new_moduli.len();

        // Compute Q̂_i mod q_j
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

        // Compute fixed-point inverses
        let mut inv_q_fp = vec![0u128; num_old];
        for i in 0..num_old {
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

    pub fn extend_coefficient(&self, residues: &[u64]) -> Vec<u64> {
        let num_old = self.old_moduli.len();
        let num_new = self.new_moduli.len();

        // Step 1: Compute α_i = x_i * (Q̂_i)^(-1) mod q_i
        // These α_i values satisfy: x ≡ Σ α_i * Q̂_i (mod Q)
        let mut alpha = vec![0u64; num_old];
        for i in 0..num_old {
            alpha[i] = mul_mod(residues[i], self.q_hat_inv_mod_q[i], self.old_moduli[i]);
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
                let term = ((alpha[i] as u128) * (self.q_hat_mod_new[i][j] as u128)) % (p as u128);
                sum = (sum + term) % (p as u128);
            }

            // Compute k * Q mod p
            let k_mod_p = k % (p as u128);
            let correction = (k_mod_p * (self.q_mod_new[j] as u128)) % (p as u128);

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

#[test]
fn test_fbe_small_positive() {
    let old_moduli = vec![17u64, 19, 23];  // Q = 7429
    let new_moduli = vec![29u64, 31];

    let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

    // Test x = 100
    // 100 mod 17 = 15, 100 mod 19 = 5, 100 mod 23 = 8
    let residues = vec![15u64, 5, 8];
    let new_residues = fbe.extend_coefficient(&residues);

    // Expected: 100 mod 29 = 13, 100 mod 31 = 7
    assert_eq!(new_residues[0], 13, "100 mod 29");
    assert_eq!(new_residues[1], 7, "100 mod 31");
    println!("test_fbe_small_positive PASSED");
}

#[test]
fn test_fbe_small_negative() {
    let old_moduli = vec![17u64, 19, 23];  // Q = 7429
    let new_moduli = vec![29u64, 31];

    let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

    // Test x = -50 (centered representation)
    // -50 mod 17 = 1, -50 mod 19 = 7, -50 mod 23 = 19
    let residues = vec![1u64, 7, 19];
    let new_residues = fbe.extend_coefficient(&residues);

    // Expected: -50 mod 29 = 8, -50 mod 31 = 12
    assert_eq!(new_residues[0], 8, "-50 mod 29");
    assert_eq!(new_residues[1], 12, "-50 mod 31");
    println!("test_fbe_small_negative PASSED");
}

#[test]
fn test_fbe_zero() {
    let old_moduli = vec![17u64, 19, 23];
    let new_moduli = vec![29u64, 31];

    let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

    let residues = vec![0u64, 0, 0];
    let new_residues = fbe.extend_coefficient(&residues);

    assert_eq!(new_residues[0], 0, "0 mod 29");
    assert_eq!(new_residues[1], 0, "0 mod 31");
    println!("test_fbe_zero PASSED");
}

#[test]
fn test_fbe_boundary() {
    let old_moduli = vec![17u64, 19, 23];  // Q = 7429
    let new_moduli = vec![29u64, 31];

    let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

    // Test x = Q/2 - 1 = 3713 (maximum positive centered value)
    // 3713 mod 17 = 6, 3713 mod 19 = 9, 3713 mod 23 = 10
    let val = 3713i64;
    let residues: Vec<u64> = old_moduli.iter()
        .map(|&q| (val as u64) % q)
        .collect();
    println!("Residues for {}: {:?}", val, residues);

    let new_residues = fbe.extend_coefficient(&residues);

    let expected: Vec<u64> = new_moduli.iter()
        .map(|&p| (val as u64) % p)
        .collect();

    assert_eq!(new_residues[0], expected[0], "{} mod 29", val);
    assert_eq!(new_residues[1], expected[1], "{} mod 31", val);
    println!("test_fbe_boundary PASSED");
}

#[test]
fn test_fbe_negative_boundary() {
    let old_moduli = vec![17u64, 19, 23];  // Q = 7429
    let new_moduli = vec![29u64, 31];

    let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

    // Test x = -Q/2 = -3714 (minimum negative centered value)
    let val = -3714i64;

    // Compute residues for negative value
    let residues: Vec<u64> = old_moduli.iter()
        .map(|&q| {
            let abs_val = (-val) as u64;
            if abs_val % q == 0 { 0 } else { q - (abs_val % q) }
        })
        .collect();
    println!("Residues for {}: {:?}", val, residues);

    let new_residues = fbe.extend_coefficient(&residues);

    let expected: Vec<u64> = new_moduli.iter()
        .map(|&p| {
            let abs_val = (-val) as u64;
            if abs_val % p == 0 { 0 } else { p - (abs_val % p) }
        })
        .collect();

    assert_eq!(new_residues[0], expected[0], "{} mod 29", val);
    assert_eq!(new_residues[1], expected[1], "{} mod 31", val);
    println!("test_fbe_negative_boundary PASSED");
}

#[test]
fn test_fbe_many_values() {
    let old_moduli = vec![17u64, 19, 23];  // Q = 7429
    let new_moduli = vec![29u64, 31, 37];

    let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

    // Test range of values from -Q/2 to Q/2
    let test_values: Vec<i64> = vec![
        0, 1, -1, 10, -10, 100, -100, 1000, -1000,
        3714, -3714, 3713, -3713, 2000, -2000
    ];

    for &val in &test_values {
        // Compute residues
        let residues: Vec<u64> = old_moduli.iter()
            .map(|&q| {
                if val >= 0 {
                    (val as u64) % q
                } else {
                    let abs_val = (-val) as u64;
                    if abs_val % q == 0 { 0 } else { q - (abs_val % q) }
                }
            })
            .collect();

        let new_residues = fbe.extend_coefficient(&residues);

        // Compute expected
        let expected: Vec<u64> = new_moduli.iter()
            .map(|&p| {
                if val >= 0 {
                    (val as u64) % p
                } else {
                    let abs_val = (-val) as u64;
                    if abs_val % p == 0 { 0 } else { p - (abs_val % p) }
                }
            })
            .collect();

        for (j, (&got, &exp)) in new_residues.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got, exp,
                "FBE failed for value {} at prime {}: got {}, expected {}",
                val, new_moduli[j], got, exp
            );
        }
    }

    println!("test_fbe_many_values PASSED: all {} values correct", test_values.len());
}

#[test]
fn test_fbe_large_primes() {
    // Use realistic 60-bit and 45-bit primes
    let old_moduli = vec![
        1152921504606584833u64,  // 60-bit NTT-friendly
        1099511678977u64,        // 41-bit
    ];
    let new_moduli = vec![
        1099511683073u64,        // 41-bit
    ];

    let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

    // Test with small value
    let val = 12345i64;
    let residues: Vec<u64> = old_moduli.iter()
        .map(|&q| (val as u64) % q)
        .collect();

    let new_residues = fbe.extend_coefficient(&residues);
    let expected = (val as u64) % new_moduli[0];

    assert_eq!(
        new_residues[0], expected,
        "Large primes: {} mod {} should be {}, got {}",
        val, new_moduli[0], expected, new_residues[0]
    );

    // Test with negative value
    let val = -12345i64;
    let residues: Vec<u64> = old_moduli.iter()
        .map(|&q| {
            let abs_val = 12345u64;
            q - (abs_val % q)
        })
        .collect();

    let new_residues = fbe.extend_coefficient(&residues);
    let expected = new_moduli[0] - (12345u64 % new_moduli[0]);

    assert_eq!(
        new_residues[0], expected,
        "Large primes negative: {} mod {} should be {}, got {}",
        val, new_moduli[0], expected, new_residues[0]
    );

    println!("test_fbe_large_primes PASSED");
}

#[test]
fn test_roundtrip_invariant() {
    let old_moduli = vec![17u64, 19, 23];
    let new_moduli = vec![29u64, 31, 37];

    let fbe = FastBasisExtension::new(&old_moduli, &new_moduli);

    // The key invariant: for any value x in [-Q/2, Q/2),
    // extend_coefficient(residues(x)) should give correct residues for new primes

    for val in -3714i64..=3714 {
        let residues: Vec<u64> = old_moduli.iter()
            .map(|&q| {
                if val >= 0 {
                    (val as u64) % q
                } else {
                    let abs_val = (-val) as u64;
                    if abs_val % q == 0 { 0 } else { q - (abs_val % q) }
                }
            })
            .collect();

        let new_residues = fbe.extend_coefficient(&residues);

        for (j, &got) in new_residues.iter().enumerate() {
            let p = new_moduli[j];
            let expected = if val >= 0 {
                (val as u64) % p
            } else {
                let abs_val = (-val) as u64;
                if abs_val % p == 0 { 0 } else { p - (abs_val % p) }
            };

            assert_eq!(
                got, expected,
                "Roundtrip failed for value {} at prime {}: got {}, expected {}",
                val, p, got, expected
            );
        }
    }

    println!("test_roundtrip_invariant PASSED: all 7429 values in range verified");
}
