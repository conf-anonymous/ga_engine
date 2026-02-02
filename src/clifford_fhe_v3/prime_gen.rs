//! NTT-Friendly Prime Generation
//!
//! Generates primes of the form q = k × 2n + 1 where n is the ring dimension.
//! These primes are "NTT-friendly" because (q-1) is divisible by 2n, which
//! guarantees the existence of a primitive 2n-th root of unity in Z_q.

use rand::Rng;

/// Miller-Rabin primality test
///
/// Tests whether n is prime with high probability.
/// Uses k random witnesses (default 20 for ~2^-40 error probability).
///
/// # Arguments
/// * `n` - Number to test for primality
/// * `k` - Number of rounds (more rounds = higher confidence)
///
/// # Returns
/// `true` if n is probably prime, `false` if n is composite
pub fn miller_rabin(n: u64, k: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    // Write n-1 as 2^r × d
    let mut r = 0u32;
    let mut d = n - 1;
    while d % 2 == 0 {
        r += 1;
        d /= 2;
    }

    let mut rng = rand::thread_rng();

    // Test with k random witnesses
    'witness: for _ in 0..k {
        let a = rng.gen_range(2..n - 1);
        let mut x = mod_exp(a, d, n);

        if x == 1 || x == n - 1 {
            continue 'witness;
        }

        for _ in 0..r - 1 {
            x = mod_exp(x, 2, n);
            if x == n - 1 {
                continue 'witness;
            }
        }

        return false; // n is composite
    }

    true // n is probably prime
}

/// Modular exponentiation: (base^exp) mod m
///
/// Uses binary exponentiation for efficiency.
fn mod_exp(base: u64, exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }

    let mut result = 1u128;
    let mut base = (base % m) as u128;
    let mut exp = exp;
    let m = m as u128;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % m;
        }
        exp >>= 1;
        base = (base * base) % m;
    }

    result as u64
}

/// Generate NTT-friendly primes for a given ring dimension
///
/// Generates `count` primes of the form q = k × 2n + 1, where:
/// - n is the ring dimension (e.g., 8192)
/// - q is in the target bit range (e.g., 40-41 bits for scaling primes)
///
/// # Arguments
/// * `n` - Ring dimension (must be a power of 2)
/// * `count` - Number of primes to generate
/// * `bit_size` - Target bit size (e.g., 40 for ~2^40 primes)
/// * `skip_first` - Skip this many primes (for generating disjoint sets)
///
/// # Returns
/// Vector of `count` NTT-friendly primes, sorted in ascending order
///
/// # Example
/// ```
/// // Generate 22 primes of ~40 bits for N=8192
/// let primes = generate_ntt_primes(8192, 22, 40, 0);
/// assert_eq!(primes.len(), 22);
/// ```
pub fn generate_ntt_primes(n: usize, count: usize, bit_size: u32, skip_first: usize) -> Vec<u64> {
    assert!(n.is_power_of_two(), "Ring dimension must be a power of 2");
    assert!(bit_size >= 30 && bit_size <= 60, "Bit size must be in range [30, 60]");

    let multiplier = (2 * n) as u64;
    let min_val = 1u64 << (bit_size - 1); // 2^(bit_size-1)
    let max_val = 1u64 << bit_size;       // 2^bit_size

    // Start searching from the minimum k value
    let mut k = (min_val / multiplier) + 1;

    let mut primes = Vec::with_capacity(count);
    let mut skipped = 0;

    println!("Generating {} NTT-friendly primes (~{}-bit) for N={}...", count, bit_size, n);
    println!("  Formula: q = k × {} + 1", multiplier);
    println!("  Target range: [{}, {})", min_val, max_val);

    if skip_first > 0 {
        println!("  Skipping first {} primes...", skip_first);
    }

    while primes.len() < count {
        let q = k * multiplier + 1;

        // Check if we've exceeded the bit range
        if q >= max_val {
            panic!(
                "Ran out of {}-bit primes after finding {} primes (needed {})",
                bit_size,
                primes.len(),
                count
            );
        }

        // Test primality with Miller-Rabin (20 rounds = ~2^-40 error probability)
        if miller_rabin(q, 20) {
            if skipped < skip_first {
                skipped += 1;
            } else {
                primes.push(q);
                if primes.len() % 5 == 0 || primes.len() == count {
                    println!("  Found {}/{} primes (latest: q={}, k={})",
                             primes.len(), count, q, k);
                }
            }
        }

        k += 1;

        // Safety check to prevent infinite loops
        if k > (max_val / multiplier) + 100000 {
            panic!(
                "Failed to find {} primes in reasonable search space (found {} primes)",
                count,
                primes.len()
            );
        }
    }

    println!("✓ Generated {} NTT-friendly primes successfully!\n", count);

    primes
}

/// Generate a single large special modulus (typically 60-bit)
///
/// The special modulus is used as the first prime in CKKS parameter sets.
/// It's larger than the scaling primes to support efficient modular reduction.
///
/// # Arguments
/// * `n` - Ring dimension
/// * `bit_size` - Target bit size (typically 60)
///
/// # Returns
/// A single NTT-friendly prime of the specified bit size
pub fn generate_special_modulus(n: usize, bit_size: u32) -> u64 {
    let primes = generate_ntt_primes(n, 1, bit_size, 0);
    primes[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_miller_rabin() {
        // Known primes
        assert!(miller_rabin(2, 10));
        assert!(miller_rabin(3, 10));
        assert!(miller_rabin(5, 10));
        assert!(miller_rabin(7, 10));
        assert!(miller_rabin(11, 10));
        assert!(miller_rabin(1099511922689, 20)); // 41-bit NTT-friendly prime

        // Known composites
        assert!(!miller_rabin(4, 10));
        assert!(!miller_rabin(6, 10));
        assert!(!miller_rabin(8, 10));
        assert!(!miller_rabin(9, 10));
        assert!(!miller_rabin(15, 10));
    }

    #[test]
    fn test_generate_ntt_primes_small() {
        let primes = generate_ntt_primes(512, 5, 30, 0);
        assert_eq!(primes.len(), 5);

        // Verify all are NTT-friendly: q ≡ 1 (mod 2n)
        let multiplier = 2 * 512;
        for &q in &primes {
            assert_eq!((q - 1) % multiplier, 0, "Prime {} is not NTT-friendly", q);
            assert!(miller_rabin(q, 20), "Generated non-prime: {}", q);
        }
    }

    #[test]
    fn test_generate_special_modulus() {
        let special = generate_special_modulus(8192, 60);

        // Verify it's NTT-friendly
        assert_eq!((special - 1) % (2 * 8192), 0);

        // Verify it's in the right bit range
        assert!(special >= (1u64 << 59));
        assert!(special < (1u64 << 60));

        // Verify it's prime
        assert!(miller_rabin(special, 20));
    }

    #[test]
    fn test_skip_first() {
        let primes1 = generate_ntt_primes(512, 3, 30, 0);
        let primes2 = generate_ntt_primes(512, 3, 30, 3);

        // The second set should be completely different
        assert_ne!(primes1[0], primes2[0]);
        assert_ne!(primes1[1], primes2[1]);
        assert_ne!(primes1[2], primes2[2]);
    }
}
