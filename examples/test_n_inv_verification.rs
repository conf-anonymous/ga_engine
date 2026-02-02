//! Verify n_inv computation per expert's diagnostic
//!
//! Expert said to check:
//! 1. (n * n_inv_normal) % q == 1
//! 2. (n * from_mont(n_inv_mont)) % q == 1
//!
//! This will tell us if the 512× bug is due to wrong n_inv constant.

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

fn mod_inverse(a: u64, q: u64) -> u64 {
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
        panic!("{} has no modular inverse mod {}", a, q);
    }

    // Ensure positive result
    if old_s < 0 {
        (old_s + q as i128) as u64
    } else {
        old_s as u64
    }
}

fn compute_q_inv(q: u64) -> u64 {
    assert!(q & 1 == 1, "q must be odd");
    let mut q_inv = q;
    for _ in 0..5 {
        q_inv = q_inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(q_inv)));
    }
    q_inv.wrapping_neg()
}

fn compute_r_squared_mod_q(q: u64) -> u64 {
    let r_mod_q = ((1u128 << 64) % q as u128) as u64;
    ((r_mod_q as u128 * r_mod_q as u128) % q as u128) as u64
}

fn to_montgomery(x: u64, r_squared: u64, q: u64, q_inv: u64) -> u64 {
    mont_mul(x, r_squared, q, q_inv)
}

fn from_montgomery(x_mont: u64, q: u64, q_inv: u64) -> u64 {
    mont_mul(x_mont, 1, q, q_inv)
}

fn mont_mul(a: u64, b: u64, q: u64, q_inv: u64) -> u64 {
    let t = a as u128 * b as u128;
    let t_lo = t as u64;
    let t_hi = (t >> 64) as u64;

    let m = t_lo.wrapping_mul(q_inv);

    let mq = m as u128 * q as u128;
    let mq_lo = mq as u64;
    let mq_hi = (mq >> 64) as u64;

    let (sum_lo, carry1) = t_lo.overflowing_add(mq_lo);
    let (sum_hi, carry2) = t_hi.overflowing_add(mq_hi);
    let sum_hi = sum_hi.wrapping_add(carry1 as u64).wrapping_add(carry2 as u64);

    if sum_hi >= q {
        sum_hi - q
    } else {
        sum_hi
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        n_inv Verification Test - Expert's Diagnostic            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n as u64;

    println!("Testing for N = {} (2^10)\n", n);
    println!("Expected: 512× error suggests 2^9 normalization bug\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Test first 3 primes (matching the test case)
    for (idx, &q) in params.moduli.iter().take(3).enumerate() {
        println!("Prime {} (q = {}):", idx, q);

        // Compute Montgomery parameters
        let q_inv = compute_q_inv(q);
        let r_squared = compute_r_squared_mod_q(q);

        // Compute n_inv in normal domain
        let n_inv_normal = mod_inverse(n, q);

        // Convert to Montgomery domain
        let n_inv_mont = to_montgomery(n_inv_normal, r_squared, q, q_inv);

        // TEST 1: Verify (n * n_inv_normal) % q == 1
        let test1 = ((n as u128 * n_inv_normal as u128) % q as u128) as u64;
        println!("  TEST 1: (n * n_inv_normal) % q = {}", test1);
        if test1 == 1 {
            println!("          ✓ PASS - n_inv_normal is correct");
        } else {
            println!("          ✗ FAIL - n_inv_normal is WRONG!");
        }

        // TEST 2: Verify (n * from_mont(n_inv_mont)) % q == 1
        let n_inv_from_mont = from_montgomery(n_inv_mont, q, q_inv);
        let test2 = ((n as u128 * n_inv_from_mont as u128) % q as u128) as u64;
        println!("  TEST 2: (n * from_mont(n_inv_mont)) % q = {}", test2);
        if test2 == 1 {
            println!("          ✓ PASS - n_inv_mont converts correctly");
        } else {
            println!("          ✗ FAIL - n_inv_mont is WRONG!");
        }

        // Additional diagnostic: What is n_inv_mont supposed to be?
        // n_inv_mont = n_inv * R mod q = n_inv * 2^64 mod q
        let expected_n_inv_mont = ((n_inv_normal as u128 * ((1u128 << 64) % q as u128)) % q as u128) as u64;
        println!("  n_inv_normal      = {}", n_inv_normal);
        println!("  n_inv_mont (got)  = {}", n_inv_mont);
        println!("  n_inv_mont (exp)  = {}", expected_n_inv_mont);

        if n_inv_mont == expected_n_inv_mont {
            println!("          ✓ Montgomery conversion matches manual calculation");
        } else {
            println!("          ✗ Montgomery conversion DIFFERS from manual!");
        }

        // TEST 3: Check if error is exactly 2^9 = 512
        // If n_inv is off by a factor of 2^9, then:
        // wrong_n_inv = correct_n_inv / 512
        let wrong_n_inv_512 = if n_inv_normal >= 512 {
            n_inv_normal / 512
        } else {
            // Compute (n_inv_normal / 512) mod q
            ((n_inv_normal as u128 * mod_inverse(512, q) as u128) % q as u128) as u64
        };

        let test3 = ((n as u128 * wrong_n_inv_512 as u128) % q as u128) as u64;
        println!("  TEST 3: If n_inv was off by 512×:");
        println!("          (n * (n_inv / 512)) % q = {}", test3);
        println!("          (should be 512 if that's the bug)");

        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("Summary:");
    println!("  If all tests PASS but we still have 512× error:");
    println!("  → Bug is NOT in n_inv computation");
    println!("  → Bug is in NTT kernel (per-stage halving or double normalization)\n");
    println!("  If TEST 1 or 2 FAIL:");
    println!("  → Bug IS in n_inv computation or Montgomery conversion\n");
}
