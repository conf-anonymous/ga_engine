//! Unit test for RNS divide-and-round formula
//!
//! This test verifies the correct RNS formula for ⌊(C + q_top/2) / q_top⌋ mod qi

fn mod_inverse(a: u64, m: u64) -> u64 {
    // Extended Euclidean algorithm
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }

    if old_s < 0 {
        old_s += m as i128;
    }
    old_s as u64
}

fn rns_divide_round_reference(r0: u64, r1: u64, r_top: u64, q0: u64, q1: u64, q_top: u64) -> (u64, u64) {
    // Reference implementation using BigInt
    use num_bigint::BigInt;
    use num_traits::ToPrimitive;

    // Step 1: CRT reconstruct C from (r0, r1, r_top)
    let q0_big = BigInt::from(q0);
    let q1_big = BigInt::from(q1);
    let qtop_big = BigInt::from(q_top);

    let Q = &q0_big * &q1_big * &qtop_big;

    // CRT formula
    let M0 = &Q / &q0_big;
    let M1 = &Q / &q1_big;
    let Mtop = &Q / &qtop_big;

    let y0 = M0.modinv(&q0_big).unwrap();
    let y1 = M1.modinv(&q1_big).unwrap();
    let ytop = Mtop.modinv(&qtop_big).unwrap();

    let mut C = BigInt::from(0);
    C += BigInt::from(r0) * &M0 * y0;
    C += BigInt::from(r1) * &M1 * y1;
    C += BigInt::from(r_top) * &Mtop * ytop;
    C %= &Q;

    // Step 2: Compute ⌊(C + q_top/2) / q_top⌋
    let half_qtop = &qtop_big / 2;
    let C_rounded = &C + &half_qtop;
    let result_big = C_rounded / &qtop_big;

    // Step 3: Reduce mod q0 and q1
    let res0_big: BigInt = &result_big % &q0_big;
    let res1_big: BigInt = &result_big % &q1_big;
    let res0 = res0_big.to_u64().unwrap();
    let res1 = res1_big.to_u64().unwrap();

    (res0, res1)
}

fn rns_divide_round_v1(r0: u64, r1: u64, r_top: u64, q0: u64, q1: u64, q_top: u64) -> (u64, u64) {
    // Version 1: Add rounding in last limb, then map
    let qtop_inv_q0 = mod_inverse(q_top, q0);
    let qtop_inv_q1 = mod_inverse(q_top, q1);

    let half_qtop = q_top / 2;
    let r_top_rounded = (r_top + half_qtop) % q_top;

    // Map rounded r_top into q0 and q1
    let mapped_top_q0 = (r_top_rounded * qtop_inv_q0) % q0;
    let mapped_top_q1 = (r_top_rounded * qtop_inv_q1) % q1;

    // Subtract and divide
    let diff0 = if r0 >= mapped_top_q0 { r0 - mapped_top_q0 } else { r0 + q0 - mapped_top_q0 };
    let diff1 = if r1 >= mapped_top_q1 { r1 - mapped_top_q1 } else { r1 + q1 - mapped_top_q1 };

    let res0 = (diff0 * qtop_inv_q0) % q0;
    let res1 = (diff1 * qtop_inv_q1) % q1;

    (res0, res1)
}

fn rns_divide_round_v2(r0: u64, r1: u64, r_top: u64, q0: u64, q1: u64, q_top: u64) -> (u64, u64) {
    // Version 2: Add rounding term to each ri
    let qtop_inv_q0 = mod_inverse(q_top, q0);
    let qtop_inv_q1 = mod_inverse(q_top, q1);

    let half_qtop_mod_q0 = (q_top / 2) % q0;
    let half_qtop_mod_q1 = (q_top / 2) % q1;

    let r0_rounded = (r0 + half_qtop_mod_q0) % q0;
    let r1_rounded = (r1 + half_qtop_mod_q1) % q1;

    let r_top_mod_q0 = r_top % q0;
    let r_top_mod_q1 = r_top % q1;

    let diff0 = if r0_rounded >= r_top_mod_q0 { r0_rounded - r_top_mod_q0 } else { r0_rounded + q0 - r_top_mod_q0 };
    let diff1 = if r1_rounded >= r_top_mod_q1 { r1_rounded - r_top_mod_q1 } else { r1_rounded + q1 - r_top_mod_q1 };

    let res0 = (diff0 * qtop_inv_q0) % q0;
    let res1 = (diff1 * qtop_inv_q1) % q1;

    (res0, res1)
}

fn rns_divide_round_v3(r0: u64, r1: u64, r_top: u64, q0: u64, q1: u64, q_top: u64) -> (u64, u64) {
    // Version 3: Compute k = (C - r_top) / q_top, then add rounding bit
    let qtop_inv_q0 = mod_inverse(q_top, q0);
    let qtop_inv_q1 = mod_inverse(q_top, q1);

    let r_top_mod_q0 = r_top % q0;
    let r_top_mod_q1 = r_top % q1;

    // k ≡ (r_i - r_top) * q_top^{-1} (mod qi) - this is the quotient without rounding
    let diff0 = if r0 >= r_top_mod_q0 { r0 - r_top_mod_q0 } else { r0 + q0 - r_top_mod_q0 };
    let diff1 = if r1 >= r_top_mod_q1 { r1 - r_top_mod_q1 } else { r1 + q1 - r_top_mod_q1 };

    let k0 = (diff0 * qtop_inv_q0) % q0;
    let k1 = (diff1 * qtop_inv_q1) % q1;

    // Rounding bit: 1 if (r_top + q_top/2) >= q_top, else 0
    let round_bit = if r_top >= (q_top - (q_top / 2)) { 1u64 } else { 0u64 };

    let res0 = (k0 + round_bit) % q0;
    let res1 = (k1 + round_bit) % q1;

    (res0, res1)
}

fn rns_divide_round_subtractive(r0: u64, r1: u64, r_top: u64, q0: u64, q1: u64, q_top: u64) -> (u64, u64) {
    // Subtractive algorithm for RNS rescaling
    let qtop_inv_q0 = mod_inverse(q_top, q0);
    let qtop_inv_q1 = mod_inverse(q_top, q1);

    // Add (q_top-1)/2 to r_top for rounding
    let half = q_top >> 1;
    let r_top_rounded = (r_top + half) % q_top;

    // For q0:
    let temp0 = r_top_rounded % q0;
    let half_mod_q0 = half % q0;
    let r_top_mod_q0 = if temp0 >= half_mod_q0 { temp0 - half_mod_q0 } else { temp0 + q0 - half_mod_q0 };
    let diff0 = if r0 >= r_top_mod_q0 { r0 - r_top_mod_q0 } else { r0 + q0 - r_top_mod_q0 };
    let res0 = (diff0 * qtop_inv_q0) % q0;

    // For q1:
    let temp1 = r_top_rounded % q1;
    let half_mod_q1 = half % q1;
    let r_top_mod_q1 = if temp1 >= half_mod_q1 { temp1 - half_mod_q1 } else { temp1 + q1 - half_mod_q1 };
    let diff1 = if r1 >= r_top_mod_q1 { r1 - r_top_mod_q1 } else { r1 + q1 - r_top_mod_q1 };
    let res1 = (diff1 * qtop_inv_q1) % q1;

    (res0, res1)
}

fn main() {
    // Test primes
    let q0: u64 = 17592186112001;  // ~45-bit NTT-friendly
    let q1: u64 = 17592186129409;  // ~45-bit NTT-friendly
    let q_top: u64 = 17592186146817; // ~45-bit NTT-friendly

    println!("Testing RNS divide-and-round formulas");
    println!("q0 = {}", q0);
    println!("q1 = {}", q1);
    println!("q_top = {}", q_top);
    println!();

    // Test with a few random values
    let test_cases = vec![
        (1000u64, 2000u64, 3000u64),
        (q0/2, q1/2, q_top/2),
        (q0-1, q1-1, q_top-1),
        (123456789, 987654321, 555555555),
    ];

    for (r0, r1, r_top) in test_cases {
        let (ref0, ref1) = rns_divide_round_reference(r0, r1, r_top, q0, q1, q_top);
        let (sub0, sub1) = rns_divide_round_subtractive(r0, r1, r_top, q0, q1, q_top);

        println!("Input: r0={}, r1={}, r_top={}", r0, r1, r_top);
        println!("  Reference (BigInt):      ({}, {})", ref0, ref1);
        println!("  Subtractive Algorithm:   ({}, {}) - {}", sub0, sub1, if sub0 == ref0 && sub1 == ref1 { "✅ MATCH" } else { "❌ FAIL" });
        println!();
    }
}
