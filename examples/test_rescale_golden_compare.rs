//! Golden Compare Test for GPU Rescaling

use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use num_bigint::BigUint;

fn main() -> Result<(), String> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║         GPU Rescaling Golden Compare Test                    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Create params
    println!("Creating GPU context...");
    let params = CliffordFHEParams::new_v3_bootstrap_metal()?;
    let n = params.n;
    let moduli_slice = &params.moduli[..3];  // Use first 3 primes
    let moduli: Vec<u64> = moduli_slice.to_vec();

    let gpu_ctx = MetalCkksContext::new(params)?;

    // Generate random test polynomial
    println!("Generating {} random coefficients...", 100);
    let num_test_coeffs = 100;
    let num_primes_in = 3;
    let mut test_poly = vec![0u64; n * num_primes_in];

    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Generate in flat RNS layout: poly_in[prime_idx * n + coeff_idx]
    for prime_idx in 0..num_primes_in {
        let q = moduli[prime_idx];
        for coeff_idx in 0..num_test_coeffs {
            test_poly[prime_idx * n + coeff_idx] = rng.gen::<u64>() % q;
        }
    }

    // CPU reference
    println!("Computing CPU reference...");
    let cpu_result = cpu_rescale_reference(&test_poly, &moduli, n)?;

    // GPU test
    println!("Computing GPU result...");
    let level = 2;
    let gpu_result = gpu_ctx.exact_rescale_gpu_fixed(&test_poly, level)?;

    // Compare
    println!("\nComparing {} coefficients...\n", num_test_coeffs);
    let num_primes_out = 2;
    let mut mismatch_count = 0;

    for coeff_idx in 0..num_test_coeffs {
        for prime_idx in 0..num_primes_out {
            // Read from flat RNS layout
            let idx = prime_idx * n + coeff_idx;
            let cpu_val = cpu_result[idx];
            let gpu_val = gpu_result[idx];

            if cpu_val != gpu_val {
                mismatch_count += 1;
                if mismatch_count == 1 {
                    println!("🔍 FIRST MISMATCH:");
                    println!("  Coeff: {}, Prime: {}", coeff_idx, prime_idx);
                    println!("  CPU: {}", cpu_val);
                    println!("  GPU: {}", gpu_val);

                    // Debug info (read from flat RNS layout)
                    let q_last = moduli[2];
                    let r_last = test_poly[2 * n + coeff_idx];
                    let r_i = test_poly[prime_idx * n + coeff_idx];
                    let q_i = moduli[prime_idx];

                    println!("\n  Inputs:");
                    println!("    q_last = {}", q_last);
                    println!("    r_last = {}", r_last);
                    println!("    q_i = {}", q_i);
                    println!("    r_i = {}", r_i);

                    // Manual computation
                    let half_last = q_last >> 1;
                    let r_last_rounded = (r_last + half_last) % q_last;
                    let r_last_mod_qi = r_last_rounded % q_i;
                    let half_mod_qi = half_last % q_i;
                    let r_last_adjusted = if r_last_mod_qi >= half_mod_qi {
                        r_last_mod_qi - half_mod_qi
                    } else {
                        r_last_mod_qi + q_i - half_mod_qi
                    };
                    let diff = if r_i >= r_last_adjusted {
                        r_i - r_last_adjusted
                    } else {
                        r_i + q_i - r_last_adjusted
                    };

                    let qtop_inv = gpu_ctx.rescale_inv_table[level][prime_idx];
                    let manual = ((diff as u128) * (qtop_inv as u128)) % (q_i as u128);

                    println!("\n  GPU-side computation:");
                    println!("    diff = {}", diff);
                    println!("    qtop_inv = {}", qtop_inv);
                    println!("    manual result = {}", manual);

                    if manual == cpu_val as u128 {
                        println!("\n  ✅ Manual matches CPU reference");
                        println!("  ❌ GPU shader mul_mod_128 has a bug");
                    } else {
                        println!("\n  ❌ Manual doesn't match CPU - algorithm issue");
                    }
                }
            }
        }
    }

    println!("\nTotal mismatches: {}/{}", mismatch_count, num_test_coeffs * num_primes_out);

    if mismatch_count == 0 {
        println!("✅ SUCCESS: All coefficients match!");
        Ok(())
    } else {
        Err(format!("Found {} mismatches", mismatch_count))
    }
}

/// CPU reference using flat RNS layout (matching GPU)
/// Input: poly_in[prime_idx * n + coeff_idx]
/// Output: result[prime_idx * n + coeff_idx]
fn cpu_rescale_reference(poly_in: &[u64], moduli: &[u64], n: usize) -> Result<Vec<u64>, String> {
    let num_primes_in = moduli.len();
    let num_primes_out = num_primes_in - 1;
    let q_last = moduli[num_primes_in - 1];
    let mut result = vec![0u64; n * num_primes_out];

    for coeff_idx in 0..n {
        let mut c = BigUint::from(0u64);
        let mut q_product = BigUint::from(1u64);
        for &q in moduli.iter() {
            q_product *= BigUint::from(q);
        }

        // Read from flat RNS layout
        for (prime_idx, &q) in moduli.iter().enumerate() {
            let r = poly_in[prime_idx * n + coeff_idx];  // Flat RNS layout
            let q_i = BigUint::from(q);
            let q_hat = &q_product / &q_i;
            let q_hat_inv = mod_inverse_bigint(&q_hat, q).ok_or("Inverse failed")?;
            let term = BigUint::from(r) * &q_hat * q_hat_inv;
            c = (c + term) % &q_product;
        }

        let q_last_big = BigUint::from(q_last);
        let c_rounded = (c + &q_last_big / 2u64) / q_last_big;

        // Write to flat RNS layout
        for (out_idx, &q) in moduli[..num_primes_out].iter().enumerate() {
            let r = (&c_rounded % BigUint::from(q)).to_u64_digits().first().copied().unwrap_or(0);
            result[out_idx * n + coeff_idx] = r;  // Flat RNS layout
        }
    }
    Ok(result)
}

fn mod_inverse_bigint(a: &BigUint, m: u64) -> Option<BigUint> {
    use num_bigint::BigInt;
    let m_big = BigUint::from(m);
    let (g, x, _) = extended_gcd_bigint(a.clone(), m_big.clone());
    if g != BigUint::from(1u64) { return None; }
    let m_bigint = BigInt::from(m);
    let x_mod = ((x % &m_bigint) + &m_bigint) % &m_bigint;
    x_mod.to_biguint()
}

fn extended_gcd_bigint(a: BigUint, b: BigUint) -> (BigUint, num_bigint::BigInt, num_bigint::BigInt) {
    use num_bigint::BigInt;
    if b == BigUint::from(0u64) {
        return (a, BigInt::from(1), BigInt::from(0));
    }
    let (g, x1, y1) = extended_gcd_bigint(b.clone(), &a % &b);
    let x = y1.clone();
    let y = x1 - BigInt::from(&a / &b) * y1;
    (g, x, y)
}
