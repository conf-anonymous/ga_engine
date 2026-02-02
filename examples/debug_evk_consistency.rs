//! Debug: Check EVK consistency across primes
//!
//! The bug shows inconsistent errors across primes. Let's check if EVK values
//! satisfy the expected relationship across primes.

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::{KeyContext, EvaluationKey},
        ntt::NttContext,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("EVK CONSISTENCY CHECK\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (_pk, sk, cpu_evk) = key_ctx.keygen();

    let device = Arc::new(MetalDevice::new()?);
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let ntt_contexts = metal_ctx.ntt_contexts();

    let metal_evk = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        20,
    )?;

    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];

    // Get EVK keys
    let (metal_evk0, metal_evk1) = metal_evk.get_coeff_keys(level)?;
    let (cpu_evk0, cpu_evk1) = evk_to_flat(&cpu_evk, num_primes);

    // Compute s² for each prime
    let mut s_sq = vec![vec![0u64; n]; num_primes];
    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];
        let ntt_ctx = NttContext::new(n, q);
        let s_poly: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[prime_idx]).collect();
        s_sq[prime_idx] = ntt_ctx.multiply_polynomials(&s_poly, &s_poly);
    }

    println!("s²[coeff=0] across primes: {:?}\n",
        (0..num_primes).map(|j| s_sq[j][0]).collect::<Vec<_>>());

    // For the EVK identity to work correctly:
    // evk0[t] - evk1[t] × s = -B^t × s² + error
    //
    // The "error" should be the SAME integer value when viewed mod each prime.
    // Let's compute the error for each prime and check consistency.

    let base_w = 20u32;
    let base = 1u64 << base_w;

    println!("=== CPU EVK Identity Check ===\n");
    for t in 0..3.min(cpu_evk0.len()) {
        // Compute B^t mod each prime
        let bt: Vec<u64> = moduli.iter().map(|&q| {
            let mut p = 1u128;
            for _ in 0..t {
                p = (p * base as u128) % q as u128;
            }
            p as u64
        }).collect();

        println!("Digit {}: B^{} mod q = {:?}", t, t, bt);

        // For coeff 0, compute error = evk0 - evk1×s + B^t×s²
        let mut errors = vec![0i64; num_primes];
        for prime_idx in 0..num_primes {
            let q = moduli[prime_idx];
            let ntt_ctx = NttContext::new(n, q);

            // Extract evk0, evk1 for this prime
            let evk0_prime: Vec<u64> = (0..n).map(|i| cpu_evk0[t][i * num_primes + prime_idx]).collect();
            let evk1_prime: Vec<u64> = (0..n).map(|i| cpu_evk1[t][i * num_primes + prime_idx]).collect();
            let s_prime: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[prime_idx]).collect();

            // evk1 × s
            let evk1_s = ntt_ctx.multiply_polynomials(&evk1_prime, &s_prime);

            // evk0 - evk1×s
            let diff = if evk0_prime[0] >= evk1_s[0] {
                evk0_prime[0] - evk1_s[0]
            } else {
                q - (evk1_s[0] - evk0_prime[0])
            };

            // + B^t × s²
            let bt_s2 = ((bt[prime_idx] as u128 * s_sq[prime_idx][0] as u128) % q as u128) as u64;
            let result = ((diff as u128 + bt_s2 as u128) % q as u128) as u64;

            // Center
            errors[prime_idx] = if result > q/2 { result as i64 - q as i64 } else { result as i64 };
        }

        println!("  Error (coeff 0): {:?}", errors);
        let consistent = errors.iter().all(|&e| (e - errors[0]).abs() < 100);
        println!("  Consistent: {}\n", if consistent { "✅" } else { "❌" });
    }

    println!("=== Metal EVK Identity Check ===\n");
    for t in 0..3.min(metal_evk0.len()) {
        let bt: Vec<u64> = moduli.iter().map(|&q| {
            let mut p = 1u128;
            for _ in 0..t {
                p = (p * base as u128) % q as u128;
            }
            p as u64
        }).collect();

        println!("Digit {}: B^{} mod q = {:?}", t, t, bt);

        let mut errors = vec![0i64; num_primes];
        for prime_idx in 0..num_primes {
            let q = moduli[prime_idx];
            let ntt_ctx = NttContext::new(n, q);

            let evk0_prime: Vec<u64> = (0..n).map(|i| metal_evk0[t][i * num_primes + prime_idx]).collect();
            let evk1_prime: Vec<u64> = (0..n).map(|i| metal_evk1[t][i * num_primes + prime_idx]).collect();
            let s_prime: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[prime_idx]).collect();

            let evk1_s = ntt_ctx.multiply_polynomials(&evk1_prime, &s_prime);

            let diff = if evk0_prime[0] >= evk1_s[0] {
                evk0_prime[0] - evk1_s[0]
            } else {
                q - (evk1_s[0] - evk0_prime[0])
            };

            let bt_s2 = ((bt[prime_idx] as u128 * s_sq[prime_idx][0] as u128) % q as u128) as u64;
            let result = ((diff as u128 + bt_s2 as u128) % q as u128) as u64;

            errors[prime_idx] = if result > q/2 { result as i64 - q as i64 } else { result as i64 };
        }

        println!("  Error (coeff 0): {:?}", errors);
        let consistent = errors.iter().all(|&e| (e - errors[0]).abs() < 100);
        println!("  Consistent: {}\n", if consistent { "✅" } else { "❌" });
    }

    // Now check if the a_t values (evk1) are actually the same polynomial
    // (they should be - same random a_t sampled for all primes)
    println!("=== EVK1 Consistency Check (should be SAME polynomial) ===\n");

    for t in 0..2.min(metal_evk1.len()) {
        println!("Digit {} evk1 values (coeff 0):", t);
        println!("  CPU:   {:?}", (0..num_primes).map(|j| cpu_evk1[t][j]).collect::<Vec<_>>());
        println!("  Metal: {:?}", (0..num_primes).map(|j| metal_evk1[t][j]).collect::<Vec<_>>());

        // The values should represent the same integer mod each prime
        // For a random polynomial, this is always true by construction
        println!();
    }

    Ok(())
}

#[cfg(feature = "v2-gpu-metal")]
fn evk_to_flat(evk: &EvaluationKey, num_primes: usize) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
    let num_digits = evk.evk0.len();
    let n = evk.evk0[0].len();
    let mut evk0 = Vec::with_capacity(num_digits);
    let mut evk1 = Vec::with_capacity(num_digits);
    for t in 0..num_digits {
        let mut e0 = vec![0u64; n * num_primes];
        let mut e1 = vec![0u64; n * num_primes];
        for i in 0..n {
            for j in 0..num_primes {
                e0[i * num_primes + j] = evk.evk0[t][i].values[j];
                e1[i * num_primes + j] = evk.evk1[t][i].values[j];
            }
        }
        evk0.push(e0);
        evk1.push(e1);
    }
    (evk0, evk1)
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
