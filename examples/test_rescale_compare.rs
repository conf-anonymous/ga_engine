//! Compare CPU and Metal rescaling on identical input

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::ckks::MetalCkksContext,
    backends::cpu_optimized::rns::RnsRepresentation,
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn mod_pow(base: u64, exp: u64, modulus: u64) -> u64 {
    let mut result = 1u128;
    let mut base = (base as u128) % (modulus as u128);
    let mut exp = exp;
    let modulus = modulus as u128;
    
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp /= 2;
        base = (base * base) % modulus;
    }
    result as u64
}

#[cfg(feature = "v2-gpu-metal")]
fn rescale_polynomial_cpu(
    poly_flat: &[u64],
    n: usize,
    num_primes: usize,
    q_last: u64,
    new_moduli: &[u64],
) -> Vec<u64> {
    let num_primes_out = new_moduli.len();
    let mut result = vec![0u64; n * num_primes_out];

    // Precompute q_last^(-1) mod qi for each remaining prime
    let q_last_inv: Vec<u64> = new_moduli
        .iter()
        .map(|&qi| mod_pow(q_last % qi, qi - 2, qi))
        .collect();

    for i in 0..n {
        // Get the value mod q_last (from last position)
        let val_mod_qlast = poly_flat[i * num_primes + (num_primes - 1)];

        // Centered lift: convert to signed value
        let val_centered = if val_mod_qlast > q_last / 2 {
            val_mod_qlast as i64 - q_last as i64
        } else {
            val_mod_qlast as i64
        };

        // For each remaining prime qi:
        for (j, &qi) in new_moduli.iter().enumerate() {
            let old_val = poly_flat[i * num_primes + j];

            // Compute (old_val - val_centered) mod qi
            let diff = if val_centered >= 0 {
                let vc = (val_centered as u64) % qi;
                if old_val >= vc {
                    old_val - vc
                } else {
                    qi - (vc - old_val)
                }
            } else {
                let vc = ((-val_centered) as u64) % qi;
                (old_val + vc) % qi
            };

            // Multiply by q_last^(-1) mod qi
            let new_val = ((diff as u128) * (q_last_inv[j] as u128)) % (qi as u128);
            result[i * num_primes_out + j] = new_val as u64;
        }
    }

    result
}

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Rescale Comparison Test");
    println!("=======================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    
    let n = params.n;
    let level = 2;
    let num_primes = level + 1;  // 3
    let moduli = &params.moduli[..num_primes];
    
    // Create test polynomial with known values
    let mut poly_flat = vec![0u64; n * num_primes];
    for i in 0..n {
        for j in 0..num_primes {
            // Use deterministic test values
            let val = ((i as u64 * 12345 + j as u64 * 67890) % moduli[j]) as u64;
            poly_flat[i * num_primes + j] = val;
        }
    }
    
    println!("Test polynomial:");
    println!("  n = {}", n);
    println!("  num_primes = {}", num_primes);
    println!("  moduli = {:?}", moduli);
    println!("  poly[0] = {:?}", (0..num_primes).map(|j| poly_flat[0 * num_primes + j]).collect::<Vec<_>>());
    println!("  poly[1] = {:?}", (0..num_primes).map(|j| poly_flat[1 * num_primes + j]).collect::<Vec<_>>());
    
    // CPU rescale
    let q_last = moduli[num_primes - 1];
    let new_moduli = &moduli[..num_primes - 1];
    let cpu_result = rescale_polynomial_cpu(&poly_flat, n, num_primes, q_last, new_moduli);
    
    println!("\nCPU rescale result:");
    println!("  output[0] = {:?}", (0..num_primes-1).map(|j| cpu_result[0 * (num_primes-1) + j]).collect::<Vec<_>>());
    println!("  output[1] = {:?}", (0..num_primes-1).map(|j| cpu_result[1 * (num_primes-1) + j]).collect::<Vec<_>>());
    
    // Metal rescale
    let metal_result = metal_ctx.exact_rescale_gpu(&poly_flat, level)?;
    
    println!("\nMetal rescale result:");
    println!("  output[0] = {:?}", (0..num_primes-1).map(|j| metal_result[0 * (num_primes-1) + j]).collect::<Vec<_>>());
    println!("  output[1] = {:?}", (0..num_primes-1).map(|j| metal_result[1 * (num_primes-1) + j]).collect::<Vec<_>>());
    
    // Compare
    let mut match_count = 0;
    let mut mismatch_count = 0;
    for i in 0..n {
        for j in 0..(num_primes - 1) {
            let cpu_val = cpu_result[i * (num_primes - 1) + j];
            let metal_val = metal_result[i * (num_primes - 1) + j];
            if cpu_val == metal_val {
                match_count += 1;
            } else {
                mismatch_count += 1;
                if mismatch_count <= 5 {
                    println!("MISMATCH at coeff[{}] prime[{}]: CPU={}, Metal={}", i, j, cpu_val, metal_val);
                }
            }
        }
    }
    
    println!("\nComparison:");
    println!("  Matches: {}", match_count);
    println!("  Mismatches: {}", mismatch_count);
    
    if mismatch_count == 0 {
        println!("\n✅ CPU and Metal rescaling match perfectly!");
        Ok(())
    } else {
        println!("\n❌ Rescaling mismatch!");
        Err(format!("{} mismatches found", mismatch_count))
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
