//! Debug CUDA inverse NTT for n=8192
//! Check if the issue is in the butterfly stages or the final n_inv scaling

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ntt::CudaNttContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext as CpuNttContext;

fn main() -> Result<(), String> {
    let n = 8192;
    let q = 1152921504606994433u64;

    let psi = find_psi(n, q)?;
    let omega = ((psi as u128 * psi as u128) % q as u128) as u64;

    println!("Testing CUDA inverse NTT for n={}", n);
    println!("q={}", q);
    println!("psi={}, omega={}\n", psi, omega);

    let cuda_ntt = CudaNttContext::new(n, q, omega)?;
    let cpu_ntt = CpuNttContext::new(n, q);

    // Start with a known NTT-domain value (result of forward NTT)
    let mut test_poly = vec![0u64; n];
    test_poly[0] = 1;
    test_poly[1] = 2;
    test_poly[2] = 3;

    // Get NTT representation using CPU (known good)
    let mut ntt_values = test_poly.clone();
    cpu_ntt.forward_ntt(&mut ntt_values);

    println!("Starting with NTT-domain values from CPU forward NTT");
    println!("ntt_values[0..5] = {:?}\n", &ntt_values[0..5]);

    // CPU inverse for reference
    let mut cpu_result = ntt_values.clone();
    cpu_ntt.inverse_ntt(&mut cpu_result);
    println!("CPU inverse result[0..5] = {:?}", &cpu_result[0..5]);

    // CUDA inverse
    let mut cuda_result = ntt_values.clone();
    cuda_ntt.inverse(&mut cuda_result)?;
    println!("CUDA inverse result[0..5] = {:?}", &cuda_result[0..5]);

    // Compare
    println!("\nComparison:");
    let mut mismatches = 0;
    for i in 0..20 {
        let match_str = if cpu_result[i] == cuda_result[i] { "✓" } else { "✗" };
        println!("[{}] Original={:5} CPU={:5} CUDA={:20} {}",
            i, test_poly[i], cpu_result[i], cuda_result[i], match_str);
        if cpu_result[i] != cuda_result[i] {
            mismatches += 1;
        }
    }

    // Check all
    let mut total_mismatches = 0;
    for i in 0..n {
        if cpu_result[i] != cuda_result[i] {
            total_mismatches += 1;
        }
    }

    if total_mismatches == 0 {
        println!("\n✅ Inverse NTT matches CPU!");
        Ok(())
    } else {
        println!("\n❌ Inverse NTT has {} mismatches", total_mismatches);

        // Check if it's just a scaling issue
        println!("\nChecking if results differ by a constant factor...");
        if cuda_result[0] != 0 && cpu_result[0] != 0 {
            let ratio = (cuda_result[0] as f64) / (cpu_result[0] as f64);
            println!("Ratio cuda[0]/cpu[0] = {}", ratio);
        }

        Err(format!("{} mismatches in inverse NTT", total_mismatches))
    }
}

fn find_psi(n: usize, q: u64) -> Result<u64, String> {
    let two_n = 2 * n as u64;
    for g in 2..100u64 {
        let psi = pow_mod(g, (q - 1) / two_n, q);
        if pow_mod(psi, n as u64, q) == q - 1 {
            return Ok(psi);
        }
    }
    Err("Could not find psi".to_string())
}

fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % m as u128) as u64;
        }
        base = ((base as u128 * base as u128) % m as u128) as u64;
        exp >>= 1;
    }
    result
}
