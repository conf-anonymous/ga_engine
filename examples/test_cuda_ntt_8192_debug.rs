//! Debug CUDA NTT for n=8192
//! Compare forward NTT output between CPU and CUDA

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ntt::CudaNttContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext as CpuNttContext;

fn main() -> Result<(), String> {
    let n = 8192;
    let q = 1152921504606994433u64;

    // Find psi
    let psi = find_psi(n, q)?;
    let omega = ((psi as u128 * psi as u128) % q as u128) as u64;

    println!("Testing CUDA vs CPU forward NTT for n={}", n);
    println!("q={}", q);
    println!("psi={}, omega={}\n", psi, omega);

    // Create contexts
    let cuda_ntt = CudaNttContext::new(n, q, omega)?;
    let cpu_ntt = CpuNttContext::new(n, q);

    // Simple test polynomial: [1, 2, 3, 0, 0, ...]
    let mut test_poly = vec![0u64; n];
    test_poly[0] = 1;
    test_poly[1] = 2;
    test_poly[2] = 3;

    // CPU forward
    let mut cpu_poly = test_poly.clone();
    cpu_ntt.forward_ntt(&mut cpu_poly);

    // CUDA forward
    let mut cuda_poly = test_poly.clone();
    cuda_ntt.forward(&mut cuda_poly)?;

    // Compare first 20 coefficients
    println!("Comparing forward NTT outputs:");
    let mut mismatches = 0;
    for i in 0..20 {
        let match_str = if cpu_poly[i] == cuda_poly[i] { "✓" } else { "✗" };
        println!("[{}] CPU={:20} CUDA={:20} {}", i, cpu_poly[i], cuda_poly[i], match_str);
        if cpu_poly[i] != cuda_poly[i] {
            mismatches += 1;
        }
    }

    // Check all
    let mut total_mismatches = 0;
    for i in 0..n {
        if cpu_poly[i] != cuda_poly[i] {
            total_mismatches += 1;
        }
    }

    if total_mismatches == 0 {
        println!("\n✅ Forward NTT matches CPU perfectly!");
        Ok(())
    } else {
        println!("\n❌ Forward NTT has {} mismatches", total_mismatches);
        Err(format!("{} mismatches in forward NTT", total_mismatches))
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
