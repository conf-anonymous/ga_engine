//! Debug CUDA NTT twiddle factors
//!
//! This test prints out the twiddle factors to verify they're computed correctly.

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ntt::CudaNttContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext as CpuNttContext;

fn main() -> Result<(), String> {
    let n = 8;
    let q = 1073872897u64;

    // Find psi and omega
    let psi = find_psi(n, q)?;
    let omega = ((psi as u128 * psi as u128) % q as u128) as u64;

    println!("n={}, q={}", n, q);
    println!("psi={}, omega={}\n", psi, omega);

    // Create contexts
    let cuda_ntt = CudaNttContext::new(n, q, omega)?;
    let cpu_ntt = CpuNttContext::new(n, q);

    println!("CUDA twiddles (forward):");
    for (i, &t) in cuda_ntt.twiddles.iter().enumerate() {
        println!("  twiddles[{}] = {}", i, t);
    }

    println!("\nCUDA twiddles_inv (inverse):");
    for (i, &t) in cuda_ntt.twiddles_inv.iter().enumerate() {
        println!("  twiddles_inv[{}] = {}", i, t);
    }

    println!("\nCPU omega = {}", cpu_ntt.omega);
    println!("CPU omega_inv = {}", cpu_ntt.omega_inv);

    // Verify omega and omega_inv are multiplicative inverses
    let prod = ((omega as u128 * cpu_ntt.omega_inv as u128) % q as u128) as u64;
    println!("\nomega * omega_inv mod q = {} (should be 1)", prod);

    if prod == 1 {
        println!("✅ omega and omega_inv are correct inverses");
    } else {
        println!("❌ ERROR: omega and omega_inv are NOT inverses!");
        return Err("omega_inv is wrong!".to_string());
    }

    // Check that twiddles[i] * twiddles_inv[i] = 1 (mod q)
    println!("\nVerifying twiddle inverses:");
    for i in 0..n {
        let fw = cuda_ntt.twiddles[i];
        let inv = cuda_ntt.twiddles_inv[i];
        let prod = ((fw as u128 * inv as u128) % q as u128) as u64;
        println!("  twiddles[{}] * twiddles_inv[{}] = {} (should be 1)", i, i, prod);
        if prod != 1 {
            return Err(format!("Twiddle {} is not an inverse!", i));
        }
    }

    println!("\n✅ All twiddles are correct!");
    Ok(())
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
