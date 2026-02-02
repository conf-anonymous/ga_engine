//! Compare CUDA and CPU twiddles_inv values

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ntt::CudaNttContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext as CpuNttContext;

fn main() -> Result<(), String> {
    let n = 8192;
    let q = 1152921504606994433u64;

    let psi = find_psi(n, q)?;
    let omega = ((psi as u128 * psi as u128) % q as u128) as u64;

    println!("Comparing twiddles_inv for n={}, q={}", n, q);
    println!("omega={}\n", omega);

    let cuda_ntt = CudaNttContext::new(n, q, omega)?;
    let cpu_ntt = CpuNttContext::new(n, q);

    // CUDA twiddles_inv are public
    println!("First 20 CUDA twiddles_inv:");
    for i in 0..20 {
        println!("  cuda_twiddles_inv[{}] = {}", i, cuda_ntt.twiddles_inv[i]);
    }

    // CPU uses omega_inv_powers_br which is bit-reversed
    println!("\nCPU omega_inv_powers_br (first 20):");
    for i in 0..20 {
        println!("  cpu_omega_inv_powers_br[{}] = {}", i, cpu_ntt.omega_inv_powers_br[i]);
    }

    // Compute omega_inv for comparison
    let omega_inv = mod_inverse(omega, q)?;
    println!("\nomega_inv = {}", omega_inv);
    println!("Verify: omega * omega_inv mod q = {}", ((omega as u128 * omega_inv as u128) % q as u128) as u64);

    // Manually compute first few omega_inv powers
    println!("\nManually computed omega_inv^i:");
    let mut power = 1u64;
    for i in 0..20 {
        println!("  omega_inv^{} = {}", i, power);
        if cuda_ntt.twiddles_inv[i] != power {
            println!("    ❌ MISMATCH with CUDA!");
        }
        power = ((power as u128 * omega_inv as u128) % q as u128) as u64;
    }

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

fn mod_inverse(a: u64, m: u64) -> Result<u64, String> {
    fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
        if a == 0 {
            (b, 0, 1)
        } else {
            let (gcd, x1, y1) = extended_gcd(b % a, a);
            (gcd, y1 - (b / a) * x1, x1)
        }
    }

    let (gcd, x, _) = extended_gcd(a as i128, m as i128);
    if gcd != 1 {
        return Err(format!("{} has no inverse mod {}", a, m));
    }

    let result = ((x % m as i128) + m as i128) % m as i128;
    Ok(result as u64)
}
