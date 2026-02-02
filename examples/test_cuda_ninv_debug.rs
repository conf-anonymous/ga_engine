//! Debug n_inv computation
//!
//! Check if n_inv is computed correctly

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ntt::CudaNttContext;

fn main() -> Result<(), String> {
    let n = 8;
    let q = 1073872897u64;
    let psi = find_psi(n, q)?;
    let omega = ((psi as u128 * psi as u128) % q as u128) as u64;

    println!("n={}, q={}", n, q);

    let cuda_ntt = CudaNttContext::new(n, q, omega)?;

    println!("CUDA n_inv = {}", cuda_ntt.n_inv);

    // Verify: n * n_inv ≡ 1 (mod q)
    let prod = ((n as u128 * cuda_ntt.n_inv as u128) % q as u128) as u64;
    println!("n * n_inv mod q = {} (should be 1)", prod);

    if prod == 1 {
        println!("✅ n_inv is correct");
        Ok(())
    } else {
        println!("❌ ERROR: n_inv is wrong!");
        Err("n_inv incorrect".to_string())
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
