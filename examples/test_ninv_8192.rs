//! Verify n_inv for n=8192

use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ntt::CudaNttContext;

fn main() -> Result<(), String> {
    let n = 8192;
    let q = 1152921504606994433u64;

    let psi = find_psi(n, q)?;
    let omega = ((psi as u128 * psi as u128) % q as u128) as u64;

    let cuda_ntt = CudaNttContext::new(n, q, omega)?;

    // Check n_inv
    let n_inv = cuda_ntt.n_inv;
    println!("n = {}", n);
    println!("q = {}", q);
    println!("n_inv = {}", n_inv);

    // Verify
    let product = ((n as u128 * n_inv as u128) % q as u128) as u64;
    println!("\nVerification:");
    println!("{} * {} mod {} = {}", n, n_inv, q, product);

    if product == 1 {
        println!("✅ n_inv is correct!");
        Ok(())
    } else {
        println!("❌ n_inv is WRONG!");
        Err("n_inv is incorrect".to_string())
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
