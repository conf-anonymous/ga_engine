use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ntt::CudaNttContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext as CpuNttContext;

fn main() -> Result<(), String> {
    let n = 8192;
    let q = 1152921504606994433u64; // First prime from params

    // Find psi (2N-th root)
    let psi = find_psi(n, q)?;
    let omega = ((psi as u128 * psi as u128) % q as u128) as u64;

    println!("Testing CUDA NTT round-trip");
    println!("n={}, q={}", n, q);
    println!("psi={}, omega={}", psi, omega);

    // Create contexts
    let cuda_ntt = CudaNttContext::new(n, q, omega)?;

    // Test polynomial
    let mut test_poly = vec![0u64; n];
    test_poly[0] = 42;
    test_poly[1] = 100;
    test_poly[100] = 999;

    // CUDA round-trip
    let mut cuda_poly = test_poly.clone();
    cuda_ntt.forward(&mut cuda_poly)?;
    cuda_ntt.inverse(&mut cuda_poly)?;

    println!("\nRound-trip test:");
    println!("Original[0]={}, CUDA[0]={}, match={}", test_poly[0], cuda_poly[0], test_poly[0] == cuda_poly[0]);
    println!("Original[1]={}, CUDA[1]={}, match={}", test_poly[1], cuda_poly[1], test_poly[1] == cuda_poly[1]);
    println!("Original[100]={}, CUDA[100]={}, match={}", test_poly[100], cuda_poly[100], test_poly[100] == cuda_poly[100]);

    // Check all coefficients
    let mut mismatches = 0;
    for i in 0..n {
        if test_poly[i] != cuda_poly[i] {
            if mismatches < 10 {
                println!("Mismatch at i={}: original={}, cuda={}", i, test_poly[i], cuda_poly[i]);
            }
            mismatches += 1;
        }
    }

    if mismatches == 0 {
        println!("\n✅ CUDA NTT round-trip PASSED");
        Ok(())
    } else {
        println!("\n❌ CUDA NTT round-trip FAILED: {} mismatches out of {}", mismatches, n);
        Err(format!("NTT round-trip failed with {} mismatches", mismatches))
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
