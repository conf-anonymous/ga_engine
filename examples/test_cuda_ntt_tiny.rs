use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ntt::CudaNttContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext as CpuNttContext;

fn main() -> Result<(), String> {
    // Use tiny n=8 for manual verification
    let n = 8;
    let q = 1073872897u64; // Small prime where q-1 = 2^6 * 16779577, so q ≡ 1 (mod 16)

    // Find psi (primitive 16-th root of unity)
    let psi = find_psi(n, q)?;
    let omega = ((psi as u128 * psi as u128) % q as u128) as u64;

    println!("Testing CUDA NTT with n={}, q={}", n, q);
    println!("psi={}, omega={}", psi, omega);

    // Verify psi and omega
    println!("psi^8 mod q = {} (should be {} = q-1)", pow_mod(psi, 8, q), q-1);
    println!("omega^8 mod q = {} (should be 1)", pow_mod(omega, 8, q));

    // Create contexts
    let cuda_ntt = CudaNttContext::new(n, q, omega)?;
    let cpu_ntt = CpuNttContext::new(n, q);

    // Simple test polynomial
    let mut test_poly = vec![0u64; n];
    test_poly[0] = 1;
    test_poly[1] = 2;

    println!("\nOriginal polynomial: {:?}", test_poly);

    // CPU round-trip
    let mut cpu_poly = test_poly.clone();
    cpu_ntt.forward_ntt(&mut cpu_poly);
    println!("CPU after forward NTT: {:?}", cpu_poly);
    cpu_ntt.inverse_ntt(&mut cpu_poly);
    println!("CPU after inverse NTT: {:?}", cpu_poly);

    // CUDA round-trip
    let mut cuda_poly = test_poly.clone();
    cuda_ntt.forward(&mut cuda_poly)?;
    println!("CUDA after forward NTT: {:?}", cuda_poly);
    cuda_ntt.inverse(&mut cuda_poly)?;
    println!("CUDA after inverse NTT: {:?}", cuda_poly);

    // Compare
    let mut mismatches = 0;
    for i in 0..n {
        if test_poly[i] != cuda_poly[i] {
            println!("Mismatch at [{}]: original={}, cuda={}, cpu={}",
                i, test_poly[i], cuda_poly[i], cpu_poly[i]);
            mismatches += 1;
        }
    }

    if mismatches == 0 {
        println!("\n✅ CUDA NTT tiny test PASSED");
        Ok(())
    } else {
        println!("\n❌ CUDA NTT tiny test FAILED: {} mismatches", mismatches);
        Err(format!("Failed with {} mismatches", mismatches))
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
