//! Test NTT multiplication

fn main() {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;

    let n = 8192;
    let q = 1152921504606748673; // First prime from params

    let ntt_ctx = NttContext::new(n, q);

    // Test: multiply [1, 0, 0, ...] by [5, 0, 0, ...]
    // Expected result: [5, 0, 0, ...]
    let mut a = vec![0u64; n];
    a[0] = 1;

    let mut b = vec![0u64; n];
    b[0] = 5;

    let result = ntt_ctx.multiply_polynomials(&a, &b);

    println!("a[0] = {}", a[0]);
    println!("b[0] = {}", b[0]);
    println!("result[0] = {} (expected 5)", result[0]);
    println!("result[1] = {} (expected 0)", result[1]);

    if result[0] == 5 && result[1] == 0 {
        println!("✅ NTT multiply works correctly!");
    } else {
        println!("❌ NTT multiply is BROKEN!");
    }
}
