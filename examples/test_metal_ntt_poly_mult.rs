//! Test Metal NTT Polynomial Multiplication
//!
//! Verifies that Metal NTT can correctly multiply polynomials.

#[cfg(feature = "v2-gpu-metal")]
fn main() {
    use ga_engine::clifford_fhe_v2::backends::gpu_metal::ntt::MetalNttContext;

    println!("Testing Metal NTT polynomial multiplication...\n");

    // Use first prime from standard params
    let n = 1024;
    let q = 1152921504606584833u64;
    let psi = 693807653563943717u64; // Pre-computed for this q,n pair

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  q = {}\n", q);

    // Create Metal NTT context
    let ntt_ctx = match MetalNttContext::new(n, q, psi) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("❌ Failed to create NTT context: {}", e);
            return;
        }
    };

    // Test multiplication: (X + 1) * (X + 2) = X^2 + 3X + 2
    let mut a = vec![0u64; n];
    a[0] = 1; // constant term
    a[1] = 1; // X term

    let mut b = vec![0u64; n];
    b[0] = 2; // constant term
    b[1] = 1; // X term

    println!("Polynomial a: {} + {}X", a[0], a[1]);
    println!("Polynomial b: {} + {}X", b[0], b[1]);
    println!("Expected result: 2 + 3X + X^2\n");

    // Forward NTT
    let mut a_ntt = a.clone();
    let mut b_ntt = b.clone();

    println!("Before forward NTT:");
    println!("  a[0..3]: {:?}", &a_ntt[0..3]);
    println!("  b[0..3]: {:?}\n", &b_ntt[0..3]);

    ntt_ctx.forward(&mut a_ntt).unwrap();
    ntt_ctx.forward(&mut b_ntt).unwrap();

    println!("After forward NTT (in Montgomery domain):");
    println!("  a_ntt[0..3]: {:?}", &a_ntt[0..3]);
    println!("  b_ntt[0..3]: {:?}\n", &b_ntt[0..3]);

    // Pointwise multiply
    let mut result_ntt = vec![0u64; n];
    ntt_ctx.pointwise_multiply(&a_ntt, &b_ntt, &mut result_ntt).unwrap();

    println!("After pointwise multiply:");
    println!("  result_ntt[0..3]: {:?}\n", &result_ntt[0..3]);

    // Inverse NTT
    ntt_ctx.inverse(&mut result_ntt).unwrap();

    println!("After inverse NTT (back to normal domain):");
    println!("  result[0..3]: {:?}\n", &result_ntt[0..3]);

    println!("Result coefficients:");
    println!("  Constant: {} (expected 2)", result_ntt[0]);
    println!("  X:        {} (expected 3)", result_ntt[1]);
    println!("  X^2:      {} (expected 1)", result_ntt[2]);

    // Check correctness
    if result_ntt[0] == 2 && result_ntt[1] == 3 && result_ntt[2] == 1 {
        println!("\n✅ TEST PASSED - Metal NTT working correctly!");
    } else {
        println!("\n❌ TEST FAILED - Incorrect results");
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    eprintln!("This example requires the v2-gpu-metal feature.");
    eprintln!("Run with: cargo run --release --features v2,v2-gpu-metal --example test_metal_ntt_poly_mult");
}
