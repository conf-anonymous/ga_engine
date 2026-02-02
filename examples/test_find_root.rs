//! Test to find what primitive root the CPU backend uses

use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;

fn main() {
    let n = 2048;
    let q = 1152921504605798401u64; // The problematic prime

    println!("Finding primitive root for n={}, q={}", n, q);
    println!("Testing q ≡ 1 (mod 2n): {}", (q - 1) % (2 * n as u64) == 0);

    // The CPU backend will find it
    let ctx = NttContext::new(n, q);

    println!("✓ CPU found psi = {}", ctx.psi);
    println!("  omega = psi^2 = {}", ctx.omega);

    // Verify
    let two_n = 2 * n as u64;
    println!("\nVerification:");
    println!("  psi^(2n) mod q = {} (should be 1)", mod_pow(ctx.psi, two_n, q));
    println!("  psi^n mod q = {} (should be q-1 = {})", mod_pow(ctx.psi, n as u64, q), q - 1);
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp >>= 1;
    }
    result
}
