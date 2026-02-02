//! Debug: Compare s² computed by CPU vs Metal EVK generation

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::MetalCkksContext,
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::KeyContext,
        ntt::NttContext,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Debug: s² computation comparison");
    println!("=================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (_, sk, cpu_evk) = key_ctx.keygen();

    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];
    let base_w = 20u32;

    // Compute s² the same way Metal does (using NttContext for each prime)
    println!("Computing s² using same method as Metal EVK generation...");
    let mut s_squared_metal_way = vec![vec![0u64; num_primes]; n];
    for (prime_idx, &q) in moduli.iter().enumerate() {
        let ntt_ctx = NttContext::new(n, q);
        let s_poly: Vec<u64> = sk.coeffs.iter().map(|rns| rns.values[prime_idx]).collect();
        let s_sq = ntt_ctx.multiply_polynomials(&s_poly, &s_poly);
        for i in 0..n {
            s_squared_metal_way[i][prime_idx] = s_sq[i];
        }
    }

    // Now compute s² using KeyContext multiply_polynomials (how CPU EVK does it)
    println!("Computing s² using CPU KeyContext method...");
    // CPU uses: s_squared = self.multiply_polynomials(&sk.coeffs, &sk.coeffs, moduli)
    // which is line 443
    let cpu_s_squared = multiply_polys_keyctx(&sk.coeffs, &sk.coeffs, &key_ctx, moduli);

    // Compare
    println!("\nComparing s² values (first 3 coefficients):");
    for i in 0..3 {
        println!("  coeff[{}]:", i);
        println!("    Metal way: {:?}", s_squared_metal_way[i]);
        println!("    CPU way:   {:?}", cpu_s_squared[i].values[..num_primes].to_vec());
        let match_all = (0..num_primes).all(|j|
            s_squared_metal_way[i][j] == cpu_s_squared[i].values[j]
        );
        println!("    Match: {}", if match_all { "✅" } else { "❌" });
    }

    // Now compute -B^t*s² for t=0 (B^0=1)
    println!("\nComputing -s² (for t=0):");
    let mut neg_s_sq_metal = vec![vec![0u64; num_primes]; n];
    for i in 0..n {
        for (j, &q) in moduli.iter().enumerate() {
            let s2 = s_squared_metal_way[i][j];
            neg_s_sq_metal[i][j] = if s2 == 0 { 0 } else { q - s2 };
        }
    }

    let mut neg_s_sq_cpu = vec![vec![0u64; num_primes]; n];
    for i in 0..n {
        for (j, &q) in moduli.iter().enumerate() {
            let s2 = cpu_s_squared[i].values[j];
            neg_s_sq_cpu[i][j] = if s2 == 0 { 0 } else { q - s2 };
        }
    }

    println!("  -s²[0] Metal way: {:?}", neg_s_sq_metal[0]);
    println!("  -s²[0] CPU way:   {:?}", neg_s_sq_cpu[0]);

    // Now check: does EVK identity use the same s²?
    // The test verifies evk0[0] - evk1[0]*s = -B^0*s² + noise = -s² + noise
    //
    // In the identity test at test_evk_all_digits.rs, line 47-58 computes s² the same way as above
    // So the identity test and the EVK generation should be using the same s²...
    //
    // But wait - the identity test PASSES. This means the EVK is correct.
    // The problem must be in how we USE the EVK during multiplication.

    // Let me check: what if the NTT contexts are different between CPU and Metal?
    println!("\n=== Checking NTT primitives ===");
    let q0 = moduli[0];
    let ntt_ctx = NttContext::new(n, q0);

    // Test polynomial: [1, 2, 3, 0, 0, ...]
    let mut test_a = vec![0u64; n];
    test_a[0] = 1;
    test_a[1] = 2;
    test_a[2] = 3;

    let mut test_b = vec![0u64; n];
    test_b[0] = 4;
    test_b[1] = 5;

    let result_cpu = ntt_ctx.multiply_polynomials(&test_a, &test_b);
    println!("CPU NTT multiply [1,2,3]*[4,5] first 5 coeffs: {:?}", &result_cpu[..5]);

    // Expected (negacyclic):
    // (1 + 2x + 3x^2) * (4 + 5x) mod (x^n + 1)
    // = 4 + 5x + 8x + 10x^2 + 12x^2 + 15x^3
    // = 4 + 13x + 22x^2 + 15x^3 (for standard polynomial)
    // But negacyclic wraps higher terms with negative sign
    // For n=1024, x^1024 = -1, so no wrapping for low degree
    println!("Expected (standard poly): 4, 13, 22, 15, 0, ...");

    Ok(())
}

#[cfg(feature = "v2-gpu-metal")]
fn multiply_polys_keyctx(
    a: &[ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation],
    b: &[ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation],
    key_ctx: &KeyContext,
    moduli: &[u64],
) -> Vec<ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation> {
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

    let n = a.len();
    let num_primes = moduli.len();

    let mut result = vec![RnsRepresentation::new(vec![0; num_primes], moduli.to_vec()); n];

    for (j, &_q) in moduli.iter().enumerate() {
        let ntt_ctx = &key_ctx.ntt_contexts[j];

        let a_poly: Vec<u64> = a.iter().map(|r| r.values[j]).collect();
        let b_poly: Vec<u64> = b.iter().map(|r| r.values[j]).collect();

        let prod = ntt_ctx.multiply_polynomials(&a_poly, &b_poly);

        for i in 0..n {
            result[i].values[j] = prod[i];
        }
    }

    result
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
