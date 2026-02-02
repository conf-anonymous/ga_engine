//! Debug Metal multiplication - compare tensor product with CPU

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::ckks::MetalCkksContext,
    backends::cpu_optimized::{
        keys::KeyContext,
        ckks::CkksContext,
    },
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Debug: Comparing CPU vs Metal tensor product\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();

    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let cpu_ctx = CkksContext::new(params.clone());

    // Simple test: 2.0 × 3.0
    let a = 2.0;
    let b = 3.0;

    println!("Test: {} × {}\n", a, b);

    // CPU version
    println!("=== CPU ===");
    let cpu_pt_a = cpu_ctx.encode(&[a]);
    let cpu_pt_b = cpu_ctx.encode(&[b]);
    let cpu_ct_a = cpu_ctx.encrypt(&cpu_pt_a, &pk);
    let cpu_ct_b = cpu_ctx.encrypt(&cpu_pt_b, &pk);

    println!("CPU input ciphertexts:");
    println!("  ct_a.c0[0]: {:?}", &cpu_ct_a.c0[0].values);
    println!("  ct_a.c1[0]: {:?}", &cpu_ct_a.c1[0].values);
    println!("  ct_b.c0[0]: {:?}", &cpu_ct_b.c0[0].values);
    println!("  ct_b.c1[0]: {:?}", &cpu_ct_b.c1[0].values);

    // Manually compute tensor product for CPU
    let c0_d0 = cpu_ctx.multiply_ntt(&cpu_ct_a.c0, &cpu_ct_b.c0);
    let c0_d1 = cpu_ctx.multiply_ntt(&cpu_ct_a.c0, &cpu_ct_b.c1);
    let c1_d0 = cpu_ctx.multiply_ntt(&cpu_ct_a.c1, &cpu_ct_b.c0);
    let c1_d1 = cpu_ctx.multiply_ntt(&cpu_ct_a.c1, &cpu_ct_b.c1);

    println!("\nCPU tensor product c0[0] (c0×d0):");
    println!("  {:?}", &c0_d0[0].values);

    println!("\nCPU tensor product c2[0] (c1×d1):");
    println!("  {:?}", &c1_d1[0].values);

    // Metal version
    println!("\n=== Metal ===");
    let metal_pt_a = metal_ctx.encode(&[a])?;
    let metal_pt_b = metal_ctx.encode(&[b])?;
    let metal_ct_a = metal_ctx.encrypt(&metal_pt_a, &pk)?;
    let metal_ct_b = metal_ctx.encrypt(&metal_pt_b, &pk)?;

    let n = params.n;
    let level = metal_ct_a.level;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];

    println!("Metal input ciphertexts:");
    print!("  ct_a.c0[0]: ");
    for j in 0..num_primes {
        print!("{} ", metal_ct_a.c0[0 * num_primes + j]);
    }
    println!();
    print!("  ct_a.c1[0]: ");
    for j in 0..num_primes {
        print!("{} ", metal_ct_a.c1[0 * num_primes + j]);
    }
    println!();
    print!("  ct_b.c0[0]: ");
    for j in 0..num_primes {
        print!("{} ", metal_ct_b.c0[0 * num_primes + j]);
    }
    println!();
    print!("  ct_b.c1[0]: ");
    for j in 0..num_primes {
        print!("{} ", metal_ct_b.c1[0 * num_primes + j]);
    }
    println!();

    // Manually compute tensor product for Metal
    let ct0_ct0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(
        &metal_ct_a.c0,
        &metal_ct_b.c0,
        moduli,
    )?;

    let ct1_ct1 = metal_ctx.multiply_polys_flat_ntt_negacyclic(
        &metal_ct_a.c1,
        &metal_ct_b.c1,
        moduli,
    )?;

    println!("\nMetal tensor product c0[0] (c0×d0):");
    print!("  ");
    for j in 0..num_primes {
        print!("{} ", ct0_ct0[0 * num_primes + j]);
    }
    println!();

    println!("\nMetal tensor product c2[0] (c1×d1):");
    print!("  ");
    for j in 0..num_primes {
        print!("{} ", ct1_ct1[0 * num_primes + j]);
    }
    println!();

    println!("\n=== Comparison ===");
    println!("Do CPU and Metal tensor products match?");

    let mut matches = true;
    for j in 0..num_primes {
        let cpu_val = c0_d0[0].values[j];
        let metal_val = ct0_ct0[0 * num_primes + j];
        if cpu_val != metal_val {
            println!("  c0[0] prime[{}]: CPU={}, Metal={} ❌", j, cpu_val, metal_val);
            matches = false;
        } else {
            println!("  c0[0] prime[{}]: {} ✅", j, cpu_val);
        }
    }

    if matches {
        println!("\n✅ CPU and Metal tensor products MATCH!");
        Ok(())
    } else {
        println!("\n❌ CPU and Metal tensor products DIFFER!");
        Err("Tensor products don't match".to_string())
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
