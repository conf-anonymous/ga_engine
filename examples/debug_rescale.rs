//! Debug rescale step by step

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::ckks::MetalCkksContext,
    params::CliffordFHEParams,
};

#[cfg(feature = "v2-gpu-metal")]
fn mod_pow(base: u64, exp: u64, modulus: u64) -> u64 {
    let mut result = 1u128;
    let mut base = (base as u128) % (modulus as u128);
    let mut exp = exp;
    let modulus = modulus as u128;
    
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp /= 2;
        base = (base * base) % modulus;
    }
    result as u64
}

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Debug Rescale Step by Step");
    println!("==========================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let metal_ctx = MetalCkksContext::new(params.clone())?;
    
    let level = 2;
    let num_primes = level + 1;  // 3
    let moduli = &params.moduli[..num_primes];
    
    println!("Moduli: {:?}\n", moduli);
    
    let q0 = moduli[0];
    let q1 = moduli[1];
    let q2 = moduli[2];  // q_last
    
    // Compute inverses using CPU method (Fermat's little theorem)
    let q2_inv_mod_q0_cpu = mod_pow(q2 % q0, q0 - 2, q0);
    let q2_inv_mod_q1_cpu = mod_pow(q2 % q1, q1 - 2, q1);
    
    println!("CPU computed inverses:");
    println!("  q2^(-1) mod q0 = {}", q2_inv_mod_q0_cpu);
    println!("  q2^(-1) mod q1 = {}", q2_inv_mod_q1_cpu);
    
    // Get Metal precomputed inverses
    let metal_inv = &metal_ctx.rescale_inv_table[level];
    println!("\nMetal precomputed inverses (rescale_inv_table[{}]):", level);
    println!("  qtop_inv[0] = {}", metal_inv[0]);
    println!("  qtop_inv[1] = {}", metal_inv[1]);
    
    // Verify they match
    if q2_inv_mod_q0_cpu == metal_inv[0] && q2_inv_mod_q1_cpu == metal_inv[1] {
        println!("\n✅ Inverse tables match!");
    } else {
        println!("\n❌ Inverse tables don't match!");
        return Err("Inverse mismatch".to_string());
    }
    
    // Now trace through one coefficient manually
    println!("\n--- Manual trace for coeff[0] ---");
    
    // Test input: [0, 67890, 135780]
    let val_q0 = 0u64;
    let val_q1 = 67890u64;
    let val_qlast = 135780u64;
    
    println!("Input: [{}, {}, {}]", val_q0, val_q1, val_qlast);
    
    // Centered lift of val_qlast
    let half_qlast = q2 / 2;
    let val_centered: i64 = if val_qlast > half_qlast {
        val_qlast as i64 - q2 as i64
    } else {
        val_qlast as i64
    };
    println!("val_centered = {} (half_qlast = {})", val_centered, half_qlast);
    
    // CPU rescale for prime 0
    let diff_0 = if val_centered >= 0 {
        let vc = (val_centered as u64) % q0;
        if val_q0 >= vc { val_q0 - vc } else { q0 - (vc - val_q0) }
    } else {
        let vc = ((-val_centered) as u64) % q0;
        (val_q0 + vc) % q0
    };
    let new_val_0_cpu = ((diff_0 as u128) * (q2_inv_mod_q0_cpu as u128)) % (q0 as u128);
    
    println!("\nPrime 0 (q0 = {}):", q0);
    println!("  old_val = {}", val_q0);
    println!("  diff = {}", diff_0);
    println!("  q2_inv_mod_q0 = {}", q2_inv_mod_q0_cpu);
    println!("  new_val (CPU) = {}", new_val_0_cpu);
    
    // CPU rescale for prime 1
    let diff_1 = if val_centered >= 0 {
        let vc = (val_centered as u64) % q1;
        if val_q1 >= vc { val_q1 - vc } else { q1 - (vc - val_q1) }
    } else {
        let vc = ((-val_centered) as u64) % q1;
        (val_q1 + vc) % q1
    };
    let new_val_1_cpu = ((diff_1 as u128) * (q2_inv_mod_q1_cpu as u128)) % (q1 as u128);
    
    println!("\nPrime 1 (q1 = {}):", q1);
    println!("  old_val = {}", val_q1);
    println!("  diff = {}", diff_1);
    println!("  q2_inv_mod_q1 = {}", q2_inv_mod_q1_cpu);
    println!("  new_val (CPU) = {}", new_val_1_cpu);
    
    // Now run Metal on just this one coefficient
    let n = 1;  // Just one coefficient for testing
    let mut poly_flat = vec![0u64; n * num_primes];
    poly_flat[0] = val_q0;
    poly_flat[1] = val_q1;
    poly_flat[2] = val_qlast;
    
    println!("\nRunning Metal rescale on single coefficient...");
    // We need to pad to match params.n for Metal
    let n_full = params.n;
    let mut poly_flat_full = vec![0u64; n_full * num_primes];
    poly_flat_full[0] = val_q0;
    poly_flat_full[1] = val_q1;
    poly_flat_full[2] = val_qlast;
    
    let metal_result = metal_ctx.exact_rescale_gpu(&poly_flat_full, level)?;
    
    println!("\nMetal result:");
    println!("  output[0] = [{}, {}]", metal_result[0], metal_result[1]);
    println!("\nCPU expected:");
    println!("  output[0] = [{}, {}]", new_val_0_cpu, new_val_1_cpu);
    
    if metal_result[0] == new_val_0_cpu as u64 && metal_result[1] == new_val_1_cpu as u64 {
        println!("\n✅ Rescale matches!");
        Ok(())
    } else {
        println!("\n❌ Rescale mismatch!");
        println!("  Diff prime 0: {} vs {}", new_val_0_cpu, metal_result[0]);
        println!("  Diff prime 1: {} vs {}", new_val_1_cpu, metal_result[1]);
        Err("Rescale mismatch".to_string())
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
