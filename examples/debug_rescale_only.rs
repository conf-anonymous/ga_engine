//! Debug: Test rescaling alone
//!
//! The Newton-Raphson debug showed rescaling from level 2 to 1
//! changed the value from 0.14285... to 13.244...

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::{MetalCkksContext, MetalCiphertext},
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::keys::KeyContext,
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  DEBUG: Rescale Operation Only                                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();

    let device = Arc::new(MetalDevice::new()?);
    let metal_ctx = MetalCkksContext::new(params.clone())?;

    // Test rescaling a freshly encrypted ciphertext
    for val in [0.14285714285714285, 1.0, 2.0, 100.0, 7.0] {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Testing rescale with value: {}", val);
        println!("═══════════════════════════════════════════════════════════════\n");

        let pt = metal_ctx.encode(&[val])?;
        let ct = metal_ctx.encrypt(&pt, &pk)?;

        println!("  Original:");
        println!("    Level: {}", ct.level);
        println!("    Scale: {}", ct.scale);
        println!("    num_primes: {}", ct.num_primes);

        let dec_orig = metal_ctx.decode(&metal_ctx.decrypt(&ct, &sk)?)?[0];
        println!("    Decrypted: {} (error: {:.2e})", dec_orig, (dec_orig - val).abs());

        // Rescale from level 2 to level 1
        println!("\n  Rescaling from level 2 to 1...");
        let rescaled_c0 = metal_ctx.exact_rescale_gpu(&ct.c0, ct.level)?;
        let rescaled_c1 = metal_ctx.exact_rescale_gpu(&ct.c1, ct.level)?;
        let new_scale = ct.scale / params.moduli[ct.level] as f64;

        let ct_rescaled = MetalCiphertext {
            c0: rescaled_c0,
            c1: rescaled_c1,
            n: ct.n,
            num_primes: ct.level,  // level drops by 1
            level: ct.level - 1,
            scale: new_scale,
        };

        println!("  After rescale:");
        println!("    Level: {}", ct_rescaled.level);
        println!("    Scale: {}", ct_rescaled.scale);
        println!("    num_primes: {}", ct_rescaled.num_primes);

        let dec_rescaled = metal_ctx.decode(&metal_ctx.decrypt(&ct_rescaled, &sk)?)?[0];
        println!("    Decrypted: {} (error: {:.2e})", dec_rescaled, (dec_rescaled - val).abs());

        if (dec_rescaled - val).abs() > 1.0 {
            println!("    ❌ RESCALE BUG DETECTED!\n");

            // Debug: Check coefficients before/after rescale
            println!("  Debug c0 coefficients (coeff 0):");
            println!("    Before rescale (level 2, 3 primes):");
            for j in 0..3 {
                println!("      prime {}: {}", j, ct.c0[0 * 3 + j]);
            }

            println!("    After rescale (level 1, 2 primes):");
            for j in 0..2 {
                println!("      prime {}: {}", j, ct_rescaled.c0[0 * 2 + j]);
            }

            println!("\n  Debug c1 coefficients (coeff 0):");
            println!("    Before rescale (level 2, 3 primes):");
            for j in 0..3 {
                println!("      prime {}: {}", j, ct.c1[0 * 3 + j]);
            }

            println!("    After rescale (level 1, 2 primes):");
            for j in 0..2 {
                println!("      prime {}: {}", j, ct_rescaled.c1[0 * 2 + j]);
            }
        } else {
            println!("    ✅ Rescale looks good\n");
        }
    }

    Ok(())
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
