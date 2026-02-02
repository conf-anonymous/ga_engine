//! Debug Metal GPU multiply_plain

use ga_engine::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

fn main() -> Result<(), String> {
    let params = CliffordFHEParams::new_test_ntt_1024();  // Smaller for debugging

    println!("Creating Metal GPU context...");
    let mut key_ctx = MetalKeyContext::new(params.clone())?;
    let (pk, sk, _evk) = key_ctx.keygen()?;
    let ckks = MetalCkksContext::new(params.clone())?;

    // Test simple multiplication
    let value = 42.0;
    let multiplier = 2.0;

    println!("\n=== Encryption ===");
    let pt1 = ckks.encode(&[value])?;
    println!("Plaintext scale: {}", pt1.scale);
    let ct1 = ckks.encrypt(&pt1, &pk)?;
    println!("Ciphertext level: {}, scale: {}", ct1.level, ct1.scale);

    // Decrypt to verify
    let dec1 = ckks.decrypt(&ct1, &sk)?;
    let val1 = ckks.decode(&dec1)?;
    println!("Decrypted value: {}", val1[0]);

    println!("\n=== Multiply by {} ===", multiplier);
    let pt_mult = ckks.encode(&[multiplier])?;
    println!("Multiplier plaintext scale: {}", pt_mult.scale);

    println!("Before multiply_plain:");
    println!("  ct level: {}, scale: {}", ct1.level, ct1.scale);
    println!("  pt level: {}, scale: {}", pt_mult.level, pt_mult.scale);

    let ct2 = ct1.multiply_plain(&pt_mult, &ckks)?;

    println!("After multiply_plain:");
    println!("  ct level: {}, scale: {}", ct2.level, ct2.scale);

    // Decrypt result
    let dec2 = ckks.decrypt(&ct2, &sk)?;
    let val2 = ckks.decode(&dec2)?;
    println!("Decrypted value: {}", val2[0]);
    println!("Expected: {}", value * multiplier);
    println!("Error: {}", (val2[0] - value * multiplier).abs());

    // Also test without rescale - convert to CPU and test
    println!("\n=== Testing CPU multiply_plain for comparison ===");
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

    let cpu_key_ctx = KeyContext::new(params.clone());
    let (cpu_pk, cpu_sk, _cpu_evk) = cpu_key_ctx.keygen();
    let cpu_ckks = CkksContext::new(params.clone());

    let cpu_pt1 = cpu_ckks.encode(&[value]);
    let cpu_ct1 = cpu_ckks.encrypt(&cpu_pt1, &cpu_pk);
    println!("CPU ct level: {}, scale: {}", cpu_ct1.level, cpu_ct1.scale);

    let cpu_pt_mult = cpu_ckks.encode(&[multiplier]);
    let cpu_ct2 = cpu_ct1.multiply_plain(&cpu_pt_mult, &cpu_ckks);
    println!("CPU ct after multiply: level={}, scale={}", cpu_ct2.level, cpu_ct2.scale);

    let cpu_dec2 = cpu_ckks.decrypt(&cpu_ct2, &cpu_sk);
    let cpu_val2 = cpu_ckks.decode(&cpu_dec2);
    println!("CPU decrypted value: {}", cpu_val2[0]);

    Ok(())
}
