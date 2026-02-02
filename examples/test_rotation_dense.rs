//! Test rotation with DENSE message (all slots filled)

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v3::bootstrapping::{generate_rotation_keys, rotate};

fn main() {
    println!("=== Dense Rotation Test ===\n");

    let params = CliffordFHEParams::new_test_ntt_1024();  // Smaller for speed
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Create DENSE message (all slots filled with pattern)
    let slots = params.n / 2;
    let mut message = Vec::with_capacity(slots);
    for i in 0..slots {
        message.push((i % 10) as f64);  // Pattern: 0,1,2,3,4,5,6,7,8,9,0,1,2,...
    }

    println!("Original pattern (first 12): {:?}", &message[..12]);

    // Encrypt
    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    // Decrypt to verify
    let pt_dec = ckks_ctx.decrypt(&ct, &sk);
    let dec = ckks_ctx.decode(&pt_dec);
    println!("Decrypted (first 12): {:?}",
             dec.iter().take(12).map(|&x| x.round() as i32).collect::<Vec<_>>());

    // Generate rotation key for k=1
    let rotations = vec![1];
    println!("\nGenerating rotation key...");
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

    // Rotate by 1
    println!("Rotating by 1...");
    let ct_rotated = rotate(&ct, 1, &rotation_keys).expect("Rotation failed");

    // Decrypt rotated
    let pt_rotated = ckks_ctx.decrypt(&ct_rotated, &sk);
    let dec_rotated = ckks_ctx.decode(&pt_rotated);

    println!("After rotation (first 12): {:?}",
             dec_rotated.iter().take(12).map(|&x| x.round() as i32).collect::<Vec<_>>());

    // Expected: [1,2,3,4,5,6,7,8,9,0,1,2] (shifted left by 1)
    println!("\nExpected: [1,2,3,4,5,6,7,8,9,0,1,2]");

    // Check if rotation worked
    let mut matches = 0;
    for i in 0..10 {
        let expected = ((i + 1) % 10) as f64;
        if (dec_rotated[i] - expected).abs() < 1.0 {
            matches += 1;
        }
    }

    println!("\nMatches: {}/10", matches);
    if matches >= 8 {
        println!("✅ Rotation appears to be working!");
    } else {
        println!("❌ Rotation not working correctly");
    }
}
