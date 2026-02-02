//! Verification test for rotation with medium-sized message

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v3::bootstrapping::{generate_rotation_keys, rotate};

fn main() {
    println!("=== Rotation Verification Test ===\n");

    // Use smaller params for faster test
    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Create message with first 20 slots filled
    let slots = params.n / 2;
    let mut message = vec![0.0; slots];
    for i in 0..20 {
        message[i] = (i + 1) as f64;  // 1,2,3,...,20
    }

    println!("Original (first 20): {:?}", &message[..20]);

    // Encrypt
    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    // Decrypt to verify
    let pt_dec = ckks_ctx.decrypt(&ct, &sk);
    let dec = ckks_ctx.decode(&pt_dec);
    println!("Decrypted (first 20): {:?}",
             dec.iter().take(20).map(|&x| x.round() as i32).collect::<Vec<_>>());

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

    println!("After rotation (first 20): {:?}",
             dec_rotated.iter().take(20).map(|&x| x.round() as i32).collect::<Vec<_>>());

    // Expected: [2,3,4,...,20,0] (shifted left by 1)
    println!("\nExpected: [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,0]");

    // Check if rotation worked
    let mut correct = true;
    for i in 0..19 {
        let expected = (i + 2) as f64;
        if (dec_rotated[i] - expected).abs() > 1.0 {
            println!("❌ Slot {} incorrect: got {:.2}, expected {:.2}", i, dec_rotated[i], expected);
            correct = false;
        }
    }
    // Check wraparound
    if dec_rotated[19].abs() > 1.0 {
        println!("❌ Slot 19 (wraparound) incorrect: got {:.2}, expected 0", dec_rotated[19]);
        correct = false;
    }

    if correct {
        println!("\n✅ SUCCESS! Rotation working correctly!");
    } else {
        println!("\n❌ Some slots incorrect");
    }
}
