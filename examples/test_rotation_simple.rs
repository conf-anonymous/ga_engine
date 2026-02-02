//! Simple rotation test to verify correctness

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v3::bootstrapping::{generate_rotation_keys, rotate};

fn main() {
    println!("=== Simple Rotation Test ===\n");

    let params = CliffordFHEParams::new_128bit();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Create a distinguishable pattern
    let mut message = vec![0.0; params.n / 2];
    message[0] = 100.0;
    message[1] = 200.0;
    message[2] = 300.0;
    message[3] = 400.0;

    println!("Original message:");
    println!("  [0]={}, [1]={}, [2]={}, [3]={}", message[0], message[1], message[2], message[3]);

    // Encrypt
    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    // Decrypt to verify
    let pt_dec = ckks_ctx.decrypt(&ct, &sk);
    let dec = ckks_ctx.decode(&pt_dec);
    println!("\nDecrypted (no rotation):");
    println!("  [0]={:.2}, [1]={:.2}, [2]={:.2}, [3]={:.2}", dec[0], dec[1], dec[2], dec[3]);

    // Generate rotation key for k=1
    let rotations = vec![1];
    println!("\nGenerating rotation key for k=1...");
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

    // Rotate by 1
    println!("Rotating by 1...");
    let ct_rotated = rotate(&ct, 1, &rotation_keys).expect("Rotation failed");

    // Decrypt rotated
    let pt_rotated = ckks_ctx.decrypt(&ct_rotated, &sk);
    let dec_rotated = ckks_ctx.decode(&pt_rotated);

    println!("\nDecrypted (after rotation by 1):");
    println!("  [0]={:.2}, [1]={:.2}, [2]={:.2}, [3]={:.2}",
             dec_rotated[0], dec_rotated[1], dec_rotated[2], dec_rotated[3]);

    // In CKKS, rotation by 1 should shift left: [100,200,300,400] → [200,300,400,100]
    println!("\nExpected (left rotation): [200, 300, 400, 100]");

    // Check
    let tol = 10.0;
    if (dec_rotated[0] - 200.0).abs() < tol &&
       (dec_rotated[1] - 300.0).abs() < tol &&
       (dec_rotated[2] - 400.0).abs() < tol &&
       (dec_rotated[3] - 100.0).abs() < tol {
        println!("\n✅ SUCCESS! Rotation is working correctly!");
    } else {
        println!("\n❌ FAILED! Rotation not producing expected results");
        println!("Got: [{:.2}, {:.2}, {:.2}, {:.2}]",
                 dec_rotated[0], dec_rotated[1], dec_rotated[2], dec_rotated[3]);
    }
}
