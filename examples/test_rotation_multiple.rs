//! Test multiple rotation amounts

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v3::bootstrapping::{generate_rotation_keys, rotate};

fn main() {
    println!("=== Multiple Rotation Test ===\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Create message: [1,2,3,4,5,6,7,8,9,10,0,0,...]
    let slots = params.n / 2;
    let mut message = vec![0.0; slots];
    for i in 0..10 {
        message[i] = (i + 1) as f64;
    }

    println!("Original: {:?}", &message[..10]);

    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    // Test rotations: 1, 2, 4
    let rotations = vec![1, 2, 4];
    println!("\nGenerating rotation keys for k=1,2,4...");
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

    // Test k=1: [1,2,3,4,5,6,7,8,9,10] → [2,3,4,5,6,7,8,9,10,0]
    println!("\n--- Rotation by 1 ---");
    let ct_rot1 = rotate(&ct, 1, &rotation_keys).expect("Rotation by 1 failed");
    let dec1 = ckks_ctx.decode(&ckks_ctx.decrypt(&ct_rot1, &sk));
    let result1: Vec<i32> = dec1.iter().take(10).map(|&x| x.round() as i32).collect();
    println!("Result: {:?}", result1);
    println!("Expected: [2, 3, 4, 5, 6, 7, 8, 9, 10, 0]");
    let ok1 = result1 == vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 0];
    println!("{}", if ok1 { "✅ PASS" } else { "❌ FAIL" });

    // Test k=2: [1,2,3,4,5,6,7,8,9,10] → [3,4,5,6,7,8,9,10,0,0]
    println!("\n--- Rotation by 2 ---");
    let ct_rot2 = rotate(&ct, 2, &rotation_keys).expect("Rotation by 2 failed");
    let dec2 = ckks_ctx.decode(&ckks_ctx.decrypt(&ct_rot2, &sk));
    let result2: Vec<i32> = dec2.iter().take(10).map(|&x| x.round() as i32).collect();
    println!("Result: {:?}", result2);
    println!("Expected: [3, 4, 5, 6, 7, 8, 9, 10, 0, 0]");
    let ok2 = result2 == vec![3, 4, 5, 6, 7, 8, 9, 10, 0, 0];
    println!("{}", if ok2 { "✅ PASS" } else { "❌ FAIL" });

    // Test k=4: [1,2,3,4,5,6,7,8,9,10] → [5,6,7,8,9,10,0,0,0,0]
    println!("\n--- Rotation by 4 ---");
    let ct_rot4 = rotate(&ct, 4, &rotation_keys).expect("Rotation by 4 failed");
    let dec4 = ckks_ctx.decode(&ckks_ctx.decrypt(&ct_rot4, &sk));
    let result4: Vec<i32> = dec4.iter().take(10).map(|&x| x.round() as i32).collect();
    println!("Result: {:?}", result4);
    println!("Expected: [5, 6, 7, 8, 9, 10, 0, 0, 0, 0]");
    let ok4 = result4 == vec![5, 6, 7, 8, 9, 10, 0, 0, 0, 0];
    println!("{}", if ok4 { "✅ PASS" } else { "❌ FAIL" });

    // Overall result
    println!("\n{}", if ok1 && ok2 && ok4 {
        "✅ ALL TESTS PASSED! Rotation fully working!"
    } else {
        "❌ Some tests failed"
    });
}
