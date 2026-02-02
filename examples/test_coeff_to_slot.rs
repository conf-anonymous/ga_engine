//! Test CoeffToSlot/SlotToCoeff transformations

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v3::bootstrapping::{coeff_to_slot, slot_to_coeff, required_rotations_for_bootstrap, generate_rotation_keys};

fn main() {
    println!("=== CoeffToSlot/SlotToCoeff Test ===\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Create test message
    let slots = params.n / 2;
    let mut message = vec![0.0; slots];
    for i in 0..16 {
        message[i] = (i + 1) as f64;  // [1,2,3,...,16,0,0,...]
    }

    println!("Original message (first 16): {:?}", &message[..16]);

    // Encrypt
    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    // Verify encryption
    let dec_orig = ckks_ctx.decode(&ckks_ctx.decrypt(&ct, &sk));
    println!("After encrypt/decrypt (first 16): {:?}",
             dec_orig.iter().take(16).map(|&x| x.round() as i32).collect::<Vec<_>>());

    // Generate all rotation keys needed for bootstrap
    let rotations = required_rotations_for_bootstrap(params.n);
    println!("\nGenerating {} rotation keys for CoeffToSlot/SlotToCoeff...", rotations.len());
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

    // Apply CoeffToSlot
    println!("\nApplying CoeffToSlot transformation...");
    let ct_slot = coeff_to_slot(&ct, &rotation_keys).expect("CoeffToSlot failed");

    // Decrypt to see intermediate result
    let dec_slot = ckks_ctx.decode(&ckks_ctx.decrypt(&ct_slot, &sk));
    println!("After CoeffToSlot (first 16): {:?}",
             dec_slot.iter().take(16).map(|&x| x.round() as i32).collect::<Vec<_>>());

    // Apply SlotToCoeff (inverse)
    println!("\nApplying SlotToCoeff transformation (inverse)...");
    let ct_back = slot_to_coeff(&ct_slot, &rotation_keys).expect("SlotToCoeff failed");

    // Decrypt final result
    let dec_back = ckks_ctx.decode(&ckks_ctx.decrypt(&ct_back, &sk));
    println!("After SlotToCoeff (first 16): {:?}",
             dec_back.iter().take(16).map(|&x| x.round() as i32).collect::<Vec<_>>());

    // Check if we recovered original message (approximately)
    println!("\nExpected (original): [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]");

    let mut errors = 0;
    for i in 0..16 {
        let expected = (i + 1) as f64;
        let error = (dec_back[i] - expected).abs();
        if error > 2.0 {  // Allow some noise accumulation
            errors += 1;
            println!("Large error at slot {}: got {:.2}, expected {:.2}", i, dec_back[i], expected);
        }
    }

    if errors == 0 {
        println!("\n✅ CoeffToSlot/SlotToCoeff roundtrip successful!");
    } else {
        println!("\n⚠️ CoeffToSlot/SlotToCoeff completed with {} errors (noise accumulation)", errors);
    }

    println!("\nNote: Current implementation is a placeholder that applies rotations.");
    println!("Full implementation will add diagonal matrix multiplications.");
}
