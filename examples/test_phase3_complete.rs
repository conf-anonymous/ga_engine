//! Phase 3 Verification Test Suite
//!
//! Comprehensive correctness verification for:
//! - Rotation key generation with CRT-consistent decomposition
//! - Homomorphic rotation via Galois automorphisms (k=1,2,4)
//! - CoeffToSlot/SlotToCoeff transformations with O(log N) complexity
//! - CKKS canonical embedding with orbit ordering

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v3::bootstrapping::{
    generate_rotation_keys, rotate, coeff_to_slot, slot_to_coeff,
    required_rotations_for_bootstrap,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         V3 Phase 3: Comprehensive Verification Test             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    println!("Test Configuration:");
    println!("  Ring dimension: N = {}", params.n);
    println!("  Number of slots: {}", params.n / 2);
    println!("  RNS moduli count: L = {}\n", params.moduli.len());

    // ========================================================================
    // TEST 1: Canonical Embedding Roundtrip
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: CKKS Canonical Embedding");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut message = vec![0.0; params.n / 2];
    for i in 0..10 {
        message[i] = (i + 1) as f64;
    }

    println!("Original message: {:?}", &message[..10]);
    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);
    let dec = ckks_ctx.decode(&ckks_ctx.decrypt(&ct, &sk));
    let result: Vec<i32> = dec.iter().take(10).map(|&x| x.round() as i32).collect();
    println!("After encode/encrypt/decrypt/decode: {:?}", result);

    let test1_pass = result == vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    println!("\n{}\n", if test1_pass {
        "✅ TEST 1 PASSED: Canonical embedding working correctly!"
    } else {
        "❌ TEST 1 FAILED"
    });

    // ========================================================================
    // TEST 2: Single Rotation
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: Homomorphic Rotation (k=1)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Generating rotation key for k=1...");
    let rotation_keys = generate_rotation_keys(&vec![1], &sk, &params);

    println!("Applying rotation...");
    let ct_rot = rotate(&ct, 1, &rotation_keys).expect("Rotation failed");
    let dec_rot = ckks_ctx.decode(&ckks_ctx.decrypt(&ct_rot, &sk));
    let result_rot: Vec<i32> = dec_rot.iter().take(10).map(|&x| x.round() as i32).collect();

    println!("Original:       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
    println!("After rotate 1: {:?}", result_rot);
    println!("Expected:       [2, 3, 4, 5, 6, 7, 8, 9, 10, 0]");

    let test2_pass = result_rot == vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 0];
    println!("\n{}\n", if test2_pass {
        "✅ TEST 2 PASSED: Single rotation working!"
    } else {
        "❌ TEST 2 FAILED"
    });

    // ========================================================================
    // TEST 3: Multiple Rotations
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 3: Multiple Rotation Amounts (k=2,4)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Generating rotation keys for k=1,2,4...");
    let multi_keys = generate_rotation_keys(&vec![1, 2, 4], &sk, &params);

    let ct_rot2 = rotate(&ct, 2, &multi_keys).expect("Rotation k=2 failed");
    let dec_rot2 = ckks_ctx.decode(&ckks_ctx.decrypt(&ct_rot2, &sk));
    let result2: Vec<i32> = dec_rot2.iter().take(10).map(|&x| x.round() as i32).collect();

    let ct_rot4 = rotate(&ct, 4, &multi_keys).expect("Rotation k=4 failed");
    let dec_rot4 = ckks_ctx.decode(&ckks_ctx.decrypt(&ct_rot4, &sk));
    let result4: Vec<i32> = dec_rot4.iter().take(10).map(|&x| x.round() as i32).collect();

    println!("k=2: {:?}", result2);
    println!("     Expected: [3, 4, 5, 6, 7, 8, 9, 10, 0, 0]");
    println!("k=4: {:?}", result4);
    println!("     Expected: [5, 6, 7, 8, 9, 10, 0, 0, 0, 0]");

    let test3a_pass = result2 == vec![3, 4, 5, 6, 7, 8, 9, 10, 0, 0];
    let test3b_pass = result4 == vec![5, 6, 7, 8, 9, 10, 0, 0, 0, 0];
    let test3_pass = test3a_pass && test3b_pass;

    println!("\n{}\n", if test3_pass {
        "✅ TEST 3 PASSED: Multiple rotations working!"
    } else {
        "❌ TEST 3 FAILED"
    });

    // ========================================================================
    // TEST 4: CoeffToSlot/SlotToCoeff
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 4: CoeffToSlot/SlotToCoeff Transformations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let rotations = required_rotations_for_bootstrap(params.n);
    println!("Generating {} rotation keys for bootstrap...", rotations.len());
    let bootstrap_keys = generate_rotation_keys(&rotations, &sk, &params);

    println!("Applying CoeffToSlot...");
    let ct_slot = coeff_to_slot(&ct, &bootstrap_keys).expect("CoeffToSlot failed");

    println!("Applying SlotToCoeff (inverse)...");
    let ct_back = slot_to_coeff(&ct_slot, &bootstrap_keys).expect("SlotToCoeff failed");

    let dec_back = ckks_ctx.decode(&ckks_ctx.decrypt(&ct_back, &sk));
    let result_back: Vec<i32> = dec_back.iter().take(10).map(|&x| x.round() as i32).collect();

    println!("Original:            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
    println!("After roundtrip:     {:?}", result_back);

    let test4_pass = result_back == vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    println!("\n{}\n", if test4_pass {
        "✅ TEST 4 PASSED: CoeffToSlot/SlotToCoeff roundtrip successful!"
    } else {
        "❌ TEST 4 FAILED"
    });

    // ========================================================================
    // FINAL SUMMARY
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("FINAL RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("TEST 1 (Canonical Embedding):  {}", if test1_pass { "✅ PASS" } else { "❌ FAIL" });
    println!("TEST 2 (Single Rotation):      {}", if test2_pass { "✅ PASS" } else { "❌ FAIL" });
    println!("TEST 3 (Multiple Rotations):   {}", if test3_pass { "✅ PASS" } else { "❌ FAIL" });
    println!("TEST 4 (CoeffToSlot/SlotToCoeff): {}", if test4_pass { "✅ PASS" } else { "❌ FAIL" });

    let all_pass = test1_pass && test2_pass && test3_pass && test4_pass;

    println!("\n{}", "═".repeat(68));
    if all_pass {
        println!("║  ALL TESTS PASSED - Phase 3 Verification Complete               ║");
        println!("{}", "═".repeat(68));
        println!("\nVerification Summary:");
        println!("  • Test Success Rate: 4/4 (100%)");
        println!("  • Maximum Error: < 0.5 (all tests)");
        println!("  • Test Parameters: N=1024, L=3 moduli");
        println!("\nComponents Verified:");
        println!("  • CKKS Canonical Embedding (orbit-ordered at roots ζ_M^(5^t))");
        println!("  • Rotation Key Generation (CRT-consistent decomposition)");
        println!("  • Homomorphic Rotation (correctness verified for k=1,2,4)");
        println!("  • CoeffToSlot Transformation (9 levels, 18 rotations)");
        println!("  • SlotToCoeff Transformation (roundtrip error < 0.5)");
        println!("\nNext: Phase 4 implementation (diagonal matrices, EvalMod)");
    } else {
        println!("║  SOME TESTS FAILED - Review output above for details            ║");
        println!("{}", "═".repeat(68));
    }
}
