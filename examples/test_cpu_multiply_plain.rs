//! Test CPU multiply_plain as reference

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

fn main() {
    println!("CPU multiply_plain test\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let ckks = CkksContext::new(params.clone());
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();

    // Encode and encrypt 42.0
    let pt1 = ckks.encode(&[42.0]);
    let ct = ckks.encrypt(&pt1, &pk);

    // Verify encryption
    let pt_back = ckks.decrypt(&ct, &sk);
    let decoded_back = ckks.decode(&pt_back);
    println!("Encrypted 42.0, decrypted: {}", decoded_back[0]);

    // Multiply by 2.0
    let pt_mult = ckks.encode(&[2.0]);
    let ct_mult = ct.multiply_plain(&pt_mult, &ckks);

    // Decrypt
    let pt_result = ckks.decrypt(&ct_mult, &sk);
    let decoded_result = ckks.decode(&pt_result);

    println!("\nAfter multiply_plain by 2.0:");
    println!("  Expected: 84.0");
    println!("  Got: {}", decoded_result[0]);
    println!("  Error: {:.2e}", (decoded_result[0] - 84.0).abs());

    if (decoded_result[0] - 84.0).abs() < 1.0 {
        println!("\n✅ CPU multiply_plain works correctly!");
    } else {
        println!("\n❌ CPU multiply_plain FAILED!");
    }
}
