//! Test EVK digit count
//! Run with: cargo test --test test_v2_evk_digits --features v2 -- --nocapture

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

#[test]
fn test_evk_digit_count() {
    println!("\n=== EVK DIGIT COUNT TEST ===\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("Moduli: {:?}", params.moduli);

    use num_bigint::BigInt;
    let q_prod: BigInt = params.moduli.iter().map(|&q| BigInt::from(q)).product();
    let q_bits = q_prod.bits() as u32;
    let base_w = 20u32;
    let num_digits = ((q_bits + base_w - 1) / base_w) as usize;

    println!("Q = {}", q_prod);
    println!("Q has {} bits", q_bits);
    println!("base_w = {}", base_w);
    println!("num_digits = ceil({} / {}) = {}", q_bits, base_w, num_digits);

    let key_ctx = KeyContext::new(params.clone());
    let (_pk, _sk, evk) = key_ctx.keygen();

    println!("\nEVK has {} components (evk0.len())", evk.evk0.len());
    println!("EVK has {} components (evk1.len())", evk.evk1.len());

    assert_eq!(evk.evk0.len(), num_digits, "EVK must have correct number of digits");
    assert_eq!(evk.evk1.len(), num_digits, "EVK must have correct number of digits");

    println!("\n✓ EVK has correct number of digits!");
}
