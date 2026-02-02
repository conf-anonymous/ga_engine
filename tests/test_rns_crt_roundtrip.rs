//! Test RNS CRT reconstruction roundtrip

use ga_engine::clifford_fhe_v1::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v1::rns::RnsPolynomial;

#[test]
fn test_crt_roundtrip_small() {
    let params = CliffordFHEParams::new_rns_mult();

    // Create simple coefficients (must match params.n)
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = 5120;

    println!("Original coeffs[0]: {}", coeffs[0]);

    // Convert to RNS
    let rns = RnsPolynomial::from_coeffs(&coeffs, &params.moduli, params.n, 0);

    println!("RNS residues[0]: {:?}", &rns.rns_coeffs[0]);
    println!("Moduli: {:?}", &params.moduli);

    // Reconstruct back
    let recovered = rns.to_coeffs(&params.moduli);

    println!("Recovered coeffs: {:?}", &recovered[..3]);

    assert_eq!(recovered[0], coeffs[0],
               "CRT roundtrip failed: got {}, expected {}", recovered[0], coeffs[0]);
}
