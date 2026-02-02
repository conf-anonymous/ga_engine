// tests/ga_full_tests.rs

use ga_engine::ga::geometric_product_full;

/// Generate the 8 basis blades as 8-component multivectors.
fn basis_blades() -> Vec<[f64; 8]> {
    (0..8)
        .map(|i| {
            let mut v = [0.0; 8];
            v[i] = 1.0;
            v
        })
        .collect()
}

#[test]
fn basis_blade_products_yield_single_nonzero() {
    let blades = basis_blades();
    for (i, a) in blades.iter().enumerate() {
        for (j, b) in blades.iter().enumerate() {
            let mut out = [0.0; 8];
            geometric_product_full(a, b, &mut out);
            // Count nonzero entries
            let nonzeros: Vec<_> = out.iter().cloned().filter(|&x| x != 0.0).collect();
            assert_eq!(
                nonzeros.len(),
                1,
                "Expected exactly one nonzero for blade {} × {}, got {:?}",
                i,
                j,
                out
            );
            // That nonzero must be +1 or -1
            let v = nonzeros[0];
            assert!(
                v == 1.0 || v == -1.0,
                "Expected ±1.0 for blade {} × {}, got {}",
                i,
                j,
                v
            );
        }
    }
}

#[test]
fn unit_vectors_square_to_scalar_one() {
    let blades = basis_blades(); // ← bind the blades here
    for blade in &blades[1..4] {
        let mut out = [0.0; 8];
        geometric_product_full(blade, blade, &mut out);
        assert_eq!(
            out,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Unit vector squared should give scalar 1.0, got {:?}",
            out
        );
    }
}

#[test]
fn pseudoscalar_square_is_minus_one() {
    // e123 * e123 = -1
    let blades = basis_blades();
    let pseu = blades[7]; // index 7 is e123
    let mut out = [0.0; 8];
    geometric_product_full(&pseu, &pseu, &mut out);
    assert_eq!(
        out,
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Pseudoscalar squared should be -1"
    );
}
