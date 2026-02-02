// tests/nd_multivector_tests.rs

use ga_engine::nd::multivector::Multivector;
use ga_engine::nd::types::Scalar;

/// Helper to build a 2-D multivector from its 4 components:
/// [scalar, e1, e2, e12]
fn mv2(data: [Scalar; 4]) -> Multivector<2> {
    Multivector::new(data.to_vec())
}

#[test]
fn basis_blade_products_2d() {
    // basis blades in 2D
    let s = mv2([1.0, 0.0, 0.0, 0.0]); // scalar 1
    let e1 = mv2([0.0, 1.0, 0.0, 0.0]);
    let e2 = mv2([0.0, 0.0, 1.0, 0.0]);
    let e12 = mv2([0.0, 0.0, 0.0, 1.0]);

    // 1 * anything = itself
    assert_eq!(s.gp(&e1).data, e1.data);
    assert_eq!(e2.gp(&s).data, e2.data);

    // e1*e1 = +1
    assert_eq!(e1.gp(&e1).data, s.data);

    // e2*e2 = +1
    assert_eq!(e2.gp(&e2).data, s.data);

    // e1 ∧ e2 = e12
    assert_eq!(e1.gp(&e2).data, e12.data);

    // anti‐commutativity: e2*e1 = –e12
    let mut neg_e12 = e12.data.clone();
    neg_e12[3] = -neg_e12[3];
    assert_eq!(e2.gp(&e1).data, neg_e12);

    // pseudoscalar square = (e12)*(e12) = -1
    let mut neg_s = s.data.clone();
    neg_s[0] = -neg_s[0];
    assert_eq!(e12.gp(&e12).data, neg_s);
}

#[test]
fn mixed_dot_and_wedge_2d() {
    // (a e1 + b e2) * (c e1 + d e2) = (ac+bd) + (ad−bc) e12
    let a = mv2([0.0, 2.0, 3.0, 0.0]);
    let b = mv2([0.0, 5.0, 7.0, 0.0]);

    let prod = a.gp(&b);
    // scalar part = 2*5 + 3*7 = 10 + 21 = 31
    assert!((prod.data[0] - 31.0).abs() < 1e-12);
    // e12 part = 2*7 - 3*5 = 14 - 15 = -1
    assert!((prod.data[3] + 1.0).abs() < 1e-12);
    // vector parts should vanish
    assert_eq!(prod.data[1], 0.0);
    assert_eq!(prod.data[2], 0.0);
}
