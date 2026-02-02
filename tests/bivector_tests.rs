// tests/bivector_tests.rs

use ga_engine::bivector::Bivector3;
use ga_engine::vector::Vec3;

const EPS: f64 = 1e-12;

#[test]
fn test_new_components() {
    let bv = Bivector3::new(1.0, 2.0, 3.0);
    assert!((bv.xy - 1.0).abs() < EPS);
    assert!((bv.yz - 2.0).abs() < EPS);
    assert!((bv.zx - 3.0).abs() < EPS);
}

#[test]
fn test_from_wedge_basis() {
    let e1 = Vec3::new(1.0, 0.0, 0.0);
    let e2 = Vec3::new(0.0, 1.0, 0.0);
    let e3 = Vec3::new(0.0, 0.0, 1.0);

    // e1 ∧ e2 = e12 → zx = 1
    let b12 = Bivector3::from_wedge(e1, e2);
    assert_eq!(b12, Bivector3::new(0.0, 0.0, 1.0));

    // e2 ∧ e3 = e23 → xy = 1
    let b23 = Bivector3::from_wedge(e2, e3);
    assert_eq!(b23, Bivector3::new(1.0, 0.0, 0.0));

    // e3 ∧ e1 = e31 → yz = 1
    let b31 = Bivector3::from_wedge(e3, e1);
    assert_eq!(b31, Bivector3::new(0.0, 1.0, 0.0));
}

#[test]
fn test_wedge_antisymmetry() {
    let e1 = Vec3::new(1.0, 0.0, 0.0);
    let e2 = Vec3::new(0.0, 1.0, 0.0);

    // e2 ∧ e1 = - (e1 ∧ e2)
    let b21 = Bivector3::from_wedge(e2, e1);
    let b12 = Bivector3::from_wedge(e1, e2);
    assert_eq!(b21, Bivector3::new(0.0, 0.0, -1.0));
    assert_eq!(b21.xy, -b12.xy);
    assert_eq!(b21.yz, -b12.yz);
    assert_eq!(b21.zx, -b12.zx);
}
