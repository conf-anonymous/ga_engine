// tests/multivector_tests.rs

use ga_engine::{Bivector3, Multivector3, Vec3};

const EPS: f64 = 1e-12;

#[test]
fn zero_multivector_is_all_zero() {
    let m = Multivector3::zero();
    assert!((m.scalar).abs() < EPS);
    assert!((m.vector.x).abs() < EPS && (m.vector.y).abs() < EPS && (m.vector.z).abs() < EPS);
    assert!(
        (m.bivector.xy).abs() < EPS && (m.bivector.yz).abs() < EPS && (m.bivector.zx).abs() < EPS
    );
    assert!((m.pseudo).abs() < EPS);
}

#[test]
fn from_scalar_sets_only_scalar_component() {
    let s = std::f64::consts::PI;
    let m = Multivector3::from_scalar(s);
    assert!((m.scalar - s).abs() < EPS);
    assert_eq!(m.vector, Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(m.bivector, Bivector3::new(0.0, 0.0, 0.0));
    assert!((m.pseudo).abs() < EPS);
}

#[test]
fn from_vector_sets_only_vector_component() {
    let v = Vec3::new(1.0, -2.0, 3.0);
    let m = Multivector3::from_vector(v);
    assert!((m.scalar).abs() < EPS);
    assert_eq!(m.vector, v);
    assert_eq!(m.bivector, Bivector3::new(0.0, 0.0, 0.0));
    assert!((m.pseudo).abs() < EPS);
}

#[test]
fn geometric_product_scalar_times_vector() {
    let m_s = Multivector3::from_scalar(2.0);
    let v = Vec3::new(1.0, 2.0, 3.0);
    let m_v = Multivector3::from_vector(v);
    let r = m_s.gp(&m_v);
    // scalar*vector → vector doubled
    assert!((r.scalar).abs() < EPS);
    assert_eq!(r.vector, Vec3::new(2.0, 4.0, 6.0));
    assert_eq!(r.bivector, Bivector3::new(0.0, 0.0, 0.0));
    assert!((r.pseudo).abs() < EPS);
}

#[test]
fn geometric_product_vector_times_vector() {
    let v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(4.0, 5.0, 6.0);
    let m1 = Multivector3::from_vector(v1);
    let m2 = Multivector3::from_vector(v2);
    let r = m1.gp(&m2);
    // dot product = 1*4 + 2*5 + 3*6 = 32
    assert!((r.scalar - 32.0).abs() < EPS);
    // vector part zero
    assert_eq!(r.vector, Vec3::new(0.0, 0.0, 0.0));
    // bivector = v1 ∧ v2
    let expected_biv = Bivector3::new(
        v1.y * v2.z - v1.z * v2.y, // xy = e23
        v1.z * v2.x - v1.x * v2.z, // yz = e31
        v1.x * v2.y - v1.y * v2.x, // zx = e12
    );
    assert_eq!(r.bivector, expected_biv);
    assert!((r.pseudo).abs() < EPS);
}

#[test]
fn reverse_flips_bivector_and_pseudoscalar() {
    let mut m = Multivector3::zero();
    m.scalar = 1.0;
    m.vector = Vec3::new(2.0, 3.0, 4.0);
    m.bivector = Bivector3::new(5.0, 6.0, 7.0);
    m.pseudo = 8.0;

    let r = m.reverse();
    // scalar and vector unchanged
    assert!((r.scalar - 1.0).abs() < EPS);
    assert_eq!(r.vector, Vec3::new(2.0, 3.0, 4.0));
    // bivector and pseudo negated
    assert_eq!(r.bivector, Bivector3::new(-5.0, -6.0, -7.0));
    assert!((r.pseudo + 8.0).abs() < EPS);
}

#[test]
fn double_reverse_restores_original() {
    let mut m = Multivector3::zero();
    m.scalar = 9.0;
    m.vector = Vec3::new(-1.0, 2.0, -3.0);
    m.bivector = Bivector3::new(4.0, -5.0, 6.0);
    m.pseudo = -7.0;

    let r2 = m.reverse().reverse();
    assert_eq!(r2, m);
}
