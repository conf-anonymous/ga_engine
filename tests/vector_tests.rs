// tests/vector_tests.rs

use ga_engine::vector::Vec3;

const EPS: f64 = 1e-12;

#[test]
fn test_new_and_fields() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);
}

#[test]
fn test_dot() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, -5.0, 6.0);
    // 1*4 + 2*(-5) + 3*6 = 4 -10 +18 = 12
    assert!((a.dot(&b) - 12.0).abs() < EPS);
}

#[test]
fn test_cross() {
    let e1 = Vec3::new(1.0, 0.0, 0.0);
    let e2 = Vec3::new(0.0, 1.0, 0.0);
    let e3 = Vec3::new(0.0, 0.0, 1.0);
    assert_eq!(e1.cross(&e2), e3);
    assert_eq!(e2.cross(&e3), e1);
    assert_eq!(e3.cross(&e1), e2);
    // anti-commutativity
    assert_eq!(e2.cross(&e1), Vec3::new(0.0, 0.0, -1.0));
}

#[test]
fn test_norm() {
    let v = Vec3::new(3.0, 4.0, 0.0);
    assert!((v.norm() - 5.0).abs() < EPS);
}

#[test]
fn test_scale() {
    let v = Vec3::new(1.5, -2.0, 0.5);
    let w = v.scale(2.0);
    assert!((w.x - 3.0).abs() < EPS);
    assert!((w.y + 4.0).abs() < EPS);
    assert!((w.z - 1.0).abs() < EPS);
}

#[test]
fn test_add() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    let c = a + b;
    assert_eq!(c, Vec3::new(5.0, 7.0, 9.0));
}

#[test]
fn test_sub() {
    let a = Vec3::new(4.0, 5.0, 6.0);
    let b = Vec3::new(1.0, 1.0, 1.0);
    let c = a - b;
    assert_eq!(c, Vec3::new(3.0, 4.0, 5.0));
}

#[test]
fn test_mul_scalar() {
    let v = Vec3::new(2.0, -3.0, 0.5);
    let w = v * 3.0;
    assert_eq!(w, Vec3::new(6.0, -9.0, 1.5));
}

#[test]
fn test_display_rounded() {
    let v = Vec3::new(1.23456789, -2.3456789, 3.456789);
    let s = format!("{}", ga_engine::vector::Rounded::new(&v, 3));
    assert_eq!(s, "Vec3 { x: 1.235, y: -2.346, z: 3.457 }");
}
