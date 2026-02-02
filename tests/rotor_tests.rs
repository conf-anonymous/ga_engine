// tests/rotor_tests.rs
// DISABLED: transform module no longer exists
#![cfg(feature = "transform_module_exists")]

use ga_engine::{transform::apply_matrix3, Rotor3, Vec3};

const EPS: f64 = 1e-12;

/// Helper to create a +90° rotation around Z via classical matrix
fn make_rotor_and_matrix() -> (Rotor3, Vec3) {
    let v = Vec3::new(1.0, 0.0, 0.0);
    let m: [f64; 9] = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let rotor = Rotor3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
    let v_mat = apply_matrix3(&m, v);
    (rotor, v_mat)
}

#[test]
fn rotate_z_with_sandwich() {
    let (rotor, v_mat) = make_rotor_and_matrix();
    let v = Vec3::new(1.0, 0.0, 0.0);
    let v_ga = rotor.rotate(v);
    assert!(
        (v_mat.x - v_ga.x).abs() < EPS,
        "x mismatch: {} vs {}",
        v_mat.x,
        v_ga.x
    );
    assert!(
        (v_mat.y - v_ga.y).abs() < EPS,
        "y mismatch: {} vs {}",
        v_mat.y,
        v_ga.y
    );
    assert!(
        (v_mat.z - v_ga.z).abs() < EPS,
        "z mismatch: {} vs {}",
        v_mat.z,
        v_ga.z
    );
}

#[test]
fn rotate_z_with_fast() {
    let (rotor, v_mat) = make_rotor_and_matrix();
    let v = Vec3::new(1.0, 0.0, 0.0);
    let v_fast = rotor.rotate_fast(v);
    assert!(
        (v_mat.x - v_fast.x).abs() < EPS,
        "x mismatch fast: {} vs {}",
        v_mat.x,
        v_fast.x
    );
    assert!(
        (v_mat.y - v_fast.y).abs() < EPS,
        "y mismatch fast: {} vs {}",
        v_mat.y,
        v_fast.y
    );
    assert!(
        (v_mat.z - v_fast.z).abs() < EPS,
        "z mismatch fast: {} vs {}",
        v_mat.z,
        v_fast.z
    );
}

#[test]
fn rotate_z_simd4_matches() {
    let (rotor, v_mat) = make_rotor_and_matrix();
    let vs = [Vec3::new(1.0, 0.0, 0.0); 4];
    let out = rotor.rotate_simd(vs);
    for &r in &out {
        assert!((r.x - v_mat.x).abs() < EPS);
        assert!((r.y - v_mat.y).abs() < EPS);
        assert!((r.z - v_mat.z).abs() < EPS);
    }
}

#[test]
fn rotate_z_simd8_matches() {
    let (rotor, v_mat) = make_rotor_and_matrix();
    let vs = [Vec3::new(1.0, 0.0, 0.0); 8];
    let out = rotor.rotate_simd8(vs);
    for &r in &out {
        assert!((r.x - v_mat.x).abs() < EPS);
        assert!((r.y - v_mat.y).abs() < EPS);
        assert!((r.z - v_mat.z).abs() < EPS);
    }
}
