// tests/transform_matrix_tests.rs
// DISABLED: transform module no longer exists
#![cfg(feature = "transform_module_exists")]

use ga_engine::{transform::apply_matrix3, vector::Vec3};

const EPS: f64 = 1e-12;

#[test]
fn identity_matrix_leaves_vector_unchanged() {
    let v = Vec3::new(3.5, -1.2, 7.8);
    let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let r = apply_matrix3(&m, v);
    assert!((r.x - v.x).abs() < EPS);
    assert!((r.y - v.y).abs() < EPS);
    assert!((r.z - v.z).abs() < EPS);
}

#[test]
fn scale_matrix_scales_vector_components() {
    let v = Vec3::new(1.0, -2.0, 3.0);
    let m = [2.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.5];
    let r = apply_matrix3(&m, v);
    assert!((r.x - 2.0).abs() < EPS);
    assert!((r.y - 6.0).abs() < EPS);
    assert!((r.z - 1.5).abs() < EPS);
}

#[test]
fn rotate_z_90_degrees_about_z() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    // +90° about Z: x→-y, y→x, z unchanged
    let m = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let r = apply_matrix3(&m, v);
    assert!((r.x - -2.0).abs() < EPS);
    assert!((r.y - 1.0).abs() < EPS);
    assert!((r.z - 3.0).abs() < EPS);
}
