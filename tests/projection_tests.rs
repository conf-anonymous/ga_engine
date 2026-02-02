// tests/projection_tests.rs

use ga_engine::ops::projection::{project_onto, project_onto_plane, reject_from, Vec3Projection};
use ga_engine::vector::Vec3;

const EPS: f64 = 1e-12;

#[test]
fn test_project_onto_axis() {
    let v = Vec3::new(3.0, 4.0, 0.0);
    let axis = Vec3::new(0.0, 1.0, 0.0);

    // free‐function version
    let p1 = project_onto(&v, &axis);
    // method version
    let p2 = v.project_onto(&axis);

    assert!((p1.x - 0.0).abs() < EPS && (p1.y - 4.0).abs() < EPS && (p1.z - 0.0).abs() < EPS);
    assert_eq!(p1, p2);
}

#[test]
fn test_reject_from_axis() {
    let v = Vec3::new(3.0, 4.0, 0.0);
    let axis = Vec3::new(0.0, 1.0, 0.0);

    let r1 = reject_from(&v, &axis);
    let r2 = v.reject_from(&axis);

    // projection was (0,4,0) so rejection is original minus that = (3,0,0)
    assert!((r1.x - 3.0).abs() < EPS && (r1.y - 0.0).abs() < EPS && (r1.z - 0.0).abs() < EPS);
    assert_eq!(r1, r2);
}

#[test]
fn test_project_onto_plane() {
    let v = Vec3::new(3.0, 4.0, 5.0);
    // project onto the plane whose normal is e_z = (0,0,1)
    let normal = Vec3::new(0.0, 0.0, 1.0);

    let p1 = project_onto_plane(&v, &normal);
    let p2 = v.project_onto_plane(&normal);

    // dropping the z‐component: (3,4,0)
    assert!((p1.x - 3.0).abs() < EPS && (p1.y - 4.0).abs() < EPS && (p1.z - 0.0).abs() < EPS);
    assert_eq!(p1, p2);
}

#[test]
fn test_project_onto_zero_axis_yields_zero() {
    let v = Vec3::new(3.0, 4.0, 5.0);
    let zero = Vec3::new(0.0, 0.0, 0.0);

    let p = project_onto(&v, &zero);
    let r = reject_from(&v, &zero);

    // projecting onto a zero‐vector → zero; rejection = original
    assert_eq!(p, Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(r, v);
}
