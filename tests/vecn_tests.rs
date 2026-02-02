// VecN tests - DISABLED: vecn module no longer exists
#![cfg(feature = "vecn_module_exists")]

use ga_engine::nd::vecn::VecN;

const EPS: f64 = 1e-12;

#[test]
fn vecn_new_and_data() {
    let v: VecN<3> = VecN::new([1.0, 2.0, 3.0]);
    assert_eq!(v.data, [1.0, 2.0, 3.0]);
}

#[test]
fn vecn_dot_and_norm() {
    let a = VecN::new([1.0, 2.0, 2.0]);
    let b = VecN::new([2.0, 1.0, 2.0]);
    // dot = 1*2 + 2*1 + 2*2 = 2+2+4 = 8
    assert_eq!(a.dot(&b), 8.0);
    // norm²(a) = 1+4+4 =9 → norm =3
    assert!((a.norm() - 3.0).abs() < EPS);
}

#[test]
fn vecn_scale_add_sub_mul_neg() {
    let v = VecN::new([1.0, -2.0, 3.0]);
    // scale
    let s = v.scale(2.0);
    assert_eq!(s.data, [2.0, -4.0, 6.0]);
    // add
    let two = VecN::new([1.0, 1.0, 1.0]);
    let sum = v.clone() + two.clone();
    assert_eq!(sum.data, [2.0, -1.0, 4.0]);
    // sub
    let diff = v.clone() - two.clone();
    assert_eq!(diff.data, [0.0, -3.0, 2.0]);
    // mul<Scalar>
    let m = v.clone() * 3.0;
    assert_eq!(m.data, [3.0, -6.0, 9.0]);
    // neg
    let n = -v.clone();
    assert_eq!(n.data, [-1.0, 2.0, -3.0]);
}

#[test]
fn vecn_generic_dimensions() {
    // try N=5
    let u = VecN::new([1.0, 2.0, 3.0, 4.0, 5.0]);
    let v = VecN::new([5.0, 4.0, 3.0, 2.0, 1.0]);
    // dot = 1*5+2*4+3*3+4*2+5*1 = 5+8+9+8+5 =35
    assert_eq!(u.dot(&v), 35.0);
    // norm² = 1+4+9+16+25 =55
    assert!((u.norm().powi(2) - 55.0).abs() < EPS);
}
