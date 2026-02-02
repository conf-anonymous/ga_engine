// DISABLED: classical module no longer exists
#![cfg(feature = "classical_module_exists")]

use ga_engine::classical::multiply_matrices;

#[test]
fn identity_2x2() {
    let a = vec![1.0, 0.0, 0.0, 1.0];
    let c = multiply_matrices(&a, &a, 2);
    assert_eq!(c, a);
}

#[test]
fn simple_2x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![3.0, 1.0, 2.0, 1.0];
    let c = multiply_matrices(&a, &b, 2);
    assert_eq!(c, vec![7.0, 3.0, 17.0, 7.0]);
}
