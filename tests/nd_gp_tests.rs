// tests/nd_gp_tests.rs
#[allow(deprecated)]
use ga_engine::nd::gp::make_gp_table;
use ga_engine::nd::types::Scalar;

#[test]
#[allow(deprecated)]
fn gp_table_2_length_and_basics() {
    let table = make_gp_table(2);
    assert_eq!(table.len(), 16);

    // Force i and j to be usize
    let idx = |i: usize, j: usize| table[i * 4 + j];

    // scalar*anything = itself
    assert_eq!(idx(0, 0), (1.0 as Scalar, 0));
    assert_eq!(idx(0, 3), (1.0 as Scalar, 3));

    // e1*e1 = +scalar
    assert_eq!(idx(1, 1), (1.0 as Scalar, 0));
    // e2*e2 = +scalar
    assert_eq!(idx(2, 2), (1.0 as Scalar, 0));

    // e1*e2 = +e12  (1⊕2=3)
    assert_eq!(idx(1, 2), (1.0 as Scalar, 3));
    // e2*e1 = -e12
    assert_eq!(idx(2, 1), (-1.0 as Scalar, 3));

    // e12 * e12 = -scalar (3⊕3=0)
    assert_eq!(idx(3, 3), (-1.0 as Scalar, 0));
}

#[test]
#[allow(deprecated)]
fn gp_table_3_some_spots() {
    let table = make_gp_table(3);
    assert_eq!(table.len(), 64);

    let idx = |i: usize, j: usize| table[i * 8 + j];

    // e1*e2 = +e12 → mask = 1⊕2 = 3
    assert_eq!(idx(1, 2), (1.0 as Scalar, 3));
    // e2*e1 = -e12
    assert_eq!(idx(2, 1), (-1.0 as Scalar, 3));

    // e1*e1 = +1
    assert_eq!(idx(1, 1), (1.0 as Scalar, 0));
    // e123 * e123 = -1 (7⊕7=0)
    assert_eq!(idx(7, 7), (-1.0 as Scalar, 0));
}
