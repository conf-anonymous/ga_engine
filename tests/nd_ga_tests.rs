// tests/nd_ga_tests.rs

use ga_engine::nd::ga::{gp_table_2, gp_table_3, make_gp_table};
use ga_engine::nd::types::Scalar;

#[test]
fn make_gp_table_2d_basic_entries() {
    let table = make_gp_table::<2>();
    // 2ⁿ=4 blades → 4×4=16 entries
    assert_eq!(table.len(), 16);

    // index helper with usize to satisfy slice indexing
    let idx = |i: usize, j: usize| table[i * 4 + j];

    // scalar (i=0) times any j yields ( +1, j )
    for j in 0..4 {
        assert_eq!(idx(0, j), (1.0 as Scalar, j));
    }

    // e1*e1 = +scalar
    assert_eq!(idx(1, 1), (1.0 as Scalar, 0));
    // e2*e2 = +scalar
    assert_eq!(idx(2, 2), (1.0 as Scalar, 0));

    // e1*e2 = +e12  (1⊕2 = 3)
    assert_eq!(idx(1, 2), (1.0 as Scalar, 3));
    // e2*e1 = -e12
    assert_eq!(idx(2, 1), (-1.0 as Scalar, 3));

    // e12*e12 = -scalar
    assert_eq!(idx(3, 3), (-1.0 as Scalar, 0));
}

#[test]
fn gp_table_2_alias_matches_make() {
    let t1 = make_gp_table::<2>();
    let t2 = gp_table_2();
    assert_eq!(t1, t2);
}

#[test]
fn make_gp_table_3d_basic_entries() {
    let table = make_gp_table::<3>();
    // 2ⁿ=8 blades → 8×8=64 entries
    assert_eq!(table.len(), 64);

    let idx = |i: usize, j: usize| table[i * 8 + j];

    // e1*e2 = +e12 (1⊕2=3)
    assert_eq!(idx(1, 2), (1.0 as Scalar, 3));
    // e2*e1 = -e12
    assert_eq!(idx(2, 1), (-1.0 as Scalar, 3));

    // e1*e1 = +1 (scalar)
    assert_eq!(idx(1, 1), (1.0 as Scalar, 0));

    // pseudoscalar e123 index is 1⊕2⊕4 = 7; so e123*e123 = -1
    assert_eq!(idx(7, 7), (-1.0 as Scalar, 0));
}

#[test]
fn gp_table_3_alias_matches_make() {
    let t1 = make_gp_table::<3>();
    let t2 = gp_table_3();
    assert_eq!(t1, t2);
}
