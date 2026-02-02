//! src/nd/gp.rs
//!
//! (Deprecated) Runtime GP table generator for N-dimensional GA.
//! Note: This logic is now inlined in `Multivector<N>::gp` for performance.
//!
//! You can delete this module if you're no longer using precomputed tables.

#[allow(dead_code)]
use crate::nd::types::Scalar;

/// Legacy: Build a runtime GP‐table for a given dimension `n`.
/// Returns a Vec of length (2ⁿ)*(2ⁿ), indexed by `i * 2ⁿ + j`.
#[deprecated(note = "GP tables are now inlined for speed; this function is unused.")]
pub fn make_gp_table(n: usize) -> Vec<(Scalar, usize)> {
    let m = 1 << n;
    let mut table = Vec::with_capacity(m * m);
    for i in 0..m {
        for j in 0..m {
            table.push(sign_and_index(i, j, n));
        }
    }
    table
}

#[deprecated(note = "GP tables are now inlined.")]
#[allow(deprecated)]
pub fn gp_table_2() -> Vec<(Scalar, usize)> {
    make_gp_table(2)
}

#[deprecated(note = "GP tables are now inlined.")]
#[allow(deprecated)]
pub fn gp_table_3() -> Vec<(Scalar, usize)> {
    make_gp_table(3)
}

#[deprecated(note = "GP tables are now inlined.")]
#[allow(deprecated)]
pub fn gp_table_4() -> Vec<(Scalar, usize)> {
    make_gp_table(4)
}

/// Legacy sign and blade index function used during precomputation.
fn sign_and_index(i: usize, j: usize, n: usize) -> (Scalar, usize) {
    let k = i ^ j;
    let mut sign = 1.0;
    for bit in 0..n {
        if ((i >> bit) & 1) != 0 {
            let mut lower = j & ((1 << bit) - 1);
            let mut count = 0;
            while lower != 0 {
                count += lower & 1;
                lower >>= 1;
            }
            if count % 2 != 0 {
                sign = -sign;
            }
        }
    }
    (sign, k)
}
