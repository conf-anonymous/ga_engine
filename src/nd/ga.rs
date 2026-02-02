// src/nd/ga.rs
//! Geometric‐product table generator for N‐dimensional Euclidean GA.
//!
//! At run time, builds a lookup table of size 2ⁿ×2ⁿ mapping
//! (i, j) → (sign, k) for blade multiplication: e_i * e_j = sign · e_k.

use crate::nd::types::Scalar;

/// Compute the sign (±1) and output blade index for blade `i` × blade `j`.
///
/// Blades are represented by bitmasks 0..2ⁿ. The resulting blade‐mask is `i ^ j`,
/// and the sign is determined by counting bit‐swaps.
fn sign_and_index<const N: usize>(i: usize, j: usize) -> (Scalar, usize) {
    let mi = i;
    let mj = j;
    let out = mi ^ mj;

    // Count the parity of swaps: for each bit set in mi, count lower bits in mj.
    let mut sgn = 1i32;
    let mut bit = 0;
    while bit < N {
        if ((mi >> bit) & 1) != 0 {
            let mut lower = mj & ((1 << bit) - 1);
            let mut cnt = 0u32;
            while lower != 0 {
                cnt += (lower & 1) as u32;
                lower >>= 1;
            }
            if (cnt & 1) != 0 {
                sgn = -sgn;
            }
        }
        bit += 1;
    }

    let sign = if sgn > 0 {
        1.0 as Scalar
    } else {
        -1.0 as Scalar
    };
    (sign, out)
}

/// Build the full geometric‐product table at run time.
/// Returns a `Vec` of length `(2ⁿ)*(2ⁿ)`, indexed by `i*m + j`.
pub fn make_gp_table<const N: usize>() -> Vec<(Scalar, usize)> {
    let m = 1 << N;
    let mut table = Vec::with_capacity(m * m);
    for i in 0..m {
        for j in 0..m {
            table.push(sign_and_index::<N>(i, j));
        }
    }
    table
}

// Example helpers for common small dims, if you want to cache them:
/// 2‐D GA: 4 blades → 16 entries
pub fn gp_table_2() -> Vec<(Scalar, usize)> {
    make_gp_table::<2>()
}
/// 3‐D GA: 8 blades → 64 entries
pub fn gp_table_3() -> Vec<(Scalar, usize)> {
    make_gp_table::<3>()
}
/// 4‐D GA: 16 blades → 256 entries
pub fn gp_table_4() -> Vec<(Scalar, usize)> {
    make_gp_table::<4>()
}
