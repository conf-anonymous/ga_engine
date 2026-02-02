//! Cryptanalysis using Geometric Algebra
//!
//! This module implements cryptanalytic techniques accelerated by GA:
//! - S-box analysis (DDT, LAT) using multilinear forms
//! - Rotor-based differential trail propagation
//! - Clifford-Fourier transforms for linear cryptanalysis

pub mod boolean_ga;
pub mod sbox_ga;
pub mod sbox_analysis;
pub mod bitsliced_sbox;
#[cfg(feature = "lattice-reduction")]
pub mod trail_propagation;
pub mod higher_order_differential;
