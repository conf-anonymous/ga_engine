//! Lattice Reduction Module
//!
//! This module implements geometric algebra-accelerated lattice reduction algorithms
//! for cryptanalysis applications. The key innovation is using rotor-based projections
//! during enumeration instead of explicit μ coefficient computations.
//!
//! # Overview
//!
//! Traditional lattice reduction (LLL, BKZ) uses Gram-Schmidt orthogonalization (GSO)
//! which requires O(n²) storage for projection coefficients (μ matrix). During enumeration,
//! many projection operations are needed: v' = v - μ₁b₁ - μ₂b₂ - ...
//!
//! Our GA approach uses rotor chains to represent the orthogonalization transformation,
//! requiring only O(n) storage. Projections become rotor sandwich products: v' = R·v·R†
//!
//! # Key Advantages
//!
//! - **Memory**: O(n) rotor chain vs O(n²) μ matrix
//! - **Re-orthogonalization**: Rotor composition (O(n)) vs μ matrix recomputation (O(n²))
//! - **Cache locality**: Compact rotor representation
//! - **Numerical stability**: Unit rotors preserve norms
//!
//! # Modules
//!
//! - `rotor_nd`: n-dimensional rotor operations (core primitive)
//! - `lll_baseline`: Standard LLL for comparison
//! - `lll_ga`: GA-accelerated LLL
//! - `bkz_ga`: GA-accelerated BKZ with enumeration
//! - `svp_challenge`: SVP Challenge format parser
//! - `verification`: Solution verification

pub mod rotor_nd;
pub mod lll_baseline;
pub mod ga_lll;
pub mod ga_lll_rotors;  // Hypothesis 1 - Hybrid rotor GSO (v1)
pub mod ga_lll_pure;    // Hypothesis 1 - Pure rotor GSO (v2) - FAILED
pub mod e8_lattice;     // E8 orbit enumeration (symmetry test)
pub mod ga_reflection;  // GA reflections for E8
pub mod stable_gso;
pub mod enumeration;
// pub mod bkz_baseline;  // Removed: old buggy implementation, replaced by bkz_stable
pub mod bkz_stable;
pub mod bkz_rotor;    // Rotor-tracked BKZ (15× GSO update speedup)
// pub mod bkz_ga;
pub mod svp_challenge;
// pub mod verification;
