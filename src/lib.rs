#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(non_snake_case)]

//! # GAEngine Quickstart
//!
//! ```rust
//! use ga_engine::prelude::*;
//!
//! // Rotate (1,0,0) 90° about the Z axis
//! let v = Vec3::new(1.0, 0.0, 0.0);
//! let r = Rotor3::from_axis_angle(
//!     Vec3::new(0.0, 0.0, 1.0),
//!     std::f64::consts::FRAC_PI_2,
//! );
//! let v_rot = r.rotate_fast(v);
//!
//! // Should end up at (0,1,0)
//! const EPS: f64 = 1e-12;
//! assert!((v_rot.x.abs()) < EPS);
//! assert!((v_rot.y - 1.0).abs() < EPS);
//! assert!((v_rot.z).abs() < EPS);
//! ```
//!
#![doc = include_str!("../README.md")]

// Core modules
pub mod bivector;

// Clifford FHE: Version selection via feature flags
#[cfg(feature = "v1")]
pub mod clifford_fhe_v1;  // V1: Baseline reference (stable, complete)

#[cfg(feature = "v2")]
pub mod clifford_fhe_v2;  // V2: Optimized implementation (active development)

#[cfg(feature = "v3")]
pub mod clifford_fhe_v3;  // V3: Bootstrapping for unlimited depth (in development)

#[cfg(feature = "v4")]
pub mod clifford_fhe_v4;  // V4: Packed slot-interleaved layout (8× memory reduction)

#[cfg(feature = "v5")]
pub mod clifford_fhe_v5;  // V5: Privacy-trace collection and analysis

#[cfg(feature = "v6")]
pub mod clifford_fhe_v6;  // V6: parallel_lift GPU acceleration (25-552× speedup)

// Default export: V1 unless V2 is explicitly requested
#[cfg(all(feature = "v1", not(feature = "v2")))]
pub use clifford_fhe_v1 as clifford_fhe;

#[cfg(feature = "v2")]
pub use clifford_fhe_v2 as clifford_fhe;

pub mod ga;
pub mod multivector;
pub mod ops;
pub mod prelude;
pub mod rotor;
pub mod vector;

// N-dimensional GA support
pub mod nd;

// Medical imaging applications
pub mod medical_imaging;

// Lattice reduction for cryptanalysis (optional, enabled by lattice-reduction feature)
#[cfg(feature = "lattice-reduction")]
pub mod lattice_reduction;

// Cryptanalysis using GA (S-box analysis, differential trail propagation)
pub mod cryptanalysis;

// --- Public API exports ---

// 3D types and operations
pub use bivector::Bivector3;
pub use ga::{geometric_product, geometric_product_full};
pub use multivector::Multivector3;
pub use rotor::Rotor3;
pub use vector::{Rounded, Vec3};

// High-level GA ops
pub use ops::motor::Motor3;
pub use ops::projection::*;
pub use ops::reflection::*;

// N-dimensional types
pub use nd::multivector::Multivector;
