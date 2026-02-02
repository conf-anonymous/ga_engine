// src/prelude.rs
//! The “everything” import for GAEngine.
//!
//! Brings you the most commonly used types and functions with one glob:
//! ```rust
//! use ga_engine::prelude::*;
//! ```

// core data types
pub use crate::bivector::Bivector3;
pub use crate::multivector::Multivector3;
pub use crate::rotor::Rotor3;
pub use crate::vector::{Rounded, Vec3};

// GA operations
pub use crate::ga::{geometric_product, geometric_product_full};
