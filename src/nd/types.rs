// src/nd/types.rs
#![allow(dead_code)]

#[cfg(feature = "f32")]
pub type Scalar = f32;
#[cfg(not(feature = "f32"))]
pub type Scalar = f64;

// the dimension is also const-generic, so we don't hard-code DIM here
