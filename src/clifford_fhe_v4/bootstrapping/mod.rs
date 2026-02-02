//! V4 Bootstrap Operations for Packed Multivectors
//!
//! V4 bootstrap leverages the V3 CUDA bootstrap infrastructure but operates
//! on packed multivectors with 8× memory reduction.
//!
//! ## Key Insight
//!
//! Since bootstrap operations (CoeffToSlot, EvalMod, SlotToCoeff) work uniformly
//! on all slots, the packed layout with 8 interleaved components is preserved
//! through the bootstrap process.
//!
//! ## Performance
//!
//! V4 bootstrap has the same computational cost as V3 for a single ciphertext,
//! but effectively bootstraps 8 components simultaneously, achieving:
//! - Same total time as V3 for equivalent data
//! - 8× memory reduction during bootstrap
//! - Useful when multiple packed multivectors need bootstrapping

#[cfg(feature = "v2-gpu-cuda")]
pub mod cuda;

#[cfg(feature = "v2-gpu-cuda")]
pub use cuda::{V4BootstrapContext, bootstrap_packed_multivector};
