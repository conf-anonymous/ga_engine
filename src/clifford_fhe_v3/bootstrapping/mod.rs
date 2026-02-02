//! CKKS Bootstrapping Module
//!
//! Implements homomorphic noise refresh to enable deep computation.
//!
//! ## Bootstrapping Pipeline
//!
//! ```text
//! Input: Noisy ciphertext (almost out of levels)
//!   ↓
//! 1. ModRaise: Raise modulus to higher level (~10ms)
//!   ↓
//! 2. CoeffToSlot: Transform to evaluation form (~200ms)
//!   ↓
//! 3. EvalMod: Homomorphic modular reduction (~500ms)
//!   ↓
//! 4. SlotToCoeff: Transform back to coefficient form (~200ms)
//!   ↓
//! Output: Fresh ciphertext (full levels restored, noise removed)
//! Total: ~1 second per ciphertext
//! ```
//!
//! ## Components
//!
//! - **ModRaise:** Modulus raising for bootstrap working room
//! - **CoeffToSlot:** FFT-like transformation to evaluation form
//! - **EvalMod:** Homomorphic modular reduction (sine approximation)
//! - **SlotToCoeff:** Inverse transformation back to coefficients
//! - **SinApprox:** Polynomial approximation of sine function
//! - **Keys:** Rotation key generation for bootstrap

mod bootstrap_context;
mod mod_raise;
mod sin_approx;
pub mod keys;
pub mod rotation;
pub mod coeff_to_slot;
pub mod slot_to_coeff;

// Phase 4 components (COMPLETE)
mod eval_mod;
mod diagonal_mult;

// CUDA GPU Bootstrap
#[cfg(feature = "v2-gpu-cuda")]
pub mod cuda_bootstrap;

#[cfg(feature = "v2-gpu-cuda")]
pub mod cuda_coeff_to_slot;

#[cfg(feature = "v2-gpu-cuda")]
pub mod cuda_slot_to_coeff;

#[cfg(feature = "v2-gpu-cuda")]
pub mod cuda_eval_mod;

pub use bootstrap_context::{BootstrapContext, BootstrapParams};
pub use mod_raise::{mod_raise, mod_down};
pub use sin_approx::{chebyshev_sin_coeffs, eval_polynomial, taylor_sin_coeffs};
pub use keys::{RotationKeys, RotationKey, galois_element_for_rotation, generate_rotation_keys, required_rotations_for_bootstrap};
pub use rotation::rotate;
pub use coeff_to_slot::coeff_to_slot;
pub use slot_to_coeff::slot_to_coeff;
pub use eval_mod::eval_mod;
pub use diagonal_mult::diagonal_mult;
