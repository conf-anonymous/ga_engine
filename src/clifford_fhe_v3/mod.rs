//! Clifford FHE V3 - Bootstrapping and Deep Computation
//!
//! V3 adds CKKS bootstrapping to enable unlimited multiplication depth,
//! unlocking encrypted deep neural networks for privacy-preserving ML.
//!
//! ## Key Features
//!
//! - **CKKS Bootstrapping:** Homomorphic noise refresh for unlimited depth
//! - **Deep Encrypted GNN:** 168+ multiplications for medical imaging
//! - **GPU Acceleration:** Metal and CUDA support (future)
//! - **SIMD Batching:** 512× throughput multiplier via slot packing
//!
//! ## Use Case
//!
//! Encrypted 3D Medical Imaging Classification with proprietary model protection:
//! - Both data AND model privacy (encrypted weights + encrypted data)
//! - Deep GNN (1→16→8→3) requiring 168 multiplications
//! - Bootstrapping enables arbitrary-depth encrypted computation
//!
//! ## Example
//!
//! ```rust,ignore
//! use ga_engine::clifford_fhe_v3::bootstrapping::{BootstrapContext, BootstrapParams};
//! use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
//!
//! // Setup parameters
//! let params = CliffordFHEParams::new_128bit();
//! let bootstrap_params = BootstrapParams::balanced();
//!
//! // Create bootstrap context (generates rotation keys)
//! let bootstrap_ctx = BootstrapContext::new(params, bootstrap_params, &secret_key)?;
//!
//! // Bootstrap a noisy ciphertext (refresh noise)
//! let ct_fresh = bootstrap_ctx.bootstrap(&ct_noisy)?;
//! ```
//!
//! ## Performance Targets
//!
//! - **CPU:** ~2 seconds per multivector refresh
//! - **GPU:** ~500ms per multivector refresh (target)
//! - **SIMD Batched:** ~5ms per sample (512× batch)
//!
//! ## Timeline
//!
//! V3 development: 2-4 weeks (6 phases)
//!
//! See [V3_BOOTSTRAPPING_DESIGN.md](../../V3_BOOTSTRAPPING_DESIGN.md) for complete architecture.

pub mod bootstrapping;
pub mod params;
pub mod batched;
pub mod prime_gen;
