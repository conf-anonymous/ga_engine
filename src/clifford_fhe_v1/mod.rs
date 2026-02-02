//! Clifford-FHE: Fully Homomorphic Encryption for Geometric Algebra
//!
//! This module implements a CKKS-based FHE scheme optimized for Clifford algebra
//! operations. Unlike the LWE-based scheme (clifford_lwe), this supports:
//!
//! - Homomorphic geometric product: Enc(a) ⊗ Enc(b) = Enc(a ⊗ b)
//! - Homomorphic rotations via rotors
//! - Arbitrary depth computations (with bootstrapping)
//!
//! # Architecture
//!
//! ```text
//! Multivector (Cl(3,0))
//!     ↓ encoding
//! Polynomial in R = Z[x]/(x^N + 1)
//!     ↓ CKKS encryption
//! Ciphertext (c0, c1)
//!     ↓ homomorphic ops
//! Result ciphertext
//!     ↓ CKKS decryption
//! Polynomial
//!     ↓ decoding
//! Multivector result
//! ```
//!
//! # Example (Future)
//!
//! ```rust,ignore
//! use clifford_fhe::*;
//!
//! // Generate keys
//! let (pk, sk, evk) = keygen(SecurityLevel::Bit128);
//!
//! // Encrypt multivectors
//! let mv_a = Multivector::from([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
//! let ct_a = encrypt(&pk, &mv_a);
//!
//! let mv_b = Multivector::from([3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
//! let ct_b = encrypt(&pk, &mv_b);
//!
//! // Homomorphic geometric product!
//! let ct_c = geometric_product(&evk, &ct_a, &ct_b);
//!
//! // Decrypt
//! let mv_c = decrypt(&sk, &ct_c);
//! // mv_c should equal mv_a ⊗ mv_b
//! ```

pub mod automorphisms;
pub mod canonical_embedding;
pub mod ckks_rns; // RNS-CKKS implementation
pub mod geometric_nn; // Geometric Deep Learning with FHE
pub mod geometric_product_rns; // RNS geometric product
pub mod keys_rns; // RNS-aware key generation
pub mod params;
pub mod rns; // RNS (Residue Number System) core
pub mod rotation_keys; // Specialized rotation keys for fast R·x·R̃
pub mod slot_encoding;

// Re-exports
pub use ckks_rns::{RnsCiphertext, RnsPlaintext, rns_decrypt, rns_encrypt};
pub use geometric_product_rns::{
    Cl2StructureConstants, Cl3StructureConstants,
    geometric_product_2d_componentwise, geometric_product_3d_componentwise,
    reverse_2d, reverse_3d,
    rotate_2d, rotate_3d,
    wedge_product_2d, wedge_product_3d,
    inner_product_2d, inner_product_3d,
    project_3d, reject_3d
};
pub use keys_rns::{RnsEvaluationKey, RnsPublicKey, RnsSecretKey, rns_keygen};
pub use params::CliffordFHEParams;
pub use slot_encoding::{encode_multivector_slots, decode_multivector_slots};
