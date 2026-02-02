//! Core traits for Clifford FHE V2
//!
//! These traits provide a common interface across different backend implementations
//! (CPU optimized, CUDA GPU, Metal GPU, SIMD batched).
//!
//! This follows common FHE library design patterns:
//! - Multiple backends with unified interface
//! - Backend abstraction for hardware-specific optimizations
//! - Trait-based compile-time backend selection

use super::types::*;

/// Main trait for Clifford FHE operations
///
/// All backends (CPU, GPU CUDA, GPU Metal, SIMD) implement this trait
pub trait CliffordFHE {
    /// Ciphertext type (backend-specific)
    type Ciphertext: Clone;

    /// Plaintext type (backend-specific)
    type Plaintext: Clone;

    /// Public key type
    type PublicKey: Clone;

    /// Secret key type
    type SecretKey: Clone;

    /// Evaluation key type (for relinearization)
    type EvaluationKey: Clone;

    /// Parameters type
    type Params: Clone;

    // === Key Generation ===

    /// Generate public key, secret key, and evaluation key
    fn keygen(params: &Self::Params) -> (Self::PublicKey, Self::SecretKey, Self::EvaluationKey);

    // === Encryption / Decryption ===

    /// Encrypt a plaintext multivector component
    fn encrypt(
        pk: &Self::PublicKey,
        pt: &Self::Plaintext,
        params: &Self::Params,
    ) -> Self::Ciphertext;

    /// Decrypt a ciphertext multivector component
    fn decrypt(
        sk: &Self::SecretKey,
        ct: &Self::Ciphertext,
        params: &Self::Params,
    ) -> Self::Plaintext;

    // === Homomorphic Operations ===

    /// Homomorphic geometric product of two Cl(3,0) multivectors
    ///
    /// Input: two encrypted multivectors (8 ciphertexts each)
    /// Output: encrypted geometric product (8 ciphertexts)
    ///
    /// This is the core operation for Clifford FHE!
    fn geometric_product_3d(
        a: &[Self::Ciphertext; 8],
        b: &[Self::Ciphertext; 8],
        evk: &Self::EvaluationKey,
        params: &Self::Params,
    ) -> [Self::Ciphertext; 8];

    /// Homomorphic reverse: ~M (flip bivector signs)
    fn reverse_3d(
        m: &[Self::Ciphertext; 8],
        params: &Self::Params,
    ) -> [Self::Ciphertext; 8];

    /// Homomorphic rotation: v' = R ⊗ v ⊗ ~R
    fn rotate_3d(
        rotor: &[Self::Ciphertext; 8],
        vec: &[Self::Ciphertext; 8],
        evk: &Self::EvaluationKey,
        params: &Self::Params,
    ) -> [Self::Ciphertext; 8];

    /// Homomorphic wedge product: (a⊗b - b⊗a)/2
    fn wedge_product_3d(
        a: &[Self::Ciphertext; 8],
        b: &[Self::Ciphertext; 8],
        evk: &Self::EvaluationKey,
        params: &Self::Params,
    ) -> [Self::Ciphertext; 8];

    /// Homomorphic inner product: (a⊗b + b⊗a)/2
    fn inner_product_3d(
        a: &[Self::Ciphertext; 8],
        b: &[Self::Ciphertext; 8],
        evk: &Self::EvaluationKey,
        params: &Self::Params,
    ) -> [Self::Ciphertext; 8];

    /// Homomorphic projection: proj_a(b)
    fn project_3d(
        a: &[Self::Ciphertext; 8],
        b: &[Self::Ciphertext; 8],
        evk: &Self::EvaluationKey,
        params: &Self::Params,
    ) -> [Self::Ciphertext; 8];

    /// Homomorphic rejection: rej_a(b) = b - proj_a(b)
    fn reject_3d(
        a: &[Self::Ciphertext; 8],
        b: &[Self::Ciphertext; 8],
        evk: &Self::EvaluationKey,
        params: &Self::Params,
    ) -> [Self::Ciphertext; 8];

    // === Performance Metadata ===

    /// Get backend name for benchmarking
    fn backend_name() -> &'static str;

    /// Get expected speedup vs V1 baseline
    fn expected_speedup() -> f64;
}

/// Helper trait for multivector encryption/decryption
pub trait MultivectorOps: CliffordFHE {
    /// Encrypt a complete Cl(3,0) multivector (8 components)
    fn encrypt_multivector_3d(
        mv: &[f64; 8],
        pk: &Self::PublicKey,
        params: &Self::Params,
    ) -> [Self::Ciphertext; 8];

    /// Decrypt a complete Cl(3,0) multivector (8 components)
    fn decrypt_multivector_3d(
        ct: &[Self::Ciphertext; 8],
        sk: &Self::SecretKey,
        params: &Self::Params,
    ) -> [f64; 8];
}

/// Backend capabilities for feature detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub has_ntt_optimization: bool,
    pub has_gpu_acceleration: bool,
    pub has_simd_batching: bool,
    pub has_rotation_keys: bool,
}

/// Trait for querying backend capabilities
pub trait BackendInfo {
    fn capabilities() -> BackendCapabilities;
    fn max_polynomial_degree() -> usize;
    fn recommended_params() -> Vec<String>;
}
