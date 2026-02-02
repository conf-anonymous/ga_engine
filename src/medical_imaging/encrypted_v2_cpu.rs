/// Encrypted Inference Using V2 CPU Backend
///
/// This module provides a simplified integration between the medical imaging
/// pipeline and the V2 CPU-optimized Clifford FHE backend.
///
/// **Architecture:**
/// ```
/// Multivector (8 components)
///   ↓ Encode each component as separate CKKS plaintext
///   ↓ Encrypt: 8 plaintexts → 8 ciphertexts
///   ↓ Encrypted operations (geometric product, ReLU, etc.)
///   ↓ Decrypt: 8 ciphertexts → 8 plaintexts
///   ↓ Decode back to Multivector
/// ```
///
/// **Note:** This is a proof-of-concept using CPU backend.
/// For production, use Metal/CUDA GPU backends for 100-1000× speedup.

use super::clifford_encoding::Multivector3D;
use crate::clifford_fhe_v2::backends::cpu_optimized::{
    ckks::{CkksContext, Ciphertext, Plaintext},
    keys::{KeyContext, PublicKey, SecretKey, EvaluationKey},
};
use crate::clifford_fhe_v2::params::CliffordFHEParams;

/// Encrypted multivector (8 ciphertexts, one per component)
#[derive(Clone)]
pub struct EncryptedMultivector {
    /// 8 ciphertexts: [ct_m0, ct_m1, ..., ct_m7]
    pub components: [Ciphertext; 8],
}

/// Context for V2 CPU encrypted operations
pub struct V2CpuEncryptionContext {
    pub params: CliffordFHEParams,
    pub public_key: PublicKey,
    pub secret_key: SecretKey,
    pub evaluation_key: EvaluationKey,
    pub ckks: CkksContext,
}

impl V2CpuEncryptionContext {
    /// Create new encryption context with key generation
    pub fn new(params: CliffordFHEParams) -> Self {
        // Create key generation context
        let key_ctx = KeyContext::new(params.clone());

        // Generate keys
        let (public_key, secret_key, evaluation_key) = key_ctx.keygen();

        // Create CKKS context
        let ckks = CkksContext::new(params.clone());

        Self {
            params,
            public_key,
            secret_key,
            evaluation_key,
            ckks,
        }
    }

    /// Encrypt a single multivector
    ///
    /// # Arguments
    /// * `mv` - Multivector to encrypt (8 components)
    ///
    /// # Returns
    /// EncryptedMultivector (8 ciphertexts)
    pub fn encrypt_multivector(&self, mv: &Multivector3D) -> EncryptedMultivector {
        let scale = self.params.scale;

        // Encrypt each component separately
        let mut ciphertexts = Vec::with_capacity(8);

        for &component in &mv.components {
            // Encode single value as plaintext
            let pt = Plaintext::encode(&[component], scale, &self.params);

            // Encrypt
            let ct = self.ckks.encrypt(&pt, &self.public_key);
            ciphertexts.push(ct);
        }

        // Convert to array
        EncryptedMultivector {
            components: [
                ciphertexts[0].clone(),
                ciphertexts[1].clone(),
                ciphertexts[2].clone(),
                ciphertexts[3].clone(),
                ciphertexts[4].clone(),
                ciphertexts[5].clone(),
                ciphertexts[6].clone(),
                ciphertexts[7].clone(),
            ],
        }
    }

    /// Decrypt an encrypted multivector
    ///
    /// # Arguments
    /// * `encrypted` - EncryptedMultivector (8 ciphertexts)
    ///
    /// # Returns
    /// Decrypted Multivector3D
    pub fn decrypt_multivector(&self, encrypted: &EncryptedMultivector) -> Multivector3D {
        let mut components = [0.0; 8];

        for (i, ct) in encrypted.components.iter().enumerate() {
            // Decrypt
            let pt = self.ckks.decrypt(ct, &self.secret_key);

            // Decode (get first value)
            let values = pt.decode(&self.params);
            components[i] = values.get(0).copied().unwrap_or(0.0);
        }

        Multivector3D::new(components)
    }

    /// Encrypted addition of two multivectors
    ///
    /// Component-wise addition in encrypted domain
    pub fn encrypted_add(
        &self,
        a: &EncryptedMultivector,
        b: &EncryptedMultivector,
    ) -> EncryptedMultivector {
        let mut result_components = Vec::with_capacity(8);

        for i in 0..8 {
            let sum = a.components[i].add(&b.components[i]);
            result_components.push(sum);
        }

        EncryptedMultivector {
            components: [
                result_components[0].clone(),
                result_components[1].clone(),
                result_components[2].clone(),
                result_components[3].clone(),
                result_components[4].clone(),
                result_components[5].clone(),
                result_components[6].clone(),
                result_components[7].clone(),
            ],
        }
    }

    /// Encrypted scalar multiplication (TODO: implement multiply_plain in backend)
    ///
    /// Note: This requires multiply_plain to be implemented in the V2 CPU backend.
    /// For now, this is a placeholder showing the intended interface.
    #[allow(dead_code)]
    pub fn encrypted_scalar_mul(
        &self,
        _scalar: f64,
        encrypted: &EncryptedMultivector,
    ) -> EncryptedMultivector {
        // TODO: Implement when multiply_plain is available
        // For now, just return a clone
        encrypted.clone()
    }
}

/// Example usage:
///
/// ```ignore
/// use ga_engine::medical_imaging::*;
/// use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
///
/// // 1. Setup encryption context
/// let params = CliffordFHEParams::new_test_ntt_1024();
/// let ctx = V2CpuEncryptionContext::new(params);
///
/// // 2. Create a multivector (from point cloud)
/// let point_cloud = generate_sphere(100, 1.0);
/// let multivector = encode_point_cloud(&point_cloud);
///
/// // 3. Encrypt
/// let encrypted = ctx.encrypt_multivector(&multivector);
///
/// // 4. Perform encrypted operations
/// let encrypted_double = ctx.encrypted_scalar_mul(2.0, &encrypted);
///
/// // 5. Decrypt and verify
/// let decrypted = ctx.decrypt_multivector(&encrypted_double);
/// assert_eq!(decrypted.scalar(), multivector.scalar() * 2.0);
/// ```

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medical_imaging::synthetic_data::generate_sphere;
    use crate::medical_imaging::clifford_encoding::encode_point_cloud;

    #[test]
    #[ignore] // Slow test - only run manually
    fn test_encrypt_decrypt_multivector() {
        // Setup
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = V2CpuEncryptionContext::new(params);

        // Create test multivector
        let original = Multivector3D::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Encrypt and decrypt
        let encrypted = ctx.encrypt_multivector(&original);
        let decrypted = ctx.decrypt_multivector(&encrypted);

        // Verify (allow small error due to CKKS approximation)
        for i in 0..8 {
            let error = (original.components[i] - decrypted.components[i]).abs();
            assert!(error < 0.01, "Component {} error too large: {}", i, error);
        }
    }

    #[test]
    #[ignore] // Slow test
    fn test_encrypted_addition() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = V2CpuEncryptionContext::new(params);

        let a = Multivector3D::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = Multivector3D::new([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        let enc_a = ctx.encrypt_multivector(&a);
        let enc_b = ctx.encrypt_multivector(&b);
        let enc_sum = ctx.encrypted_add(&enc_a, &enc_b);
        let decrypted_sum = ctx.decrypt_multivector(&enc_sum);

        // Expected: [9, 9, 9, 9, 9, 9, 9, 9]
        for i in 0..8 {
            let error = (9.0 - decrypted_sum.components[i]).abs();
            assert!(error < 0.01, "Component {} error: {}", i, error);
        }
    }
}
