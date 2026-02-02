//! Batch Encoding and Decoding
//!
//! Packs multiple multivectors into CKKS slots for parallel processing.

use super::BatchedMultivector;
use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{CkksContext, Plaintext};
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey;
use crate::clifford_fhe_v2::params::CliffordFHEParams;

/// Encode multiple multivectors into a batched ciphertext
///
/// # Layout
///
/// For batch of size B with 8 components each:
/// - Slot 0: mv[0].c0, Slot 1: mv[0].c1, ..., Slot 7: mv[0].c7
/// - Slot 8: mv[1].c0, Slot 9: mv[1].c1, ..., Slot 15: mv[1].c7
/// - ...
/// - Slot 8B: mv[B-1].c0, ..., Slot 8B+7: mv[B-1].c7
///
/// Remaining slots (8B to N/2-1) are filled with zeros.
///
/// # Arguments
///
/// * `multivectors` - Array of multivectors, each with 8 components
/// * `ckks_ctx` - CKKS context for encoding
/// * `pk` - Public key for encryption
///
/// # Returns
///
/// Batched ciphertext encrypting all multivectors
///
/// # Panics
///
/// Panics if batch size exceeds maximum for ring dimension
pub fn encode_batch(
    multivectors: &[[f64; 8]],
    ckks_ctx: &CkksContext,
    pk: &PublicKey,
) -> BatchedMultivector {
    let batch_size = multivectors.len();
    let n = ckks_ctx.params.n;
    let max_batch = n / 2 / 8;

    assert!(
        batch_size <= max_batch,
        "Batch size {} exceeds maximum {} for N={}",
        batch_size, max_batch, n
    );

    // Pack multivectors into slots
    let num_slots = n / 2;
    let mut slots = vec![0.0; num_slots];

    for (i, mv) in multivectors.iter().enumerate() {
        let base_slot = i * 8;
        for (comp_idx, &component) in mv.iter().enumerate() {
            slots[base_slot + comp_idx] = component;
        }
    }

    // Encode and encrypt
    let pt = ckks_ctx.encode(&slots);
    let ct = ckks_ctx.encrypt(&pt, pk);

    BatchedMultivector::new(ct, batch_size)
}

/// Decode batched ciphertext to multiple multivectors
///
/// # Arguments
///
/// * `batched` - Batched ciphertext
/// * `ckks_ctx` - CKKS context for decoding
/// * `sk` - Secret key for decryption
///
/// # Returns
///
/// Vector of multivectors (length = batch_size)
pub fn decode_batch(
    batched: &BatchedMultivector,
    ckks_ctx: &CkksContext,
    sk: &crate::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
) -> Vec<[f64; 8]> {
    // Decrypt and decode
    let pt = ckks_ctx.decrypt(&batched.ciphertext, sk);
    let slots = ckks_ctx.decode(&pt);

    // Extract multivectors
    let mut multivectors = Vec::with_capacity(batched.batch_size);
    for i in 0..batched.batch_size {
        let base_slot = i * 8;
        let mut mv = [0.0; 8];
        for comp_idx in 0..8 {
            mv[comp_idx] = slots[base_slot + comp_idx];
        }
        multivectors.push(mv);
    }

    multivectors
}

/// Encode single multivector (convenience wrapper)
///
/// Packs one multivector into first 8 slots, rest are zero.
pub fn encode_single(
    multivector: &[f64; 8],
    ckks_ctx: &CkksContext,
    pk: &PublicKey,
) -> BatchedMultivector {
    encode_batch(&[*multivector], ckks_ctx, pk)
}

/// Decode single multivector (convenience wrapper)
pub fn decode_single(
    batched: &BatchedMultivector,
    ckks_ctx: &CkksContext,
    sk: &crate::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
) -> [f64; 8] {
    let multivectors = decode_batch(batched, ckks_ctx, sk);
    assert!(!multivectors.is_empty(), "Batch is empty");
    multivectors[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use crate::clifford_fhe_v2::params::CliffordFHEParams;

    #[test]
    fn test_single_multivector_roundtrip() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params);

        let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let batched = encode_single(&mv, &ckks_ctx, &pk);
        let decoded = decode_single(&batched, &ckks_ctx, &sk);

        for i in 0..8 {
            let error = (decoded[i] - mv[i]).abs();
            assert!(error < 0.1, "Component {} error too large: {}", i, error);
        }
    }

    #[test]
    fn test_batch_multivector_roundtrip() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        // Create batch of 10 multivectors
        let mut multivectors = Vec::new();
        for i in 0..10 {
            let base = (i as f64) * 10.0;
            multivectors.push([
                base + 1.0, base + 2.0, base + 3.0, base + 4.0,
                base + 5.0, base + 6.0, base + 7.0, base + 8.0,
            ]);
        }

        let batched = encode_batch(&multivectors, &ckks_ctx, &pk);
        assert_eq!(batched.batch_size, 10);
        assert_eq!(batched.slots_used(), 80);

        let decoded = decode_batch(&batched, &ckks_ctx, &sk);
        assert_eq!(decoded.len(), 10);

        for (i, (original, decoded)) in multivectors.iter().zip(decoded.iter()).enumerate() {
            for comp in 0..8 {
                let error = (decoded[comp] - original[comp]).abs();
                assert!(
                    error < 0.1,
                    "Multivector {} component {} error: {}",
                    i, comp, error
                );
            }
        }
    }

    #[test]
    fn test_max_batch_size() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        let max_batch = BatchedMultivector::max_batch_size(params.n);
        assert_eq!(max_batch, 64); // 1024/2/8 = 64

        // Create max batch
        let multivectors: Vec<[f64; 8]> = (0..max_batch)
            .map(|i| {
                let base = i as f64;
                [base, base + 1.0, base + 2.0, base + 3.0,
                 base + 4.0, base + 5.0, base + 6.0, base + 7.0]
            })
            .collect();

        let batched = encode_batch(&multivectors, &ckks_ctx, &pk);
        assert_eq!(batched.batch_size, max_batch);
        assert_eq!(batched.slot_utilization(), 100.0);

        let decoded = decode_batch(&batched, &ckks_ctx, &sk);
        assert_eq!(decoded.len(), max_batch);
    }

    #[test]
    #[should_panic(expected = "exceeds maximum")]
    fn test_batch_size_overflow() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, _, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        let max_batch = BatchedMultivector::max_batch_size(params.n);
        let too_many: Vec<[f64; 8]> = vec![[0.0; 8]; max_batch + 1];

        encode_batch(&too_many, &ckks_ctx, &pk); // Should panic
    }
}
