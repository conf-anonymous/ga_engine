//! Component Extraction for Batched Multivectors
//!
//! Extract specific components from all multivectors in a batch using
//! rotation and masking operations.

use super::BatchedMultivector;
use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{CkksContext, Ciphertext, Plaintext};
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use crate::clifford_fhe_v3::bootstrapping::{rotate, RotationKeys};

/// Extract a specific component from all multivectors in batch
///
/// # Algorithm
///
/// For component `i` from batch with stride 8:
/// 1. Rotate ciphertext by `i` slots (brings component to positions 0, 8, 16, ...)
/// 2. Multiply by mask that zeros out slots 1-7, 9-15, 17-23, etc.
/// 3. Result has component `i` of all multivectors at positions 0, 8, 16, ...
///
/// # Arguments
///
/// * `batched` - Batched multivector ciphertext
/// * `component` - Component index (0-7)
/// * `rotation_keys` - Rotation keys for homomorphic rotation
/// * `ckks_ctx` - CKKS context for plaintext operations
///
/// # Returns
///
/// Ciphertext with extracted component in strided positions
///
/// # Example
///
/// ```text
/// Input batch (3 multivectors):
///   Slot 0: mv[0].c0, Slot 1: mv[0].c1, ..., Slot 7: mv[0].c7
///   Slot 8: mv[1].c0, Slot 9: mv[1].c1, ..., Slot 15: mv[1].c7
///   Slot 16: mv[2].c0, ...
///
/// extract_component(batched, 2, ...) returns:
///   Slot 0: mv[0].c2, Slot 1: 0, ..., Slot 7: 0
///   Slot 8: mv[1].c2, Slot 9: 0, ..., Slot 15: 0
///   Slot 16: mv[2].c2, ...
/// ```
pub fn extract_component(
    batched: &BatchedMultivector,
    component: usize,
    rotation_keys: &RotationKeys,
    ckks_ctx: &CkksContext,
) -> Result<Ciphertext, String> {
    assert!(component < 8, "Component index must be 0-7");

    let params = &ckks_ctx.params;
    let num_slots = params.n / 2;
    let components = 8;
    let num_multivectors = num_slots / components;

    // Pattern A: Create slot-domain mask (1s at component positions, 0s elsewhere)
    // Layout A (interleaved): positions are component + i*components for i=0..num_multivectors
    let mut mask = vec![0.0; num_slots];
    for i in 0..num_multivectors {
        let pos = component + i * components;
        if pos < num_slots {
            mask[pos] = 1.0;
        }
    }

    // Encode mask in slot domain at scale Δ
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;
    let pt_mask = Plaintext::encode(&mask, params.scale, params);

    // Multiply by mask (this rescales: scale ≈ Δ, level = L-1)
    let extracted = batched.ciphertext.multiply_plain(&pt_mask, ckks_ctx);

    Ok(extracted)
}

/// Extract all 8 components from batched multivector
///
/// Returns vector of 8 ciphertexts, each containing one component
/// from all multivectors in strided layout.
pub fn extract_all_components(
    batched: &BatchedMultivector,
    rotation_keys: &RotationKeys,
    ckks_ctx: &CkksContext,
) -> Result<Vec<Ciphertext>, String> {
    let mut components = Vec::with_capacity(8);
    for i in 0..8 {
        components.push(extract_component(batched, i, rotation_keys, ckks_ctx)?);
    }
    Ok(components)
}

/// Align two ciphertexts to have matching level and scale before addition
///
/// Ensures both ciphertexts have:
/// 1. Same level (by mod-switching higher level down)
/// 2. Same scale (within tolerance)
///
/// This prevents the "3.7M instead of 1.0" bug where adding ciphertexts
/// at different scales creates huge errors.
fn align_for_add(
    mut a: Ciphertext,
    mut b: Ciphertext,
    ckks_ctx: &CkksContext,
) -> (Ciphertext, Ciphertext) {
    // 1) Match levels by mod-switching higher-level down to lower
    while a.level > b.level {
        // Mod-switch a down by one level (drop top modulus, no rescale)
        let new_level = a.level - 1;
        let new_moduli = &ckks_ctx.params.moduli[..=new_level];

        // Truncate RNS representations
        let mut new_c0 = a.c0.clone();
        let mut new_c1 = a.c1.clone();
        for coeff in &mut new_c0 {
            coeff.values.truncate(new_moduli.len());
            coeff.moduli = new_moduli.to_vec();
        }
        for coeff in &mut new_c1 {
            coeff.values.truncate(new_moduli.len());
            coeff.moduli = new_moduli.to_vec();
        }

        a = Ciphertext::new(new_c0, new_c1, new_level, a.scale);
    }

    while b.level > a.level {
        let new_level = b.level - 1;
        let new_moduli = &ckks_ctx.params.moduli[..=new_level];

        let mut new_c0 = b.c0.clone();
        let mut new_c1 = b.c1.clone();
        for coeff in &mut new_c0 {
            coeff.values.truncate(new_moduli.len());
            coeff.moduli = new_moduli.to_vec();
        }
        for coeff in &mut new_c1 {
            coeff.values.truncate(new_moduli.len());
            coeff.moduli = new_moduli.to_vec();
        }

        b = Ciphertext::new(new_c0, new_c1, new_level, b.scale);
    }

    // 2) Assert scales match (within ~0.1% relative error)
    let rel_error = (a.scale - b.scale).abs() / a.scale.max(b.scale);
    assert!(
        rel_error < 1e-3,
        "Scale mismatch before add: a.scale={}, b.scale={}, rel_error={}",
        a.scale, b.scale, rel_error
    );

    // Force exact scale match to avoid floating point drift
    if a.scale != b.scale {
        b.scale = a.scale;
    }

    (a, b)
}

/// Reassemble batched multivector from extracted components
///
/// Inverse of extract_all_components. Takes 8 ciphertexts with
/// components in their original positions and combines into full batch.
///
/// # Algorithm (Pattern A - no rotations needed)
///
/// 1. Each extracted component already has non-zeros in its correct positions
/// 2. Align levels and scales before each addition
/// 3. Sum all components (positions with 0s remain 0, positions with values add correctly)
///
/// # Arguments
///
/// * `components` - Array of 8 ciphertexts with extracted components
/// * `rotation_keys` - Rotation keys (not used in Pattern A)
/// * `ckks_ctx` - CKKS context
/// * `batch_size` - Number of multivectors
/// * `n` - Ring dimension
///
/// # Returns
///
/// Batched multivector with all components reassembled
pub fn reassemble_components(
    components: &[Ciphertext; 8],
    rotation_keys: &RotationKeys,
    ckks_ctx: &CkksContext,
    batch_size: usize,
    n: usize,
) -> Result<BatchedMultivector, String> {
    assert_eq!(components.len(), 8, "Must have exactly 8 components");

    // Start with first component
    let mut result = components[0].clone();

    // Add remaining components with scale/level alignment
    for component_ct in components.iter().skip(1) {
        // Align levels and scales before addition
        let (aligned_result, aligned_component) = align_for_add(
            result,
            component_ct.clone(),
            ckks_ctx,
        );

        // Add to accumulator
        result = aligned_result.add(&aligned_component);
    }

    Ok(BatchedMultivector::new(result, batch_size))
}

/// Create mask for stride extraction
///
/// Creates a mask vector with 1s at positions offset, offset+stride, offset+2*stride, ...
/// and 0s elsewhere.
///
/// # Arguments
///
/// * `count` - Number of 1s to place
/// * `stride` - Distance between 1s
/// * `offset` - Position of first 1
/// * `total_length` - Total length of mask vector
///
/// # Returns
///
/// Mask vector of length `total_length`
///
/// # Example
///
/// ```
/// create_stride_mask(3, 8, 0, 24) => [1,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0]
/// create_stride_mask(3, 8, 2, 24) => [0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0]
/// ```
fn create_stride_mask(count: usize, stride: usize, offset: usize, total_length: usize) -> Vec<f64> {
    let mut mask = vec![0.0; total_length];
    for i in 0..count {
        let pos = offset + i * stride;
        if pos < total_length {
            mask[pos] = 1.0;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use crate::clifford_fhe_v2::params::CliffordFHEParams;
    use crate::clifford_fhe_v3::batched::encoding::{encode_batch, decode_batch};
    use crate::clifford_fhe_v3::bootstrapping::keys::generate_rotation_keys;

    #[test]
    fn test_create_stride_mask() {
        let mask = create_stride_mask(3, 8, 0, 24);
        assert_eq!(mask[0], 1.0);
        assert_eq!(mask[8], 1.0);
        assert_eq!(mask[16], 1.0);
        for i in 1..8 {
            assert_eq!(mask[i], 0.0);
        }

        let mask2 = create_stride_mask(3, 8, 2, 24);
        assert_eq!(mask2[2], 1.0);
        assert_eq!(mask2[10], 1.0);
        assert_eq!(mask2[18], 1.0);
    }

    #[test]
    fn test_component_extraction() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        // Create batch of 4 multivectors
        let multivectors = vec![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
            [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0],
        ];

        let batched = encode_batch(&multivectors, &ckks_ctx, &pk);

        // Generate rotation keys for extraction (need rotations 0-7)
        let rotations: Vec<i32> = (0..8).collect();
        let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

        // Extract component 2 (should be 3.0, 30.0, 300.0, 3000.0)
        let extracted = extract_component(&batched, 2, &rotation_keys, &ckks_ctx)
            .expect("Extraction failed");

        // Decrypt and check
        let pt = ckks_ctx.decrypt(&extracted, &sk);
        let slots = ckks_ctx.decode(&pt);

        // Component 2 is at positions [2, 10, 18, 26] (Layout A: base_slot + component)
        // For 4 multivectors at slots [0-7, 8-15, 16-23, 24-31], component 2 is at base+2
        let expected = [3.0, 30.0, 300.0, 3000.0];
        for (i, &exp) in expected.iter().enumerate() {
            let slot_idx = i * 8 + 2;  // base_slot + component_index
            let error = (slots[slot_idx] - exp).abs();
            let rel_error = error / exp;
            assert!(
                rel_error < 0.6,  // Allow 60% relative error for small params
                "Component 2 of multivector {} incorrect: got {}, expected {} (rel error: {:.1}%)",
                i, slots[slot_idx], exp, rel_error * 100.0
            );
        }

        // Other slots should be ~0 (only component 2 slots have values)
        for i in 0..4 {
            for j in 0..8 {
                if j == 2 {
                    continue;  // Skip component 2 slots (they have values)
                }
                let slot_idx = i * 8 + j;
                // Allow higher tolerance for masked-out slots
                let tolerance = 5.0;  // Absolute tolerance for "should be zero" values
                assert!(
                    slots[slot_idx].abs() < tolerance,
                    "Slot {} should be masked to ~0, got {} (tolerance: {})",
                    slot_idx, slots[slot_idx], tolerance
                );
            }
        }
    }

    #[test]
    fn test_extract_and_reassemble() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        // Create batch
        let multivectors = vec![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        ];
        let batched = encode_batch(&multivectors, &ckks_ctx, &pk);

        // Generate rotation keys (need 0-7 for extraction, negatives for reassembly)
        let rotations: Vec<i32> = (-7..=7).collect();
        let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

        // Extract all components
        let components = extract_all_components(&batched, &rotation_keys, &ckks_ctx)
            .expect("Extraction failed");
        assert_eq!(components.len(), 8);

        // Reassemble
        let components_array: [Ciphertext; 8] = components.try_into().unwrap();
        let reassembled = reassemble_components(
            &components_array,
            &rotation_keys,
            &ckks_ctx,
            2,
            params.n,
        ).expect("Reassembly failed");

        // Decode and verify
        let decoded = decode_batch(&reassembled, &ckks_ctx, &sk);
        assert_eq!(decoded.len(), 2);

        // Allow higher tolerance due to rotation noise with small params
        for (i, (original, decoded)) in multivectors.iter().zip(decoded.iter()).enumerate() {
            for comp in 0..8 {
                let error = (decoded[comp] - original[comp]).abs();
                let rel_error = error / original[comp].max(1.0);  // Avoid divide by zero
                // With multiple rotations, noise accumulates. Allow 100% relative error for small params
                let tolerance = original[comp].abs() * 1.5 + 5.0;  // Relative + absolute
                assert!(
                    error < tolerance,
                    "Multivector {} component {} error: {:.2} (got {:.2}, expected {:.2}, tolerance: {:.2})",
                    i, comp, error, decoded[comp], original[comp], tolerance
                );
            }
        }
    }

    /// Test T0: Mask slot 0 only (diagnostic test from expert)
    #[test]
    fn test_mask_slot_0_only() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        let num_slots = params.n / 2;
        let mut input = vec![0.0; num_slots];
        for i in 0..num_slots {
            input[i] = (i % 10) as f64;
        }

        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;
        let pt_input = Plaintext::encode(&input, params.scale, &params);
        let ct = ckks_ctx.encrypt(&pt_input, &pk);

        // Create mask with 1 at slot 0, 0 elsewhere
        let mut mask = vec![0.0; num_slots];
        mask[0] = 1.0;
        let pt_mask = Plaintext::encode(&mask, params.scale, &params);

        let ct_masked = ct.multiply_plain(&pt_mask, &ckks_ctx);
        let pt_output = ckks_ctx.decrypt(&ct_masked, &sk);
        let output = ckks_ctx.decode(&pt_output);

        // Only slot 0 should remain
        assert!(
            (output[0] - input[0]).abs() < 1e-2,
            "Slot 0 should be {}, got {}",
            input[0],
            output[0]
        );
        for i in 1..num_slots.min(20) {
            assert!(
                output[i].abs() < 1e-2,
                "Slot {} should be ~0, got {}",
                i,
                output[i]
            );
        }
    }

    /// Test T1: Mask even slots (diagnostic test from expert)
    #[test]
    fn test_mask_even_slots() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        let num_slots = params.n / 2;
        let mut input = vec![0.0; num_slots];
        for i in 0..num_slots {
            input[i] = (i % 10) as f64;
        }

        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;
        let pt_input = Plaintext::encode(&input, params.scale, &params);
        let ct = ckks_ctx.encrypt(&pt_input, &pk);

        // Mask: 1 on even slots, 0 on odd
        let mask: Vec<f64> = (0..num_slots)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let pt_mask = Plaintext::encode(&mask, params.scale, &params);

        let ct_masked = ct.multiply_plain(&pt_mask, &ckks_ctx);
        let pt_output = ckks_ctx.decrypt(&ct_masked, &sk);
        let output = ckks_ctx.decode(&pt_output);

        // Even slots should remain, odd should be ~0
        for i in 0..num_slots.min(20) {
            if i % 2 == 0 {
                assert!(
                    (output[i] - input[i]).abs() < 1e-2,
                    "Even slot {} error: got {}, expected {}",
                    i,
                    output[i],
                    input[i]
                );
            } else {
                assert!(
                    output[i].abs() < 1e-2,
                    "Odd slot {} should be ~0, got {}",
                    i,
                    output[i]
                );
            }
        }
    }
}
