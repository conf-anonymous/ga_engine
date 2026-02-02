///! CUDA-specific packing functions for V4
///!
///! Implements pack/unpack operations using CUDA backend API.

use super::packed_multivector::PackedMultivector;
use crate::clifford_fhe_v2::backends::gpu_cuda::{
    ckks::{CudaCiphertext, CudaCkksContext, CudaPlaintext},
    rotation::CudaRotationContext,
    rotation_keys::CudaRotationKeys,
};
use std::sync::Arc;

/// Pack 8 component ciphertexts into a single PackedMultivector (CUDA version)
///
/// Takes 8 separate ciphertexts (one per Clifford algebra component) and
/// interleaves them into a single packed ciphertext.
///
/// Slot layout: [s₀, e1₀, e2₀, e3₀, e12₀, e23₀, e31₀, I₀, s₁, e1₁, ...]
pub fn pack_multivector_cuda(
    components: &[CudaCiphertext; 8],
    rot_keys: &CudaRotationKeys,
    rot_ctx: &CudaRotationContext,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    let n = components[0].n;
    let batch_size = n / 16; // 8 components × 2 (for packing overhead)

    // Start with zeros
    let zero_values = vec![0.0; n / 2];
    let level = components[0].level;
    let scale = components[0].scale;

    let zero_pt = ckks_ctx.encode(&zero_values, scale, level)?;

    // Create zero ciphertext (will accumulate into this)
    let mut packed_ct = CudaCiphertext {
        c0: zero_pt.poly.clone(),
        c1: vec![0u64; zero_pt.poly.len()],
        n,
        num_primes: components[0].num_primes,
        level,
        scale,
    };

    // Rotate and add each component
    for (i, component) in components.iter().enumerate() {
        // Rotate component i to position i (interleaved with other components)
        let rotated = if i == 0 {
            component.clone()
        } else {
            rotate_cuda(component, i as i32, rot_keys, rot_ctx, ckks_ctx)?
        };

        // Add to accumulator
        packed_ct = add_cuda(&packed_ct, &rotated, ckks_ctx)?;
    }

    Ok(PackedMultivector::new(
        packed_ct,
        batch_size,
        n,
        components[0].num_primes,
        level,
        scale,
    ))
}

/// Unpack a PackedMultivector into 8 component ciphertexts (CUDA version)
pub fn unpack_multivector_cuda(
    packed: &PackedMultivector,
    rot_keys: &CudaRotationKeys,
    rot_ctx: &CudaRotationContext,
    ckks_ctx: &CudaCkksContext,
) -> Result<[CudaCiphertext; 8], String> {
    let mut components = Vec::with_capacity(8);

    for i in 0..8 {
        let component = if i == 0 {
            packed.ct.clone()
        } else {
            rotate_cuda(&packed.ct, -(i as i32), rot_keys, rot_ctx, ckks_ctx)?
        };

        components.push(component);
    }

    Ok(components.try_into().unwrap())
}

/// Extract a single component from packed multivector (CUDA version)
pub fn extract_component_cuda(
    packed: &PackedMultivector,
    component_idx: usize,
    rot_keys: &CudaRotationKeys,
    rot_ctx: &CudaRotationContext,
    ckks_ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String> {
    if component_idx >= 8 {
        return Err(format!("Component index {} out of range [0,8)", component_idx));
    }

    // Rotate to align component to position 0
    let rotated = if component_idx == 0 {
        packed.ct.clone()
    } else {
        rotate_cuda(&packed.ct, -(component_idx as i32), rot_keys, rot_ctx, ckks_ctx)?
    };

    // Apply extraction mask to zero out other components
    let mask = create_extraction_mask_cuda(packed.batch_size, packed.n, packed.scale, packed.level, ckks_ctx)?;
    multiply_plain_cuda(&rotated, &mask, ckks_ctx)
}

/// Helper: Add two CUDA ciphertexts
fn add_cuda(
    ct1: &CudaCiphertext,
    ct2: &CudaCiphertext,
    ckks_ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String> {
    ckks_ctx.add(ct1, ct2)
}

/// Helper: Rotate CUDA ciphertext
fn rotate_cuda(
    ct: &CudaCiphertext,
    steps: i32,
    rot_keys: &CudaRotationKeys,
    _rot_ctx: &CudaRotationContext,
    ckks_ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String> {
    // Delegate to extension method
    ct.rotate_by_steps(steps, rot_keys, ckks_ctx)
}

/// Helper: Multiply ciphertext by plaintext
fn multiply_plain_cuda(
    ct: &CudaCiphertext,
    pt: &CudaPlaintext,
    ckks_ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String> {
    // Delegate to extension method
    ct.multiply_plain(pt, ckks_ctx)
}

// Old implementation (kept for reference, remove later):
fn _multiply_plain_cuda_old(
    ct: &CudaCiphertext,
    pt: &CudaPlaintext,
    ckks_ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String> {
    let c0_mult = ckks_ctx.pointwise_multiply_polynomials_gpu_strided(
        &ct.c0,
        &pt.poly,
        ct.num_primes,
        ct.num_primes,
    )?;

    let c1_mult = ckks_ctx.pointwise_multiply_polynomials_gpu_strided(
        &ct.c1,
        &pt.poly,
        ct.num_primes,
        ct.num_primes,
    )?;

    let c0_rescaled = ckks_ctx.exact_rescale_gpu_strided(&c0_mult, ct.level)?;
    let c1_rescaled = ckks_ctx.exact_rescale_gpu_strided(&c1_mult, ct.level)?;

    Ok(CudaCiphertext {
        c0: c0_rescaled,
        c1: c1_rescaled,
        n: ct.n,
        num_primes: ct.num_primes,
        level: ct.level.saturating_sub(1),
        scale: ct.scale * pt.scale / ckks_ctx.params().scale,
    })
}

/// Create extraction mask (1 at position 0, 8, 16, ... and 0 elsewhere)
fn create_extraction_mask_cuda(
    batch_size: usize,
    n: usize,
    scale: f64,
    level: usize,
    ckks_ctx: &CudaCkksContext,
) -> Result<CudaPlaintext, String> {
    let num_slots = n / 2;
    let mut mask_values = vec![0.0; num_slots];

    // Place 1.0 at positions 0, 8, 16, ... (every 8th slot)
    for i in 0..batch_size {
        let slot_idx = i * 8;
        if slot_idx < num_slots {
            mask_values[slot_idx] = 1.0;
        }
    }

    ckks_ctx.encode(&mask_values, scale, level)
}

/// Compute Galois element for rotation
fn compute_galois_element(rotation_steps: i32, n: usize) -> Result<usize, String> {
    let two_n = 2 * n;
    let k = if rotation_steps >= 0 {
        rotation_steps as usize % (n / 2)
    } else {
        let abs_steps = (-rotation_steps) as usize % (n / 2);
        (n / 2) - abs_steps
    };

    let mut result = 1usize;
    let mut base = 5usize;
    let mut exp = k;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % two_n;
        }
        base = (base * base) % two_n;
        exp >>= 1;
    }

    Ok(result)
}
