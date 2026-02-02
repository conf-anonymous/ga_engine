/// Packing and Unpacking Operations
///
/// Convert between V2/V3 naive component-separate layout and V4 packed slot-interleaved layout.

use super::packed_multivector::PackedMultivector;

#[cfg(feature = "v2-gpu-cuda")]
use crate::clifford_fhe_v2::backends::gpu_cuda::{
    ckks::{CudaCiphertext as Ciphertext, CudaCkksContext, CudaPlaintext as Plaintext},
    rotation_keys::CudaRotationKeys as RotationKeys,
};

#[cfg(all(feature = "v2-gpu-metal", not(feature = "v2-gpu-cuda")))]
use crate::clifford_fhe_v2::backends::gpu_metal::{
    ckks::{MetalCiphertext as Ciphertext, MetalCkksContext as CudaCkksContext, MetalPlaintext as Plaintext},
    rotation_keys::MetalRotationKeys as RotationKeys,
};

#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
use crate::clifford_fhe_v2::backends::cpu_optimized::{
    ckks::{Ciphertext, CpuCkksContext as CudaCkksContext, Plaintext},
    // Note: CPU backend may not have rotation keys implemented yet
};

/// Create a mask plaintext for extracting a single component from packed layout
///
/// The mask has 1.0 at positions 0, 8, 16, 24, ... (every 8th slot)
/// and 0.0 everywhere else. This extracts one component from the interleaved layout.
///
/// After rotating the packed ciphertext to align the desired component to position 0,
/// multiplying by this mask zeros out all other components.
///
/// NOTE: This is Metal/CPU only. CUDA uses packing_cuda.rs instead.
#[cfg(all(feature = "v2-gpu-metal", not(feature = "v2-gpu-cuda")))]
fn create_extraction_mask(
    batch_size: usize,
    n: usize,
    ckks_ctx: &CudaCkksContext,
) -> Result<Plaintext, String> {
    let num_slots = n / 2;

    // Create mask: 1.0 at positions 0, 8, 16, ..., 0.0 elsewhere
    let mut mask_values = vec![0.0; num_slots];
    for i in 0..batch_size {
        let slot_idx = i * 8; // Every 8th position
        if slot_idx < num_slots {
            mask_values[slot_idx] = 1.0;
        }
    }

    // Encode the mask into a plaintext (Metal 1-param signature)
    ckks_ctx.encode(&mask_values)
}

/// Pack multiple component-separate ciphertexts into a single packed ciphertext
///
/// Input: 8 ciphertexts [ct_s, ct_e1, ct_e2, ct_e3, ct_e12, ct_e23, ct_e31, ct_I]
///        Each encrypts batch_size scalar values
///
/// Output: Single ciphertext with interleaved slots
///         [s₀, e1₀, e2₀, e3₀, e12₀, e23₀, e31₀, I₀, s₁, e1₁, ...]
///
/// Algorithm:
/// 1. For each component i:
///    - Rotate ct[i] left by i positions
/// 2. Sum all rotated ciphertexts
///
/// This interleaves the components by rotating each one into its designated position
/// within each 8-slot group.
///
/// NOTE: Metal/CPU only. CUDA uses pack_multivector_cuda from packing_cuda.rs.
#[cfg(all(feature = "v2-gpu-metal", not(feature = "v2-gpu-cuda")))]
pub fn pack_multivector(
    components: &[Ciphertext; 8],
    batch_size: usize,
    rot_keys: &RotationKeys,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    // Get parameters from the first ciphertext (they should all match)
    let n = components[0].n;
    let num_primes = components[0].num_primes;
    let level = components[0].level;
    let scale = components[0].scale;

    // Verify all components are compatible
    for (i, ct) in components.iter().enumerate() {
        if ct.n != n {
            return Err(format!("Component {} has mismatched n", i));
        }
        if ct.num_primes != num_primes {
            return Err(format!("Component {} has mismatched num_primes", i));
        }
        if ct.level != level {
            return Err(format!("Component {} has mismatched level", i));
        }
        if (ct.scale - scale).abs() > 1e-6 {
            return Err(format!("Component {} has mismatched scale", i));
        }
    }

    // Verify batch size is valid
    if batch_size * 8 > n / 2 {
        return Err(format!(
            "Batch size {} × 8 components = {} exceeds n/2 = {}",
            batch_size, batch_size * 8, n / 2
        ));
    }

    // Start with component 0 (scalar) - no rotation needed
    let mut packed_ct = components[0].clone();

    // Rotate and accumulate remaining components
    for i in 1..8 {
        // Rotate component i left by i positions
        let rotated = components[i].rotate_by_steps(i as i32, rot_keys, ckks_ctx)?;

        // Add to accumulator
        packed_ct = packed_ct.add(&rotated, ckks_ctx)?;
    }

    Ok(PackedMultivector::new(
        packed_ct,
        batch_size,
        n,
        num_primes,
        level,
        scale,
    ))
}

/// CPU-optimized version (placeholder - rotation may not be available)
#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
pub fn pack_multivector(
    components: &[Ciphertext; 8],
    batch_size: usize,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    Err("Packing not yet implemented for CPU backend - requires rotation keys".to_string())
}

/// Unpack a packed ciphertext into 8 component-separate ciphertexts
///
/// Input: Single packed ciphertext with interleaved slots
///        [s₀, e1₀, e2₀, e3₀, e12₀, e23₀, e31₀, I₀, s₁, e1₁, ...]
///
/// Output: 8 ciphertexts [ct_s, ct_e1, ct_e2, ct_e3, ct_e12, ct_e23, ct_e31, ct_I]
///
/// Algorithm:
/// 1. Create extraction mask: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, ...] (1 at every 8th position)
/// 2. For each component i:
///    - Rotate packed_ct right by i positions (to align component i to position 0)
///    - Multiply by mask to extract only that component
///
/// NOTE: Metal/CPU only. CUDA uses unpack_multivector_cuda from packing_cuda.rs.
#[cfg(all(feature = "v2-gpu-metal", not(feature = "v2-gpu-cuda")))]
pub fn unpack_multivector(
    packed: &PackedMultivector,
    rot_keys: &RotationKeys,
    ckks_ctx: &CudaCkksContext,
) -> Result<[Ciphertext; 8], String> {
    // Create extraction mask (reuse for all components)
    let mask = create_extraction_mask(packed.batch_size, packed.n, ckks_ctx)?;

    let mut components = Vec::with_capacity(8);

    for i in 0..8 {
        // Step 1: Rotate right by i positions to align component i to position 0
        let rotated = if i == 0 {
            // No rotation needed for component 0
            packed.ct.clone()
        } else {
            packed.ct.rotate_by_steps(-(i as i32), rot_keys, ckks_ctx)?
        };

        // Step 2: Apply mask to extract only positions 0, 8, 16, 24, ...
        // This zeros out all other components
        let masked = rotated.multiply_plain(&mask, ckks_ctx)?;

        components.push(masked);
    }

    Ok([
        components[0].clone(),
        components[1].clone(),
        components[2].clone(),
        components[3].clone(),
        components[4].clone(),
        components[5].clone(),
        components[6].clone(),
        components[7].clone(),
    ])
}

/// CPU version (placeholder)
#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
pub fn unpack_multivector(
    packed: &PackedMultivector,
    ckks_ctx: &CudaCkksContext,
) -> Result<[Ciphertext; 8], String> {
    Err("Unpacking not yet implemented for CPU backend - requires rotation keys".to_string())
}

/// Extract a single component from a packed ciphertext
///
/// This is more efficient than unpacking all 8 components when you only need one.
///
/// Algorithm:
/// 1. Rotate packed_ct right by component_idx positions to align to position 0
/// 2. Multiply by extraction mask to zero out all other components
///
/// NOTE: Metal/CPU only. CUDA uses extract_component_cuda from packing_cuda.rs.
#[cfg(all(feature = "v2-gpu-metal", not(feature = "v2-gpu-cuda")))]
pub fn extract_component(
    packed: &PackedMultivector,
    component_idx: usize,
    rot_keys: &RotationKeys,
    ckks_ctx: &CudaCkksContext,
) -> Result<Ciphertext, String> {
    if component_idx >= 8 {
        return Err(format!("Component index {} out of range [0,8)", component_idx));
    }

    // Step 1: Rotate right by component_idx positions
    let rotated = if component_idx == 0 {
        packed.ct.clone()
    } else {
        packed.ct.rotate_by_steps(-(component_idx as i32), rot_keys, ckks_ctx)?
    };

    // Step 2: Apply extraction mask
    let mask = create_extraction_mask(packed.batch_size, packed.n, ckks_ctx)?;
    let masked = rotated.multiply_plain(&mask, ckks_ctx)?;

    Ok(masked)
}

/// CPU version (placeholder)
#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
pub fn extract_component(
    packed: &PackedMultivector,
    component_idx: usize,
    ckks_ctx: &CudaCkksContext,
) -> Result<Ciphertext, String> {
    Err("Component extraction not yet implemented for CPU backend".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Note: Actual tests will require V2 CKKS context initialization
    // For now, just test the API structure
    
    #[test]
    fn test_api_structure() {
        // This test just verifies the API compiles
        // Real tests will be added once packing is implemented
    }
}
