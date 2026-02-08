//! CUDA-Accelerated Batched Multivector Operations
//!
//! Provides a CUDA-resident batched multivector type and GPU-accelerated
//! geometric product. All intermediate operations stay on GPU; conversion
//! happens only at encode (once) and decode (once).
//!
//! # Layout
//!
//! CPU Ciphertext: `ct.c0[coeff_idx].values[prime_idx]` (Vec<RnsRepresentation>)
//! CUDA Ciphertext: `ct.c0[coeff_idx * num_primes + prime_idx]` (flat Vec<u64>)

use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{CkksContext, Ciphertext, Plaintext};
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{PublicKey, SecretKey};
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use crate::clifford_fhe_v2::backends::gpu_cuda::ckks::{CudaCkksContext, CudaCiphertext, CudaPlaintext};
use crate::clifford_fhe_v2::backends::gpu_cuda::relin_keys::CudaRelinKeys;
use crate::clifford_fhe_v2::backends::gpu_cuda::rotation_keys::CudaRotationKeys;
use crate::clifford_fhe_v2::backends::gpu_cuda::inversion::multiply_ciphertexts_gpu;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use super::BatchedMultivector;
use super::encoding::{encode_batch, decode_batch};

/// CUDA-resident batched multivector
///
/// Wraps a CudaCiphertext containing SIMD-packed multivectors.
/// Data stays on GPU between operations; conversion only at encode/decode.
#[derive(Clone, Debug)]
pub struct CudaBatchedMultivector {
    /// Underlying CUDA ciphertext with packed slots
    pub ciphertext: CudaCiphertext,

    /// Number of multivectors in this batch
    pub batch_size: usize,

    /// Ring dimension
    pub n: usize,
}

/// Convert a CPU Ciphertext to CudaCiphertext (layout transform)
///
/// CPU layout: ct.c0[coeff_idx].values[prime_idx]
/// CUDA layout: ct.c0[coeff_idx * num_primes + prime_idx]
pub fn cpu_ciphertext_to_cuda(ct: &Ciphertext) -> CudaCiphertext {
    let n = ct.n;
    let num_primes = ct.c0[0].values.len();

    let mut c0 = vec![0u64; n * num_primes];
    let mut c1 = vec![0u64; n * num_primes];

    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            let idx = coeff_idx * num_primes + prime_idx;
            c0[idx] = ct.c0[coeff_idx].values[prime_idx];
            c1[idx] = ct.c1[coeff_idx].values[prime_idx];
        }
    }

    CudaCiphertext {
        c0,
        c1,
        n,
        num_primes,
        level: ct.level,
        scale: ct.scale,
    }
}

/// Convert a CudaCiphertext back to CPU Ciphertext
///
/// CUDA layout: ct.c0[coeff_idx * num_primes + prime_idx]
/// CPU layout: ct.c0[coeff_idx].values[prime_idx]
pub fn cuda_ciphertext_to_cpu(ct: &CudaCiphertext, moduli: &[u64]) -> Ciphertext {
    let n = ct.n;
    let num_primes = ct.num_primes;
    let active_moduli = moduli[..num_primes].to_vec();

    let mut c0 = Vec::with_capacity(n);
    let mut c1 = Vec::with_capacity(n);

    for coeff_idx in 0..n {
        let mut vals0 = Vec::with_capacity(num_primes);
        let mut vals1 = Vec::with_capacity(num_primes);

        for prime_idx in 0..num_primes {
            let idx = coeff_idx * num_primes + prime_idx;
            vals0.push(ct.c0[idx]);
            vals1.push(ct.c1[idx]);
        }

        c0.push(RnsRepresentation::new(vals0, active_moduli.clone()));
        c1.push(RnsRepresentation::new(vals1, active_moduli.clone()));
    }

    Ciphertext::new(c0, c1, ct.level, ct.scale)
}

/// Encode multivectors and convert to CUDA batch (encode on CPU, convert once)
pub fn encode_batch_cuda(
    multivectors: &[[f64; 8]],
    ckks_ctx: &CkksContext,
    pk: &PublicKey,
) -> CudaBatchedMultivector {
    let cpu_batch = encode_batch(multivectors, ckks_ctx, pk);
    let cuda_ct = cpu_ciphertext_to_cuda(&cpu_batch.ciphertext);

    CudaBatchedMultivector {
        batch_size: cpu_batch.batch_size,
        n: cpu_batch.n,
        ciphertext: cuda_ct,
    }
}

/// Decode CUDA batch (convert once, decode on CPU)
pub fn decode_batch_cuda(
    batched: &CudaBatchedMultivector,
    ckks_ctx: &CkksContext,
    sk: &SecretKey,
    moduli: &[u64],
) -> Vec<[f64; 8]> {
    let cpu_ct = cuda_ciphertext_to_cpu(&batched.ciphertext, moduli);
    let cpu_batch = BatchedMultivector::new(cpu_ct, batched.batch_size);
    decode_batch(&cpu_batch, ckks_ctx, sk)
}

/// Extract component i from a CUDA batched multivector via rotate-then-mask
///
/// 1. Rotate by +component to bring component i to position 0 of each block
/// 2. Mask at positions [0, 8, 16, ...] to isolate the target component
/// Result: component i values at positions [0, 8, 16, ...] (same for all components)
pub fn extract_component_cuda(
    batched: &CudaBatchedMultivector,
    component: usize,
    rot_keys: &CudaRotationKeys,
    cuda_ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String> {
    assert!(component < 8, "Component index must be 0-7");

    let params = cuda_ctx.params();
    let num_slots = params.n / 2;
    let num_multivectors = num_slots / 8;

    // Step 1: Rotate to bring component i to position 0 of each block
    let rotated_ct = if component > 0 {
        batched.ciphertext.rotate_by_steps(component as i32, rot_keys, cuda_ctx)?
    } else {
        batched.ciphertext.clone()
    };

    // Step 2: Mask at position 0 of each 8-slot block [0, 8, 16, ...]
    let mut mask = vec![0.0; num_slots];
    for i in 0..num_multivectors {
        mask[i * 8] = 1.0;
    }

    // Encode mask at the rotated ciphertext's level
    let pt_mask = CudaPlaintext::encode_at_level(&mask, params.scale, params, rotated_ct.level);

    // Multiply by mask (consumes 1 level)
    rotated_ct.multiply_plain(&pt_mask, cuda_ctx)
}

/// Extract all 8 components from CUDA batched multivector
pub fn extract_all_components_cuda(
    batched: &CudaBatchedMultivector,
    rot_keys: &CudaRotationKeys,
    cuda_ctx: &CudaCkksContext,
) -> Result<Vec<CudaCiphertext>, String> {
    let mut components = Vec::with_capacity(8);
    for i in 0..8 {
        components.push(extract_component_cuda(batched, i, rot_keys, cuda_ctx)?);
    }
    Ok(components)
}

/// Reassemble 8 component CTs into a CUDA batched multivector
///
/// Each component ciphertext has values at positions [0, 8, 16, ...].
/// Rotates component i by -i to shift to position i, then sums all.
pub fn reassemble_components_cuda(
    components: &[CudaCiphertext],
    rot_keys: &CudaRotationKeys,
    cuda_ctx: &CudaCkksContext,
    batch_size: usize,
    n: usize,
) -> Result<CudaBatchedMultivector, String> {
    assert_eq!(components.len(), 8, "Must have exactly 8 components");

    // Component 0 is already at position 0 - no rotation needed
    let mut result = components[0].clone();

    // Components 1-7: rotate by -i to shift from position 0 to position i
    for i in 1..8 {
        let shifted = components[i].rotate_by_steps(-(i as i32), rot_keys, cuda_ctx)?;

        // Align levels if needed via mod_switch
        let (a, b) = if result.level > shifted.level {
            (result.mod_switch_to_level(shifted.level), shifted)
        } else if shifted.level > result.level {
            (result.clone(), shifted.mod_switch_to_level(result.level))
        } else {
            (result.clone(), shifted)
        };

        result = cuda_ctx.add(&a, &b)?;
    }

    Ok(CudaBatchedMultivector {
        ciphertext: result,
        batch_size,
        n,
    })
}

/// Negate a CUDA ciphertext: compute -ct (mod q for each prime)
///
/// Phase 4C: Uses GPU kernel instead of CPU loop
fn negate_cuda_ciphertext(ct: &CudaCiphertext, cuda_ctx: &CudaCkksContext) -> CudaCiphertext {
    let n = ct.n;
    let num_primes = ct.num_primes;

    let neg_c0 = cuda_ctx.negate_strided_gpu(&ct.c0, num_primes, num_primes)
        .expect("GPU negate c0 failed");
    let neg_c1 = cuda_ctx.negate_strided_gpu(&ct.c1, num_primes, num_primes)
        .expect("GPU negate c1 failed");

    CudaCiphertext {
        c0: neg_c0,
        c1: neg_c1,
        n,
        num_primes,
        level: ct.level,
        scale: ct.scale,
    }
}

/// Cl(3,0) structure constants for geometric product (reused from batched/geometric.rs)
struct Cl3StructureConstants {
    products: Vec<Vec<(i64, usize, usize)>>,
}

impl Cl3StructureConstants {
    fn new() -> Self {
        let mut products = vec![Vec::new(); 8];

        products[0] = vec![
            (1, 0, 0), (1, 1, 1), (1, 2, 2), (1, 3, 3),
            (-1, 4, 4), (-1, 5, 5), (-1, 6, 6), (-1, 7, 7),
        ];
        products[1] = vec![
            (1, 0, 1), (1, 1, 0), (1, 2, 4), (-1, 4, 2),
            (1, 3, 5), (-1, 5, 3), (-1, 6, 7), (1, 7, 6),
        ];
        products[2] = vec![
            (1, 0, 2), (1, 2, 0), (-1, 1, 4), (1, 4, 1),
            (1, 3, 6), (-1, 6, 3), (-1, 5, 7), (1, 7, 5),
        ];
        products[3] = vec![
            (1, 0, 3), (1, 3, 0), (-1, 1, 5), (1, 5, 1),
            (-1, 2, 6), (1, 6, 2), (-1, 4, 7), (1, 7, 4),
        ];
        products[4] = vec![
            (1, 0, 4), (1, 4, 0), (1, 1, 2), (-1, 2, 1),
            (1, 3, 7), (-1, 7, 3), (1, 5, 6), (-1, 6, 5),
        ];
        products[5] = vec![
            (1, 0, 5), (1, 5, 0), (1, 1, 3), (-1, 3, 1),
            (-1, 2, 7), (1, 7, 2), (-1, 4, 6), (1, 6, 4),
        ];
        products[6] = vec![
            (1, 0, 6), (1, 6, 0), (1, 2, 3), (-1, 3, 2),
            (1, 1, 7), (-1, 7, 1), (1, 4, 5), (-1, 5, 4),
        ];
        products[7] = vec![
            (1, 0, 7), (1, 7, 0), (1, 1, 6), (-1, 6, 1),
            (-1, 2, 5), (1, 5, 2), (1, 3, 4), (-1, 4, 3),
        ];

        Cl3StructureConstants { products }
    }
}

/// CUDA batched geometric product
///
/// Mirrors the algorithm of `geometric_product_batched()` from CPU batched:
/// 1. Extract 8 components from each batch (16 multiply_plain on GPU)
/// 2. Compute 8 output components: 64 multiply_ciphertexts_gpu + accumulate
/// 3. Reassemble via addition
///
/// All 64+ operations stay entirely on GPU. CPU-CUDA transfer happens only at
/// encode_batch_cuda (once) and decode_batch_cuda (once).
pub fn geometric_product_batched_cuda(
    a_batch: &CudaBatchedMultivector,
    b_batch: &CudaBatchedMultivector,
    relin_keys: &CudaRelinKeys,
    rot_keys: &CudaRotationKeys,
    cuda_ctx: &CudaCkksContext,
) -> Result<CudaBatchedMultivector, String> {
    assert_eq!(
        a_batch.batch_size, b_batch.batch_size,
        "Batch sizes must match"
    );

    let constants = Cl3StructureConstants::new();

    // Step 1: Extract all 8 components from each batch
    let a_components = extract_all_components_cuda(a_batch, rot_keys, cuda_ctx)?;
    let b_components = extract_all_components_cuda(b_batch, rot_keys, cuda_ctx)?;

    // Step 2: Compute 8 output components
    let mut result_components: Vec<CudaCiphertext> = Vec::with_capacity(8);

    for out_idx in 0..8 {
        let product_terms = &constants.products[out_idx];

        // Compute first term
        let (first_coeff, first_a_idx, first_b_idx) = product_terms[0];
        let mut accumulated = multiply_ciphertexts_gpu(
            &a_components[first_a_idx],
            &b_components[first_b_idx],
            relin_keys,
            cuda_ctx,
        )?;

        if first_coeff < 0 {
            accumulated = negate_cuda_ciphertext(&accumulated, cuda_ctx);
        }

        // Accumulate remaining 7 terms
        for &(coeff, a_idx, b_idx) in &product_terms[1..] {
            let mut ct_product = multiply_ciphertexts_gpu(
                &a_components[a_idx],
                &b_components[b_idx],
                relin_keys,
                cuda_ctx,
            )?;

            if coeff < 0 {
                ct_product = negate_cuda_ciphertext(&ct_product, cuda_ctx);
            }

            accumulated = cuda_ctx.add(&accumulated, &ct_product)?;
        }

        result_components.push(accumulated);
    }

    // Step 3: Reassemble into CUDA batched multivector
    reassemble_components_cuda(
        &result_components,
        rot_keys,
        cuda_ctx,
        a_batch.batch_size,
        a_batch.n,
    )
}
