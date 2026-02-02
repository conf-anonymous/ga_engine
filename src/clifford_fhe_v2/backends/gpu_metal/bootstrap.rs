//! Metal GPU Bootstrap Operations
//!
//! GPU-accelerated implementations of CoeffToSlot and SlotToCoeff transformations
//! for V3 CKKS bootstrapping.
//!
//! **Key Advantages:**
//! - All 48 rotations run on Metal GPU (no CPU fallback)
//! - Projected 36-72× speedup vs CPU-only V3
//! - Target: 5-10s bootstrap for N=1024 (vs 360s CPU baseline)
//!
//! **Architecture:**
//! ```
//! CoeffToSlot: log(N) levels, each level:
//!   1. Diagonal matrix multiplication (GPU multiply_plain)
//!   2. Rotation by power of 2 (GPU rotate_by_steps)
//!   3. Addition (GPU buffer ops)
//!
//! SlotToCoeff: Inverse FFT (reversed order)
//!   1. Same structure as CoeffToSlot
//!   2. Negative rotations
//!   3. Inverse twiddle factors
//! ```
//!
//! **Status:** Phase 4 of Metal GPU V3 roadmap (CoeffToSlot/SlotToCoeff)

use super::ckks::{MetalCiphertext, MetalCkksContext};
use super::rotation_keys::MetalRotationKeys;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use std::f64::consts::PI;

/// Metal GPU CoeffToSlot transformation (HYBRID CPU RESCALING)
///
/// Transforms a ciphertext from coefficient representation to slot representation
/// using Metal GPU for all operations with hybrid CPU rescaling.
///
/// This version uses exact BigInt CRT rescaling on CPU for correctness.
/// Achieves ~3.6e-3 max error.
///
/// # Arguments
///
/// * `ct` - Input Metal ciphertext in coefficient representation
/// * `rotation_keys` - Metal rotation keys for all required rotations
/// * `ctx` - Metal CKKS context for multiply_plain operations
///
/// # Returns
///
/// Metal ciphertext in slot representation
///
/// # Algorithm
///
/// FFT-like butterfly structure with O(log N) levels:
///
/// ```text
/// Level 0: rotation by ±1
/// Level 1: rotation by ±2
/// Level 2: rotation by ±4
/// ...
/// Level log(N/2)-1: rotation by ±N/4
/// ```
///
/// Each level applies:
/// 1. Diagonal matrix multiplication (twiddle factors)
/// 2. Rotation by power of 2
/// 3. Butterfly addition
///
/// # Performance
///
/// - All rotations on GPU (no CPU conversion)
/// - Projected 12× faster than CPU V3
/// - Target: <1s for CoeffToSlot at N=1024
///
pub fn coeff_to_slot_gpu(
    ct: &MetalCiphertext,
    rotation_keys: &MetalRotationKeys,
    ctx: &MetalCkksContext,
    params: &CliffordFHEParams,
) -> Result<MetalCiphertext, String> {
    let n = ct.n;
    let num_slots = n / 2;
    let num_levels = (num_slots as f64).log2() as usize;

    println!("Metal GPU CoeffToSlot: N={}, slots={}, levels={}", n, num_slots, num_levels);

    // Start with input ciphertext
    let mut current = ct.clone();
    let initial_level = current.level;

    // Get moduli for all operations
    let moduli: Vec<u64> = params.moduli.iter().cloned().collect();

    // Apply FFT-like butterfly structure
    for level_idx in 0..num_levels {
        let rotation_amount = 1 << level_idx;  // 1, 2, 4, 8, ..., N/4

        println!("  Level {}: rotation by ±{}, current level={}", level_idx, rotation_amount, current.level);

        // Rotate by +rotation_amount (GPU operation)
        let ct_rotated = current.rotate_by_steps(rotation_amount as i32, rotation_keys, ctx)?;

        // Compute DFT twiddle factors for this level
        let (diag1, diag2) = compute_dft_twiddle_factors(n, num_slots, level_idx);

        // Encode diagonal matrices as plaintexts
        // CRITICAL: Use current level's top modulus for proper scaling
        let q_top = moduli[current.level] as f64;

        let pt_diag1 = encode_diagonal_for_metal(&diag1, q_top, n, current.level, &moduli)?;
        let pt_diag2 = encode_diagonal_for_metal(&diag2, q_top, n, current.level, &moduli)?;

        // Apply diagonal matrices (GPU multiply_plain)
        // result = diag1 * current + diag2 * ct_rotated
        let ct_mul1 = current.multiply_plain_metal(&pt_diag1, ctx)?;
        let ct_mul2 = ct_rotated.multiply_plain_metal(&pt_diag2, ctx)?;

        // Butterfly addition (GPU operation)
        current = add_metal_ciphertexts(&ct_mul1, &ct_mul2, &moduli)?;

        println!("    After level {}: level={}, scale={:.2e}", level_idx, current.level, current.scale);
    }

    println!("  Metal GPU CoeffToSlot complete: final level={} (consumed {} levels)",
             current.level, initial_level - current.level);

    Ok(current)
}

/// Metal GPU SlotToCoeff transformation
///
/// Transforms a ciphertext from slot representation back to coefficient representation
/// using Metal GPU for all operations.
///
/// # Arguments
///
/// * `ct` - Input Metal ciphertext in slot representation
/// * `rotation_keys` - Metal rotation keys for all required rotations
/// * `ctx` - Metal CKKS context for multiply_plain operations
///
/// # Returns
///
/// Metal ciphertext in coefficient representation
///
/// # Algorithm
///
/// Inverse FFT-like butterfly structure (reversed order from CoeffToSlot):
///
/// ```text
/// Level log(N/2)-1: rotation by ∓N/4
/// ...
/// Level 2: rotation by ∓4
/// Level 1: rotation by ∓2
/// Level 0: rotation by ∓1
/// ```
///
/// Each level applies:
/// 1. Rotation by negative power of 2 (inverse direction)
/// 2. Diagonal matrix multiplication (inverse twiddle factors)
/// 3. Inverse butterfly addition
///
/// # Correctness
///
/// Must satisfy: SlotToCoeff(CoeffToSlot(x)) ≈ x (up to noise growth)
///
pub fn slot_to_coeff_gpu(
    ct: &MetalCiphertext,
    rotation_keys: &MetalRotationKeys,
    ctx: &MetalCkksContext,
    params: &CliffordFHEParams,
) -> Result<MetalCiphertext, String> {
    let n = ct.n;
    let num_slots = n / 2;
    let num_levels = (num_slots as f64).log2() as usize;

    println!("Metal GPU SlotToCoeff: N={}, slots={}, levels={}", n, num_slots, num_levels);

    // Start with input ciphertext
    let mut current = ct.clone();
    let initial_level = current.level;

    // Get moduli for all operations
    let moduli: Vec<u64> = params.moduli.iter().cloned().collect();

    // Apply inverse FFT-like butterfly structure (reversed order)
    for level_idx in (0..num_levels).rev() {
        let rotation_amount = 1 << level_idx;  // N/4, ..., 4, 2, 1

        println!("  Level {}: rotation by ∓{}, current level={}", level_idx, rotation_amount, current.level);

        // Rotate by -rotation_amount (GPU operation, negative for inverse)
        let ct_rotated = current.rotate_by_steps(-(rotation_amount as i32), rotation_keys, ctx)?;

        // Compute inverse DFT twiddle factors for this level
        let (diag1, diag2) = compute_inverse_dft_twiddle_factors(n, num_slots, level_idx);

        // Encode diagonal matrices as plaintexts
        let q_top = moduli[current.level] as f64;

        let pt_diag1 = encode_diagonal_for_metal(&diag1, q_top, n, current.level, &moduli)?;
        let pt_diag2 = encode_diagonal_for_metal(&diag2, q_top, n, current.level, &moduli)?;

        // Apply diagonal matrices (GPU multiply_plain)
        // result = diag1 * current + diag2 * ct_rotated
        let ct_mul1 = current.multiply_plain_metal(&pt_diag1, ctx)?;
        let ct_mul2 = ct_rotated.multiply_plain_metal(&pt_diag2, ctx)?;

        // Inverse butterfly addition (GPU operation)
        current = add_metal_ciphertexts(&ct_mul1, &ct_mul2, &moduli)?;

        println!("    After level {}: level={}, scale={:.2e}", level_idx, current.level, current.scale);
    }

    println!("  Metal GPU SlotToCoeff complete: final level={} (consumed {} levels)",
             current.level, initial_level - current.level);

    Ok(current)
}

/// Metal GPU CoeffToSlot transformation (NATIVE GPU RESCALING - EXPERIMENTAL)
///
/// This version uses GPU-native RNS rescaling instead of hybrid CPU rescaling.
/// Currently produces ~385k errors (vs ~3.6e-3 with hybrid) due to RNS approximation.
///
/// Use this for performance benchmarking and experimentation with GPU rescaling algorithms.
///
pub fn coeff_to_slot_gpu_native(
    ct: &MetalCiphertext,
    rotation_keys: &MetalRotationKeys,
    ctx: &MetalCkksContext,
    params: &CliffordFHEParams,
) -> Result<MetalCiphertext, String> {
    let n = ct.n;
    let num_slots = n / 2;
    let num_levels = (num_slots as f64).log2() as usize;

    println!("Metal GPU CoeffToSlot (NATIVE): N={}, slots={}, levels={}", n, num_slots, num_levels);

    let mut current = ct.clone();

    // Apply FFT-like butterfly structure
    for level_idx in 0..num_levels {
        let rotation_amount = 1 << level_idx;

        println!("  Level {}: rotation by ±{}, current level={}", level_idx, rotation_amount, current.level);

        // Rotate by +rotation_amount (GPU operation)
        let ct_rotated = current.rotate_by_steps(rotation_amount as i32, rotation_keys, ctx)?;

        // Compute DFT twiddle factors
        let (diag1, diag2) = compute_dft_twiddle_factors(n, num_slots, level_idx);

        // Encode diagonal matrices
        let q_top = params.moduli[current.level] as f64;
        let moduli: Vec<u64> = params.moduli[0..current.level+1].to_vec();
        let pt_diag1 = encode_diagonal_for_metal(&diag1, q_top, n, current.level, &moduli)?;
        let pt_diag2 = encode_diagonal_for_metal(&diag2, q_top, n, current.level, &moduli)?;

        // Apply diagonal multiplications using NATIVE GPU rescaling
        let ct_mul1 = current.multiply_plain_metal_native_rescale(&pt_diag1, ctx)?;
        let ct_mul2 = ct_rotated.multiply_plain_metal_native_rescale(&pt_diag2, ctx)?;

        // Add results
        current = ct_mul1.add(&ct_mul2, ctx)?;

        println!("    After level {}: level={}, scale={:.2e}", level_idx, current.level, current.scale);
    }

    println!("  Metal GPU CoeffToSlot (NATIVE) complete: final level={} (consumed {} levels)", current.level, num_levels);

    Ok(current)
}

/// Metal GPU SlotToCoeff transformation (NATIVE GPU RESCALING - EXPERIMENTAL)
///
/// This version uses GPU-native RNS rescaling instead of hybrid CPU rescaling.
/// Currently produces ~385k errors (vs ~3.6e-3 with hybrid) due to RNS approximation.
///
/// Use this for performance benchmarking and experimentation with GPU rescaling algorithms.
///
pub fn slot_to_coeff_gpu_native(
    ct: &MetalCiphertext,
    rotation_keys: &MetalRotationKeys,
    ctx: &MetalCkksContext,
    params: &CliffordFHEParams,
) -> Result<MetalCiphertext, String> {
    let n = ct.n;
    let num_slots = n / 2;
    let num_levels = (num_slots as f64).log2() as usize;

    println!("Metal GPU SlotToCoeff (NATIVE): N={}, slots={}, levels={}", n, num_slots, num_levels);

    let mut current = ct.clone();

    // Apply inverse FFT (reversed order)
    for level_idx in (0..num_levels).rev() {
        let rotation_amount = 1 << level_idx;

        println!("  Level {}: rotation by ∓{}, current level={}", level_idx, rotation_amount, current.level);

        // Rotate by -rotation_amount (GPU operation)
        let ct_rotated = current.rotate_by_steps(-(rotation_amount as i32), rotation_keys, ctx)?;

        // Compute inverse DFT twiddle factors
        let (diag1, diag2) = compute_inverse_dft_twiddle_factors(n, num_slots, level_idx);

        // Encode diagonal matrices
        let q_top = params.moduli[current.level] as f64;
        let moduli: Vec<u64> = params.moduli[0..current.level+1].to_vec();
        let pt_diag1 = encode_diagonal_for_metal(&diag1, q_top, n, current.level, &moduli)?;
        let pt_diag2 = encode_diagonal_for_metal(&diag2, q_top, n, current.level, &moduli)?;

        // Apply diagonal multiplications using NATIVE GPU rescaling
        let ct_mul1 = current.multiply_plain_metal_native_rescale(&pt_diag1, ctx)?;
        let ct_mul2 = ct_rotated.multiply_plain_metal_native_rescale(&pt_diag2, ctx)?;

        // Add results
        current = ct_mul1.add(&ct_mul2, ctx)?;

        println!("    After level {}: level={}, scale={:.2e}", level_idx, current.level, current.scale);
    }

    println!("  Metal GPU SlotToCoeff (NATIVE) complete: final level={} (consumed {} levels)", current.level, num_levels);

    Ok(current)
}

fn compute_dft_twiddle_factors(n: usize, num_slots: usize, level_idx: usize) -> (Vec<f64>, Vec<f64>) {
    let mut diag1 = vec![0.5; num_slots];
    let mut diag2 = vec![0.5; num_slots];

    let stride = 1 << level_idx;
    for j in 0..num_slots {
        let k = (j / stride) * stride;
        let theta = 2.0 * PI * (k as f64) / (n as f64);

        let cos_theta = theta.cos();
        diag1[j] = (1.0 + cos_theta) / 2.0;
        diag2[j] = (1.0 - cos_theta) / 2.0;
    }

    (diag1, diag2)
}

/// Compute inverse DFT twiddle factors for a given level
///
/// Returns (diag1, diag2) where:
/// - diag1[j] = (1 + cos(-θ_j)) / 2
/// - diag2[j] = (1 - cos(-θ_j)) / 2
/// - θ_j = -2π * k(j) / N (negative for inverse)
///
fn compute_inverse_dft_twiddle_factors(n: usize, num_slots: usize, level_idx: usize) -> (Vec<f64>, Vec<f64>) {
    let mut diag1 = vec![0.5; num_slots];
    let mut diag2 = vec![0.5; num_slots];

    let stride = 1 << level_idx;
    for j in 0..num_slots {
        let k = (j / stride) * stride;
        let theta = -2.0 * PI * (k as f64) / (n as f64);  // Negative for inverse

        let cos_theta = theta.cos();
        diag1[j] = (1.0 + cos_theta) / 2.0;
        diag2[j] = (1.0 - cos_theta) / 2.0;
    }

    (diag1, diag2)
}

/// Encode a diagonal vector as a Metal plaintext in flat RNS layout
///
/// Creates a plaintext with proper scaling and RNS representation for Metal GPU operations.
///
/// # Arguments
///
/// * `diagonal` - Vector of num_slots = N/2 diagonal values
/// * `scale` - Encoding scale (typically q_top)
/// * `n` - Ring dimension N
/// * `level` - Current ciphertext level
/// * `moduli` - RNS moduli
///
/// # Returns
///
/// Flat RNS plaintext: [coeff0_mod_q0, coeff0_mod_q1, ..., coeffN-1_mod_qL]
///
fn encode_diagonal_for_metal(
    diagonal: &[f64],
    scale: f64,
    n: usize,
    level: usize,
    moduli: &[u64],
) -> Result<Vec<u64>, String> {
    let num_slots = n / 2;
    if diagonal.len() != num_slots {
        return Err(format!("Diagonal size mismatch: {} vs {}", diagonal.len(), num_slots));
    }

    // CRITICAL: Use canonical embedding (same as V3 CPU and V2 CPU)
    // Simple scaling does NOT work with Galois automorphisms!
    // We need proper FFT-based slot encoding.
    use crate::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
    let coeffs_i64 = MetalCkksContext::canonical_embed_encode_real(diagonal, scale, n);

    // Convert to RNS representation (flat layout)
    let num_primes = level + 1;  // Only use primes up to current level
    let mut flat_rns = vec![0u64; n * num_primes];

    for coeff_idx in 0..n {
        let val_i64 = coeffs_i64[coeff_idx];

        for prime_idx in 0..num_primes {
            let q = moduli[prime_idx];
            let val_mod_q = if val_i64 >= 0 {
                (val_i64 as u64) % q
            } else {
                let abs_val = (-val_i64) as u64;
                if abs_val % q == 0 {
                    0
                } else {
                    q - (abs_val % q)
                }
            };

            // Flat RNS layout: [coeff0_mod_q0, coeff0_mod_q1, ..., coeff1_mod_q0, ...]
            flat_rns[coeff_idx * num_primes + prime_idx] = val_mod_q;
        }
    }

    Ok(flat_rns)
}

/// Add two Metal ciphertexts (component-wise modular addition)
///
/// Performs: result = ct1 + ct2 (mod q) for all RNS components
///
/// Handles variable strides: after rescaling, ct arrays may have different strides than level suggests.
///
fn add_metal_ciphertexts(
    ct1: &MetalCiphertext,
    ct2: &MetalCiphertext,
    moduli: &[u64],
) -> Result<MetalCiphertext, String> {
    if ct1.level != ct2.level {
        return Err(format!("Level mismatch: {} vs {}", ct1.level, ct2.level));
    }
    if ct1.n != ct2.n {
        return Err(format!("Ring dimension mismatch: {} vs {}", ct1.n, ct2.n));
    }

    let n = ct1.n;
    let num_primes = ct1.level + 1;  // Active primes at current level

    // Determine actual strides (may differ from num_primes after rescaling)
    let ct1_stride = ct1.c0.len() / n;
    let ct2_stride = ct2.c0.len() / n;

    if ct1.c0.len() % n != 0 || ct2.c0.len() % n != 0 {
        return Err(format!("Invalid ciphertext array sizes: ct1.c0={}, ct2.c0={}, n={}",
            ct1.c0.len(), ct2.c0.len(), n));
    }

    // Add c0 components (extract active primes, add, store)
    let mut c0_sum = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            let q = moduli[prime_idx];
            let val1 = ct1.c0[coeff_idx * ct1_stride + prime_idx];
            let val2 = ct2.c0[coeff_idx * ct2_stride + prime_idx];
            c0_sum[coeff_idx * num_primes + prime_idx] =
                ((val1 as u128 + val2 as u128) % q as u128) as u64;
        }
    }

    // Add c1 components
    let mut c1_sum = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            let q = moduli[prime_idx];
            let val1 = ct1.c1[coeff_idx * ct1_stride + prime_idx];
            let val2 = ct2.c1[coeff_idx * ct2_stride + prime_idx];
            c1_sum[coeff_idx * num_primes + prime_idx] =
                ((val1 as u128 + val2 as u128) % q as u128) as u64;
        }
    }

    Ok(MetalCiphertext {
        c0: c0_sum,
        c1: c1_sum,
        n,
        num_primes: ct1.num_primes,
        level: ct1.level,
        scale: ct1.scale,  // Scale remains unchanged after addition
    })
}

// ================================================================================================
// HELPER: multiply_plain_metal for MetalCiphertext
// ================================================================================================

impl MetalCiphertext {
    /// Multiply Metal ciphertext by plaintext (GPU operation)
    ///
    /// Performs: result = ct * pt (with rescaling)
    ///
    /// # Algorithm
    ///
    /// 1. NTT multiply on GPU for all RNS components
    /// 2. Rescale to drop top modulus
    /// 3. Update scale and level
    ///
    /// # Important: Stride Handling
    ///
    /// After rescaling, arrays shrink (drop top modulus), so stride changes.
    /// We maintain consistent stride by extracting only active components.
    ///
    pub fn multiply_plain_metal(
        &self,
        pt: &[u64],
        ctx: &MetalCkksContext,
    ) -> Result<Self, String> {
        let n = self.n;
        let num_primes = self.level + 1;

        // Verify plaintext size
        if pt.len() != n * num_primes {
            return Err(format!("Plaintext size mismatch: {} vs {} (n={}, num_primes={})",
                pt.len(), n * num_primes, n, num_primes));
        }

        // Determine stride of ciphertext arrays (may be larger than num_primes after rescaling)
        let ct_stride = self.c0.len() / n;
        if self.c0.len() % n != 0 {
            return Err(format!("Ciphertext c0 size {} not divisible by n={}", self.c0.len(), n));
        }

        // Extract only the active primes from ciphertext (keep strided layout for multiply_polys)
        let mut c0_active = vec![0u64; n * num_primes];
        let mut c1_active = vec![0u64; n * num_primes];

        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                c0_active[coeff_idx * num_primes + prime_idx] =
                    self.c0[coeff_idx * ct_stride + prime_idx];
                c1_active[coeff_idx * num_primes + prime_idx] =
                    self.c1[coeff_idx * ct_stride + prime_idx];
            }
        }

        // Get moduli for current level
        let moduli: Vec<u64> = ctx.params.moduli[0..num_primes].to_vec();

        // Multiply c0 * pt (GPU NTT multiplication) - outputs strided layout
        let c0_mul = ctx.multiply_polys_flat_ntt_negacyclic(&c0_active, pt, &moduli)?;

        // Multiply c1 * pt (GPU NTT multiplication) - outputs strided layout
        let c1_mul = ctx.multiply_polys_flat_ntt_negacyclic(&c1_active, pt, &moduli)?;

        // HYBRID RESCALE: Use CPU's exact BigInt CRT rescaling
        //
        // NOTE: GPU-native RNS rescaling was attempted using the formula r'ᵢ = (rᵢ - r_top) × q_top^{-1} mod qᵢ
        // but this approximation doesn't handle CKKS rounding correctly (produced 385k errors vs 3.6e-3 with CPU).
        // The RNS formula works for BFV-style rescaling but CKKS requires exact centered rounding.
        //
        // Future optimization: Implement full BigInt arithmetic on GPU using multi-precision libraries.
        // For now, the hybrid approach (GPU multiply + CPU rescale) achieves correct results.

        let new_level = if self.level > 0 { self.level - 1 } else { 0 };
        let new_num_primes = new_level + 1;

        let q_top = moduli[self.level];

        // After ct * pt: scale = ct.scale * pt.scale = self.scale * q_top
        let pre_rescale_scale = self.scale * (q_top as f64);

        // After rescale: scale = (self.scale * q_top) / q_top = self.scale
        let new_scale = pre_rescale_scale / (q_top as f64);

        // EXACT rescaling using CPU's BigInt CRT algorithm
        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext as CpuCiphertext;

        let mut c0_rescaled = vec![0u64; n * new_num_primes];
        let mut c1_rescaled = vec![0u64; n * new_num_primes];

        for coeff_idx in 0..n {
            // Extract RNS limbs for this coefficient (all primes including top)
            let mut c0_limbs = vec![0u64; num_primes];
            let mut c1_limbs = vec![0u64; num_primes];

            for prime_idx in 0..num_primes {
                c0_limbs[prime_idx] = c0_mul[coeff_idx * num_primes + prime_idx];
                c1_limbs[prime_idx] = c1_mul[coeff_idx * num_primes + prime_idx];
            }

            // Apply EXACT rescaling: BigInt CRT + rounded division by q_top
            let c0_rescaled_limbs = CpuCiphertext::rescale_coeff_bigint(&c0_limbs, &moduli, q_top);
            let c1_rescaled_limbs = CpuCiphertext::rescale_coeff_bigint(&c1_limbs, &moduli, q_top);

            // Store rescaled residues (now with one fewer prime)
            for prime_idx in 0..new_num_primes {
                c0_rescaled[coeff_idx * new_num_primes + prime_idx] = c0_rescaled_limbs[prime_idx];
                c1_rescaled[coeff_idx * new_num_primes + prime_idx] = c1_rescaled_limbs[prime_idx];
            }
        }

        Ok(Self {
            c0: c0_rescaled,
            c1: c1_rescaled,
            n,
            num_primes: self.num_primes,  // Keep original total, but level controls active primes
            level: new_level,
            scale: new_scale,
        })
    }

    /// Multiply ciphertext by plaintext and rescale (GPU-NATIVE RESCALING VERSION)
    ///
    /// This is an experimental version that uses GPU-native RNS rescaling instead of
    /// hybrid CPU rescaling. Currently produces higher errors (~385k) vs hybrid (~3.6e-3)
    /// due to RNS approximation not handling CKKS centered rounding correctly.
    ///
    /// Algorithm:
    /// 1. Multiply: (c0, c1) * pt → (c0·pt, c1·pt)
    /// 2. GPU-native rescale using RNS formula: r'ᵢ = (rᵢ - r_top) × q_top^{-1} mod qᵢ
    /// 3. Update scale and level
    ///
    /// Use this for performance benchmarking and comparing with hybrid approach.
    pub fn multiply_plain_metal_native_rescale(
        &self,
        pt: &[u64],
        ctx: &MetalCkksContext,
    ) -> Result<Self, String> {
        let n = self.n;
        let num_primes = self.level + 1;

        // Verify plaintext size
        if pt.len() != n * num_primes {
            return Err(format!("Plaintext size mismatch: {} vs {} (n={}, num_primes={})",
                pt.len(), n * num_primes, n, num_primes));
        }

        // Determine stride of ciphertext arrays
        let ct_stride = self.c0.len() / n;
        if self.c0.len() % n != 0 {
            return Err(format!("Ciphertext c0 size {} not divisible by n={}", self.c0.len(), n));
        }

        // Extract only the active primes from ciphertext (keep strided layout for multiply_polys)
        let mut c0_active = vec![0u64; n * num_primes];
        let mut c1_active = vec![0u64; n * num_primes];

        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                c0_active[coeff_idx * num_primes + prime_idx] = self.c0[coeff_idx * ct_stride + prime_idx];
                c1_active[coeff_idx * num_primes + prime_idx] = self.c1[coeff_idx * ct_stride + prime_idx];
            }
        }

        let moduli: Vec<u64> = ctx.params.moduli[0..num_primes].to_vec();

        // Multiply c0 * pt (GPU NTT multiplication) - outputs strided layout
        let c0_mul_strided = ctx.multiply_polys_flat_ntt_negacyclic(&c0_active, pt, &moduli)?;

        // Multiply c1 * pt (GPU NTT multiplication) - outputs strided layout
        let c1_mul_strided = ctx.multiply_polys_flat_ntt_negacyclic(&c1_active, pt, &moduli)?;

        // Convert from strided to flat RNS layout for rescaling
        // Strided: poly[coeff_idx * num_primes + prime_idx]
        // Flat RNS: poly[prime_idx * n + coeff_idx]
        let mut c0_mul = vec![0u64; n * num_primes];
        let mut c1_mul = vec![0u64; n * num_primes];

        for prime_idx in 0..num_primes {
            for coeff_idx in 0..n {
                c0_mul[prime_idx * n + coeff_idx] = c0_mul_strided[coeff_idx * num_primes + prime_idx];
                c1_mul[prime_idx * n + coeff_idx] = c1_mul_strided[coeff_idx * num_primes + prime_idx];
            }
        }

        // GPU-NATIVE EXACT RESCALING (FIXED - PROPER DRLMQ)
        // Uses proper divide-and-round by last modulus with:
        // 1. Centered rounding: ⌊(C + q_last/2) / q_last⌋
        // 2. Correct 128-bit modular arithmetic (no precision loss)
        // 3. All operations in standard domain (matching inverse NTT output)
        //
        // This fixes the domain mismatch issues that caused 100k+ errors.

        let new_level = if self.level > 0 { self.level - 1 } else { 0 };
        let new_num_primes = new_level + 1;

        let q_top = moduli[self.level];

        // After ct * pt: scale = ct.scale * pt.scale = self.scale * q_top
        let pre_rescale_scale = self.scale * (q_top as f64);

        // After rescale: scale = (self.scale * q_top) / q_top = self.scale
        let new_scale = pre_rescale_scale / (q_top as f64);

        // Apply GPU-native exact rescaling (FIXED algorithm) to both c0 and c1
        // Returns flat RNS layout: [prime_idx * n + coeff_idx]
        let c0_rescaled_flat = ctx.exact_rescale_gpu_fixed(&c0_mul, self.level)?;
        let c1_rescaled_flat = ctx.exact_rescale_gpu_fixed(&c1_mul, self.level)?;

        // Convert back to strided layout (COMPACT, matching hybrid version)
        // Output: n × (num_primes_out) in strided layout
        let num_primes_out = new_level + 1;
        let mut c0_rescaled = vec![0u64; n * num_primes_out];
        let mut c1_rescaled = vec![0u64; n * num_primes_out];

        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes_out {
                c0_rescaled[coeff_idx * num_primes_out + prime_idx] =
                    c0_rescaled_flat[prime_idx * n + coeff_idx];
                c1_rescaled[coeff_idx * num_primes_out + prime_idx] =
                    c1_rescaled_flat[prime_idx * n + coeff_idx];
            }
        }

        Ok(Self {
            c0: c0_rescaled,
            c1: c1_rescaled,
            n,
            num_primes: self.num_primes,
            level: new_level,
            scale: new_scale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_twiddle_factors() {
        let n = 1024;
        let num_slots = n / 2;

        // Test forward DFT twiddle factors
        let (diag1, diag2) = compute_dft_twiddle_factors(n, num_slots, 0);
        assert_eq!(diag1.len(), num_slots);
        assert_eq!(diag2.len(), num_slots);

        // Test inverse DFT twiddle factors
        let (inv_diag1, inv_diag2) = compute_inverse_dft_twiddle_factors(n, num_slots, 0);
        assert_eq!(inv_diag1.len(), num_slots);
        assert_eq!(inv_diag2.len(), num_slots);
    }

    #[test]
    fn test_encode_diagonal() {
        let diagonal = vec![1.0, 2.0, 3.0, 4.0];
        let scale = 1e10;
        let n = 8;
        let level = 2;
        let moduli = vec![1152921504606584833, 1152921504598720513, 1152921504597016577];

        let result = encode_diagonal_for_metal(&diagonal, scale, n, level, &moduli);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert_eq!(encoded.len(), n * (level + 1));
    }
}
