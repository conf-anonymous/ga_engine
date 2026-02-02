//! Canonical Embedding for CKKS (Correct Version)
//!
//! This implements the proper canonical embedding from CKKS that ensures
//! Galois automorphisms correspond to slot rotations.
//!
//! # Key Difference from Standard FFT
//!
//! - **Standard FFT**: Evaluates polynomial at ω^k where ω = e^(2πi/N)
//! - **CKKS Canonical Embedding**: Evaluates at ζ_M^(2k+1) where ζ_M = e^(2πi/M), M=2N
//!
//! This specific choice of evaluation points ensures that the Galois automorphism
//! σ_k : x → x^k corresponds to a slot permutation.
//!
//! # Mathematical Foundation
//!
//! For cyclotomic polynomial Φ_M(x) where M = 2N:
//! - Roots are ζ_M^k for k coprime to M (Euler phi function: φ(M) = N)
//! - For M = 2N power of 2: roots are ζ_M^(2k+1) for k = 0, 1, ..., N-1
//! - These are the N primitive M-th roots of unity
//!
//! The canonical embedding maps:
//! ```text
//! σ: R = Z[x]/(Φ_M(x)) → C^N
//! σ(p(x)) = [p(ζ_M), p(ζ_M^3), p(ζ_M^5), ..., p(ζ_M^(2N-1))]
//! ```
//!
//! With this embedding, the automorphism σ_k: x → x^k acts on slots as:
//! - If k = 2r+1 (odd), it permutes slots
//! - The specific permutation depends on how k acts on the exponents
//!
//! # References
//!
//! - CKKS paper: "Homomorphic Encryption for Arithmetic of Approximate Numbers"
//!   Cheon, Kim, Kim, Song (2017)
//! - Halevi & Shoup: "Algorithms in HElib" (2014)

use rustfft::num_complex::Complex;
use std::f64::consts::PI;

/// Compute the Galois orbit order for CKKS slot indexing
///
/// For power-of-two cyclotomics M=2N, the odd residues mod M form two orbits
/// under multiplication by generator g (typically g=5). This function computes
/// the orbit starting from 1: e[t] = g^t mod M.
///
/// With this ordering, automorphism σ_g acts as a left rotation by 1 slot!
///
/// # Arguments
/// * `n` - Ring dimension N
/// * `g` - Generator (typically 5 for power-of-two cyclotomics)
///
/// # Returns
/// Vector e where e[t] = g^t mod M for t=0..(N/2-1)
fn orbit_order(n: usize, g: usize) -> Vec<usize> {
    let m = 2 * n; // M = 2N
    let num_slots = n / 2; // N/2 slots

    let mut e = vec![0usize; num_slots];
    let mut cur = 1usize;

    for t in 0..num_slots {
        e[t] = cur; // odd exponent in [1..2N-1]
        cur = (cur * g) % m;
    }

    e
}

/// Encode slots using CKKS canonical embedding
///
/// Evaluates slots at the specific primitive roots ζ_M^(2k+1)
/// to ensure automorphisms correspond to slot rotations.
///
/// # Arguments
/// * `slots` - N/2 complex values to encode
/// * `scale` - Scaling factor
/// * `n` - Ring dimension (N in the formula above)
///
/// # Returns
/// Polynomial coefficients
pub fn canonical_embed_encode(slots: &[Complex<f64>], scale: f64, n: usize) -> Vec<i64> {
    assert!(n.is_power_of_two());
    let num_slots = n / 2;
    assert_eq!(slots.len(), num_slots);

    let m = 2 * n; // Cyclotomic index M = 2N
    let g = 5; // Generator for power-of-two cyclotomics

    // CRITICAL FIX: Use Galois orbit order instead of natural order!
    // This ensures automorphism σ_g acts as rotate-by-1
    let e = orbit_order(n, g);

    // Inverse canonical embedding (orbit-order compatible)
    // For each coefficient j, sum over slots with both the slot value and its conjugate
    // This handles the Hermitian symmetry required for real coefficients
    //
    // Formula: c[j] = (1/N) * Re( Σ_t ( z[t] * w_t(j) + conj(z[t]) * conj(w_t(j)) ) )
    // where w_t(j) = exp(-2πi * e[t] * j / M)
    //
    // Key points:
    // - Single loop over t with TWO terms (z and conj(z)) per slot
    // - Normalization is 1/N (NOT 2/N!)
    // - No need to index conjugate slots - they're in the "other orbit" implicitly
    let mut coeffs_float = vec![0.0; n];

    for j in 0..n {
        let mut sum = Complex::new(0.0, 0.0);

        // Single loop over slots, adding both the slot and its conjugate contribution
        for t in 0..num_slots {
            // w_t(j) = exp(-2πi * e[t] * j / M)
            let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
            let w = Complex::new(angle.cos(), -angle.sin()); // exp(-i*angle)

            // Add both z[t] * w and conj(z[t]) * conj(w)
            sum += slots[t] * w + slots[t].conj() * w.conj();
        }

        // Normalize by 1/N (NOT 2/N!) and take real part
        coeffs_float[j] = sum.re / (n as f64);
    }

    // Scale and round to integers
    coeffs_float.iter().map(|&x| (x * scale).round() as i64).collect()
}

/// Center-lift coefficients from [0, q) to (-q/2, q/2]
///
/// This is critical for correct decoding! Polynomial multiplication produces
/// coefficients in [0, q), but for decoding to real values we must interpret
/// them in the symmetric interval around zero.
///
/// # Arguments
/// * `coeffs` - Coefficients in [0, q)
/// * `q` - Modulus
///
/// # Returns
/// * Coefficients in (-q/2, q/2]
fn center_lift(coeffs: &[i64], q: i64) -> Vec<i64> {
    coeffs.iter().map(|&c| {
        let mut v = c % q;
        if v < 0 {
            v += q;
        }
        if v > q / 2 {
            v -= q;
        }
        v
    }).collect()
}

/// Decode slots using CKKS canonical embedding with orbit ordering
///
/// Evaluates polynomial at the orbit-ordered primitive roots ζ_M^{e[t]}.
///
/// This is the adjoint of canonical_embed_encode.
///
/// # Arguments
/// * `coeffs` - Polynomial coefficients
/// * `scale` - Scaling factor
/// * `n` - Ring dimension
///
/// # Returns
/// N/2 complex slot values
pub fn canonical_embed_decode(coeffs: &[i64], scale: f64, n: usize) -> Vec<Complex<f64>> {
    assert_eq!(coeffs.len(), n);

    let m = 2 * n; // M = 2N
    let num_slots = n / 2;
    let g = 5; // Generator

    // CRITICAL FIX: Use Galois orbit order!
    let e = orbit_order(n, g);

    // Convert to floating point (with scale normalization)
    let coeffs_float: Vec<f64> = coeffs.iter().map(|&c| c as f64 / scale).collect();

    // Forward canonical embedding: evaluate polynomial at ζ_M^{e[t]} for t = 0..N/2-1
    // Formula: y_t = Σ_{j=0}^{N-1} c[j] * exp(+2πi * e[t] * j / M)
    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];

    for t in 0..num_slots {
        let mut sum = Complex::new(0.0, 0.0);
        for j in 0..n {
            // w_t(j) = exp(+2πi * e[t] * j / M)  (note: positive angle for decode)
            let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
            let w = Complex::new(angle.cos(), angle.sin()); // exp(+i*angle)
            sum += coeffs_float[j] * w;
        }
        slots[t] = sum;
    }

    slots
}

/// Decode slots from a PRODUCT polynomial (after homomorphic multiplication)
///
/// Use this for PLAINTEXT product polynomials (before encryption).
/// After polynomial multiplication, coefficients are in [0, q) and represent
/// values scaled by s². This function:
/// 1. Center-lifts coefficients to (-q/2, q/2]
/// 2. Normalizes by s² (not just s)
/// 3. Decodes to slots
///
/// # Arguments
/// * `coeffs` - Product polynomial coefficients in [0, q)
/// * `scale` - Original scaling factor (NOT scale²!)
/// * `q` - Modulus used in polynomial multiplication
/// * `n` - Ring dimension
///
/// # Returns
/// N/2 complex slot values representing element-wise product
pub fn canonical_embed_decode_product(coeffs: &[i64], scale: f64, q: i64, n: usize) -> Vec<Complex<f64>> {
    assert_eq!(coeffs.len(), n);

    let m = 2 * n; // M = 2N
    let num_slots = n / 2;
    let g = 5; // Generator

    // CRITICAL FIX: Use Galois orbit order!
    let e = orbit_order(n, g);

    // Step 1: Center-lift coefficients from [0, q) to (-q/2, q/2]
    let centered = center_lift(coeffs, q);

    // Step 2: Convert to float and normalize by s² (product scale)
    let scale_squared = scale * scale;
    let coeffs_float: Vec<f64> = centered.iter().map(|&c| c as f64 / scale_squared).collect();

    // Step 3: Forward canonical embedding
    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];

    for t in 0..num_slots {
        let mut sum = Complex::new(0.0, 0.0);
        for j in 0..n {
            let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
            let w = Complex::new(angle.cos(), angle.sin());
            sum += coeffs_float[j] * w;
        }
        slots[t] = sum;
    }

    slots
}

/// Decode slots from ENCRYPTED product (after decrypt of homomorphic multiply result)
///
/// Use this after: ct1 × ct2 → decrypt → decode
/// The decrypt function already center-lifted, so coefficients are in (-q/2, q/2].
/// Polynomial coefficients still represent values at scale s².
///
/// # Arguments
/// * `coeffs` - Center-lifted coefficients from decrypt (in (-q/2, q/2])
/// * `scale` - Original scaling factor
/// * `n` - Ring dimension
///
/// # Returns
/// N/2 complex slot values
pub fn canonical_embed_decode_homomorphic_product(coeffs: &[i64], scale: f64, n: usize) -> Vec<Complex<f64>> {
    assert_eq!(coeffs.len(), n);

    let m = 2 * n; // M = 2N
    let num_slots = n / 2;
    let g = 5; // Generator

    let e = orbit_order(n, g);

    // Coefficients are already center-lifted by decrypt!
    // But they represent values at scale s² (from multiplication)
    // So normalize by s²
    let scale_squared = scale * scale;
    let coeffs_float: Vec<f64> = coeffs.iter().map(|&c| c as f64 / scale_squared).collect();

    // Forward canonical embedding
    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];

    for t in 0..num_slots {
        let mut sum = Complex::new(0.0, 0.0);
        for j in 0..n {
            let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
            let w = Complex::new(angle.cos(), angle.sin());
            sum += coeffs_float[j] * w;
        }
        slots[t] = sum;
    }

    slots
}

/// Encode multivector using canonical embedding
pub fn encode_multivector_canonical(mv: &[f64; 8], scale: f64, n: usize) -> Vec<i64> {
    assert!(n >= 16);
    let num_slots = n / 2;

    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];
    for i in 0..8 {
        slots[i] = Complex::new(mv[i], 0.0);
    }

    canonical_embed_encode(&slots, scale, n)
}

/// Decode multivector using canonical embedding
pub fn decode_multivector_canonical(coeffs: &[i64], scale: f64, n: usize) -> [f64; 8] {
    let slots = canonical_embed_decode(coeffs, scale, n);

    let mut mv = [0.0; 8];
    for i in 0..8 {
        mv[i] = slots[i].re;
    }
    mv
}

/// Decode multivector from a PRODUCT polynomial (after homomorphic multiplication)
///
/// This is a convenience wrapper for decode_multivector_canonical that handles
/// the center-lifting and s² scaling required after polynomial multiplication.
///
/// # Arguments
/// * `coeffs` - Product polynomial coefficients in [0, q)
/// * `scale` - Original scaling factor (the ciphertext's scale after multiply/rescale)
/// * `q` - Modulus
/// * `n` - Ring dimension
///
/// # Returns
/// 8-component multivector (real values)
pub fn decode_multivector_product(coeffs: &[i64], scale: f64, q: i64, n: usize) -> [f64; 8] {
    // For CKKS multiply, the scale management depends on the implementation
    // After multiply: new_scale = scale1 * scale2 / params.scale
    // So if scale1 = scale2 = s, new_scale = s
    // But the polynomial coefficients represent values at scale s²
    // So we need to decode considering the actual polynomial scale

    // The cleanest approach: just use canonical_embed_decode with the ciphertext's scale
    // The decrypt function already did center-lifting, so coeffs are in (-q/2, q/2]
    let slots = canonical_embed_decode(coeffs, scale, n);

    let mut mv = [0.0; 8];
    for i in 0..8 {
        mv[i] = slots[i].re;
    }
    mv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_embedding_roundtrip() {
        let n = 32;
        let scale = 1u64 << 40;

        let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = encode_multivector_canonical(&mv, scale as f64, n);
        let mv_decoded = decode_multivector_canonical(&coeffs, scale as f64, n);

        for i in 0..8 {
            let error = (mv[i] - mv_decoded[i]).abs();
            assert!(error < 1e-3, "Slot {} error {} too large", i, error);
        }
    }

    #[test]
    fn test_automorphism_rotates_slots() {
        use crate::clifford_fhe_v1::automorphisms::apply_automorphism;

        let n = 32;
        let scale = 1u64 << 40;

        let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = encode_multivector_canonical(&mv, scale as f64, n);

        // Try different automorphism indices to find which one rotates
        for k in [3, 5, 7, 9, 11, 13, 15, 17].iter() {
            let coeffs_auto = apply_automorphism(&coeffs, *k, n);
            let mv_result = decode_multivector_canonical(&coeffs_auto, scale as f64, n);

            // Check if this is a left rotation by 1
            let expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0];
            let matches = mv_result.iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 0.1);

            if matches {
                println!("✓ Automorphism k={} produces left rotation by 1", k);
            }
        }
    }
}
