//! SIMD Slot Encoding for CKKS using RustFFT
//!
//! This module implements proper CKKS SIMD slot encoding using the battle-tested
//! `rustfft` library for FFT transforms. This ensures correctness and performance.
//!
//! # Mathematical Foundation
//!
//! The polynomial ring R = Z[x]/(Φ_M(x)) can be viewed as having N/2 complex slots:
//! ```text
//! R ≅ C^(N/2)  via Chinese Remainder Theorem
//! ```
//!
//! Where M = 2N and Φ_M is the M-th cyclotomic polynomial.
//!
//! # Encoding Process
//!
//! To encode vector [z₀, z₁, ..., z_{N/2-1}] into polynomial p(x):
//! 1. Create complex vector with proper conjugate symmetry
//! 2. Apply inverse FFT to get polynomial coefficients
//! 3. Scale and round for fixed-point arithmetic
//!
//! # Implementation
//!
//! Uses `rustfft` for efficient, correct FFT computation instead of manual implementation.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Encode multivector into SIMD slots
///
/// Takes an 8-component multivector and encodes it into the first 8 slots
/// of a CKKS ciphertext. Remaining slots are set to zero.
///
/// # Arguments
/// * `mv` - Multivector components [scalar, e1, e2, e3, e12, e13, e23, e123]
/// * `scale` - Scaling factor for fixed-point encoding
/// * `n` - Ring dimension (must be power of 2, typically 4096-32768)
///
/// # Returns
/// Vector of scaled polynomial coefficients ready for CKKS encryption
pub fn encode_multivector_slots(mv: &[f64; 8], scale: f64, n: usize) -> Vec<i64> {
    assert!(n.is_power_of_two(), "Ring dimension must be power of 2");
    assert!(n >= 16, "Ring dimension too small for 8 slots");

    let num_slots = n / 2;

    // Step 1: Create complex slot vector
    // For CKKS with real values, we use conjugate symmetry
    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];

    // Place multivector components in first 8 slots (as real values)
    for i in 0..8 {
        slots[i] = Complex::new(mv[i], 0.0);
    }

    // Step 2: Convert slots to coefficients via inverse FFT
    slots_to_coefficients(&slots, scale, n)
}

/// Decode SIMD slots back to multivector
///
/// Extracts the first 8 slots from a polynomial and returns them as a multivector.
///
/// # Arguments
/// * `coeffs` - Polynomial coefficients
/// * `scale` - Scaling factor used during encoding
/// * `n` - Ring dimension
///
/// # Returns
/// Multivector with 8 components
pub fn decode_multivector_slots(coeffs: &[i64], scale: f64, n: usize) -> [f64; 8] {
    assert_eq!(coeffs.len(), n, "Coefficient vector must have length n");

    // Convert coefficients to slots via forward FFT
    let slots = coefficients_to_slots(coeffs, scale, n);

    // Extract first 8 slots (real parts only)
    let mut mv = [0.0f64; 8];
    for i in 0..8 {
        mv[i] = slots[i].re;
    }

    mv
}

/// Convert SIMD slots to polynomial coefficients using FFT
///
/// This is the core CKKS encoding operation. For a vector of N/2 complex slots,
/// we compute the polynomial coefficients via FFT.
///
/// # Implementation Notes
///
/// CKKS uses a special encoding based on:
/// 1. Slots → Extended vector with conjugate symmetry (length N)
/// 2. Inverse FFT of extended vector
/// 3. Real parts give polynomial coefficients
///
/// For simplicity with real-valued multivectors, we use a direct approach:
/// - Treat slots as complex numbers (real values only for MVs)
/// - Use standard inverse FFT
/// - Scale for fixed-point arithmetic
pub fn slots_to_coefficients(slots: &[Complex<f64>], scale: f64, n: usize) -> Vec<i64> {
    let num_slots = n / 2;
    assert_eq!(
        slots.len(),
        num_slots,
        "Must have N/2 slots for ring dimension N"
    );

    // For CKKS, we need to extend slots to length N with conjugate symmetry
    // For real inputs: extended[i] = conj(extended[N-i])
    let mut extended = vec![Complex::new(0.0, 0.0); n];

    // Copy slots to first half
    for i in 0..num_slots {
        extended[i] = slots[i];
    }

    // Conjugate symmetry for second half
    // extended[N/2 + i] = conj(extended[N/2 - i]) for i = 1..N/2-1
    // This ensures inverse FFT produces real values
    for i in 1..num_slots {
        extended[n - i] = slots[i].conj();
    }

    // Apply inverse FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);
    fft.process(&mut extended);

    // Extract real parts, scale, and round to integers
    // Note: inverse FFT from rustfft doesn't normalize, so we divide by n
    let mut coeffs = vec![0i64; n];
    for i in 0..n {
        let value = extended[i].re / (n as f64) * scale;
        coeffs[i] = value.round() as i64;
    }

    coeffs
}

/// Convert polynomial coefficients to SIMD slots using FFT
///
/// This is the core CKKS decoding operation.
///
/// # Implementation
///
/// 1. Unscale coefficients to floating point
/// 2. Apply forward FFT
/// 3. Extract first N/2 complex values as slots
pub fn coefficients_to_slots(coeffs: &[i64], scale: f64, n: usize) -> Vec<Complex<f64>> {
    assert_eq!(coeffs.len(), n, "Must have N coefficients");

    let num_slots = n / 2;

    // Convert coefficients to complex numbers (unscale)
    let mut complex_coeffs = vec![Complex::new(0.0, 0.0); n];
    for i in 0..n {
        complex_coeffs[i] = Complex::new(coeffs[i] as f64 / scale, 0.0);
    }

    // Apply forward FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut complex_coeffs);

    // Extract first N/2 slots
    // Note: forward FFT from rustfft doesn't normalize
    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];
    for i in 0..num_slots {
        slots[i] = complex_coeffs[i];
    }

    slots
}

/// Create plaintext with value in specific slot
///
/// Useful for masking operations: creates a polynomial that has value 1.0
/// in one slot and 0.0 in all others.
pub fn create_slot_mask(slot_index: usize, scale: f64, n: usize) -> Vec<i64> {
    let num_slots = n / 2;
    assert!(
        slot_index < num_slots,
        "Slot index {} out of range [0, {})",
        slot_index,
        num_slots
    );

    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];
    slots[slot_index] = Complex::new(1.0, 0.0);

    slots_to_coefficients(&slots, scale, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        let error = (a - b).abs();
        assert!(
            error < tol,
            "{}: error {} (got {}, expected {})",
            msg,
            error,
            a,
            b
        );
    }

    #[test]
    fn test_encoding_roundtrip() {
        let n = 64; // Small for testing
        let scale = 2f64.powi(20);

        // Test multivector
        let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Encode
        let coeffs = encode_multivector_slots(&mv, scale, n);
        assert_eq!(coeffs.len(), n);

        // Decode
        let mv_decoded = decode_multivector_slots(&coeffs, scale, n);

        // Check roundtrip accuracy
        for i in 0..8 {
            assert_close(
                mv_decoded[i],
                mv[i],
                1e-5, // Slightly relaxed due to FFT + fixed-point rounding
                &format!("Component {}", i),
            );
        }
    }

    #[test]
    fn test_slot_mask() {
        let n = 64;
        let scale = 2f64.powi(20);

        // Create mask for slot 3
        let mask_coeffs = create_slot_mask(3, scale, n);

        // Decode to verify
        let slots = coefficients_to_slots(&mask_coeffs, scale, n);

        // Slot 3 should be 1.0, others should be ~0.0
        for i in 0..8 {
            let expected = if i == 3 { 1.0 } else { 0.0 };
            assert_close(
                slots[i].re,
                expected,
                1e-5, // Slightly relaxed due to FFT rounding
                &format!("Slot {}", i),
            );
        }
    }

    #[test]
    fn test_slots_to_coefficients_to_slots() {
        let n = 64;
        let scale = 2f64.powi(20);
        let num_slots = n / 2;

        // Create random slots
        let mut original_slots = vec![Complex::new(0.0, 0.0); num_slots];
        for i in 0..8 {
            original_slots[i] = Complex::new(i as f64 + 1.0, 0.0);
        }

        // slots → coefficients → slots
        let coeffs = slots_to_coefficients(&original_slots, scale, n);
        let recovered_slots = coefficients_to_slots(&coeffs, scale, n);

        // Check roundtrip
        for i in 0..8 {
            assert_close(
                recovered_slots[i].re,
                original_slots[i].re,
                1e-5, // Slightly relaxed due to FFT + fixed-point rounding
                &format!("Slot {} real", i),
            );
            assert_close(
                recovered_slots[i].im,
                original_slots[i].im,
                1e-5,
                &format!("Slot {} imag", i),
            );
        }
    }

    #[test]
    fn test_zero_multivector() {
        let n = 64;
        let scale = 2f64.powi(20);

        let mv = [0.0; 8];
        let coeffs = encode_multivector_slots(&mv, scale, n);
        let mv_decoded = decode_multivector_slots(&coeffs, scale, n);

        for i in 0..8 {
            assert!(mv_decoded[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_large_values() {
        let n = 64;
        let scale = 2f64.powi(20);

        // Test with large values
        let mv = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];

        let coeffs = encode_multivector_slots(&mv, scale, n);
        let mv_decoded = decode_multivector_slots(&coeffs, scale, n);

        for i in 0..8 {
            assert_close(
                mv_decoded[i],
                mv[i],
                1e-3, // Slightly relaxed for large values
                &format!("Component {}", i),
            );
        }
    }

    #[test]
    fn test_conjugate_symmetry() {
        // Verify that our encoding produces real coefficients
        let n = 64;
        let scale = 2f64.powi(20);
        let num_slots = n / 2;

        let mut slots = vec![Complex::new(0.0, 0.0); num_slots];
        slots[0] = Complex::new(1.0, 0.0);
        slots[1] = Complex::new(2.0, 0.0);

        let coeffs = slots_to_coefficients(&slots, scale, n);

        // All coefficients should be real (imaginary part ~0)
        // We verify this by checking the roundtrip works perfectly
        let decoded_slots = coefficients_to_slots(&coeffs, scale, n);

        for i in 0..2 {
            assert_close(
                decoded_slots[i].re,
                slots[i].re,
                1e-6,
                &format!("Slot {} real", i),
            );
        }
    }
}
