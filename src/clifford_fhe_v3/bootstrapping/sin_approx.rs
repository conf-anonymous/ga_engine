//! Sine Polynomial Approximation
//!
//! Computes Chebyshev/Taylor polynomial coefficients for sine function.
//! Used in EvalMod to approximate modular reduction: x mod q ≈ x - (q/2π) · sin(2πx/q)

use std::f64::consts::PI;

/// Compute Chebyshev polynomial coefficients for sin(x) on [-π, π]
///
/// Returns coefficients [c0, c1, c2, ..., c_degree]
///
/// # Arguments
///
/// * `degree` - Degree of polynomial (must be odd, >= 5)
///
/// # Returns
///
/// Vector of coefficients where result[i] is coefficient for x^i
///
/// # Example
///
/// ```
/// use ga_engine::clifford_fhe_v3::bootstrapping::chebyshev_sin_coeffs;
///
/// let coeffs = chebyshev_sin_coeffs(15);
/// assert_eq!(coeffs.len(), 16);
/// ```
pub fn chebyshev_sin_coeffs(degree: usize) -> Vec<f64> {
    assert!(degree >= 5, "Need at least degree 5 for reasonable accuracy");
    assert!(degree % 2 == 1, "Sine is odd function, use odd degree");

    // Compute Chebyshev approximation for sin(x) on [-π, π]
    //
    // Strategy:
    // 1. Map [-π, π] to [-1, 1] via t = x/π
    // 2. Compute Chebyshev coefficients for sin(πt) on [-1, 1]
    // 3. Convert Chebyshev basis to monomial basis
    // 4. Adjust coefficients for the scaling x = πt

    // Step 1-2: Compute Chebyshev coefficients for sin(πt) on [-1, 1]
    // Using Chebyshev-Gauss quadrature with N points:
    // c_k ≈ (2/N) Σ_{j=0}^{N-1} f(cos(π(j+0.5)/N)) * cos(πk(j+0.5)/N)
    // (c_0 uses factor 1/N instead of 2/N)

    let n_points = 128; // Enough points for accurate integration
    let mut cheb_coeffs = vec![0.0; degree + 1];

    for k in 0..=degree {
        let mut sum = 0.0;
        for j in 0..n_points {
            let theta = PI * (j as f64 + 0.5) / n_points as f64;
            let t = theta.cos(); // Chebyshev node in [-1, 1]
            let f_t = (PI * t).sin(); // sin(πt)
            sum += f_t * (k as f64 * theta).cos();
        }
        cheb_coeffs[k] = if k == 0 {
            sum / n_points as f64
        } else {
            2.0 * sum / n_points as f64
        };
    }

    // Since sin is odd, even Chebyshev coefficients should be ~0
    // (T_0, T_2, T_4, ... are even functions)
    for k in (0..=degree).step_by(2) {
        cheb_coeffs[k] = 0.0;
    }

    // Step 3: Convert Chebyshev basis to monomial basis
    // T_k(t) can be expressed as a polynomial in t
    // We need coefficients for: Σ c_k T_k(t) = Σ a_j t^j
    let mono_coeffs_t = chebyshev_to_monomial(&cheb_coeffs);

    // Step 4: Adjust for scaling x = πt, so t = x/π
    // If p(t) = Σ a_j t^j, then p(x/π) = Σ a_j (x/π)^j = Σ (a_j/π^j) x^j
    let mut mono_coeffs_x = vec![0.0; degree + 1];
    let mut pi_power = 1.0;
    for j in 0..=degree {
        mono_coeffs_x[j] = mono_coeffs_t[j] / pi_power;
        pi_power *= PI;
    }

    // Ensure even coefficients are exactly zero (sine is odd)
    for j in (0..=degree).step_by(2) {
        mono_coeffs_x[j] = 0.0;
    }

    mono_coeffs_x
}

/// Convert Chebyshev polynomial coefficients to monomial (power) basis
///
/// Given coefficients c_k for Σ c_k T_k(x), compute coefficients a_j for Σ a_j x^j
///
/// Uses the recurrence relation for Chebyshev polynomials:
/// T_0(x) = 1
/// T_1(x) = x
/// T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
fn chebyshev_to_monomial(cheb_coeffs: &[f64]) -> Vec<f64> {
    let n = cheb_coeffs.len();
    if n == 0 {
        return vec![];
    }

    // Store monomial coefficients for each T_k
    // t_polys[k] contains coefficients [a_0, a_1, ..., a_k] for T_k(x)
    let mut t_polys: Vec<Vec<f64>> = Vec::with_capacity(n);

    // T_0(x) = 1
    t_polys.push(vec![1.0]);

    if n > 1 {
        // T_1(x) = x
        t_polys.push(vec![0.0, 1.0]);
    }

    // T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)
    for k in 2..n {
        let mut new_poly = vec![0.0; k + 1];

        // 2x T_{k-1}(x): multiply T_{k-1} by x and scale by 2
        let prev = &t_polys[k - 1];
        for (i, &coeff) in prev.iter().enumerate() {
            new_poly[i + 1] += 2.0 * coeff;
        }

        // Subtract T_{k-2}(x)
        let prev_prev = &t_polys[k - 2];
        for (i, &coeff) in prev_prev.iter().enumerate() {
            new_poly[i] -= coeff;
        }

        t_polys.push(new_poly);
    }

    // Combine: Σ c_k T_k(x) = Σ_j (Σ_k c_k * t_polys[k][j]) x^j
    let mut result = vec![0.0; n];
    for (k, &c_k) in cheb_coeffs.iter().enumerate() {
        if c_k.abs() > 1e-15 {
            for (j, &t_kj) in t_polys[k].iter().enumerate() {
                result[j] += c_k * t_kj;
            }
        }
    }

    result
}

/// Compute Taylor series coefficients for sin(x)
///
/// sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
///
/// # Arguments
///
/// * `degree` - Maximum degree of polynomial
///
/// # Returns
///
/// Vector of coefficients where result[i] is coefficient for x^i
///
/// # Example
///
/// ```
/// use ga_engine::clifford_fhe_v3::bootstrapping::taylor_sin_coeffs;
///
/// let coeffs = taylor_sin_coeffs(7);
/// // sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
/// assert!((coeffs[1] - 1.0).abs() < 1e-10);  // x term
/// assert!((coeffs[3] + 1.0/6.0).abs() < 1e-10);  // -x³/6 term
/// ```
pub fn taylor_sin_coeffs(degree: usize) -> Vec<f64> {
    let mut coeffs = vec![0.0; degree + 1];

    // sin(x) has only odd powers
    for k in 0..=(degree / 2) {
        let power = 2 * k + 1;
        if power <= degree {
            // Coefficient for x^power is (-1)^k / (2k+1)!
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            let factorial = factorial(power);
            coeffs[power] = sign / factorial;
        }
    }

    coeffs
}

/// Compute factorial
///
/// # Arguments
///
/// * `n` - Input value
///
/// # Returns
///
/// n! as f64
///
/// # Example
///
/// ```
/// # use ga_engine::clifford_fhe_v3::bootstrapping::sin_approx::factorial;
/// assert_eq!(factorial(5), 120.0);
/// ```
fn factorial(n: usize) -> f64 {
    if n <= 1 {
        1.0
    } else {
        (2..=n).fold(1.0, |acc, x| acc * x as f64)
    }
}

/// Evaluate polynomial with given coefficients
///
/// p(x) = c0 + c1*x + c2*x² + ... + cn*x^n
///
/// Uses Horner's method for numerical stability.
///
/// # Arguments
///
/// * `coeffs` - Polynomial coefficients [c0, c1, ..., cn]
/// * `x` - Point at which to evaluate
///
/// # Returns
///
/// Value of polynomial at x
///
/// # Example
///
/// ```
/// use ga_engine::clifford_fhe_v3::bootstrapping::eval_polynomial;
///
/// // Evaluate p(x) = 1 + 2x + 3x²
/// let coeffs = vec![1.0, 2.0, 3.0];
/// let result = eval_polynomial(&coeffs, 2.0);
/// assert_eq!(result, 1.0 + 2.0*2.0 + 3.0*4.0);  // 17.0
/// ```
pub fn eval_polynomial(coeffs: &[f64], x: f64) -> f64 {
    // Use Horner's method for numerical stability
    let mut result = 0.0;
    for &coeff in coeffs.iter().rev() {
        result = result * x + coeff;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(7), 5040.0);
    }

    #[test]
    fn test_taylor_sin_coeffs() {
        let coeffs = taylor_sin_coeffs(7);
        assert_eq!(coeffs.len(), 8);

        // Check specific coefficients
        assert!((coeffs[0] - 0.0).abs() < 1e-10, "Constant term should be 0");
        assert!((coeffs[1] - 1.0).abs() < 1e-10, "x term should be 1");
        assert!((coeffs[2] - 0.0).abs() < 1e-10, "x² term should be 0");
        assert!((coeffs[3] + 1.0/6.0).abs() < 1e-10, "x³ term should be -1/6");
        assert!((coeffs[4] - 0.0).abs() < 1e-10, "x⁴ term should be 0");
        assert!((coeffs[5] - 1.0/120.0).abs() < 1e-10, "x⁵ term should be 1/120");
    }

    #[test]
    fn test_eval_polynomial() {
        // Test p(x) = 1 + 2x + 3x²
        let coeffs = vec![1.0, 2.0, 3.0];
        let result = eval_polynomial(&coeffs, 2.0);
        assert_eq!(result, 17.0);  // 1 + 4 + 12

        // Test p(x) = 0
        let coeffs = vec![0.0, 0.0, 0.0];
        let result = eval_polynomial(&coeffs, 5.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_taylor_sin_accuracy() {
        let coeffs = taylor_sin_coeffs(31);

        // Test on [-π, π]
        let test_points = vec![
            0.0,
            PI / 4.0,
            PI / 2.0,
            3.0 * PI / 4.0,
            PI,
            -PI / 4.0,
            -PI / 2.0,
        ];

        for x in test_points {
            let approx = eval_polynomial(&coeffs, x);
            let exact = x.sin();
            let error = (approx - exact).abs();

            println!("sin({:.4}) = {:.6} (approx: {:.6}, error: {:.6})",
                     x, exact, approx, error);

            assert!(error < 1e-6, "Taylor approximation error too large: {} at x={}", error, x);
        }
    }

    #[test]
    fn test_chebyshev_sin_coeffs() {
        let coeffs = chebyshev_sin_coeffs(15);
        assert_eq!(coeffs.len(), 16);

        // Sine has no even powers
        for k in 0..coeffs.len() {
            if k % 2 == 0 {
                assert!(coeffs[k].abs() < 1e-14, "Even coefficient {} should be zero, got {}", k, coeffs[k]);
            }
        }

        // The first coefficient (x term) should be close to 1 but not exactly
        // (Chebyshev distributes error, so c_1 ≠ 1 exactly unlike Taylor)
        assert!((coeffs[1] - 1.0).abs() < 0.1, "x coefficient should be near 1, got {}", coeffs[1]);
    }

    #[test]
    fn test_chebyshev_sin_accuracy() {
        let coeffs = chebyshev_sin_coeffs(15);

        // Test on [-π, π] including boundary points
        let test_points = vec![
            0.0,
            PI / 6.0,
            PI / 4.0,
            PI / 3.0,
            PI / 2.0,
            2.0 * PI / 3.0,
            3.0 * PI / 4.0,
            5.0 * PI / 6.0,
            PI,
            -PI / 4.0,
            -PI / 2.0,
            -3.0 * PI / 4.0,
            -PI,
        ];

        let mut max_error = 0.0f64;
        for x in &test_points {
            let approx = eval_polynomial(&coeffs, *x);
            let exact = x.sin();
            let error = (approx - exact).abs();
            max_error = max_error.max(error);

            println!("Chebyshev sin({:.4}) = {:.10} (approx: {:.10}, error: {:.2e})",
                     x, exact, approx, error);
        }

        // Chebyshev degree-15 should achieve ~1e-6 max error on [-π, π]
        assert!(max_error < 1e-5, "Chebyshev max error too large: {:.2e}", max_error);
    }

    #[test]
    fn test_chebyshev_vs_taylor_at_boundaries() {
        // Chebyshev should have better (or comparable) worst-case error than Taylor
        // especially near the boundaries of [-π, π]
        let degree = 15;
        let cheb_coeffs = chebyshev_sin_coeffs(degree);
        let taylor_coeffs = taylor_sin_coeffs(degree);

        // Test near the boundaries where Taylor typically struggles
        let boundary_points = vec![
            0.95 * PI,
            0.99 * PI,
            PI,
            -0.95 * PI,
            -0.99 * PI,
            -PI,
        ];

        let mut cheb_max_error = 0.0f64;
        let mut taylor_max_error = 0.0f64;

        for x in &boundary_points {
            let exact = x.sin();
            let cheb_approx = eval_polynomial(&cheb_coeffs, *x);
            let taylor_approx = eval_polynomial(&taylor_coeffs, *x);

            let cheb_error = (cheb_approx - exact).abs();
            let taylor_error = (taylor_approx - exact).abs();

            cheb_max_error = cheb_max_error.max(cheb_error);
            taylor_max_error = taylor_max_error.max(taylor_error);

            println!("x={:.4}: Chebyshev error={:.2e}, Taylor error={:.2e}",
                     x, cheb_error, taylor_error);
        }

        println!("Max errors - Chebyshev: {:.2e}, Taylor: {:.2e}", cheb_max_error, taylor_max_error);

        // Chebyshev should be at least as good as Taylor at boundaries
        // (In practice, Chebyshev distributes error more evenly)
        assert!(cheb_max_error <= taylor_max_error * 1.5,
                "Chebyshev should not be significantly worse than Taylor at boundaries");
    }

    #[test]
    fn test_chebyshev_to_monomial() {
        // Test with known Chebyshev polynomials:
        // T_0(x) = 1 -> [1]
        // T_1(x) = x -> [0, 1]
        // T_2(x) = 2x² - 1 -> [-1, 0, 2]
        // T_3(x) = 4x³ - 3x -> [0, -3, 0, 4]

        // Test: 1*T_0 + 0*T_1 + 0*T_2 = 1 -> [1]
        let cheb = vec![1.0];
        let mono = chebyshev_to_monomial(&cheb);
        assert!((mono[0] - 1.0).abs() < 1e-10);

        // Test: 0*T_0 + 1*T_1 = x -> [0, 1]
        let cheb = vec![0.0, 1.0];
        let mono = chebyshev_to_monomial(&cheb);
        assert!((mono[0] - 0.0).abs() < 1e-10);
        assert!((mono[1] - 1.0).abs() < 1e-10);

        // Test: 0*T_0 + 0*T_1 + 1*T_2 = 2x² - 1 -> [-1, 0, 2]
        let cheb = vec![0.0, 0.0, 1.0];
        let mono = chebyshev_to_monomial(&cheb);
        assert!((mono[0] - (-1.0)).abs() < 1e-10, "T_2 constant term should be -1, got {}", mono[0]);
        assert!((mono[1] - 0.0).abs() < 1e-10, "T_2 linear term should be 0, got {}", mono[1]);
        assert!((mono[2] - 2.0).abs() < 1e-10, "T_2 quadratic term should be 2, got {}", mono[2]);

        // Test: 0*T_0 + 0*T_1 + 0*T_2 + 1*T_3 = 4x³ - 3x -> [0, -3, 0, 4]
        let cheb = vec![0.0, 0.0, 0.0, 1.0];
        let mono = chebyshev_to_monomial(&cheb);
        assert!((mono[0] - 0.0).abs() < 1e-10);
        assert!((mono[1] - (-3.0)).abs() < 1e-10, "T_3 linear term should be -3, got {}", mono[1]);
        assert!((mono[2] - 0.0).abs() < 1e-10);
        assert!((mono[3] - 4.0).abs() < 1e-10, "T_3 cubic term should be 4, got {}", mono[3]);
    }

    #[test]
    #[should_panic(expected = "Need at least degree 5")]
    fn test_chebyshev_degree_too_small() {
        chebyshev_sin_coeffs(3);
    }

    #[test]
    #[should_panic(expected = "Sine is odd function")]
    fn test_chebyshev_even_degree() {
        chebyshev_sin_coeffs(8);
    }
}
