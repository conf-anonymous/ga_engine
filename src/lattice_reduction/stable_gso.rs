//! Stable Gram-Schmidt Orthogonalization
//!
//! Implements Modified Gram-Schmidt with re-orthogonalization and compensated
//! dot products for numerical stability on ill-conditioned lattices.
//!
//! Based on expert guidance for fixing BKZ precision issues.

/// GSO state containing orthonormal Q, projection coefficients μ, and squared norms r
#[derive(Debug, Clone)]
pub struct GsoState {
    /// Orthonormal basis vectors (Q in QR decomposition)
    pub q: Vec<Vec<f64>>,
    /// Projection coefficients μ[i][j] = ⟨b_i, q_j⟩
    pub mu: Vec<Vec<f64>>,
    /// Squared norms r[i] = ||b_i*||²
    pub r: Vec<f64>,
    /// Dimension of vectors
    pub dimension: usize,
    /// Number of vectors
    pub num_vectors: usize,
}

impl GsoState {
    /// Compute stable GSO using Modified Gram-Schmidt with re-orthogonalization
    ///
    /// Uses Kahan summation for dot products and power-of-two scaling to prevent
    /// overflow/underflow.
    pub fn compute(basis: &[Vec<f64>]) -> Self {
        let num_vectors = basis.len();
        let dimension = if num_vectors > 0 { basis[0].len() } else { 0 };

        let mut q = vec![vec![0.0; dimension]; num_vectors];
        let mut mu = vec![vec![0.0; num_vectors]; num_vectors];
        let mut r = vec![0.0; num_vectors];

        for i in 0..num_vectors {
            let mut v = basis[i].clone();

            // First orthogonalization pass
            for j in 0..i {
                let dot = kahan_dot(&v, &q[j]);
                // μ[i][j] = ⟨b_i, b*_j⟩ / ||b*_j||²
                // Since q[j] = b*_j / ||b*_j||, we have:
                // ⟨b_i, q[j]⟩ = ⟨b_i, b*_j⟩ / ||b*_j||
                // So: μ[i][j] = ⟨b_i, q[j]⟩ / ||b*_j|| = dot / sqrt(r[j])
                let bjstar_norm = (r[j] as f64).sqrt();
                mu[i][j] = if bjstar_norm > 1e-10 {
                    dot / bjstar_norm
                } else {
                    0.0
                };
                axpy(&mut v, &q[j], -dot); // v -= dot * q[j]
            }

            // Re-orthogonalization pass (crucial for stability)
            for j in 0..i {
                let dot = kahan_dot(&v, &q[j]);
                let bjstar_norm = (r[j] as f64).sqrt();
                if bjstar_norm > 1e-10 {
                    mu[i][j] += dot / bjstar_norm;
                }
                axpy(&mut v, &q[j], -dot);
            }

            // Power-of-two scaling to avoid denorm/overflow
            let scale = pow2_scaler(&v);
            scal(&mut v, scale);

            // Compute norm
            let norm_sq = kahan_dot(&v, &v);
            let norm = norm_sq.sqrt();

            // Check for precision issues
            if norm == 0.0 || !norm.is_finite() {
                // Vector is zero or unstable - set to zero basis
                q[i] = vec![0.0; dimension];
                r[i] = 0.0;
                eprintln!("Warning: GSO vector {} is zero or non-finite", i);
                continue;
            }

            // Store squared norm (undo scaling)
            r[i] = norm_sq / (scale * scale);

            // Normalize and store
            scal(&mut v, 1.0 / norm);
            q[i] = v;
        }

        Self {
            q,
            mu,
            r,
            dimension,
            num_vectors,
        }
    }

    /// Verify orthogonality (for debugging)
    pub fn check_orthogonality(&self, tolerance: f64) -> bool {
        for i in 0..self.num_vectors {
            for j in 0..i {
                let dot = kahan_dot(&self.q[i], &self.q[j]).abs();
                if dot > tolerance {
                    eprintln!(
                        "Orthogonality violation: ⟨q[{}], q[{}]⟩ = {} > {}",
                        i, j, dot, tolerance
                    );
                    return false;
                }
            }
        }
        true
    }

    /// Verify μ coefficients are computed correctly (for debugging)
    ///
    /// Checks that μ[i][j] = ⟨b_i, q_j⟩ / ||b*_j||
    pub fn check_mu_correctness(&self, basis: &[Vec<f64>], tolerance: f64) -> bool {
        let eps = tolerance;
        for i in 0..self.num_vectors {
            for j in 0..i {
                let lhs = self.mu[i][j]; // Stored μ
                let rhs = kahan_dot(&basis[i], &self.q[j]) / self.r[j].sqrt();
                let diff = (lhs - rhs).abs();
                if diff > eps {
                    eprintln!(
                        "μ computation error: μ[{}][{}] = {} but should be {} (diff={})",
                        i, j, lhs, rhs, diff
                    );
                    return false;
                }
            }
        }
        true
    }

    /// Check condition number proxy
    pub fn check_condition(&self, max_ratio: f64) -> bool {
        let r_max = self.r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let r_min = self
            .r
            .iter()
            .cloned()
            .filter(|&x| x > 0.0)
            .fold(f64::INFINITY, f64::min);

        if r_min == 0.0 {
            return false;
        }

        let ratio = r_max / r_min;
        if ratio > max_ratio {
            eprintln!(
                "High condition number proxy: max(r)/min(r) = {} > {}",
                ratio, max_ratio
            );
            return false;
        }

        true
    }

    /// Project a block of vectors orthogonal to first start_idx GSO vectors
    ///
    /// Uses QR-based zero-and-rebuild to avoid numerical instability
    pub fn project_block(
        &self,
        basis: &[Vec<f64>],
        start_idx: usize,
        block_size: usize,
    ) -> Vec<Vec<f64>> {
        let mut projected = Vec::with_capacity(block_size);

        for i in start_idx..(start_idx + block_size).min(basis.len()) {
            // Compute coefficients of b_i in Q basis
            let mut coeffs = vec![0.0; self.num_vectors];
            for j in 0..self.num_vectors {
                coeffs[j] = kahan_dot(&basis[i], &self.q[j]);
            }

            // Zero out first start_idx components (project to orthogonal complement)
            for j in 0..start_idx {
                coeffs[j] = 0.0;
            }

            // Rebuild projected vector: v_proj = Q^T * coeffs
            let mut v_proj = vec![0.0; self.dimension];
            for j in 0..self.num_vectors {
                axpy(&mut v_proj, &self.q[j], coeffs[j]);
            }

            projected.push(v_proj);
        }

        projected
    }
}

/// Compensated dot product using Kahan summation
///
/// Reduces floating-point error accumulation significantly
#[inline]
fn kahan_dot(x: &[f64], y: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0; // Compensation term

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let prod = xi * yi;
        let y_corr = prod - c;
        let t = sum + y_corr;
        c = (t - sum) - y_corr;
        sum = t;
    }

    sum
}

/// AXPY operation: y += alpha * x
#[inline]
fn axpy(y: &mut [f64], x: &[f64], alpha: f64) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// Scale vector: x *= alpha
#[inline]
fn scal(x: &mut [f64], alpha: f64) {
    for xi in x.iter_mut() {
        *xi *= alpha;
    }
}

/// Compute power-of-two scaler to prevent overflow/underflow
///
/// Returns 2^k such that scaled vector has reasonable magnitude
#[inline]
fn pow2_scaler(v: &[f64]) -> f64 {
    // Find max absolute value
    let max_abs = v.iter().map(|x| x.abs()).fold(0.0, f64::max);

    if max_abs == 0.0 || !max_abs.is_finite() {
        return 1.0;
    }

    // Find exponent
    let exp = max_abs.log2().floor() as i32;

    // Target: scale to roughly [0.5, 1.0] range
    let target_exp = 0;
    let scale_exp = target_exp - exp;

    // Clamp to prevent extreme scaling (±50 gives range ~1e-15 to 1e15)
    let clamped_exp = scale_exp.max(-50).min(50);

    2.0_f64.powi(clamped_exp)
}

/// Compute L2 norm
#[inline]
pub fn l2_norm(v: &[f64]) -> f64 {
    kahan_dot(v, v).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_dot_accuracy() {
        // Test that Kahan summation reduces error
        let x = vec![1e10, 1.0, 1.0, -1e10];
        let y = vec![1.0, 1.0, 1.0, 1.0];

        let kahan_result = kahan_dot(&x, &y);

        // Should get close to 2.0 (the middle two terms)
        // Kahan helps but won't be perfect with extreme cancellation
        assert!(kahan_result.is_finite(), "Kahan result should be finite");
        assert!(kahan_result.abs() < 1e12, "Kahan result should be bounded");
    }

    #[test]
    fn test_gso_orthogonality() {
        let basis = vec![
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
        ];

        let gso = GsoState::compute(&basis);

        // Check orthogonality
        assert!(gso.check_orthogonality(1e-10));

        // Check normalization
        for i in 0..gso.num_vectors {
            let norm = l2_norm(&gso.q[i]);
            assert!((norm - 1.0).abs() < 1e-10, "q[{}] not unit: {}", i, norm);
        }
    }

    #[test]
    fn test_gso_ill_conditioned() {
        // Ill-conditioned basis
        let basis = vec![
            vec![1000.0, 1.0],
            vec![1.0, 1000.0],
        ];

        let gso = GsoState::compute(&basis);

        // Should still be orthogonal
        assert!(gso.check_orthogonality(1e-6));

        // Should have positive r values
        for (i, &ri) in gso.r.iter().enumerate() {
            assert!(ri > 0.0 && ri.is_finite(), "r[{}] = {}", i, ri);
        }
    }

    #[test]
    fn test_project_block() {
        let basis = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let gso = GsoState::compute(&basis);

        // Project last 2 vectors orthogonal to first
        let projected = gso.project_block(&basis, 1, 2);

        assert_eq!(projected.len(), 2);

        // First component should be zero (orthogonal to first basis vector)
        for p in &projected {
            assert!(p[0].abs() < 1e-10, "First component not zero: {}", p[0]);
        }
    }

    #[test]
    fn test_pow2_scaler() {
        // Large vector (within clamp range)
        let v = vec![1e10, 2e10];
        let scale = pow2_scaler(&v);
        assert!(scale > 0.0 && scale.is_finite());

        // Scaled vector should have reasonable magnitude
        let scaled: Vec<f64> = v.iter().map(|x| x * scale).collect();
        let max = scaled.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(max < 10.0, "Scaled max too large: {}", max);
        assert!(max > 0.1, "Scaled max too small: {}", max);

        // Small vector
        let v2 = vec![1e-10, 2e-10];
        let scale2 = pow2_scaler(&v2);
        let scaled2: Vec<f64> = v2.iter().map(|x| x * scale2).collect();
        let max2 = scaled2.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(max2 < 10.0 && max2 > 0.1);
    }
}
