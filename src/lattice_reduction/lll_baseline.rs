//! Standard LLL (Lenstra-Lenstra-Lovász) Lattice Reduction
//!
//! This is the baseline implementation using standard Gram-Schmidt orthogonalization
//! with μ coefficient storage (O(n²) memory).
//!
//! This serves as the reference implementation to compare against the GA-accelerated version.
//!
//! # Algorithm
//!
//! LLL reduces a lattice basis B = [b₁, b₂, ..., bₙ] to a "reduced" basis with shorter vectors.
//!
//! **Steps:**
//! 1. Compute Gram-Schmidt orthogonalization (GSO): B* = [b₁*, b₂*, ..., bₙ*]
//! 2. Compute projection coefficients: μᵢⱼ = ⟨bᵢ, bⱼ*⟩ / ⟨bⱼ*, bⱼ*⟩
//! 3. Size reduction: ensure |μᵢⱼ| ≤ 1/2 for all i > j
//! 4. Lovász condition: check ||bᵢ*||² ≥ (δ - μ²ᵢ,ᵢ₋₁)||bᵢ₋₁*||²
//! 5. If violated, swap bᵢ and bᵢ₋₁, recompute GSO
//! 6. Repeat until all conditions satisfied
//!
//! # Parameters
//!
//! - **δ**: Lovász constant, typically 0.75 ≤ δ < 1 (common: 0.99)
//! - Higher δ → better reduction, more work
//! - Lower δ → faster, less reduction
//!
//! # Complexity
//!
//! - Time: O(n⁴ log B) where B is max basis vector norm
//! - Space: O(n²) for μ matrix + O(n²) for orthogonal basis
//!
//! # References
//!
//! - Original paper: Lenstra, Lenstra, Lovász (1982)
//! - Modern reference: fplll library

use std::fmt;

/// LLL lattice reduction using standard GSO
pub struct LLL {
    /// Original basis vectors (will be modified during reduction)
    basis: Vec<Vec<f64>>,

    /// Dimension of the lattice
    dimension: usize,

    /// Number of basis vectors
    num_vectors: usize,

    /// Lovász constant (typically 0.99)
    delta: f64,

    /// Gram-Schmidt orthogonal basis
    orthogonal_basis: Vec<Vec<f64>>,

    /// Projection coefficients μ[i][j] = ⟨b_i, b*_j⟩ / ||b*_j||²
    mu: Vec<Vec<f64>>,

    /// Norms squared of orthogonal basis vectors ||b*_i||²
    b_star_norms_sq: Vec<f64>,

    /// Statistics
    stats: LLLStats,
}

/// Statistics collected during LLL reduction
#[derive(Debug, Clone, Default)]
pub struct LLLStats {
    /// Number of size reduction operations
    pub size_reductions: usize,

    /// Number of basis vector swaps
    pub swaps: usize,

    /// Number of GSO recomputations
    pub gso_updates: usize,

    /// Total operations (for comparison with GA approach)
    pub total_operations: usize,
}

impl LLL {
    /// Create new LLL reducer
    ///
    /// # Arguments
    ///
    /// * `basis` - Initial lattice basis (column vectors)
    /// * `delta` - Lovász constant (typically 0.99)
    ///
    /// # Panics
    ///
    /// Panics if basis is empty or vectors have inconsistent dimensions.
    pub fn new(basis: Vec<Vec<f64>>, delta: f64) -> Self {
        assert!(!basis.is_empty(), "Basis must be non-empty");
        assert!(delta > 0.25 && delta < 1.0, "Delta must be in (0.25, 1.0)");

        let num_vectors = basis.len();
        let dimension = basis[0].len();

        // Verify all vectors have same dimension
        for (i, v) in basis.iter().enumerate() {
            assert_eq!(v.len(), dimension, "Vector {} has wrong dimension", i);
        }

        let orthogonal_basis = vec![vec![0.0; dimension]; num_vectors];
        let mu = vec![vec![0.0; num_vectors]; num_vectors];
        let b_star_norms_sq = vec![0.0; num_vectors];

        let mut lll = Self {
            basis,
            dimension,
            num_vectors,
            delta,
            orthogonal_basis,
            mu,
            b_star_norms_sq,
            stats: LLLStats::default(),
        };

        // Initial GSO computation
        lll.compute_gso(0);
        lll.stats.gso_updates += 1;

        lll
    }

    /// Run LLL reduction algorithm
    ///
    /// Modifies the basis in-place to produce a reduced basis.
    pub fn reduce(&mut self) {
        let mut k = 1;  // Current index

        while k < self.num_vectors {
            // Size reduce b_k with respect to b_0, ..., b_{k-1}
            for j in (0..k).rev() {
                self.size_reduce(k, j);
            }

            // Check Lovász condition
            if self.lovasz_condition(k) {
                // Condition satisfied, move to next vector
                k += 1;
            } else {
                // Swap b_k and b_{k-1}
                self.swap_vectors(k, k - 1);
                self.stats.swaps += 1;

                // Update GSO from position k-1
                self.compute_gso(k - 1);
                self.stats.gso_updates += 1;

                // Move back (unless we're at the start)
                k = k.saturating_sub(1).max(1);
            }
        }
    }

    /// Size reduce b_k with respect to b_j
    ///
    /// Ensures |μ_kj| ≤ 1/2 by subtracting integer multiples of b_j from b_k
    fn size_reduce(&mut self, k: usize, j: usize) {
        if self.mu[k][j].abs() > 0.5 {
            let q = self.mu[k][j].round();

            // b_k := b_k - q * b_j
            for i in 0..self.dimension {
                self.basis[k][i] -= q * self.basis[j][i];
            }

            // Update μ coefficients
            for i in 0..=j {
                self.mu[k][i] -= q * self.mu[j][i];
            }

            self.stats.size_reductions += 1;
            self.stats.total_operations += self.dimension;  // Vector subtraction
        }
    }

    /// Check Lovász condition at index k
    ///
    /// Returns true if ||b*_k||² ≥ (δ - μ²_{k,k-1})||b*_{k-1}||²
    fn lovasz_condition(&self, k: usize) -> bool {
        if k == 0 {
            return true;
        }

        let mu_sq = self.mu[k][k - 1] * self.mu[k][k - 1];
        let lhs = self.b_star_norms_sq[k];
        let rhs = (self.delta - mu_sq) * self.b_star_norms_sq[k - 1];

        lhs >= rhs
    }

    /// Swap basis vectors at positions i and j
    fn swap_vectors(&mut self, i: usize, j: usize) {
        self.basis.swap(i, j);
    }

    /// Compute Gram-Schmidt orthogonalization starting from index start
    ///
    /// Updates orthogonal_basis, mu, and b_star_norms_sq
    fn compute_gso(&mut self, start: usize) {
        for i in start..self.num_vectors {
            // b*_i = b_i - Σ_{j=0}^{i-1} μ_ij b*_j
            let mut b_star = self.basis[i].clone();

            for j in 0..i {
                // Compute μ_ij = ⟨b_i, b*_j⟩ / ||b*_j||²
                let dot_product = dot(&self.basis[i], &self.orthogonal_basis[j]);
                self.mu[i][j] = dot_product / self.b_star_norms_sq[j];

                // b*_i -= μ_ij * b*_j
                for k in 0..self.dimension {
                    b_star[k] -= self.mu[i][j] * self.orthogonal_basis[j][k];
                }

                self.stats.total_operations += self.dimension + 2;  // dot + division + subtraction
            }

            // Store orthogonal vector and its norm squared
            self.b_star_norms_sq[i] = dot(&b_star, &b_star);
            self.orthogonal_basis[i] = b_star;
        }
    }

    /// Get the reduced basis
    pub fn get_basis(&self) -> &[Vec<f64>] {
        &self.basis
    }

    /// Get statistics
    pub fn get_stats(&self) -> &LLLStats {
        &self.stats
    }

    /// Compute Hermite factor: ||b_1|| / (det(L))^{1/n}
    ///
    /// Measures quality of reduction. Lower is better.
    /// Optimal is 1.0, LLL typically achieves ~1.02^n
    pub fn hermite_factor(&self) -> f64 {
        let b1_norm = norm(&self.basis[0]);
        let det = self.determinant();
        b1_norm / det.powf(1.0 / self.num_vectors as f64)
    }

    /// Compute lattice determinant (volume)
    fn determinant(&self) -> f64 {
        // det(L) = Π ||b*_i||
        self.b_star_norms_sq.iter().map(|x| x.sqrt()).product()
    }
}

impl fmt::Display for LLL {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LLL Reducer")?;
        writeln!(f, "  Dimension: {}", self.dimension)?;
        writeln!(f, "  Basis vectors: {}", self.num_vectors)?;
        writeln!(f, "  Delta: {}", self.delta)?;
        writeln!(f, "  Stats:")?;
        writeln!(f, "    Size reductions: {}", self.stats.size_reductions)?;
        writeln!(f, "    Swaps: {}", self.stats.swaps)?;
        writeln!(f, "    GSO updates: {}", self.stats.gso_updates)?;
        writeln!(f, "    Total operations: {}", self.stats.total_operations)?;
        writeln!(f, "  Hermite factor: {:.6}", self.hermite_factor())
    }
}

// Helper functions

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lll_simple_2d() {
        // Simple 2D basis
        let basis = vec![
            vec![1.0, 1.0],
            vec![1.0, 0.0],
        ];

        let mut lll = LLL::new(basis, 0.75);
        lll.reduce();

        let reduced = lll.get_basis();

        // First vector should be shortest
        let n0 = norm(&reduced[0]);
        let n1 = norm(&reduced[1]);
        assert!(n0 <= n1);
    }

    #[test]
    fn test_lll_identity_unchanged() {
        // Identity basis should be unchanged
        let basis = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let mut lll = LLL::new(basis.clone(), 0.99);
        lll.reduce();

        let reduced = lll.get_basis();

        // Should be essentially unchanged (up to floating point)
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((reduced[i][j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_hermite_factor() {
        let basis = vec![
            vec![1.0, 1.0],
            vec![1.0, 0.0],
        ];

        let mut lll = LLL::new(basis, 0.99);
        lll.reduce();

        let hf = lll.hermite_factor();
        // Hermite factor should be reasonable (close to 1.0)
        assert!(hf > 0.9 && hf < 2.0);
    }
}
