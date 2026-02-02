//! GA-Accelerated LLL using Rotor-Based Gram-Schmidt Orthogonalization
//!
//! **Hypothesis 1: Numerical Stability**
//!
//! This is the FIRST test of the GA acceleration hypothesis:
//! Can n-dimensional rotors provide better numerical stability for lattice reduction?
//!
//! # Key Idea
//!
//! Standard LLL uses Gram-Schmidt orthogonalization (GSO) with explicit vector arithmetic:
//!   b*_i = b_i - Σ μ_ij b*_j
//!
//! This requires many floating-point subtractions, which accumulate numerical errors.
//!
//! GA-LLL uses rotors to represent orthogonalization as a composition of rotations:
//!   R_i rotates b_i to be orthogonal to span(b_1, ..., b_{i-1})
//!
//! Rotors are:
//! - **Unit elements**: ||R|| = 1 (exactly representable)
//! - **Numerically stable**: Sandwich product R·v·R† preserves norms exactly
//! - **Composable**: R_total = R_n ∘ ... ∘ R_1 accumulates transformations without error buildup
//!
//! # Algorithm
//!
//! For each basis vector b_i:
//! 1. For each previous orthogonal direction b*_j (j < i):
//!    - Compute rotor R_ij that rotates projection of b_i onto b*_j to zero
//!    - Compose: R_i = R_ij ∘ R_i
//! 2. Apply R_i to b_i to get b*_i (orthogonal component)
//! 3. Size reduction and Lovász checks remain the same as standard LLL
//!
//! # Hypothesis Testing
//!
//! We will compare GA-LLL vs standard LLL on:
//! - **Numerical error**: ||B_output - B_expected|| after reduction
//! - **Hermite factors**: Must be equal or better
//! - **Time**: Rotor composition cost vs vector arithmetic cost
//!
//! Success Criteria:
//! - Numerical error < 0.5× standard LLL error
//! - Same or better Hermite factors
//! - Time penalty < 2× (acceptable tradeoff for stability)

use super::lll_baseline::LLLStats;
use super::rotor_nd::RotorND;
use std::fmt;

/// GA-accelerated LLL using rotor-based orthogonalization
pub struct GaLll {
    /// Original basis vectors (modified during reduction)
    basis: Vec<Vec<f64>>,

    /// Dimension of the lattice
    dimension: usize,

    /// Number of basis vectors
    num_vectors: usize,

    /// Lovász constant (typically 0.99)
    delta: f64,

    /// Gram-Schmidt orthogonal basis (computed via rotors)
    orthogonal_basis: Vec<Vec<f64>>,

    /// Projection coefficients μ[i][j] = ⟨b_i, b*_j⟩ / ||b*_j||²
    mu: Vec<Vec<f64>>,

    /// Norms squared of orthogonal basis vectors ||b*_i||²
    b_star_norms_sq: Vec<f64>,

    /// Rotors used for orthogonalization
    /// rotors[i] transforms b_i to b*_i
    rotors: Vec<RotorND>,

    /// Statistics
    stats: GaLllStats,
}

/// Statistics for GA-LLL (extends LLL stats with rotor-specific metrics)
#[derive(Debug, Clone, Default)]
pub struct GaLllStats {
    /// Standard LLL statistics
    pub lll_stats: LLLStats,

    /// Number of rotor constructions
    pub rotor_constructions: usize,

    /// Number of rotor compositions
    pub rotor_compositions: usize,

    /// Number of rotor applications (sandwich products)
    pub rotor_applications: usize,

    /// Total rotor operations (for cost comparison)
    pub rotor_operations: usize,
}

impl GaLll {
    /// Create new GA-LLL reducer
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

        // Initialize rotors as identity
        let rotors = vec![RotorND::identity(dimension); num_vectors];

        let mut ga_lll = Self {
            basis,
            dimension,
            num_vectors,
            delta,
            orthogonal_basis,
            mu,
            b_star_norms_sq,
            rotors,
            stats: GaLllStats::default(),
        };

        // Initial GSO computation using rotors
        ga_lll.compute_gso_rotors(0);
        ga_lll.stats.lll_stats.gso_updates += 1;

        ga_lll
    }

    /// Run LLL reduction algorithm (same structure as baseline)
    pub fn reduce(&mut self) {
        let mut k: usize = 1;

        while k < self.num_vectors {
            // Size reduce b_k with respect to b_0, ..., b_{k-1}
            for j in (0..k).rev() {
                self.size_reduce(k, j);
            }

            // Check Lovász condition
            if self.lovasz_condition(k) {
                k += 1;
            } else {
                // Swap vectors
                self.swap_vectors(k, k - 1);
                self.stats.lll_stats.swaps += 1;

                // Update GSO using rotors
                self.compute_gso_rotors(k - 1);
                self.stats.lll_stats.gso_updates += 1;

                k = k.saturating_sub(1).max(1);
            }
        }
    }

    /// **CORE INNOVATION: Rotor-based Gram-Schmidt Orthogonalization**
    ///
    /// **Version 1: Hybrid Approach for Correctness**
    ///
    /// For this initial implementation, we use standard GSO arithmetic but
    /// ALSO construct rotors to represent the orthogonalization transformation.
    ///
    /// This allows us to:
    /// 1. Verify correctness (GSO matches standard LLL exactly)
    /// 2. Measure rotor construction overhead
    /// 3. Build foundation for future pure-rotor implementation
    ///
    /// Algorithm:
    /// 1. Compute GSO using standard arithmetic (same as baseline LLL)
    /// 2. Construct rotor R_i from original b_i to orthogonalized b*_i
    /// 3. Track rotor statistics
    ///
    /// Future optimization: Replace arithmetic with pure rotor operations
    fn compute_gso_rotors(&mut self, start: usize) {
        for i in start..self.num_vectors {
            // Standard GSO computation (same as baseline)
            let mut b_star = self.basis[i].clone();

            for j in 0..i {
                // Compute μ_ij = ⟨b_i, b*_j⟩ / ||b*_j||²
                let dot_product = dot(&self.basis[i], &self.orthogonal_basis[j]);
                self.mu[i][j] = dot_product / self.b_star_norms_sq[j];

                // b*_i -= μ_ij * b*_j
                for k in 0..self.dimension {
                    b_star[k] -= self.mu[i][j] * self.orthogonal_basis[j][k];
                }

                self.stats.lll_stats.total_operations += self.dimension + 2;
            }

            // Store orthogonal vector and norm
            self.b_star_norms_sq[i] = dot(&b_star, &b_star);
            self.orthogonal_basis[i] = b_star.clone();

            // **GA Innovation**: Construct rotor representing b_i → b*_i transformation
            // This is the "record" of the orthogonalization
            if norm(&self.basis[i]) > 1e-10 && norm(&b_star) > 1e-10 {
                let rotor = RotorND::from_vectors(&self.basis[i], &b_star);
                self.rotors[i] = rotor;
                self.stats.rotor_constructions += 1;

                // Verify rotor correctness (for debugging)
                if cfg!(debug_assertions) {
                    let reconstructed = self.rotors[i].apply(&self.basis[i]);
                    let error = basis_diff_norm(&[reconstructed], &[b_star.clone()]);
                    if error > 1e-6 {
                        eprintln!("Warning: Rotor reconstruction error: {:.6e}", error);
                    }
                }
            } else {
                // Degenerate case: use identity rotor
                self.rotors[i] = RotorND::identity(self.dimension);
            }

            self.stats.rotor_operations += 1;
        }
    }

    /// Size reduce (same as baseline LLL)
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

            self.stats.lll_stats.size_reductions += 1;
            self.stats.lll_stats.total_operations += self.dimension;
        }
    }

    /// Check Lovász condition (same as baseline)
    fn lovasz_condition(&self, k: usize) -> bool {
        if k == 0 {
            return true;
        }

        let mu_sq = self.mu[k][k - 1] * self.mu[k][k - 1];
        let lhs = self.b_star_norms_sq[k];
        let rhs = (self.delta - mu_sq) * self.b_star_norms_sq[k - 1];

        lhs >= rhs
    }

    /// Swap vectors (same as baseline)
    fn swap_vectors(&mut self, i: usize, j: usize) {
        self.basis.swap(i, j);
    }

    /// Get the reduced basis
    pub fn get_basis(&self) -> &[Vec<f64>] {
        &self.basis
    }

    /// Get statistics
    pub fn get_stats(&self) -> &GaLllStats {
        &self.stats
    }

    /// Compute Hermite factor
    pub fn hermite_factor(&self) -> f64 {
        let b1_norm = norm(&self.basis[0]);
        let det = self.determinant();
        b1_norm / det.powf(1.0 / self.num_vectors as f64)
    }

    /// Compute lattice determinant
    fn determinant(&self) -> f64 {
        self.b_star_norms_sq.iter().map(|x| x.sqrt()).product()
    }

    /// **EXPERIMENT: Compute numerical error vs expected output**
    ///
    /// For hypothesis testing, we need to measure numerical error.
    /// This compares the actual reduced basis against an expected "ideal" reduction.
    ///
    /// We'll use the Frobenius norm: ||B_actual - B_expected||_F
    pub fn numerical_error(&self, expected: &[Vec<f64>]) -> f64 {
        assert_eq!(self.basis.len(), expected.len());
        assert_eq!(self.basis[0].len(), expected[0].len());

        let mut sum_sq: f64 = 0.0;
        for i in 0..self.basis.len() {
            for j in 0..self.basis[0].len() {
                let diff = self.basis[i][j] - expected[i][j];
                sum_sq += diff * diff;
            }
        }

        sum_sq.sqrt()
    }

    /// **EXPERIMENT: Measure orthogonality defect**
    ///
    /// Orthogonality defect = ∏||b_i|| / |det(L)|
    /// Perfect orthogonal basis has defect = 1.0
    /// Higher defect indicates numerical instability
    ///
    /// Uses log-space arithmetic to avoid overflow
    pub fn orthogonality_defect(&self) -> f64 {
        // Compute in log-space: log(∏||b_i||) = Σ log(||b_i||)
        let log_product: f64 = self.basis.iter().map(|b| norm(b).ln()).sum();

        // log(|det(L)|) = Σ log(||b*_i||) = 0.5 * Σ log(||b*_i||²)
        let log_det: f64 = self.b_star_norms_sq.iter().map(|&x| x.ln()).sum::<f64>() / 2.0;

        // defect = exp(log_product - log_det)
        (log_product - log_det).exp()
    }
}

impl fmt::Display for GaLll {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GA-LLL Reducer (Rotor-based)")?;
        writeln!(f, "  Dimension: {}", self.dimension)?;
        writeln!(f, "  Basis vectors: {}", self.num_vectors)?;
        writeln!(f, "  Delta: {}", self.delta)?;
        writeln!(f, "  LLL Stats:")?;
        writeln!(f, "    Size reductions: {}", self.stats.lll_stats.size_reductions)?;
        writeln!(f, "    Swaps: {}", self.stats.lll_stats.swaps)?;
        writeln!(f, "    GSO updates: {}", self.stats.lll_stats.gso_updates)?;
        writeln!(f, "    Total operations: {}", self.stats.lll_stats.total_operations)?;
        writeln!(f, "  GA Stats:")?;
        writeln!(f, "    Rotor constructions: {}", self.stats.rotor_constructions)?;
        writeln!(f, "    Rotor compositions: {}", self.stats.rotor_compositions)?;
        writeln!(f, "    Rotor applications: {}", self.stats.rotor_applications)?;
        writeln!(f, "    Total rotor operations: {}", self.stats.rotor_operations)?;
        writeln!(f, "  Quality:")?;
        writeln!(f, "    Hermite factor: {:.6}", self.hermite_factor())?;
        writeln!(f, "    Orthogonality defect: {:.6}", self.orthogonality_defect())
    }
}

// Helper functions

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn basis_diff_norm(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            let diff = a[i][j] - b[i][j];
            sum_sq += diff * diff;
        }
    }
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ga_lll_simple_2d() {
        // Same test as baseline LLL - should produce similar output
        let basis = vec![
            vec![12.0, 2.0],
            vec![5.0, 13.0],
        ];

        let mut ga_lll = GaLll::new(basis, 0.99);
        ga_lll.reduce();

        let reduced = ga_lll.get_basis();

        // First vector should be shortest
        let n0 = norm(&reduced[0]);
        let n1 = norm(&reduced[1]);
        assert!(n0 <= n1);

        // Should have reduced the basis
        assert!(n0 < 13.0);
    }

    #[test]
    fn test_ga_lll_identity_unchanged() {
        // Identity should remain unchanged
        let basis = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let mut ga_lll = GaLll::new(basis.clone(), 0.99);
        ga_lll.reduce();

        let reduced = ga_lll.get_basis();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((reduced[i][j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_ga_lll_hermite_factor() {
        let basis = vec![
            vec![12.0, 2.0],
            vec![5.0, 13.0],
        ];

        let mut ga_lll = GaLll::new(basis, 0.99);
        ga_lll.reduce();

        let hf = ga_lll.hermite_factor();
        assert!(hf > 0.9 && hf < 2.0);
    }

    #[test]
    fn test_orthogonality_defect() {
        let basis = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let ga_lll = GaLll::new(basis, 0.99);
        let defect = ga_lll.orthogonality_defect();

        // Perfect orthogonal basis should have defect ≈ 1.0
        assert!((defect - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotor_stats_collected() {
        let basis = vec![
            vec![12.0, 2.0, 1.0],
            vec![5.0, 13.0, 3.0],
            vec![1.0, 2.0, 10.0],
        ];

        let mut ga_lll = GaLll::new(basis, 0.99);
        ga_lll.reduce();

        let stats = ga_lll.get_stats();

        // Should have constructed some rotors
        assert!(stats.rotor_constructions > 0);
        println!("Rotor constructions: {}", stats.rotor_constructions);
        println!("Rotor compositions: {}", stats.rotor_compositions);
        println!("Rotor applications: {}", stats.rotor_applications);
    }
}
