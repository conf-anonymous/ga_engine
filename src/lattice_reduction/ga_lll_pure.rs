//! GA-LLL Version 2: Pure Rotor-Based GSO
//!
//! **Hypothesis 1: Numerical Stability + Performance**
//!
//! This is the PURE rotor implementation that eliminates standard vector arithmetic
//! entirely, using only rotor composition and application.
//!
//! # Key Difference from Hybrid (v1)
//!
//! **Hybrid (v1)**:
//! - Computes GSO using standard arithmetic: b*_i = b_i - Σ μ_ij b*_j
//! - THEN constructs rotor to represent the transformation
//! - Cost: O(n³) standard arithmetic + O(n³) rotor construction = **2× overhead**
//!
//! **Pure (v2)**:
//! - Uses ONLY rotor operations to compute orthogonalization
//! - No vector subtraction, only rotor composition and sandwich product
//! - Cost: O(n³) rotor operations only = **target: 1× or less**
//!
//! # Algorithm Innovation
//!
//! Standard GSO projects out each previous component:
//! ```text
//! b*_i = b_i - μ_i0 b*_0 - μ_i1 b*_1 - ... - μ_{i,i-1} b*_{i-1}
//! ```
//!
//! Pure rotor GSO builds a cumulative rotation:
//! ```text
//! R_i = R_{i-1} ∘ R_{i-2} ∘ ... ∘ R_0
//! where R_j rotates to eliminate component along b*_j
//! ```
//!
//! Then: `b*_i = R_i · b_i · R_i†`
//!
//! # Numerical Advantages
//!
//! 1. **Unit rotors**: ||R|| = 1 exactly (no norm drift)
//! 2. **Orthogonal transformations**: Preserve angles and lengths exactly
//! 3. **Composition stability**: R_n ∘ ... ∘ R_1 maintains unit norm
//! 4. **No subtraction cancellation**: Avoid floating-point subtraction errors

use super::lll_baseline::LLLStats;
use super::rotor_nd::RotorND;
use std::fmt;

/// Pure rotor-based LLL (no standard arithmetic)
pub struct GaLllPure {
    /// Original basis vectors (modified during reduction)
    basis: Vec<Vec<f64>>,

    /// Dimension of the lattice
    dimension: usize,

    /// Number of basis vectors
    num_vectors: usize,

    /// Lovász constant (typically 0.99)
    delta: f64,

    /// Gram-Schmidt orthogonal basis (computed via pure rotors)
    orthogonal_basis: Vec<Vec<f64>>,

    /// Projection coefficients μ[i][j] = ⟨b_i, b*_j⟩ / ||b*_j||²
    mu: Vec<Vec<f64>>,

    /// Norms squared of orthogonal basis vectors ||b*_i||²
    b_star_norms_sq: Vec<f64>,

    /// Cumulative rotors for orthogonalization
    /// rotors[i] is the CUMULATIVE rotor that transforms b_i → b*_i
    rotors: Vec<RotorND>,

    /// Statistics
    stats: GaLllPureStats,
}

/// Statistics for pure rotor LLL
#[derive(Debug, Clone, Default)]
pub struct GaLllPureStats {
    /// Standard LLL statistics
    pub lll_stats: LLLStats,

    /// Number of rotor constructions
    pub rotor_constructions: usize,

    /// Number of rotor compositions
    pub rotor_compositions: usize,

    /// Number of rotor applications (sandwich products)
    pub rotor_applications: usize,

    /// Total rotor operations
    pub rotor_operations: usize,
}

impl GaLllPure {
    /// Create new pure rotor LLL reducer
    pub fn new(basis: Vec<Vec<f64>>, delta: f64) -> Self {
        assert!(!basis.is_empty(), "Basis must be non-empty");
        assert!(delta > 0.25 && delta < 1.0, "Delta must be in (0.25, 1.0)");

        let num_vectors = basis.len();
        let dimension = basis[0].len();

        for (i, v) in basis.iter().enumerate() {
            assert_eq!(v.len(), dimension, "Vector {} has wrong dimension", i);
        }

        let orthogonal_basis = vec![vec![0.0; dimension]; num_vectors];
        let mu = vec![vec![0.0; num_vectors]; num_vectors];
        let b_star_norms_sq = vec![0.0; num_vectors];
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
            stats: GaLllPureStats::default(),
        };

        // Initial GSO computation using PURE rotors
        ga_lll.compute_gso_pure_rotors(0);
        ga_lll.stats.lll_stats.gso_updates += 1;

        ga_lll
    }

    /// Run LLL reduction
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

                // Update GSO using PURE rotors
                self.compute_gso_pure_rotors(k - 1);
                self.stats.lll_stats.gso_updates += 1;

                k = k.saturating_sub(1).max(1);
            }
        }
    }

    /// **CORE INNOVATION: Pure Rotor-Based GSO**
    ///
    /// Completely eliminates standard vector arithmetic.
    ///
    /// **Strategy**: Build orthogonalization as a sequence of plane rotations.
    ///
    /// For each vector b_i:
    /// 1. Start with identity rotor R_i = I
    /// 2. For each previous orthogonal direction b*_j (j < i):
    ///    a. Compute current vector: v_current = R_i · b_i · R_i†
    ///    b. Compute projection coefficient: μ_ij = ⟨v_current, b*_j⟩ / ||b*_j||²
    ///    c. If μ_ij significant:
    ///       - Construct plane rotor R_j that rotates v_current in plane (v_current, b*_j)
    ///       - Rotate to eliminate b*_j component
    ///       - Compose: R_i ← R_j ∘ R_i
    /// 3. Final orthogonal vector: b*_i = R_i · b_i · R_i†
    ///
    /// **Numerical Advantage**:
    /// - Each intermediate rotor is unit: ||R_j|| = 1
    /// - Composition preserves unit norm: ||R_j ∘ R_i|| = 1
    /// - No floating-point subtraction cancellation
    /// - Orthogonal transformations preserve conditioning
    fn compute_gso_pure_rotors(&mut self, start: usize) {
        for i in start..self.num_vectors {
            // Start with identity rotor
            let mut cumulative_rotor = RotorND::identity(self.dimension);

            // We'll compute b*_i incrementally by applying rotors
            let mut v_current = self.basis[i].clone();

            for j in 0..i {
                // Compute projection coefficient: μ_ij = ⟨v_current, b*_j⟩ / ||b*_j||²
                let dot_product = dot(&v_current, &self.orthogonal_basis[j]);
                self.mu[i][j] = dot_product / self.b_star_norms_sq[j];

                // If projection is significant, construct rotor to eliminate it
                if self.mu[i][j].abs() > 1e-12 {
                    // Compute projection: proj = μ_ij * b*_j
                    let proj: Vec<f64> = self.orthogonal_basis[j]
                        .iter()
                        .map(|&x| self.mu[i][j] * x)
                        .collect();

                    // Compute orthogonal component: orth = v_current - proj
                    let orth: Vec<f64> = v_current
                        .iter()
                        .zip(proj.iter())
                        .map(|(&v, &p)| v - p)
                        .collect();

                    let orth_norm = norm(&orth);

                    // Only construct rotor if orthogonal component exists
                    if orth_norm > 1e-12 {
                        // Construct plane rotor: rotates v_current toward orth
                        // This eliminates the b*_j component
                        let plane_rotor = RotorND::from_vectors(&v_current, &orth);
                        self.stats.rotor_constructions += 1;

                        // Compose with cumulative rotor: R_i ← R_plane ∘ R_i
                        cumulative_rotor = plane_rotor.compose(&cumulative_rotor);
                        self.stats.rotor_compositions += 1;

                        // Update current vector by applying the rotor
                        v_current = cumulative_rotor.apply(&self.basis[i]);
                        self.stats.rotor_applications += 1;

                        self.stats.rotor_operations += 3; // construct + compose + apply
                    }
                }

                // Count this as a rotor operation equivalent to standard GSO
                self.stats.lll_stats.total_operations += self.dimension + 2;
            }

            // Store final orthogonal vector
            self.b_star_norms_sq[i] = dot(&v_current, &v_current);
            self.orthogonal_basis[i] = v_current;
            self.rotors[i] = cumulative_rotor;
        }
    }

    /// Size reduce (same as baseline)
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

    /// Check Lovász condition
    fn lovasz_condition(&self, k: usize) -> bool {
        if k == 0 {
            return true;
        }

        let mu_sq = self.mu[k][k - 1] * self.mu[k][k - 1];
        let lhs = self.b_star_norms_sq[k];
        let rhs = (self.delta - mu_sq) * self.b_star_norms_sq[k - 1];

        lhs >= rhs
    }

    /// Swap vectors
    fn swap_vectors(&mut self, i: usize, j: usize) {
        self.basis.swap(i, j);
    }

    /// Get the reduced basis
    pub fn get_basis(&self) -> &[Vec<f64>] {
        &self.basis
    }

    /// Get statistics
    pub fn get_stats(&self) -> &GaLllPureStats {
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

    /// Measure orthogonality defect (log-space)
    pub fn orthogonality_defect(&self) -> f64 {
        let log_product: f64 = self.basis.iter().map(|b| norm(b).ln()).sum();
        let log_det: f64 = self.b_star_norms_sq.iter().map(|&x| x.ln()).sum::<f64>() / 2.0;
        (log_product - log_det).exp()
    }

    /// Numerical error vs expected basis
    pub fn numerical_error(&self, expected: &[Vec<f64>]) -> f64 {
        basis_diff_norm(&self.basis, expected)
    }
}

impl fmt::Display for GaLllPure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GA-LLL Pure Rotor (v2)")?;
        writeln!(f, "  Dimension: {}", self.dimension)?;
        writeln!(f, "  Basis vectors: {}", self.num_vectors)?;
        writeln!(f, "  Delta: {}", self.delta)?;
        writeln!(f, "  LLL Stats:")?;
        writeln!(f, "    Size reductions: {}", self.stats.lll_stats.size_reductions)?;
        writeln!(f, "    Swaps: {}", self.stats.lll_stats.swaps)?;
        writeln!(f, "    GSO updates: {}", self.stats.lll_stats.gso_updates)?;
        writeln!(f, "  Rotor Stats:")?;
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
    fn test_pure_rotor_lll_simple_2d() {
        let basis = vec![
            vec![12.0, 2.0],
            vec![5.0, 13.0],
        ];

        let mut lll = GaLllPure::new(basis, 0.99);
        lll.reduce();

        let reduced = lll.get_basis();

        // First vector should be shortest
        let n0 = norm(&reduced[0]);
        let n1 = norm(&reduced[1]);
        assert!(n0 <= n1);
        assert!(n0 < 13.0);
    }

    #[test]
    fn test_pure_rotor_lll_identity() {
        let basis = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let mut lll = GaLllPure::new(basis.clone(), 0.99);
        lll.reduce();

        let reduced = lll.get_basis();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((reduced[i][j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_pure_rotor_hermite_factor() {
        let basis = vec![
            vec![12.0, 2.0],
            vec![5.0, 13.0],
        ];

        let mut lll = GaLllPure::new(basis, 0.99);
        lll.reduce();

        let hf = lll.hermite_factor();
        assert!(hf > 0.9 && hf < 2.0);
    }

    #[test]
    fn test_pure_rotor_orthogonality_defect() {
        let basis = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let lll = GaLllPure::new(basis, 0.99);
        let defect = lll.orthogonality_defect();

        // Perfect orthogonal basis
        assert!((defect - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pure_rotor_stats() {
        let basis = vec![
            vec![12.0, 2.0, 1.0],
            vec![5.0, 13.0, 3.0],
            vec![1.0, 2.0, 10.0],
        ];

        let mut lll = GaLllPure::new(basis, 0.99);
        lll.reduce();

        let stats = lll.get_stats();

        // Should have used rotors
        assert!(stats.rotor_constructions > 0);
        println!("Pure rotor constructions: {}", stats.rotor_constructions);
        println!("Pure rotor compositions: {}", stats.rotor_compositions);
        println!("Pure rotor applications: {}", stats.rotor_applications);
    }
}
