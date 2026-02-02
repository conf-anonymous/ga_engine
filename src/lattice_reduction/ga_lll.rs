//! GA-LLL: Geometric Algebra Based LLL Reduction
//!
//! This module implements LLL lattice reduction using geometric algebra primitives,
//! specifically using rotors for the Gram-Schmidt orthogonalization step.
//!
//! # Motivation
//!
//! Standard LLL uses explicit matrix operations for projections. In geometric algebra,
//! rotations are native operations via rotors. This implementation explores whether
//! GA-based GSO offers numerical or algorithmic advantages.
//!
//! # Algorithm
//!
//! Similar to standard LLL but:
//! 1. **GSO**: Use rotors to incrementally rotate basis vectors into orthogonal complement
//! 2. **Size reduction**: Standard (integer operations, unchanged)
//! 3. **Lovász test**: Standard (norm comparison, unchanged)
//!
//! # References
//!
//! - Lenstra, Lenstra, Lovász (1982): "Factoring polynomials with rational coefficients"
//! - Dorst, Fontijne, Mann (2007): "Geometric Algebra for Computer Science"

use crate::lattice_reduction::rotor_nd::RotorND;
use std::time::{Duration, Instant};

/// Statistics for GA-LLL reduction
#[derive(Debug, Clone, Default)]
pub struct GALLLStats {
    /// Number of basis swaps
    pub swaps: usize,
    /// Number of rotor computations
    pub rotor_computations: usize,
    /// Time spent computing rotors
    pub rotor_time: Duration,
    /// Time spent applying rotors
    pub rotor_apply_time: Duration,
    /// Total reduction time
    pub total_time: Duration,
}

/// GA-based LLL lattice reduction
#[allow(non_camel_case_types)]
pub struct GA_LLL {
    /// Basis vectors (row vectors)
    basis: Vec<Vec<f64>>,
    /// Dimension of ambient space
    dimension: usize,
    /// Number of basis vectors
    num_vectors: usize,
    /// Lovász parameter (typically 0.75 or 0.99)
    delta: f64,

    /// Rotors for GSO (one per basis vector)
    rotors: Vec<RotorND>,

    /// Statistics
    stats: GALLLStats,
}

impl GA_LLL {
    /// Create new GA-LLL reducer
    ///
    /// # Arguments
    ///
    /// * `basis` - Initial basis vectors (row vectors)
    /// * `delta` - Lovász parameter (0.75 for theoretical guarantee, 0.99 for quality)
    pub fn new(basis: Vec<Vec<f64>>, delta: f64) -> Self {
        let dimension = if basis.is_empty() { 0 } else { basis[0].len() };
        let num_vectors = basis.len();

        Self {
            basis,
            dimension,
            num_vectors,
            delta,
            rotors: Vec::new(),
            stats: GALLLStats::default(),
        }
    }

    /// Perform GA-LLL reduction
    pub fn reduce(&mut self) {
        let start = Instant::now();

        // Main LLL loop
        let mut k = 1;
        while k < self.num_vectors {
            // Compute GSO using rotors
            let (gso, mu) = self.compute_gso_rotors();

            // Size reduce k-th vector
            for j in (0..k).rev() {
                let mu_kj = mu[k][j];
                if mu_kj.abs() > 0.5 {
                    let coeff = mu_kj.round();
                    // b_k ← b_k - ⌊μ_{k,j}⌉ * b_j
                    for i in 0..self.dimension {
                        self.basis[k][i] -= coeff * self.basis[j][i];
                    }
                }
            }

            // Recompute GSO after size reduction
            let (gso, mu) = self.compute_gso_rotors();

            // Lovász condition
            let norm_k_sq: f64 = gso[k].iter().map(|x| x * x).sum();
            let norm_k1_sq: f64 = if k > 0 {
                gso[k - 1].iter().map(|x| x * x).sum()
            } else {
                1.0
            };

            let mu_k_k1 = if k > 0 { mu[k][k - 1] } else { 0.0 };

            if norm_k_sq >= (self.delta - mu_k_k1 * mu_k_k1) * norm_k1_sq {
                // Lovász condition satisfied, move to next vector
                k += 1;
            } else {
                // Lovász condition violated, swap and backtrack
                if k > 0 {
                    self.basis.swap(k, k - 1);
                    self.stats.swaps += 1;
                    k -= 1;
                } else {
                    k += 1;
                }
            }
        }

        self.stats.total_time = start.elapsed();
    }

    /// Compute Gram-Schmidt Orthogonalization using rotors
    ///
    /// Returns (GSO basis, μ coefficients)
    fn compute_gso_rotors(&mut self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = self.num_vectors;
        let d = self.dimension;

        let mut gso: Vec<Vec<f64>> = Vec::new();
        let mut mu = vec![vec![0.0; n]; n];
        self.rotors.clear();

        for i in 0..n {
            let start_rotor = Instant::now();

            if i == 0 {
                // First vector: use as-is
                gso.push(self.basis[0].clone());
                self.rotors.push(RotorND::identity(d));
            } else {
                // Use rotor-based orthogonalization
                let (b_star, rotor) = self.orthogonalize_with_rotors(i, &gso);
                gso.push(b_star);
                self.rotors.push(rotor);
            }

            self.stats.rotor_time += start_rotor.elapsed();
            self.stats.rotor_computations += 1;

            // Compute μ coefficients for this vector
            for j in 0..i {
                // μᵢⱼ = ⟨bᵢ, b*ⱼ⟩ / ⟨b*ⱼ, b*ⱼ⟩
                let numerator: f64 = (0..d).map(|k| self.basis[i][k] * gso[j][k]).sum();
                let denominator: f64 = gso[j].iter().map(|x| x * x).sum();

                if denominator > 1e-10 {
                    mu[i][j] = numerator / denominator;
                }
            }
        }

        (gso, mu)
    }

    /// Orthogonalize vector i against span(b*_0, ..., b*_{i-1}) using rotors
    ///
    /// Returns (orthogonalized vector, composed rotor)
    fn orthogonalize_with_rotors(
        &mut self,
        idx: usize,
        gso: &[Vec<f64>],
    ) -> (Vec<f64>, RotorND) {
        let mut v = self.basis[idx].clone();
        let mut total_rotor = RotorND::identity(self.dimension);

        let rotor_start = Instant::now();

        // Incrementally rotate v to be orthogonal to each previous GSO vector
        for j in 0..idx {
            // Compute projection of v onto gso[j]
            let dot: f64 = (0..self.dimension).map(|k| v[k] * gso[j][k]).sum();
            let norm_sq: f64 = gso[j].iter().map(|x| x * x).sum();

            if norm_sq < 1e-10 {
                continue;
            }

            let proj_coeff = dot / norm_sq;

            // If already orthogonal, skip
            if proj_coeff.abs() < 1e-10 {
                continue;
            }

            // Compute parallel component: v_∥ = (v·gso[j]/||gso[j]||²) * gso[j]
            let v_parallel: Vec<f64> = gso[j].iter().map(|x| proj_coeff * x).collect();

            // Compute perpendicular component: v_⊥ = v - v_∥
            let v_perp: Vec<f64> = v
                .iter()
                .zip(v_parallel.iter())
                .map(|(a, b)| a - b)
                .collect();

            // Check if v_perp has reasonable norm
            let v_perp_norm_sq: f64 = v_perp.iter().map(|x| x * x).sum();
            if v_perp_norm_sq < 1e-10 {
                // v is parallel to gso[j] - can't create rotor
                // Just project directly
                v = v_perp;
                continue;
            }

            // Create rotor that rotates v → v_⊥
            // This rotor lives in the plane spanned by v and v_⊥
            let rotor = RotorND::from_vectors(&v, &v_perp);

            // Apply rotor to rotate v
            let apply_start = Instant::now();
            v = rotor.apply(&v);
            self.stats.rotor_apply_time += apply_start.elapsed();

            // Compose rotors (track total transformation)
            total_rotor = total_rotor.compose(&rotor);
        }

        self.stats.rotor_time += rotor_start.elapsed();

        (v, total_rotor)
    }

    /// Get reduced basis
    pub fn get_basis(&self) -> &[Vec<f64>] {
        &self.basis
    }

    /// Get statistics
    pub fn get_stats(&self) -> &GALLLStats {
        &self.stats
    }

    /// Compute Hermite factor: ||b₁|| / (det(L))^(1/n)
    pub fn hermite_factor(&self) -> f64 {
        if self.basis.is_empty() {
            return 1.0;
        }

        let first_norm: f64 = self.basis[0].iter().map(|x| x * x).sum::<f64>().sqrt();

        // Approximate det by product of GSO norms
        let (gso, _) = self.compute_gso_standard();
        let det_approx: f64 = gso
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt())
            .product();

        if det_approx > 1e-10 {
            first_norm / det_approx.powf(1.0 / self.num_vectors as f64)
        } else {
            1.0
        }
    }

    /// Compute GSO using standard method (for Hermite factor calculation)
    fn compute_gso_standard(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = self.num_vectors;
        let d = self.dimension;

        let mut gso: Vec<Vec<f64>> = Vec::new();
        let mut mu = vec![vec![0.0; n]; n];

        for i in 0..n {
            let mut b_star = self.basis[i].clone();

            for j in 0..i {
                let numerator: f64 = (0..d).map(|k| self.basis[i][k] * gso[j][k]).sum();
                let denominator: f64 = gso[j].iter().map(|x| x * x).sum();

                if denominator > 1e-10 {
                    mu[i][j] = numerator / denominator;

                    for k in 0..d {
                        b_star[k] -= mu[i][j] * gso[j][k];
                    }
                }
            }

            gso.push(b_star);
        }

        (gso, mu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ga_lll_identity() {
        // Identity-like basis should remain unchanged
        let basis = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let mut ga_lll = GA_LLL::new(basis.clone(), 0.99);
        ga_lll.reduce();

        let reduced = ga_lll.get_basis();

        // Should be essentially unchanged
        for i in 0..3 {
            for j in 0..3 {
                let diff = (reduced[i][j] - basis[i][j]).abs();
                assert!(diff < 1e-10, "Identity basis changed");
            }
        }
    }

    #[test]
    fn test_ga_lll_simple_2d() {
        // Classic 2D example
        let basis = vec![vec![1.0, 1.0], vec![1.0, 0.0]];

        let mut ga_lll = GA_LLL::new(basis, 0.75);
        ga_lll.reduce();

        let reduced = ga_lll.get_basis();
        let stats = ga_lll.get_stats();

        println!("GA-LLL 2D reduced: {:?}", reduced);
        println!("Stats: {:?}", stats);

        // First vector should be short
        let first_norm: f64 = reduced[0].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(first_norm < 1.5, "First vector should be short");
    }

    #[test]
    fn test_ga_lll_3d() {
        let basis = vec![
            vec![1.0, 1.0, 1.0],
            vec![-1.0, 0.0, 2.0],
            vec![3.0, 5.0, 6.0],
        ];

        let mut ga_lll = GA_LLL::new(basis, 0.99);
        ga_lll.reduce();

        let reduced = ga_lll.get_basis();
        let stats = ga_lll.get_stats();

        println!("GA-LLL 3D reduced: {:?}", reduced);
        println!("Stats: {:?}", stats);

        // Hermite factor should be reasonable
        let hf = ga_lll.hermite_factor();
        println!("Hermite factor: {:.6}", hf);
        assert!(hf < 1.5);
    }

    #[test]
    fn test_ga_lll_quality() {
        // Test that GA-LLL produces valid reduced basis
        let basis = vec![
            vec![10.0, 5.0, 2.0],
            vec![3.0, 12.0, 4.0],
            vec![1.0, 2.0, 15.0],
        ];

        let mut ga_lll = GA_LLL::new(basis, 0.99);
        ga_lll.reduce();

        let reduced = ga_lll.get_basis();

        // First vector should be short
        let first_norm: f64 = reduced[0].iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("First norm: {:.6}", first_norm);
        assert!(first_norm < 20.0);

        // Hermite factor should be good
        let hf = ga_lll.hermite_factor();
        println!("Hermite factor: {:.6}", hf);
        assert!(hf < 1.5);
    }
}
