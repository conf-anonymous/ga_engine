//! Shortest Vector Problem (SVP) Enumeration
//!
//! Implements Schnorr-Euchner enumeration for finding shortest vectors in lattices.
//! This is the core subroutine used by BKZ.
//!
//! # Algorithm
//!
//! Given a basis B and radius R, enumerate all lattice vectors v = Σ xᵢbᵢ
//! such that ||v|| ≤ R, and return the shortest one found.
//!
//! Uses recursive depth-first search with pruning to avoid exponential blowup.

use std::f64;

/// Result of enumeration
#[derive(Debug, Clone)]
pub struct EnumResult {
    /// Coefficient vector (x₀, x₁, ..., xₙ₋₁) where v = Σ xᵢbᵢ
    pub coefficients: Vec<i32>,
    /// Norm of the vector found
    pub norm: f64,
    /// Number of nodes explored in search tree
    pub nodes_explored: u64,
}

/// Statistics for enumeration
#[derive(Debug, Clone, Default)]
pub struct EnumStats {
    pub nodes_explored: u64,
    pub solutions_found: usize,
    pub pruned_branches: u64,
}

/// Enumerate shortest vector in a lattice
///
/// # Arguments
///
/// * `basis` - GSO basis (orthogonal basis)
/// * `mu` - GSO coefficients (μᵢⱼ)
/// * `radius` - Search radius (typically based on current shortest vector)
/// * `max_nodes` - Maximum nodes to explore (prevent timeout)
///
/// # Returns
///
/// Best solution found, or None if no vector within radius
pub fn enumerate_svp(
    basis: &[Vec<f64>],
    mu: &[Vec<f64>],
    radius: f64,
    max_nodes: u64,
) -> Option<EnumResult> {
    let n = basis.len();
    if n == 0 {
        return None;
    }

    // Precompute squared norms of GSO basis vectors
    let mut b_star_norms_sq: Vec<f64> = basis
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum())
        .collect();

    // Handle numerical issues
    for i in 0..n {
        if !b_star_norms_sq[i].is_finite() || b_star_norms_sq[i] <= 0.0 {
            b_star_norms_sq[i] = 1.0; // Fallback
        }
    }

    let mut stats = EnumStats::default();
    let mut best_solution: Option<(Vec<i32>, f64)> = None;

    // Start enumeration from top level (index n-1, working down to 0)
    let mut coeffs = vec![0i32; n];
    let mut centers = vec![0.0; n];
    let mut distances = vec![0.0; n];

    // Initial center at level n-1
    centers[n - 1] = 0.0;

    enumerate_recursive(
        n - 1,
        radius * radius, // Use squared radius for efficiency
        &mut coeffs,
        &mut centers,
        &mut distances,
        mu,
        &b_star_norms_sq,
        max_nodes,
        &mut stats,
        &mut best_solution,
    );

    best_solution.map(|(coeffs, norm)| EnumResult {
        coefficients: coeffs,
        norm: norm.sqrt(),
        nodes_explored: stats.nodes_explored,
    })
}

/// Recursive enumeration at level k
///
/// # Algorithm
///
/// At level k, we're choosing coefficient xₖ such that the partial vector
/// v_k = Σᵢ≥ₖ xᵢb*ᵢ has squared norm ≤ remaining_radius.
///
/// The squared partial norm is:
///   ||v_k||² = ||v_{k+1}||² + (xₖ - centerₖ)² · ||b*ₖ||²
///
/// We enumerate integer xₖ in order of increasing distance from centerₖ.
#[allow(clippy::too_many_arguments)]
fn enumerate_recursive(
    k: usize,
    remaining_radius_sq: f64,
    coeffs: &mut [i32],
    centers: &mut [f64],
    distances: &mut [f64],
    mu: &[Vec<f64>],
    b_star_norms_sq: &[f64],
    max_nodes: u64,
    stats: &mut EnumStats,
    best_solution: &mut Option<(Vec<i32>, f64)>,
) {
    // Check node limit
    stats.nodes_explored += 1;
    if stats.nodes_explored >= max_nodes {
        return;
    }

    let n = coeffs.len();

    // Base case: reached bottom level
    if k == 0 {
        // Compute center for level 0
        let mut center = 0.0;
        for j in 1..n {
            center += mu[0][j] * coeffs[j] as f64;
        }
        centers[0] = -center;

        // Try coefficient closest to center
        let x = centers[0].round() as i32;
        let dist = (x as f64 - centers[0]).abs();

        let dist_sq = dist * dist * b_star_norms_sq[0];

        if dist_sq <= remaining_radius_sq {
            coeffs[0] = x;

            // Found a solution - check if it's the best
            let norm_sq = distances[1] + dist_sq;

            if norm_sq > 1e-10 {
                // Non-zero vector
                match best_solution {
                    None => {
                        stats.solutions_found += 1;
                        *best_solution = Some((coeffs.to_vec(), norm_sq));
                    }
                    Some((_, best_norm_sq)) => {
                        if norm_sq < *best_norm_sq {
                            stats.solutions_found += 1;
                            *best_solution = Some((coeffs.to_vec(), norm_sq));
                        }
                    }
                }
            }
        }

        return;
    }

    // Recursive case: enumerate at level k

    // Compute center for level k
    let mut center = 0.0;
    for j in (k + 1)..n {
        center += mu[k][j] * coeffs[j] as f64;
    }
    centers[k] = -center;

    // Enumerate integers around the center
    // Try x = round(center), then round(center) ± 1, ± 2, ...
    let x_center = centers[k].round() as i32;

    // Maximum distance we can go from center
    let max_dist_sq = remaining_radius_sq / b_star_norms_sq[k];
    if !max_dist_sq.is_finite() || max_dist_sq < 0.0 {
        // Bad geometry - abort
        return;
    }
    let max_dist = max_dist_sq.sqrt();
    let max_offset = (max_dist.ceil() as i32).max(1).min(1000); // Cap at ±1000

    // Enumerate in order: 0, +1, -1, +2, -2, ...
    for offset in 0..=max_offset {
        if stats.nodes_explored >= max_nodes {
            return;
        }

        let candidates = if offset == 0 {
            vec![x_center]
        } else {
            vec![x_center + offset, x_center - offset]
        };

        for x in candidates {
            let dist = x as f64 - centers[k];
            let dist_sq = dist * dist * b_star_norms_sq[k];

            if dist_sq > remaining_radius_sq {
                stats.pruned_branches += 1;
                continue;
            }

            // Recurse to next level
            coeffs[k] = x;
            distances[k] = dist_sq;

            let new_radius_sq = remaining_radius_sq - dist_sq;

            enumerate_recursive(
                k - 1,
                new_radius_sq,
                coeffs,
                centers,
                distances,
                mu,
                b_star_norms_sq,
                max_nodes,
                stats,
                best_solution,
            );
        }
    }
}

/// Simple enumeration for small dimensions (testing)
///
/// Brute-force enumerate all vectors with bounded coefficients
pub fn enumerate_simple(basis: &[Vec<f64>], max_coeff: i32) -> Option<(Vec<i32>, f64)> {
    let n = basis.len();
    if n == 0 || n > 5 {
        // Only for very small dimensions
        return None;
    }

    let mut best: Option<(Vec<i32>, f64)> = None;

    // Generate all combinations
    enumerate_combinations(
        n,
        max_coeff,
        &mut vec![0; n],
        0,
        basis,
        &mut best,
    );

    best
}

fn enumerate_combinations(
    n: usize,
    max_coeff: i32,
    coeffs: &mut Vec<i32>,
    level: usize,
    basis: &[Vec<f64>],
    best: &mut Option<(Vec<i32>, f64)>,
) {
    if level == n {
        // Compute vector
        let dim = basis[0].len();
        let mut v = vec![0.0; dim];

        for i in 0..n {
            for j in 0..dim {
                v[j] += coeffs[i] as f64 * basis[i][j];
            }
        }

        // Compute norm
        let norm_sq: f64 = v.iter().map(|x| x * x).sum();

        if norm_sq > 1e-10 {
            // Non-zero
            match best {
                None => *best = Some((coeffs.clone(), norm_sq.sqrt())),
                Some((_, best_norm)) => {
                    let norm = norm_sq.sqrt();
                    if norm < *best_norm {
                        *best = Some((coeffs.clone(), norm));
                    }
                }
            }
        }

        return;
    }

    // Try coefficients for this level
    for c in -max_coeff..=max_coeff {
        coeffs[level] = c;
        enumerate_combinations(n, max_coeff, coeffs, level + 1, basis, best);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_simple_2d() {
        // Simple 2D lattice
        let basis = vec![vec![3.0, 0.0], vec![0.0, 4.0]];

        let result = enumerate_simple(&basis, 2);
        assert!(result.is_some());

        let (coeffs, norm) = result.unwrap();
        println!("2D: coeffs = {:?}, norm = {:.6}", coeffs, norm);

        // Shortest should be ±(1,0,0) or ±(0,1,0) with norm 3 or 4
        assert!(norm >= 2.9 && norm <= 4.1);
    }

    #[test]
    fn test_enumerate_simple_3d() {
        // 3D lattice
        let basis = vec![
            vec![10.0, 0.0, 0.0],
            vec![1.0, 10.0, 0.0],
            vec![1.0, 1.0, 10.0],
        ];

        let result = enumerate_simple(&basis, 2);
        assert!(result.is_some());

        let (coeffs, norm) = result.unwrap();
        println!("3D: coeffs = {:?}, norm = {:.6}", coeffs, norm);

        // Should find something reasonably short
        assert!(norm < 15.0);
    }

    #[test]
    fn test_enumerate_with_gso() {
        // Test the main enumeration function
        // Use a simple 2D orthogonal basis
        let basis = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let mu = vec![vec![0.0, 0.0], vec![0.0, 0.0]];

        let result = enumerate_svp(&basis, &mu, 2.0, 10000);
        assert!(result.is_some());

        let enum_result = result.unwrap();
        println!(
            "GSO enum: coeffs = {:?}, norm = {:.6}, nodes = {}",
            enum_result.coefficients, enum_result.norm, enum_result.nodes_explored
        );

        // Should find (±1, 0) or (0, ±1) with norm 1
        assert!(enum_result.norm >= 0.9 && enum_result.norm <= 1.1);
        assert!(enum_result.nodes_explored > 0);
    }
}
