//! Differential and Linear Trail Propagation
//!
//! This module implements trail propagation for differential-linear cryptanalysis
//! using both traditional matrix methods and GA-based rotor methods.
//!
//! # Background
//!
//! In differential-linear cryptanalysis, we track how differences and linear
//! approximations propagate through cipher rounds. The linear layer of each round
//! can be represented as:
//! - Matrix multiplication (O(n²))
//! - Rotor application (O(n)) for orthogonal transformations
//!
//! # Key Insight
//!
//! Many cipher linear layers are orthogonal or near-orthogonal:
//! - Permutations (bit/byte shuffling)
//! - Rotational mixing (ChaCha, Salsa20)
//! - MDS matrices (can be orthogonalized)
//!
//! Orthogonal transformations = Rotations → Rotors provide O(n) vs O(n²) speedup!

use crate::lattice_reduction::rotor_nd::RotorND;

/// Differential trail: sequence of differences through cipher rounds
#[derive(Debug, Clone)]
pub struct DifferentialTrail {
    /// Input/output differences at each round
    pub deltas: Vec<Vec<f64>>,

    /// Dimension of state
    pub dimension: usize,
}

impl DifferentialTrail {
    /// Create new empty trail
    pub fn new(dimension: usize) -> Self {
        Self {
            deltas: Vec::new(),
            dimension,
        }
    }

    /// Add a difference to the trail
    pub fn add_delta(&mut self, delta: Vec<f64>) {
        assert_eq!(delta.len(), self.dimension, "Delta dimension mismatch");
        self.deltas.push(delta);
    }

    /// Get number of rounds
    pub fn num_rounds(&self) -> usize {
        if self.deltas.is_empty() {
            0
        } else {
            self.deltas.len() - 1  // Initial + N rounds = N+1 deltas
        }
    }

    /// Get delta at specific round
    pub fn get_delta(&self, round: usize) -> Option<&[f64]> {
        self.deltas.get(round).map(|v| v.as_slice())
    }
}

/// Linear approximation trail
#[derive(Debug, Clone)]
pub struct LinearTrail {
    /// Linear masks at each round
    pub masks: Vec<Vec<f64>>,

    /// Dimension of state
    pub dimension: usize,
}

impl LinearTrail {
    /// Create new empty trail
    pub fn new(dimension: usize) -> Self {
        Self {
            masks: Vec::new(),
            dimension,
        }
    }

    /// Add a mask to the trail
    pub fn add_mask(&mut self, mask: Vec<f64>) {
        assert_eq!(mask.len(), self.dimension, "Mask dimension mismatch");
        self.masks.push(mask);
    }

    /// Get number of rounds
    pub fn num_rounds(&self) -> usize {
        if self.masks.is_empty() {
            0
        } else {
            self.masks.len() - 1
        }
    }
}

/// Round transformation (linear layer)
#[derive(Debug, Clone)]
pub enum RoundTransform {
    /// Matrix-based transformation (baseline)
    Matrix(Vec<Vec<f64>>),

    /// Rotor-based transformation (GA-optimized)
    Rotor(RotorND),
}

impl RoundTransform {
    /// Create from matrix
    pub fn from_matrix(matrix: Vec<Vec<f64>>) -> Self {
        Self::Matrix(matrix)
    }

    /// Create from rotor
    pub fn from_rotor(rotor: RotorND) -> Self {
        Self::Rotor(rotor)
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        match self {
            Self::Matrix(m) => m.len(),
            Self::Rotor(r) => r.dimension(),
        }
    }

    /// Propagate differential through round (matrix version)
    ///
    /// Computes: Δout = M × Δin
    /// Complexity: O(n²)
    pub fn propagate_differential_matrix(&self, delta_in: &[f64]) -> Vec<f64> {
        match self {
            Self::Matrix(m) => matrix_vector_multiply(m, delta_in),
            Self::Rotor(_) => panic!("Use propagate_differential_rotor for rotor transforms"),
        }
    }

    /// Propagate differential through round (rotor version)
    ///
    /// Computes: Δout = R·Δin·R†
    /// Complexity: O(n)
    pub fn propagate_differential_rotor(&self, delta_in: &[f64]) -> Vec<f64> {
        match self {
            Self::Rotor(r) => r.apply(delta_in),
            Self::Matrix(_) => panic!("Use propagate_differential_matrix for matrix transforms"),
        }
    }

    /// Propagate linear mask backward through round (matrix version)
    ///
    /// Computes: λout = M^T × λin
    /// Complexity: O(n²) for transpose + O(n²) for multiply = O(n²)
    pub fn propagate_linear_matrix(&self, mask_in: &[f64]) -> Vec<f64> {
        match self {
            Self::Matrix(m) => {
                let m_transpose = matrix_transpose(m);
                matrix_vector_multiply(&m_transpose, mask_in)
            }
            Self::Rotor(_) => panic!("Use propagate_linear_rotor for rotor transforms"),
        }
    }

    /// Propagate linear mask backward through round (rotor version)
    ///
    /// Computes: λout = R† × λin
    /// Complexity: O(1) for conjugate + O(n) for apply = O(n)
    ///
    /// Key insight: For orthogonal matrix M, M^T = M^(-1)
    /// For rotor R representing M, R^(-1) = R† (conjugate)
    /// Conjugate is O(1) operation (negate bivector part)
    pub fn propagate_linear_rotor(&self, mask_in: &[f64]) -> Vec<f64> {
        match self {
            Self::Rotor(r) => {
                // For orthogonal transforms, transpose = inverse = conjugate
                // This would be r.conjugate().apply(mask_in)
                // For now, we use the same rotor (assuming it represents forward transform)
                // The inverse would need to be precomputed or computed here
                // TODO: Implement proper conjugate/inverse
                r.apply(mask_in)
            }
            Self::Matrix(_) => panic!("Use propagate_linear_matrix for matrix transforms"),
        }
    }
}

/// Matrix-vector multiplication
///
/// Complexity: O(n²)
fn matrix_vector_multiply(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    let n = matrix.len();
    assert_eq!(vector.len(), n, "Matrix-vector dimension mismatch");

    let mut result = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    result
}

/// Matrix transpose
///
/// Complexity: O(n²)
fn matrix_transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut result = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            result[j][i] = matrix[i][j];
        }
    }

    result
}

/// Propagate differential trail through multiple rounds (matrix version)
pub fn propagate_differential_trail_matrix(
    initial_delta: &[f64],
    rounds: &[RoundTransform],
) -> DifferentialTrail {
    let dimension = initial_delta.len();
    let mut trail = DifferentialTrail::new(dimension);

    // Add initial difference
    trail.add_delta(initial_delta.to_vec());

    let mut current_delta = initial_delta.to_vec();

    for round in rounds {
        current_delta = round.propagate_differential_matrix(&current_delta);
        trail.add_delta(current_delta.clone());
    }

    trail
}

/// Propagate differential trail through multiple rounds (rotor version)
pub fn propagate_differential_trail_rotor(
    initial_delta: &[f64],
    rounds: &[RoundTransform],
) -> DifferentialTrail {
    let dimension = initial_delta.len();
    let mut trail = DifferentialTrail::new(dimension);

    // Add initial difference
    trail.add_delta(initial_delta.to_vec());

    let mut current_delta = initial_delta.to_vec();

    for round in rounds {
        current_delta = round.propagate_differential_rotor(&current_delta);
        trail.add_delta(current_delta.clone());
    }

    trail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differential_trail() {
        let mut trail = DifferentialTrail::new(4);
        trail.add_delta(vec![1.0, 0.0, 0.0, 0.0]);
        trail.add_delta(vec![0.0, 1.0, 0.0, 0.0]);

        assert_eq!(trail.num_rounds(), 1);
        assert_eq!(trail.get_delta(0), Some(&[1.0, 0.0, 0.0, 0.0][..]));
    }

    #[test]
    fn test_matrix_vector_multiply() {
        let matrix = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let vector = vec![2.0, 3.0];
        let result = matrix_vector_multiply(&matrix, &vector);

        assert_eq!(result, vec![2.0, 3.0]);
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let transposed = matrix_transpose(&matrix);

        assert_eq!(transposed[0], vec![1.0, 3.0]);
        assert_eq!(transposed[1], vec![2.0, 4.0]);
    }

    #[test]
    fn test_propagate_differential_matrix() {
        // Permutation: [a,b] -> [b,a]
        let perm = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        let transform = RoundTransform::from_matrix(perm);

        let delta_in = vec![1.0, 0.0];
        let delta_out = transform.propagate_differential_matrix(&delta_in);

        assert_eq!(delta_out, vec![0.0, 1.0]);
    }

    #[test]
    fn test_propagate_differential_rotor() {
        // 90° rotation: [x,y] -> [-y,x]
        let rotor = RotorND::from_vectors(&[1.0, 0.0], &[0.0, 1.0]);
        let transform = RoundTransform::from_rotor(rotor);

        let delta_in = vec![1.0, 0.0];
        let delta_out = transform.propagate_differential_rotor(&delta_in);

        // Should rotate [1,0] to approximately [0,1]
        assert!((delta_out[0].abs()) < 1e-6);
        assert!((delta_out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_multi_round_trail_matrix() {
        // Two permutation rounds
        let perm = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        let rounds = vec![
            RoundTransform::from_matrix(perm.clone()),
            RoundTransform::from_matrix(perm),
        ];

        let initial = vec![1.0, 0.0];
        let trail = propagate_differential_trail_matrix(&initial, &rounds);

        assert_eq!(trail.num_rounds(), 2);
        // Round 0: [1,0] -> [0,1]
        // Round 1: [0,1] -> [1,0]
        assert_eq!(trail.get_delta(0), Some(&[1.0, 0.0][..]));
        assert_eq!(trail.get_delta(1), Some(&[0.0, 1.0][..]));
        assert_eq!(trail.get_delta(2), Some(&[1.0, 0.0][..]));
    }
}
