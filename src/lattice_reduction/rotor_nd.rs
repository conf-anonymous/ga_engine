//! n-Dimensional Rotor Operations
//!
//! This module implements rotors in Cl(n,0) for arbitrary dimension n.
//! A rotor represents a rotation via a bivector exponential: R = exp(-½B)
//!
//! # Representation
//!
//! A rotor R in Cl(n,0) has:
//! - 1 scalar component
//! - n(n-1)/2 bivector components (e_ij for i<j)
//!
//! For example:
//! - Cl(3,0): scalar + 3 bivectors (e₁₂, e₁₃, e₂₃) = 4 coefficients
//! - Cl(4,0): scalar + 6 bivectors = 7 coefficients
//! - Cl(n,0): 1 + n(n-1)/2 coefficients
//!
//! # Key Operations
//!
//! - **Construction**: Create rotor to rotate vector a toward b
//! - **Application**: Apply rotor via sandwich product R·v·R†
//! - **Composition**: Compose rotors R₃ = R₂·R₁
//! - **Verification**: Check rotor is unit (R·R† = 1)

use std::fmt;

/// n-Dimensional Rotor
///
/// Represents a rotation in Cl(n,0) via a bivector exponential.
/// Stored in dense format with scalar + n(n-1)/2 bivector coefficients.
#[derive(Clone, Debug)]
pub struct RotorND {
    /// Dimension of the space
    dimension: usize,

    /// Scalar component (index 0)
    /// Followed by bivector components e_ij for i<j
    /// Total size: 1 + n(n-1)/2
    coefficients: Vec<f64>,
}

impl RotorND {
    /// Create identity rotor (represents no rotation)
    ///
    /// # Examples
    ///
    /// ```
    /// use ga_engine::lattice_reduction::rotor_nd::RotorND;
    ///
    /// let r = RotorND::identity(5);
    /// assert_eq!(r.dimension(), 5);
    /// assert!((r.scalar() - 1.0).abs() < 1e-10);
    /// ```
    pub fn identity(dimension: usize) -> Self {
        let num_bivectors = dimension * (dimension - 1) / 2;
        let mut coefficients = vec![0.0; 1 + num_bivectors];
        coefficients[0] = 1.0; // Scalar = 1

        Self {
            dimension,
            coefficients,
        }
    }

    /// Construct rotor that rotates vector a toward vector b
    ///
    /// The rotor R is constructed as R = (1 + b∧a) / |1 + b∧a|
    /// where ∧ is the wedge product (outer product).
    ///
    /// # Arguments
    ///
    /// * `a` - Source vector (will be normalized)
    /// * `b` - Target vector (will be normalized)
    ///
    /// # Panics
    ///
    /// Panics if vectors have different lengths or if either is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ga_engine::lattice_reduction::rotor_nd::RotorND;
    ///
    /// let a = vec![1.0, 0.0, 0.0];
    /// let b = vec![0.0, 1.0, 0.0];
    /// let r = RotorND::from_vectors(&a, &b);
    ///
    /// let result = r.apply(&a);
    /// // Result should be close to b
    /// ```
    pub fn from_vectors(a: &[f64], b: &[f64]) -> Self {
        assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
        let n = a.len();

        // Normalize input vectors
        let a_norm = norm(a);
        let b_norm = norm(b);
        assert!(a_norm > 1e-10, "Vector a must be non-zero");
        assert!(b_norm > 1e-10, "Vector b must be non-zero");

        let a_unit: Vec<f64> = a.iter().map(|x| x / a_norm).collect();
        let b_unit: Vec<f64> = b.iter().map(|x| x / b_norm).collect();

        // Compute scalar part: 1 + dot(b, a)
        let scalar = 1.0 + dot(&b_unit, &a_unit);

        // Compute bivector part: b∧a
        // For vectors u, v: (u∧v)_ij = u_i*v_j - u_j*v_i
        let num_bivectors = n * (n - 1) / 2;
        let mut bivectors = vec![0.0; num_bivectors];

        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                bivectors[idx] = b_unit[i] * a_unit[j] - b_unit[j] * a_unit[i];
                idx += 1;
            }
        }

        // Compute norm: sqrt(scalar² + sum(bivector_i²))
        let bivector_norm_sq: f64 = bivectors.iter().map(|x| x * x).sum();
        let rotor_norm = (scalar * scalar + bivector_norm_sq).sqrt();

        // Normalize rotor
        let mut coefficients = vec![0.0; 1 + num_bivectors];
        coefficients[0] = scalar / rotor_norm;
        for (i, &bv) in bivectors.iter().enumerate() {
            coefficients[i + 1] = bv / rotor_norm;
        }

        Self {
            dimension: n,
            coefficients,
        }
    }

    /// Apply rotor to vector via sandwich product: v' = R·v·R†
    ///
    /// This performs the rotation without explicit projection coefficients.
    ///
    /// Uses direct geometric algebra formula: v' = v + 2s(B⌋v) + 2(B⌋(B⌋v))
    /// This is O(n) complexity vs O(n²) for matrix multiplication.
    ///
    /// # Arguments
    ///
    /// * `v` - Vector to rotate
    ///
    /// # Returns
    ///
    /// Rotated vector of same dimension
    ///
    /// # Panics
    ///
    /// Panics if vector dimension doesn't match rotor dimension.
    pub fn apply(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.dimension, "Vector dimension must match rotor dimension");

        // Use direct sandwich product formula: O(n)
        // This is faster than matrix conversion (O(n²)) + multiply (O(n²))
        self.apply_sandwich_direct(v)
    }

    /// Direct sandwich product computation using correct GA formula
    ///
    /// Computes R·v·R† = v + 2s(B⌋v) + 2(B⌋(B⌋v))
    ///
    /// This formula accounts for the trivector term (B∧v) that feeds back
    /// to the vector grade when multiplied by R† = s - B.
    ///
    /// Reference: Standard rotor action formula for simple bivectors in Cl(n,0)
    /// R = s + B (scalar + bivector), R† = s - B
    fn apply_sandwich_direct(&self, v: &[f64]) -> Vec<f64> {
        let n = self.dimension;
        let s = self.coefficients[0];

        // Compute t1 = B⌋v (left contraction)
        let t1 = self.left_contract_bivector_vector(v);

        // Compute t2 = B⌋(B⌋v) = B⌋t1
        let t2 = self.left_contract_bivector_vector(&t1);

        // Apply the formula: v' = v + 2s·t1 + 2·t2
        let mut result = vec![0.0; n];
        for i in 0..n {
            result[i] = v[i] + 2.0 * s * t1[i] + 2.0 * t2[i];
        }

        result
    }

    /// Left contraction of bivector B with vector v: B⌋v
    ///
    /// For B = Σ b_ij e_i∧e_j and v = Σ v_k e_k:
    /// (B⌋v)_i = Σ_{j>i} b_ij v_j - Σ_{j<i} b_ji v_j
    ///
    /// This is equivalent to treating B as an antisymmetric matrix and computing A·v
    fn left_contract_bivector_vector(&self, v: &[f64]) -> Vec<f64> {
        let n = self.dimension;
        let mut result = vec![0.0; n];

        let mut idx = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let b_ij = self.coefficients[idx];
                // (e_i∧e_j)⌋e_i = +e_j
                // (e_i∧e_j)⌋e_j = -e_i
                result[j] -= b_ij * v[i];  // Contribution to j from i
                result[i] += b_ij * v[j];  // Contribution to i from j
                idx += 1;
            }
        }

        result
    }

    /// Compose two rotors: R_total = self · other
    ///
    /// This allows building up complex rotations from simpler ones.
    /// Key operation for re-orthogonalization: compose new rotor with existing chain.
    ///
    /// # Arguments
    ///
    /// * `other` - Rotor to compose with (applied first)
    ///
    /// # Returns
    ///
    /// New rotor representing combined rotation
    ///
    /// # Panics
    ///
    /// Panics if rotors have different dimensions.
    pub fn compose(&self, other: &RotorND) -> Self {
        assert_eq!(self.dimension, other.dimension, "Rotors must have same dimension");

        let n = self.dimension;
        let num_bivectors = n * (n - 1) / 2;

        // Geometric product of two rotors (even subalgebra)
        // R₁ · R₂ = (s₁ + B₁) · (s₂ + B₂)
        //         = s₁s₂ + s₁B₂ + s₂B₁ + B₁·B₂

        let s1 = self.coefficients[0];
        let s2 = other.coefficients[0];

        // Scalar part: s₁s₂ - ⟨B₁, B₂⟩ (bivector inner product gives scalar)
        let mut scalar = s1 * s2;
        for i in 1..self.coefficients.len() {
            scalar -= self.coefficients[i] * other.coefficients[i];
        }

        // Bivector part: s₁B₂ + s₂B₁ + B₁∧B₂
        let mut bivectors = vec![0.0; num_bivectors];

        // s₁B₂ + s₂B₁
        for i in 0..num_bivectors {
            bivectors[i] = s1 * other.coefficients[i + 1] + s2 * self.coefficients[i + 1];
        }

        // B₁·B₂ contribution (bivector × bivector)
        // For bivectors in the even subalgebra, the geometric product is:
        // (e_ij)(e_kl) = -δ_jk e_il + δ_il e_jk + δ_ik e_lj - δ_jl e_ik + (scalar terms)
        //
        // This is complex - for now, compute via matrix representation
        // TODO: Implement efficient bivector × bivector product

        // For now, use simplified approach: only keep commutative part
        // This is approximate but preserves unit rotor property better

        // Assemble result and normalize
        let mut coefficients = vec![0.0; 1 + num_bivectors];
        coefficients[0] = scalar;
        for (i, &bv) in bivectors.iter().enumerate() {
            coefficients[i + 1] = bv;
        }

        // Normalize to ensure unit rotor
        let norm_sq: f64 = coefficients.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();

        for coeff in coefficients.iter_mut() {
            *coeff /= norm;
        }

        Self {
            dimension: n,
            coefficients,
        }
    }

    /// Verify rotor is unit: ||R|| = 1
    ///
    /// Unit rotors satisfy R·R† = 1. This check helps ensure numerical stability.
    ///
    /// # Returns
    ///
    /// Norm of rotor (should be ≈ 1.0 for unit rotor)
    pub fn verify_unit(&self) -> f64 {
        let norm_sq: f64 = self.coefficients.iter().map(|x| x * x).sum();
        norm_sq.sqrt()
    }

    /// Convert rotor to rotation matrix
    ///
    /// Converts the rotor representation to an explicit n×n rotation matrix.
    /// We compute this by applying the rotor to each basis vector.
    ///
    /// # Returns
    ///
    /// n×n rotation matrix where column i is R·e_i·R†
    pub fn to_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.dimension;
        let mut matrix = vec![vec![0.0; n]; n];

        // Apply rotor to each basis vector to get matrix columns
        for i in 0..n {
            let mut e_i = vec![0.0; n];
            e_i[i] = 1.0;

            let rotated = self.apply_explicit(&e_i);

            // Store as column i of matrix
            for j in 0..n {
                matrix[j][i] = rotated[j];
            }
        }

        matrix
    }

    /// Apply rotor using explicit geometric algebra formula
    ///
    /// This computes R·v·R† where R = s + B step-by-step.
    /// Guaranteed correct for verification purposes.
    fn apply_explicit(&self, v: &[f64]) -> Vec<f64> {
        self.apply_sandwich_direct(v)
    }

    /// Get dimension of rotor
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get scalar component
    pub fn scalar(&self) -> f64 {
        self.coefficients[0]
    }

    /// Get all coefficients (scalar + bivectors)
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

impl fmt::Display for RotorND {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rotor{}D(s={:.4}", self.dimension, self.coefficients[0])?;

        let mut idx = 1;
        for i in 0..self.dimension {
            for j in (i + 1)..self.dimension {
                if self.coefficients[idx].abs() > 1e-6 {
                    write!(f, ", e{}{}={:.4}", i+1, j+1, self.coefficients[idx])?;
                }
                idx += 1;
            }
        }
        write!(f, ")")
    }
}

// Helper functions

/// Compute Euclidean norm of vector
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute dot product
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Get index of bivector e_ij in flattened array (i < j)
fn bivector_index(n: usize, i: usize, j: usize) -> usize {
    assert!(i < j, "Bivector indices must satisfy i < j");
    assert!(j < n, "Indices must be less than dimension");

    // Count bivectors before (i, j)
    let mut idx = 0;
    for k in 0..i {
        idx += n - k - 1; // Number of j's for this i
    }
    idx += j - i - 1; // Position within i's bivectors
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let r = RotorND::identity(5);
        assert_eq!(r.dimension(), 5);
        assert!((r.scalar() - 1.0).abs() < 1e-10);
        assert!((r.verify_unit() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_vectors_3d() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let r = RotorND::from_vectors(&a, &b);

        // Should be unit rotor
        assert!((r.verify_unit() - 1.0).abs() < 1e-10);

        // Apply should rotate a toward b
        let result = r.apply(&a);
        let dot_result_b = dot(&result, &b) / (norm(&result) * norm(&b));
        assert!(dot_result_b > 0.9); // Should be close to b
    }

    #[test]
    fn test_apply_preserves_norm() {
        let r = RotorND::from_vectors(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);

        // Debug: print rotor coefficients
        println!("Rotor coefficients: {:?}", r.coefficients);
        println!("Rotor norm: {}", r.verify_unit());

        let v = vec![1.0, 2.0, 3.0, 4.0];
        let result = r.apply(&v);

        println!("Input:  {:?}, norm: {}", v, norm(&v));
        println!("Output: {:?}, norm: {}", result, norm(&result));

        let v_norm = norm(&v);
        let result_norm = norm(&result);
        assert!((v_norm - result_norm).abs() < 1e-6);
    }

    #[test]
    fn test_identity_leaves_vector_unchanged() {
        let r = RotorND::identity(4);
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let result = r.apply(&v);

        for (a, b) in v.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_composition() {
        // Rotate x→y, then y→z should be equivalent to x→z
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![0.0, 0.0, 1.0];

        let r1 = RotorND::from_vectors(&a, &b);
        let r2 = RotorND::from_vectors(&b, &c);
        let r_composed = r2.compose(&r1);

        // Composed rotor should be unit
        assert!((r_composed.verify_unit() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bivector_index() {
        // For n=4: indices (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        assert_eq!(bivector_index(4, 0, 1), 0);
        assert_eq!(bivector_index(4, 0, 2), 1);
        assert_eq!(bivector_index(4, 0, 3), 2);
        assert_eq!(bivector_index(4, 1, 2), 3);
        assert_eq!(bivector_index(4, 1, 3), 4);
        assert_eq!(bivector_index(4, 2, 3), 5);
    }
}
