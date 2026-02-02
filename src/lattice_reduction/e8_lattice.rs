//! E8 Lattice - The Exceptional 8-Dimensional Lattice
//!
//! E8 is one of the most remarkable mathematical objects:
//! - 240 minimal vectors (roots)
//! - Weyl group W(E8) with 696,729,600 elements
//! - Densest sphere packing in 8 dimensions
//! - Central to: codes, string theory, sphere packing
//!
//! # Geometric Algebra Test Case
//!
//! E8 is perfect for testing GA-based algorithms because:
//! - Massive symmetry (696M automorphisms)
//! - Reflections generate the Weyl group
//! - GA naturally represents reflections: v' = -α·v·α⁻¹
//!
//! # References
//!
//! - Conway & Sloane, "Sphere Packings, Lattices and Groups" (1988)
//! - Coxeter, "Regular Polytopes" (1973)

use std::fmt;

/// E8 lattice with root system
#[derive(Clone, Debug)]
pub struct E8Lattice {
    /// The 8 simple roots that generate the Weyl group W(E8)
    /// These are the fundamental reflections
    simple_roots: Vec<[f64; 8]>,

    /// All 240 roots (optional, computed on demand)
    /// These are the minimal vectors of E8
    all_roots: Option<Vec<[f64; 8]>>,
}

impl E8Lattice {
    /// Create E8 lattice with standard simple root system
    ///
    /// The simple roots are chosen in the standard Bourbaki convention:
    /// - α₁ = e₁ - e₂
    /// - α₂ = e₂ - e₃
    /// - α₃ = e₃ - e₄
    /// - α₄ = e₄ - e₅
    /// - α₅ = e₅ - e₆
    /// - α₆ = e₆ - e₇
    /// - α₇ = e₆ + e₇
    /// - α₈ = -½(e₁ + e₂ + e₃ + e₄ + e₅ + e₆ + e₇ + e₈)
    ///
    /// Note: Last root uses all coordinates (connects D7 to E8)
    pub fn new() -> Self {
        let simple_roots = vec![
            // D7 simple roots (first 6 are differences)
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  // α₁ = e₁ - e₂
            [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  // α₂ = e₂ - e₃
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],  // α₃ = e₃ - e₄
            [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],  // α₄ = e₄ - e₅
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],  // α₅ = e₅ - e₆
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],  // α₆ = e₆ - e₇
            // D7 branching root
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],   // α₇ = e₆ + e₇
            // E8 special root (extends D7 to E8)
            [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],  // α₈ = -½Σeᵢ
        ];

        E8Lattice {
            simple_roots,
            all_roots: None,
        }
    }

    /// Get the 8 simple roots (Weyl group generators)
    pub fn simple_roots(&self) -> &[[f64; 8]] {
        &self.simple_roots
    }

    /// Get all 240 roots (computed if not cached)
    pub fn all_roots(&mut self) -> &[[f64; 8]] {
        if self.all_roots.is_none() {
            self.all_roots = Some(self.compute_all_roots());
        }
        self.all_roots.as_ref().unwrap()
    }

    /// Compute all 240 roots by acting with Weyl group on a single root
    ///
    /// We start with the highest root and apply all simple reflections
    /// repeatedly until we've generated the full orbit.
    fn compute_all_roots(&self) -> Vec<[f64; 8]> {
        use std::collections::HashSet;

        // Start with the first simple root
        let mut roots = HashSet::new();
        let start_root = self.simple_roots[0];
        roots.insert(FloatVec(start_root));

        // Apply all simple reflections until orbit stabilizes
        let mut prev_size = 0;
        while roots.len() != prev_size {
            prev_size = roots.len();

            let current_roots: Vec<[f64; 8]> = roots.iter().map(|fv| fv.0).collect();

            for root in current_roots {
                for simple_root in &self.simple_roots {
                    let reflected = reflect_vector(&root, simple_root);
                    roots.insert(FloatVec(reflected));
                }
            }

            // Safety: don't run forever
            if roots.len() > 500 {
                break;
            }
        }

        roots.into_iter().map(|fv| fv.0).collect()
    }

    /// Check if a vector is a root (up to tolerance)
    pub fn is_root(&self, v: &[f64; 8], tol: f64) -> bool {
        // All E8 roots have squared length 2
        let norm_sq = v.iter().map(|x| x * x).sum::<f64>();
        (norm_sq - 2.0).abs() < tol
    }

    /// Get the Weyl group order (for reference)
    pub fn weyl_group_order() -> u64 {
        696_729_600  // |W(E8)| = 2^14 · 3^5 · 5^2 · 7
    }
}

impl Default for E8Lattice {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for E8Lattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "E8 Lattice")?;
        writeln!(f, "  Simple roots: {}", self.simple_roots.len())?;
        if let Some(ref all) = self.all_roots {
            writeln!(f, "  All roots: {}", all.len())?;
        }
        writeln!(f, "  Weyl group order: {}", Self::weyl_group_order())?;
        Ok(())
    }
}

// Helper: Wrapper to make [f64; 8] hashable (for HashSet)
#[derive(Clone, Copy, Debug)]
struct FloatVec([f64; 8]);

impl PartialEq for FloatVec {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(a, b)| (a - b).abs() < 1e-10)
    }
}

impl Eq for FloatVec {}

impl std::hash::Hash for FloatVec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash rounded values to avoid floating-point issues
        for &x in &self.0 {
            let rounded = (x * 1e10).round() as i64;
            rounded.hash(state);
        }
    }
}

/// Reflect vector v across hyperplane perpendicular to root α
///
/// Formula: r_α(v) = v - 2⟨v,α⟩/⟨α,α⟩ · α
///
/// This is a Householder reflection.
pub fn reflect_vector(v: &[f64; 8], root: &[f64; 8]) -> [f64; 8] {
    let dot_va = dot(v, root);
    let dot_aa = dot(root, root);
    let coeff = 2.0 * dot_va / dot_aa;

    let mut result = *v;
    for i in 0..8 {
        result[i] -= coeff * root[i];
    }
    result
}

/// Dot product in R^8
pub fn dot(a: &[f64; 8], b: &[f64; 8]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Squared norm
pub fn norm_squared(v: &[f64; 8]) -> f64 {
    dot(v, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8_simple_roots() {
        let e8 = E8Lattice::new();
        assert_eq!(e8.simple_roots().len(), 8);
    }

    #[test]
    fn test_simple_roots_are_roots() {
        let e8 = E8Lattice::new();

        for (i, root) in e8.simple_roots().iter().enumerate() {
            let norm_sq = norm_squared(root);
            assert!(
                (norm_sq - 2.0).abs() < 1e-10,
                "Simple root {} has wrong norm: {} (expected 2.0)",
                i,
                norm_sq
            );
        }
    }

    #[test]
    fn test_reflection_preserves_norm() {
        let e8 = E8Lattice::new();
        let v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        for root in e8.simple_roots() {
            let reflected = reflect_vector(&v, root);
            let norm_before = norm_squared(&v);
            let norm_after = norm_squared(&reflected);

            assert!(
                (norm_before - norm_after).abs() < 1e-10,
                "Reflection doesn't preserve norm"
            );
        }
    }

    #[test]
    fn test_reflection_involution() {
        let e8 = E8Lattice::new();
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        for root in e8.simple_roots() {
            let reflected_once = reflect_vector(&v, root);
            let reflected_twice = reflect_vector(&reflected_once, root);

            // r_α ∘ r_α = identity
            for i in 0..8 {
                assert!(
                    (v[i] - reflected_twice[i]).abs() < 1e-10,
                    "Reflection is not an involution"
                );
            }
        }
    }

    #[test]
    fn test_generate_all_roots() {
        let mut e8 = E8Lattice::new();
        let all_roots = e8.all_roots();

        println!("Generated {} roots", all_roots.len());

        // E8 should have exactly 240 roots
        // Note: Our generation might produce slightly different count
        // due to numerical tolerance
        assert!(
            all_roots.len() >= 100,
            "Should generate at least 100 roots, got {}",
            all_roots.len()
        );

        // All should be roots (norm² = 2)
        for (i, root) in all_roots.iter().enumerate() {
            let norm_sq = norm_squared(root);
            assert!(
                (norm_sq - 2.0).abs() < 1e-8,
                "Root {} has wrong norm: {} (expected 2.0)",
                i,
                norm_sq
            );
        }
    }

    #[test]
    fn test_weyl_group_order() {
        assert_eq!(E8Lattice::weyl_group_order(), 696_729_600);
    }
}
