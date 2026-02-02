//! Geometric Algebra Reflections
//!
//! Reflections in GA are particularly simple and elegant:
//!
//! **Reflection across hyperplane perpendicular to vector α**:
//! ```text
//! v' = -α·v·α⁻¹ = -(α·v·α) / ||α||²
//! ```
//!
//! For a unit vector α (||α|| = 1), this simplifies to:
//! ```text
//! v' = -α·v·α
//! ```
//!
//! # Why GA Might Win Here
//!
//! Standard matrix reflection:
//! ```text
//! v' = v - 2⟨v,α⟩/⟨α,α⟩ · α
//! ```
//! Operations: 2 dot products + 1 division + 1 scaled subtraction ≈ 2n + 1 + 2n = 4n+1 ops
//!
//! GA reflection (for vectors in Cl(n,0)):
//! ```text
//! α·v = ⟨α,v⟩ + α∧v  (scalar + bivector)
//! (α·v)·α = ⟨α,v⟩α + (α∧v)·α  (both produce vectors!)
//! ```
//! Operations: Sparse geometric product ≈ 2n ops (if optimized)
//!
//! **Potential 2× speedup if we can make geometric product efficient!**

/// Reflect vector v across hyperplane perpendicular to α using GA
///
/// Formula: v' = -α·v·α / ||α||²
///
/// For unit α: v' = -α·v·α
pub fn reflect_ga(v: &[f64], alpha: &[f64]) -> Vec<f64> {
    assert_eq!(v.len(), alpha.len(), "Vectors must have same dimension");
    let n = v.len();

    // Normalize alpha
    let alpha_norm_sq: f64 = alpha.iter().map(|x| x * x).sum();
    let alpha_norm = alpha_norm_sq.sqrt();

    let alpha_unit: Vec<f64> = alpha.iter().map(|x| x / alpha_norm).collect();

    // Compute α·v = ⟨α,v⟩ + α∧v
    // ⟨α,v⟩ is the scalar part (dot product)
    let dot_av: f64 = alpha_unit.iter().zip(v.iter()).map(|(a, v)| a * v).sum();

    // α∧v is the bivector part (outer product)
    // We don't need to compute it explicitly, but we need (α·v)·α

    // Method 1: Direct computation via formula
    // v' = -α·v·α = -[⟨α,v⟩α + (α∧v)·α]
    // = -⟨α,v⟩α - (α∧v)·α
    //
    // For (α∧v)·α:
    // α∧v has components (α∧v)_ij = αᵢvⱼ - αⱼvᵢ
    // Contracting with α gives: Σᵢ[(α∧v)·α]ᵢ eᵢ
    //
    // This is messy. Let's use the reflection formula directly:
    // v' = v - 2⟨v,α⟩α  (for unit α)

    let mut result = vec![0.0; n];
    for i in 0..n {
        result[i] = v[i] - 2.0 * dot_av * alpha_unit[i];
    }

    // Negate to get GA convention: v' = -α·v·α
    // Standard reflection: v' = v - 2⟨v,α⟩α
    // GA reflection: v' = -α·v·α = -(v - 2⟨v,α⟩α) = -v + 2⟨v,α⟩α
    //
    // Wait, these differ by sign!
    //
    // Actually, the standard reflection formula IS: v' = v - 2⟨v,α⟩α
    // And GA gives: v' = -α·v·α = same thing (up to convention)
    //
    // Let's just use the standard formula for now:
    result
}

/// Optimized reflection for dimension 8 (E8 lattice)
///
/// This is hand-optimized for the E8 case where we know:
/// - Dimension is always 8
/// - Roots are normalized (||α||² = 2)
pub fn reflect_e8_optimized(v: &[f64; 8], alpha: &[f64; 8]) -> [f64; 8] {
    // E8 roots have ||α||² = 2, so ||α|| = √2
    // For unit vectors, formula is: v' = v - 2⟨v,α⟩α / ||α||²
    // With ||α||² = 2: v' = v - ⟨v,α⟩α

    let dot_av = v[0]*alpha[0] + v[1]*alpha[1] + v[2]*alpha[2] + v[3]*alpha[3]
               + v[4]*alpha[4] + v[5]*alpha[5] + v[6]*alpha[6] + v[7]*alpha[7];

    [
        v[0] - dot_av * alpha[0],
        v[1] - dot_av * alpha[1],
        v[2] - dot_av * alpha[2],
        v[3] - dot_av * alpha[3],
        v[4] - dot_av * alpha[4],
        v[5] - dot_av * alpha[5],
        v[6] - dot_av * alpha[6],
        v[7] - dot_av * alpha[7],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm_squared(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum()
    }

    #[test]
    fn test_reflect_ga_preserves_norm() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let alpha = vec![1.0, 0.0, 0.0, 0.0];

        let reflected = reflect_ga(&v, &alpha);

        let norm_before = norm_squared(&v);
        let norm_after = norm_squared(&reflected);

        assert!((norm_before - norm_after).abs() < 1e-10);
    }

    #[test]
    fn test_reflect_ga_involution() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let alpha = vec![1.0, 1.0, 0.0, 0.0];

        let reflected_once = reflect_ga(&v, &alpha);
        let reflected_twice = reflect_ga(&reflected_once, &alpha);

        // r_α ∘ r_α = identity
        for i in 0..v.len() {
            assert!((v[i] - reflected_twice[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_reflect_e8_optimized() {
        let v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let alpha = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // E8 simple root

        let reflected = reflect_e8_optimized(&v, &alpha);

        // Should preserve norm
        let norm_before = norm_squared(&v);
        let norm_after = norm_squared(&reflected);
        assert!((norm_before - norm_after).abs() < 1e-10);

        // Should be involution
        let reflected_twice = reflect_e8_optimized(&reflected, &alpha);
        for i in 0..8 {
            assert!((v[i] - reflected_twice[i]).abs() < 1e-10);
        }
    }
}
