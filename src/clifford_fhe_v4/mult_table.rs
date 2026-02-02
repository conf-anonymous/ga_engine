/// Multiplication Table for Packed Clifford Algebra Operations
///
/// Translates the Clifford algebra multiplication table into operations on packed ciphertexts.
///
/// **Key Difference from V2/V3:**
/// - V2/V3: Each component is a separate ciphertext → 64 ciphertext multiplications
/// - V4: All components packed → Use diagonal multiply + rotation
///
/// **Packed Layout:**
/// Slots: [s₀, e1₀, e2₀, e3₀, e12₀, e23₀, e31₀, I₀, s₁, e1₁, ...]
///
/// **Algorithm:**
/// For each output component, we need to:
/// 1. Extract relevant input components (via rotation)
/// 2. Apply diagonal masks to multiply specific components together
/// 3. Rotate and sum to get final result

/// Multiplication table entry for packed operations
///
/// Describes how to compute one term in the geometric product.
#[derive(Clone, Debug)]
pub struct PackedMultTerm {
    /// Coefficient (+1 or -1)
    pub coeff: i64,
    
    /// Component index from first multivector (0-7)
    pub a_comp: usize,
    
    /// Component index from second multivector (0-7)
    pub b_comp: usize,
    
    /// Rotation needed to align a_comp to position 0 in its 8-slot group
    pub a_rotation: i32,
    
    /// Rotation needed to align b_comp to position 0 in its 8-slot group
    pub b_rotation: i32,
}

impl PackedMultTerm {
    /// Create a new multiplication term
    pub fn new(coeff: i64, a_comp: usize, b_comp: usize) -> Self {
        Self {
            coeff,
            a_comp,
            b_comp,
            a_rotation: -(a_comp as i32), // Rotate right to align to position 0
            b_rotation: -(b_comp as i32),
        }
    }
}

/// Complete multiplication table for packed Clifford algebra Cl(3,0)
///
/// For each output component (0-7), stores the list of multiplication terms needed.
pub struct PackedMultTable {
    /// For each output component, list of terms to compute and sum
    pub terms: [Vec<PackedMultTerm>; 8],
}

impl PackedMultTable {
    /// Create the multiplication table for Cl(3,0) packed layout
    ///
    /// Basis: {1, e₁, e₂, e₃, e₁₂, e₂₃, e₃₁, e₁₂₃}
    /// Component order: [s, e1, e2, e3, e12, e23, e31, I]
    /// 
    /// Note: We use e₃₁ instead of e₁₃ to match the standard order
    pub fn new() -> Self {
        let mut terms: [Vec<PackedMultTerm>; 8] = Default::default();

        // Component 0 (scalar): vectors square to +1, bivectors to -1
        terms[0] = vec![
            PackedMultTerm::new(1, 0, 0),   // 1⊗1
            PackedMultTerm::new(1, 1, 1),   // e₁⊗e₁
            PackedMultTerm::new(1, 2, 2),   // e₂⊗e₂
            PackedMultTerm::new(1, 3, 3),   // e₃⊗e₃
            PackedMultTerm::new(-1, 4, 4),  // e₁₂⊗e₁₂
            PackedMultTerm::new(-1, 5, 5),  // e₂₃⊗e₂₃
            PackedMultTerm::new(-1, 6, 6),  // e₃₁⊗e₃₁
            PackedMultTerm::new(-1, 7, 7),  // I⊗I
        ];

        // Component 1 (e₁)
        terms[1] = vec![
            PackedMultTerm::new(1, 0, 1),   // 1⊗e₁
            PackedMultTerm::new(1, 1, 0),   // e₁⊗1
            PackedMultTerm::new(1, 2, 4),   // e₂⊗e₁₂
            PackedMultTerm::new(-1, 4, 2),  // e₁₂⊗e₂
            PackedMultTerm::new(-1, 3, 6),  // e₃⊗e₃₁
            PackedMultTerm::new(1, 6, 3),   // e₃₁⊗e₃
            PackedMultTerm::new(-1, 5, 7),  // e₂₃⊗I
            PackedMultTerm::new(1, 7, 5),   // I⊗e₂₃
        ];

        // Component 2 (e₂)
        terms[2] = vec![
            PackedMultTerm::new(1, 0, 2),   // 1⊗e₂
            PackedMultTerm::new(1, 2, 0),   // e₂⊗1
            PackedMultTerm::new(-1, 1, 4),  // e₁⊗e₁₂
            PackedMultTerm::new(1, 4, 1),   // e₁₂⊗e₁
            PackedMultTerm::new(1, 3, 5),   // e₃⊗e₂₃
            PackedMultTerm::new(-1, 5, 3),  // e₂₃⊗e₃
            PackedMultTerm::new(1, 6, 7),   // e₃₁⊗I
            PackedMultTerm::new(-1, 7, 6),  // I⊗e₃₁
        ];

        // Component 3 (e₃)
        terms[3] = vec![
            PackedMultTerm::new(1, 0, 3),   // 1⊗e₃
            PackedMultTerm::new(1, 3, 0),   // e₃⊗1
            PackedMultTerm::new(1, 1, 6),   // e₁⊗e₃₁
            PackedMultTerm::new(-1, 6, 1),  // e₃₁⊗e₁
            PackedMultTerm::new(-1, 2, 5),  // e₂⊗e₂₃
            PackedMultTerm::new(1, 5, 2),   // e₂₃⊗e₂
            PackedMultTerm::new(-1, 4, 7),  // e₁₂⊗I
            PackedMultTerm::new(1, 7, 4),   // I⊗e₁₂
        ];

        // Component 4 (e₁₂)
        terms[4] = vec![
            PackedMultTerm::new(1, 0, 4),   // 1⊗e₁₂
            PackedMultTerm::new(1, 4, 0),   // e₁₂⊗1
            PackedMultTerm::new(1, 1, 2),   // e₁⊗e₂
            PackedMultTerm::new(-1, 2, 1),  // e₂⊗e₁
            PackedMultTerm::new(1, 3, 7),   // e₃⊗I
            PackedMultTerm::new(-1, 7, 3),  // I⊗e₃
            PackedMultTerm::new(-1, 6, 5),  // e₃₁⊗e₂₃
            PackedMultTerm::new(1, 5, 6),   // e₂₃⊗e₃₁
        ];

        // Component 5 (e₂₃)
        terms[5] = vec![
            PackedMultTerm::new(1, 0, 5),   // 1⊗e₂₃
            PackedMultTerm::new(1, 5, 0),   // e₂₃⊗1
            PackedMultTerm::new(1, 2, 3),   // e₂⊗e₃
            PackedMultTerm::new(-1, 3, 2),  // e₃⊗e₂
            PackedMultTerm::new(-1, 1, 7),  // e₁⊗I
            PackedMultTerm::new(1, 7, 1),   // I⊗e₁
            PackedMultTerm::new(1, 4, 6),   // e₁₂⊗e₃₁
            PackedMultTerm::new(-1, 6, 4),  // e₃₁⊗e₁₂
        ];

        // Component 6 (e₃₁)
        terms[6] = vec![
            PackedMultTerm::new(1, 0, 6),   // 1⊗e₃₁
            PackedMultTerm::new(1, 6, 0),   // e₃₁⊗1
            PackedMultTerm::new(1, 3, 1),   // e₃⊗e₁
            PackedMultTerm::new(-1, 1, 3),  // e₁⊗e₃
            PackedMultTerm::new(1, 2, 7),   // e₂⊗I
            PackedMultTerm::new(-1, 7, 2),  // I⊗e₂
            PackedMultTerm::new(-1, 4, 5),  // e₁₂⊗e₂₃
            PackedMultTerm::new(1, 5, 4),   // e₂₃⊗e₁₂
        ];

        // Component 7 (I = e₁₂₃)
        terms[7] = vec![
            PackedMultTerm::new(1, 0, 7),   // 1⊗I
            PackedMultTerm::new(1, 7, 0),   // I⊗1
            PackedMultTerm::new(1, 1, 5),   // e₁⊗e₂₃
            PackedMultTerm::new(-1, 5, 1),  // e₂₃⊗e₁
            PackedMultTerm::new(-1, 2, 6),  // e₂⊗e₃₁
            PackedMultTerm::new(1, 6, 2),   // e₃₁⊗e₂
            PackedMultTerm::new(1, 3, 4),   // e₃⊗e₁₂
            PackedMultTerm::new(-1, 4, 3),  // e₁₂⊗e₃
        ];

        Self { terms }
    }

    /// Get terms for computing a specific output component
    pub fn get_terms(&self, output_comp: usize) -> &[PackedMultTerm] {
        assert!(output_comp < 8, "Component index must be 0-7");
        &self.terms[output_comp]
    }

    /// Count total number of multiplication terms (for performance analysis)
    pub fn total_terms(&self) -> usize {
        self.terms.iter().map(|v| v.len()).sum()
    }
}

impl Default for PackedMultTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mult_table_structure() {
        let table = PackedMultTable::new();

        // Each component should have 8 terms
        for comp in 0..8 {
            assert_eq!(table.get_terms(comp).len(), 8,
                "Component {} should have 8 terms", comp);
        }

        // Total: 8 components × 8 terms = 64 multiplications
        assert_eq!(table.total_terms(), 64);
    }

    #[test]
    fn test_scalar_component() {
        let table = PackedMultTable::new();
        let terms = table.get_terms(0);

        // First term should be 1⊗1 → scalar
        assert_eq!(terms[0].coeff, 1);
        assert_eq!(terms[0].a_comp, 0);
        assert_eq!(terms[0].b_comp, 0);

        // e₁⊗e₁ should contribute to scalar with coeff +1
        assert_eq!(terms[1].coeff, 1);
        assert_eq!(terms[1].a_comp, 1);
        assert_eq!(terms[1].b_comp, 1);

        // e₁₂⊗e₁₂ should contribute to scalar with coeff -1
        assert_eq!(terms[4].coeff, -1);
        assert_eq!(terms[4].a_comp, 4);
        assert_eq!(terms[4].b_comp, 4);
    }

    #[test]
    fn test_rotation_calculations() {
        let table = PackedMultTable::new();
        
        // For e₂ component extraction, should rotate right by 2
        let e2_term = table.get_terms(1).iter()
            .find(|t| t.a_comp == 2 && t.b_comp == 4)
            .unwrap();
        
        assert_eq!(e2_term.a_rotation, -2);
        assert_eq!(e2_term.b_rotation, -4);
    }
}
