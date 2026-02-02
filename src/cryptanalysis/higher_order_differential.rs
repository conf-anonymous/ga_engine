//! Higher-Order Differential Cryptanalysis using Geometric Algebra
//!
//! This module implements higher-order differential attacks using GA to achieve
//! a STRUCTURAL ADVANTAGE: fewer S-box evaluations by encoding differential
//! structure as multivectors.
//!
//! # Key Insight (from expert)
//!
//! Traditional approach:
//! - 2nd-order: Evaluate 4 points explicitly
//! - 3rd-order: Evaluate 8 points explicitly
//! - 4th-order: Evaluate 16 points explicitly
//!
//! GA approach:
//! - Encode differential structure as: a ∧ b ∧ c ∧ ∇f
//! - Extract properties without evaluating all points
//! - **Structural shortcut**: Skip intermediate computations
//!
//! This is NOT "GA is faster per operation" but "GA needs fewer operations total"

use crate::cryptanalysis::boolean_ga::BooleanMultivector;
use crate::cryptanalysis::sbox_ga::SBoxGA;

/// Statistics for differential computation
#[derive(Debug, Clone, Default)]
pub struct DifferentialStats {
    /// Number of S-box evaluations
    pub sbox_evals: usize,
    /// Number of XOR operations
    pub xor_ops: usize,
    /// Number of multivector operations (GA only)
    pub multivector_ops: usize,
}

/// 2nd-order differential: Δ²f(x, a, b)
///
/// Computes: f(x) ⊕ f(x⊕a) ⊕ f(x⊕b) ⊕ f(x⊕a⊕b)
///
/// This is the "differential cube" - evaluates function at 4 corners of a cube.
pub struct SecondOrderDifferential {
    /// The S-box being analyzed
    sbox: SBoxGA,
}

impl SecondOrderDifferential {
    /// Create analyzer for given S-box
    pub fn new(sbox: SBoxGA) -> Self {
        Self { sbox }
    }

    /// Compute 2nd-order differential - BASELINE method
    ///
    /// Explicitly evaluates all 4 points:
    /// - f(x)
    /// - f(x⊕a)
    /// - f(x⊕b)
    /// - f(x⊕a⊕b)
    ///
    /// Returns: (result, statistics)
    pub fn compute_baseline(&self, x: u8, a: u8, b: u8) -> (u8, DifferentialStats) {
        let mut stats = DifferentialStats::default();

        // Evaluate all 4 corners
        let f_x = self.sbox.apply(x);
        stats.sbox_evals += 1;

        let f_x_a = self.sbox.apply(x ^ a);
        stats.sbox_evals += 1;

        let f_x_b = self.sbox.apply(x ^ b);
        stats.sbox_evals += 1;

        let f_x_ab = self.sbox.apply(x ^ a ^ b);
        stats.sbox_evals += 1;

        // XOR all results
        let result = f_x ^ f_x_a ^ f_x_b ^ f_x_ab;
        stats.xor_ops += 3;

        (result, stats)
    }

    /// Compute 2nd-order differential - GA method
    ///
    /// Uses geometric algebra to encode the differential structure.
    /// Key insight: Can extract properties without evaluating all points!
    ///
    /// For now, this is a placeholder showing the structure.
    /// Full implementation would use wedge products to collapse evaluations.
    pub fn compute_ga(&self, x: u8, a: u8, b: u8) -> (u8, DifferentialStats) {
        let mut stats = DifferentialStats::default();

        // Create multivectors for directions a and b
        let mv_a = BooleanMultivector::from_bitvec(a, self.sbox.n);
        let mv_b = BooleanMultivector::from_bitvec(b, self.sbox.n);

        // Wedge product encodes the differential direction
        let diff_direction = mv_a.wedge(&mv_b);
        stats.multivector_ops += 1;

        // Key optimization: If a ∧ b = 0 (linearly dependent),
        // the 2nd-order differential is automatically zero!
        if diff_direction.is_zero() {
            // Saved 4 S-box evaluations!
            return (0, stats);
        }

        // For non-zero case, we still need to evaluate points
        // But we can potentially use the multivector structure to:
        // 1. Identify which evaluations are redundant
        // 2. Cache intermediate results
        // 3. Exploit symmetries

        // For initial implementation, fall back to baseline
        // (Future: implement true GA shortcut)
        let f_x = self.sbox.apply(x);
        stats.sbox_evals += 1;

        let f_x_a = self.sbox.apply(x ^ a);
        stats.sbox_evals += 1;

        let f_x_b = self.sbox.apply(x ^ b);
        stats.sbox_evals += 1;

        let f_x_ab = self.sbox.apply(x ^ a ^ b);
        stats.sbox_evals += 1;

        let result = f_x ^ f_x_a ^ f_x_b ^ f_x_ab;
        stats.xor_ops += 3;

        (result, stats)
    }

    /// Compute 2nd-order differential - OPTIMIZED GA method
    ///
    /// Key insight: For grade-1 multivectors (bit vectors), the wedge product test
    /// reduces to a simple bit-AND operation:
    ///   a ∧ b = 0  ⟺  a & b ≠ 0 (share a common bit)
    ///
    /// This is O(1) instead of O(2^n × 2^n) for the full multivector wedge product!
    pub fn compute_ga_optimized(&self, x: u8, a: u8, b: u8) -> (u8, DifferentialStats) {
        let mut stats = DifferentialStats::default();

        // Fast check: if a and b share any bits, they're linearly dependent
        // In that case, a ∧ b = 0 and the 2nd-order differential is zero
        if a & b != 0 {
            stats.multivector_ops += 1; // Just one AND operation
            return (0, stats);
        }

        // If a and b are independent, compute the differential normally
        let f_x = self.sbox.apply(x);
        stats.sbox_evals += 1;

        let f_x_a = self.sbox.apply(x ^ a);
        stats.sbox_evals += 1;

        let f_x_b = self.sbox.apply(x ^ b);
        stats.sbox_evals += 1;

        let f_x_ab = self.sbox.apply(x ^ a ^ b);
        stats.sbox_evals += 1;

        let result = f_x ^ f_x_a ^ f_x_b ^ f_x_ab;
        stats.xor_ops += 3;

        (result, stats)
    }

    /// Compute distribution of 2nd-order differentials
    ///
    /// For all x, compute Δ²f(x, a, b) and build histogram.
    /// This is used in higher-order differential attacks.
    ///
    /// Returns: (distribution, total_stats)
    pub fn compute_distribution_baseline(&self, a: u8, b: u8) -> (Vec<usize>, DifferentialStats) {
        let size = self.sbox.size();
        let mut distribution = vec![0usize; size];
        let mut total_stats = DifferentialStats::default();

        for x in 0..size {
            let (diff, stats) = self.compute_baseline(x as u8, a, b);
            distribution[diff as usize] += 1;

            total_stats.sbox_evals += stats.sbox_evals;
            total_stats.xor_ops += stats.xor_ops;
        }

        (distribution, total_stats)
    }

    /// Compute distribution using GA method
    pub fn compute_distribution_ga(&self, a: u8, b: u8) -> (Vec<usize>, DifferentialStats) {
        let size = self.sbox.size();
        let mut distribution = vec![0usize; size];
        let mut total_stats = DifferentialStats::default();

        for x in 0..size {
            let (diff, stats) = self.compute_ga(x as u8, a, b);
            distribution[diff as usize] += 1;

            total_stats.sbox_evals += stats.sbox_evals;
            total_stats.xor_ops += stats.xor_ops;
            total_stats.multivector_ops += stats.multivector_ops;
        }

        (distribution, total_stats)
    }
}

/// 3rd-order differential: Δ³f(x, a, b, c)
///
/// Computes XOR of 8 points (corners of a 3D cube)
pub struct ThirdOrderDifferential {
    sbox: SBoxGA,
}

impl ThirdOrderDifferential {
    pub fn new(sbox: SBoxGA) -> Self {
        Self { sbox }
    }

    /// Compute 3rd-order differential - BASELINE method
    ///
    /// Explicitly evaluates all 8 points
    pub fn compute_baseline(&self, x: u8, a: u8, b: u8, c: u8) -> (u8, DifferentialStats) {
        let mut stats = DifferentialStats::default();

        // 8 corners of the cube
        let points = [
            x,
            x ^ a,
            x ^ b,
            x ^ a ^ b,
            x ^ c,
            x ^ a ^ c,
            x ^ b ^ c,
            x ^ a ^ b ^ c,
        ];

        // Evaluate all 8 points
        let mut result = 0u8;
        for &point in &points {
            result ^= self.sbox.apply(point);
            stats.sbox_evals += 1;
        }
        stats.xor_ops += 7;

        (result, stats)
    }

    /// Compute 3rd-order differential - GA method
    ///
    /// Uses a ∧ b ∧ c to encode the structure
    pub fn compute_ga(&self, x: u8, a: u8, b: u8, c: u8) -> (u8, DifferentialStats) {
        let mut stats = DifferentialStats::default();

        // Create multivectors
        let mv_a = BooleanMultivector::from_bitvec(a, self.sbox.n);
        let mv_b = BooleanMultivector::from_bitvec(b, self.sbox.n);
        let mv_c = BooleanMultivector::from_bitvec(c, self.sbox.n);

        // Triple wedge product
        let ab = mv_a.wedge(&mv_b);
        let abc = ab.wedge(&mv_c);
        stats.multivector_ops += 2;

        // If a, b, c are linearly dependent, differential is zero
        if abc.is_zero() {
            // Saved 8 S-box evaluations!
            return (0, stats);
        }

        // For non-zero case, evaluate (future: optimize using structure)
        let points = [
            x,
            x ^ a,
            x ^ b,
            x ^ a ^ b,
            x ^ c,
            x ^ a ^ c,
            x ^ b ^ c,
            x ^ a ^ b ^ c,
        ];

        let mut result = 0u8;
        for &point in &points {
            result ^= self.sbox.apply(point);
            stats.sbox_evals += 1;
        }
        stats.xor_ops += 7;

        (result, stats)
    }

    /// Compute 3rd-order differential - OPTIMIZED GA method
    ///
    /// Key insight: For grade-1 vectors (bit vectors), linear dependence test is:
    ///   a ∧ b ∧ c = 0  ⟺  any pair shares bits OR c is in span{a,b}
    ///
    /// Simplified test for bit vectors:
    ///   - If any two of {a,b,c} share bits: dependent
    ///   - If c = 0 or a = 0 or b = 0: dependent
    ///
    /// This is O(1) instead of O(2^n × 2^n) multivector operations!
    pub fn compute_ga_optimized(&self, x: u8, a: u8, b: u8, c: u8) -> (u8, DifferentialStats) {
        let mut stats = DifferentialStats::default();

        // Fast linear dependence check for bit vectors
        // Three vectors are dependent if any pair overlaps
        if (a & b) != 0 || (a & c) != 0 || (b & c) != 0 {
            stats.multivector_ops += 1;
            return (0, stats);
        }

        // Also check if any vector is zero
        if a == 0 || b == 0 || c == 0 {
            stats.multivector_ops += 1;
            return (0, stats);
        }

        // If all independent, evaluate the 8 points
        let points = [
            x,
            x ^ a,
            x ^ b,
            x ^ a ^ b,
            x ^ c,
            x ^ a ^ c,
            x ^ b ^ c,
            x ^ a ^ b ^ c,
        ];

        let mut result = 0u8;
        for &point in &points {
            result ^= self.sbox.apply(point);
            stats.sbox_evals += 1;
        }
        stats.xor_ops += 7;

        (result, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_second_order_differential_identity() {
        // Identity S-box
        let sbox = SBoxGA::from_lut((0..16).collect(), 4);
        let diff = SecondOrderDifferential::new(sbox);

        // For identity S-box: Δ²f(x,a,b) = x ⊕ (x⊕a) ⊕ (x⊕b) ⊕ (x⊕a⊕b) = 0
        let (result, _stats) = diff.compute_baseline(5, 3, 6);
        assert_eq!(result, 0, "2nd-order diff of identity should be 0");
    }

    #[test]
    fn test_second_order_ga_matches_baseline() {
        let sbox = SBoxGA::from_lut(
            vec![0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8, 0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7],
            4,
        );
        let diff = SecondOrderDifferential::new(sbox);

        // Test multiple cases
        for x in 0..16u8 {
            for a in 1..16u8 {
                for b in 1..16u8 {
                    let (result_baseline, _) = diff.compute_baseline(x, a, b);
                    let (result_ga, _) = diff.compute_ga(x, a, b);
                    assert_eq!(
                        result_baseline, result_ga,
                        "GA and baseline should match for x={}, a={}, b={}",
                        x, a, b
                    );
                }
            }
        }
    }

    #[test]
    fn test_third_order_differential() {
        let sbox = SBoxGA::from_lut((0..16).collect(), 4);
        let diff = ThirdOrderDifferential::new(sbox);

        // For identity: 3rd-order diff should be 0
        let (result, stats) = diff.compute_baseline(5, 1, 2, 4);
        assert_eq!(result, 0);
        assert_eq!(stats.sbox_evals, 8, "Should evaluate 8 points");
    }

    #[test]
    fn test_ga_saves_evals_on_dependent_vectors() {
        let sbox = SBoxGA::from_lut((0..16).collect(), 4);
        let diff = SecondOrderDifferential::new(sbox);

        // If a = b, they're linearly dependent → a ∧ b = 0
        let (result_ga, stats_ga) = diff.compute_ga(5, 3, 3);

        assert_eq!(result_ga, 0, "Dependent vectors → zero differential");
        assert_eq!(stats_ga.sbox_evals, 0, "Should save all S-box evals!");
        assert_eq!(stats_ga.multivector_ops, 1, "Just one wedge product");
    }
}
