//! Workload Definitions for Privacy Analysis
//!
//! This module defines the workloads used for comparing CKKS and CliffordFHE
//! representations. Each workload is designed to perform equivalent computations
//! in both representations, enabling fair comparison of execution traces.

use serde::{Deserialize, Serialize};

/// Type of workload being executed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkloadType {
    /// Plain cosine similarity: v₁·v₂ / (|v₁||v₂|)
    Similarity,

    /// Vector normalization: v / |v|
    Normalization,

    /// Normalization followed by similarity (common pattern)
    NormalizeThenSimilarity,

    /// Dot product only: v₁·v₂
    DotProduct,

    /// L2 norm: sqrt(v·v)
    L2Norm,

    /// Projection: proj_a(b) = ((b·a)/(a·a)) * a
    Projection,

    /// Geometric product (CliffordFHE only): a ⊗ b
    GeometricProduct,

    /// Full rotation: R·v·R̃ (CliffordFHE only)
    GaRotation,
}

impl WorkloadType {
    /// Get a short identifier for the workload
    pub fn id(&self) -> &'static str {
        match self {
            WorkloadType::Similarity => "similarity",
            WorkloadType::Normalization => "normalize",
            WorkloadType::NormalizeThenSimilarity => "norm_then_sim",
            WorkloadType::DotProduct => "dot_product",
            WorkloadType::L2Norm => "l2_norm",
            WorkloadType::Projection => "projection",
            WorkloadType::GeometricProduct => "geom_product",
            WorkloadType::GaRotation => "ga_rotation",
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            WorkloadType::Similarity => "Cosine similarity: v₁·v₂ / (|v₁||v₂|)",
            WorkloadType::Normalization => "Vector normalization: v / |v|",
            WorkloadType::NormalizeThenSimilarity => "Normalize both vectors, then compute similarity",
            WorkloadType::DotProduct => "Dot product: v₁·v₂",
            WorkloadType::L2Norm => "L2 norm: sqrt(v·v)",
            WorkloadType::Projection => "Vector projection: proj_a(b)",
            WorkloadType::GeometricProduct => "Geometric product: a ⊗ b",
            WorkloadType::GaRotation => "GA rotation: R·v·R̃",
        }
    }

    /// Expected number of multiplications for this workload (rough estimate)
    pub fn expected_mults(&self) -> usize {
        match self {
            WorkloadType::Similarity => 3,        // dot + 2 norms
            WorkloadType::Normalization => 1,     // norm
            WorkloadType::NormalizeThenSimilarity => 5, // 2 norms + dot + 2 divs
            WorkloadType::DotProduct => 1,
            WorkloadType::L2Norm => 1,
            WorkloadType::Projection => 2,        // dot + scalar mult
            WorkloadType::GeometricProduct => 64, // 8×8 structure constants
            WorkloadType::GaRotation => 128,      // 2 geometric products
        }
    }
}

/// Configuration for a workload execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadConfig {
    /// Type of workload
    pub workload_type: WorkloadType,

    /// Number of input dimensions
    pub dimensions: usize,

    /// Sparsity ratio (0.0 = dense, 1.0 = all zeros)
    pub sparsity: f64,

    /// Whether to use padding to fixed size
    pub use_padding: bool,

    /// Padded size (if use_padding is true)
    pub padded_size: Option<usize>,

    /// Number of repetitions for timing
    pub repetitions: usize,

    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl WorkloadConfig {
    /// Create a new workload configuration
    pub fn new(workload_type: WorkloadType, dimensions: usize) -> Self {
        Self {
            workload_type,
            dimensions,
            sparsity: 0.0,
            use_padding: false,
            padded_size: None,
            repetitions: 1,
            seed: None,
        }
    }

    /// Set sparsity
    pub fn with_sparsity(mut self, sparsity: f64) -> Self {
        self.sparsity = sparsity.clamp(0.0, 1.0);
        self
    }

    /// Enable padding to fixed size
    pub fn with_padding(mut self, padded_size: usize) -> Self {
        self.use_padding = true;
        self.padded_size = Some(padded_size);
        self
    }

    /// Set number of repetitions
    pub fn with_repetitions(mut self, reps: usize) -> Self {
        self.repetitions = reps;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get effective size (padded or original)
    pub fn effective_size(&self) -> usize {
        self.padded_size.unwrap_or(self.dimensions)
    }
}

/// Trait for workload implementations
pub trait Workload {
    /// Get the workload type
    fn workload_type(&self) -> WorkloadType;

    /// Get the configuration
    fn config(&self) -> &WorkloadConfig;

    /// Get the representation type ("ckks" or "clifford")
    fn representation(&self) -> &'static str;

    /// Prepare the workload (encrypt inputs, etc.)
    fn prepare(&mut self);

    /// Execute the workload and return whether it succeeded
    fn execute(&mut self) -> bool;

    /// Get the result (for verification)
    fn result(&self) -> Option<Vec<f64>>;
}

/// Input vector pair for similarity workloads
#[derive(Debug, Clone)]
pub struct VectorPair {
    pub v1: Vec<f64>,
    pub v2: Vec<f64>,
}

impl VectorPair {
    /// Create a new vector pair
    pub fn new(v1: Vec<f64>, v2: Vec<f64>) -> Self {
        assert_eq!(v1.len(), v2.len(), "Vectors must have same length");
        Self { v1, v2 }
    }

    /// Create random vectors
    pub fn random(dim: usize, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let v1: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let v2: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        Self { v1, v2 }
    }

    /// Create random sparse vectors
    pub fn random_sparse(dim: usize, sparsity: f64, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let make_sparse = |rng: &mut ChaCha8Rng| -> Vec<f64> {
            (0..dim)
                .map(|_| {
                    if rng.gen::<f64>() < sparsity {
                        0.0
                    } else {
                        rng.gen_range(-1.0..1.0)
                    }
                })
                .collect()
        };

        Self {
            v1: make_sparse(&mut rng),
            v2: make_sparse(&mut rng),
        }
    }

    /// Pad vectors to a fixed size
    pub fn pad_to(&mut self, size: usize) {
        if size > self.v1.len() {
            self.v1.resize(size, 0.0);
            self.v2.resize(size, 0.0);
        }
    }

    /// Compute actual sparsity
    pub fn actual_sparsity(&self) -> f64 {
        let zeros1 = self.v1.iter().filter(|&&x| x == 0.0).count();
        let zeros2 = self.v2.iter().filter(|&&x| x == 0.0).count();
        let total = self.v1.len() + self.v2.len();
        (zeros1 + zeros2) as f64 / total as f64
    }

    /// Compute expected similarity (plaintext, for verification)
    pub fn expected_similarity(&self) -> f64 {
        let dot: f64 = self.v1.iter().zip(&self.v2).map(|(a, b)| a * b).sum();
        let norm1: f64 = self.v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = self.v2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot / (norm1 * norm2)
        }
    }

    /// Compute expected dot product
    pub fn expected_dot_product(&self) -> f64 {
        self.v1.iter().zip(&self.v2).map(|(a, b)| a * b).sum()
    }

    /// Dimension
    pub fn dim(&self) -> usize {
        self.v1.len()
    }
}

/// Multivector input for CliffordFHE workloads
#[derive(Debug, Clone)]
pub struct MultivectorPair {
    /// First multivector (8 components for Cl(3,0))
    pub mv1: [f64; 8],
    /// Second multivector
    pub mv2: [f64; 8],
}

impl MultivectorPair {
    /// Create a new multivector pair
    pub fn new(mv1: [f64; 8], mv2: [f64; 8]) -> Self {
        Self { mv1, mv2 }
    }

    /// Create from vector pair (embed vectors into GA)
    ///
    /// Maps a 3D vector to multivector: [0, x, y, z, 0, 0, 0, 0]
    /// (scalar=0, vector components, bivectors=0, trivector=0)
    pub fn from_vectors(v1: &[f64], v2: &[f64]) -> Self {
        let mut mv1 = [0.0; 8];
        let mut mv2 = [0.0; 8];

        // Copy up to 3 vector components
        for (i, &val) in v1.iter().take(3).enumerate() {
            mv1[i + 1] = val; // e₁, e₂, e₃ are components 1, 2, 3
        }

        for (i, &val) in v2.iter().take(3).enumerate() {
            mv2[i + 1] = val;
        }

        Self { mv1, mv2 }
    }

    /// Create random multivectors
    pub fn random(seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut mv1 = [0.0; 8];
        let mut mv2 = [0.0; 8];

        for i in 0..8 {
            mv1[i] = rng.gen_range(-1.0..1.0);
            mv2[i] = rng.gen_range(-1.0..1.0);
        }

        Self { mv1, mv2 }
    }

    /// Create random pure vectors (only components 1,2,3 non-zero)
    pub fn random_vectors(seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut mv1 = [0.0; 8];
        let mut mv2 = [0.0; 8];

        for i in 1..=3 {
            mv1[i] = rng.gen_range(-1.0..1.0);
            mv2[i] = rng.gen_range(-1.0..1.0);
        }

        Self { mv1, mv2 }
    }

    /// Compute geometric product in plaintext (for verification)
    pub fn expected_geometric_product(&self) -> [f64; 8] {
        geometric_product_plain(&self.mv1, &self.mv2)
    }
}

/// Plaintext geometric product for Cl(3,0)
fn geometric_product_plain(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
    let mut result = [0.0; 8];

    // Component 0 (scalar)
    result[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
        - a[4] * b[4] - a[5] * b[5] - a[6] * b[6] - a[7] * b[7];

    // Component 1 (e₁)
    result[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[4] - a[4] * b[2]
        + a[3] * b[5] - a[5] * b[3] - a[6] * b[7] + a[7] * b[6];

    // Component 2 (e₂)
    result[2] = a[0] * b[2] + a[2] * b[0] - a[1] * b[4] + a[4] * b[1]
        + a[3] * b[6] - a[6] * b[3] - a[5] * b[7] + a[7] * b[5];

    // Component 3 (e₃)
    result[3] = a[0] * b[3] + a[3] * b[0] - a[1] * b[5] + a[5] * b[1]
        - a[2] * b[6] + a[6] * b[2] - a[4] * b[7] + a[7] * b[4];

    // Component 4 (e₁₂)
    result[4] = a[0] * b[4] + a[4] * b[0] + a[1] * b[2] - a[2] * b[1]
        + a[3] * b[7] - a[7] * b[3] + a[5] * b[6] - a[6] * b[5];

    // Component 5 (e₁₃)
    result[5] = a[0] * b[5] + a[5] * b[0] + a[1] * b[3] - a[3] * b[1]
        - a[2] * b[7] + a[7] * b[2] - a[4] * b[6] + a[6] * b[4];

    // Component 6 (e₂₃)
    result[6] = a[0] * b[6] + a[6] * b[0] + a[2] * b[3] - a[3] * b[2]
        + a[1] * b[7] - a[7] * b[1] + a[4] * b[5] - a[5] * b[4];

    // Component 7 (e₁₂₃)
    result[7] = a[0] * b[7] + a[7] * b[0] + a[1] * b[6] - a[6] * b[1]
        - a[2] * b[5] + a[5] * b[2] + a[3] * b[4] - a[4] * b[3];

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_pair_random() {
        let pair = VectorPair::random(100, 42);
        assert_eq!(pair.dim(), 100);
        assert!(!pair.v1.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_vector_pair_sparse() {
        let pair = VectorPair::random_sparse(100, 0.8, 42);
        let sparsity = pair.actual_sparsity();
        // Should be roughly 80% sparse (with some variance)
        assert!(sparsity > 0.5);
    }

    #[test]
    fn test_vector_pair_similarity() {
        let pair = VectorPair::new(vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]);
        assert!((pair.expected_similarity() - 1.0).abs() < 1e-10);

        let pair2 = VectorPair::new(vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]);
        assert!(pair2.expected_similarity().abs() < 1e-10);
    }

    #[test]
    fn test_multivector_from_vectors() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];

        let pair = MultivectorPair::from_vectors(&v1, &v2);

        assert_eq!(pair.mv1, [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(pair.mv2, [0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_geometric_product_scalar() {
        // 2 ⊗ 3 = 6 (scalar × scalar)
        let a = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = geometric_product_plain(&a, &b);
        assert!((result[0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_product_vectors() {
        // e₁ ⊗ e₁ = 1 (in Cl(3,0))
        let e1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = geometric_product_plain(&e1, &e1);
        assert!((result[0] - 1.0).abs() < 1e-10);

        // e₁ ⊗ e₂ = e₁₂
        let e2 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result2 = geometric_product_plain(&e1, &e2);
        assert!((result2[4] - 1.0).abs() < 1e-10); // e₁₂ component
    }

    #[test]
    fn test_workload_config() {
        let config = WorkloadConfig::new(WorkloadType::Similarity, 512)
            .with_sparsity(0.1)
            .with_padding(1024)
            .with_repetitions(10)
            .with_seed(42);

        assert_eq!(config.dimensions, 512);
        assert_eq!(config.effective_size(), 1024);
        assert_eq!(config.repetitions, 10);
    }
}
