//! CliffordPointNet Model
//!
//! Complete neural network architecture for 3D point cloud classification
//! using Clifford algebra geometric products.

use super::encoding::encode_points;
use super::layers::{
    Activation, CliffordAggregation, CliffordClassifier, CliffordEmbedding, CliffordPooling,
    PoolingMethod,
};
use super::multivector::Multivector;
use crate::datasets::point_cloud::PointCloud;

/// Configuration for CliffordPointNet
#[derive(Debug, Clone)]
pub struct CliffordPointNetConfig {
    /// Number of points per sample
    pub num_points: usize,
    /// Embedding dimension (output of embedding layer)
    pub embed_dim: usize,
    /// Number of local aggregation layers
    pub num_agg_layers: usize,
    /// Hidden dimensions for classifier
    pub classifier_hidden: Vec<usize>,
    /// Number of output classes
    pub num_classes: usize,
    /// Activation function for hidden layers
    pub activation: Activation,
    /// Pooling method
    pub pooling: PoolingMethod,
    /// Use polynomial normalization in attention (FHE-friendly)
    pub use_poly_norm: bool,
}

impl Default for CliffordPointNetConfig {
    fn default() -> Self {
        CliffordPointNetConfig {
            num_points: 1024,
            embed_dim: 32,
            num_agg_layers: 2,
            classifier_hidden: vec![128, 64],
            num_classes: 40,
            activation: Activation::Square,
            pooling: PoolingMethod::Mean,
            use_poly_norm: true,
        }
    }
}

impl CliffordPointNetConfig {
    /// Configuration for quick experiments (smaller model)
    pub fn small() -> Self {
        CliffordPointNetConfig {
            num_points: 512,
            embed_dim: 16,
            num_agg_layers: 1,
            classifier_hidden: vec![64],
            num_classes: 40,
            activation: Activation::Square,
            pooling: PoolingMethod::Mean,
            use_poly_norm: true,
        }
    }

    /// Configuration for full experiments
    pub fn standard() -> Self {
        Self::default()
    }

    /// Configuration for maximum accuracy (larger model)
    pub fn large() -> Self {
        CliffordPointNetConfig {
            num_points: 1024,
            embed_dim: 64,
            num_agg_layers: 3,
            classifier_hidden: vec![256, 128, 64],
            num_classes: 40,
            activation: Activation::GeluApprox,
            pooling: PoolingMethod::Mean,
            use_poly_norm: true,
        }
    }

    pub fn with_num_classes(mut self, n: usize) -> Self {
        self.num_classes = n;
        self
    }

    pub fn with_num_points(mut self, n: usize) -> Self {
        self.num_points = n;
        self
    }

    pub fn with_activation(mut self, act: Activation) -> Self {
        self.activation = act;
        self
    }

    pub fn with_pooling(mut self, method: PoolingMethod) -> Self {
        self.pooling = method;
        self
    }
}

/// CliffordPointNet: Privacy-Preserving 3D Point Cloud Classifier
///
/// Architecture:
/// 1. Point Encoding: (x,y,z) → Cl(3,0) multivector
/// 2. Embedding: pointwise linear transformation
/// 3. Local Aggregation: geometric attention-based aggregation
/// 4. Global Pooling: permutation-invariant pooling
/// 5. Classification: FC layers → class logits
#[derive(Debug, Clone)]
pub struct CliffordPointNet {
    /// Point embedding layer
    pub embedding: CliffordEmbedding,
    /// Local aggregation layers
    pub aggregation_layers: Vec<CliffordAggregation>,
    /// Global pooling layer
    pub pooling: CliffordPooling,
    /// Classification head
    pub classifier: CliffordClassifier,
    /// Configuration
    pub config: CliffordPointNetConfig,
}

impl CliffordPointNet {
    /// Create new CliffordPointNet with given configuration
    pub fn new(config: CliffordPointNetConfig) -> Self {
        // Embedding: 1 input multivector per point → embed_dim channels
        let embedding = CliffordEmbedding::new(1, config.embed_dim, config.activation);

        // Local aggregation layers
        let aggregation_layers: Vec<CliffordAggregation> = (0..config.num_agg_layers)
            .map(|_| CliffordAggregation::new(config.embed_dim, config.activation, config.use_poly_norm))
            .collect();

        // Global pooling
        let pooling = CliffordPooling::new(config.embed_dim, config.embed_dim, config.pooling);

        // Classification head
        let classifier = CliffordClassifier::new(
            config.embed_dim,
            config.classifier_hidden.clone(),
            config.num_classes,
            config.activation,
        );

        CliffordPointNet {
            embedding,
            aggregation_layers,
            pooling,
            classifier,
            config,
        }
    }

    /// Forward pass on encoded point cloud
    ///
    /// Input: [num_points] multivectors (one per point)
    /// Output: [num_classes] logits
    pub fn forward_encoded(&self, points: &[Multivector]) -> Vec<f64> {
        let global_features = self.forward_features(points);
        self.classifier.forward(&global_features)
    }

    /// Forward pass through feature extraction layers only
    ///
    /// Input: [num_points] multivectors (one per point)
    /// Output: [embed_dim] global feature multivectors
    pub fn forward_features(&self, points: &[Multivector]) -> Vec<Multivector> {
        // 1. Embedding: each point → embed_dim channels
        //    Input shape: [num_points][1]
        //    Output shape: [num_points][embed_dim]
        let embedded: Vec<Vec<Multivector>> = points
            .iter()
            .map(|p| self.embedding.forward_single(&[*p]))
            .collect();

        // 2. Local aggregation layers
        let mut features = embedded;
        for agg_layer in &self.aggregation_layers {
            features = agg_layer.forward(&features);
        }

        // 3. Global pooling
        //    Input shape: [num_points][embed_dim]
        //    Output shape: [embed_dim]
        self.pooling.forward(&features)
    }

    /// Forward pass on raw point cloud
    pub fn forward(&self, pc: &PointCloud) -> Vec<f64> {
        let encoded = encode_points(pc);
        self.forward_encoded(&encoded)
    }

    /// Predict class for a point cloud
    pub fn predict(&self, pc: &PointCloud) -> usize {
        let logits = self.forward(pc);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Predict class from encoded points
    pub fn predict_encoded(&self, points: &[Multivector]) -> usize {
        let logits = self.forward_encoded(points);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute softmax probabilities from logits
    pub fn softmax(&self, logits: &[f64]) -> Vec<f64> {
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|l| (l - max_logit).exp()).sum();

        logits
            .iter()
            .map(|l| (l - max_logit).exp() / exp_sum)
            .collect()
    }

    /// Get class probabilities for a point cloud
    pub fn predict_proba(&self, pc: &PointCloud) -> Vec<f64> {
        let logits = self.forward(pc);
        self.softmax(&logits)
    }

    /// Compute cross-entropy loss
    pub fn loss(&self, pc: &PointCloud, target: usize) -> f64 {
        let probs = self.predict_proba(pc);
        let p = probs[target].max(1e-10).min(1.0 - 1e-10);
        -p.ln()
    }

    /// Compute loss from encoded points
    pub fn loss_encoded(&self, points: &[Multivector], target: usize) -> f64 {
        let logits = self.forward_encoded(points);
        let probs = self.softmax(&logits);
        let p = probs[target].max(1e-10).min(1.0 - 1e-10);
        -p.ln()
    }

    /// Evaluate accuracy on a dataset
    pub fn evaluate(&self, samples: &[PointCloud]) -> (f64, f64) {
        let mut correct = 0;
        let mut total_loss = 0.0;

        for sample in samples {
            let pred = self.predict(sample);
            let target = sample.label.unwrap_or(0);

            if pred == target {
                correct += 1;
            }
            total_loss += self.loss(sample, target);
        }

        let accuracy = correct as f64 / samples.len() as f64;
        let avg_loss = total_loss / samples.len() as f64;

        (accuracy, avg_loss)
    }

    /// Get total number of parameters
    pub fn num_params(&self) -> usize {
        let mut total = self.embedding.num_params();
        for layer in &self.aggregation_layers {
            total += layer.num_params();
        }
        total += self.pooling.num_params();
        total += self.classifier.num_params();
        total
    }

    /// Print model summary
    pub fn summary(&self) {
        println!("\nCliffordPointNet Summary");
        println!("========================");
        println!("Input points: {}", self.config.num_points);
        println!("Embed dim: {}", self.config.embed_dim);
        println!("Aggregation layers: {}", self.config.num_agg_layers);
        println!("Classifier hidden: {:?}", self.config.classifier_hidden);
        println!("Output classes: {}", self.config.num_classes);
        println!("Activation: {:?}", self.config.activation);
        println!("Pooling: {:?}", self.config.pooling);
        println!("Total parameters: {}", self.num_params());
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::point_cloud::Point3D;

    fn make_test_cloud(num_points: usize, label: usize) -> PointCloud {
        let points: Vec<Point3D> = (0..num_points)
            .map(|i| {
                let t = i as f64 / num_points as f64 * 2.0 * std::f64::consts::PI;
                Point3D::new(t.cos(), t.sin(), 0.5)
            })
            .collect();
        PointCloud::from_points_with_label(points, label)
    }

    #[test]
    fn test_model_creation() {
        let config = CliffordPointNetConfig::small().with_num_classes(10);
        let model = CliffordPointNet::new(config);

        assert!(model.num_params() > 0);
    }

    #[test]
    fn test_forward_pass() {
        let config = CliffordPointNetConfig {
            num_points: 64,
            embed_dim: 8,
            num_agg_layers: 1,
            classifier_hidden: vec![16],
            num_classes: 3,
            activation: Activation::Square,
            pooling: PoolingMethod::Mean,
            use_poly_norm: true,
        };

        let model = CliffordPointNet::new(config);
        let pc = make_test_cloud(64, 0);

        let logits = model.forward(&pc);

        assert_eq!(logits.len(), 3);

        // Logits should be finite
        for &l in &logits {
            assert!(l.is_finite(), "Logit should be finite: {}", l);
        }
    }

    #[test]
    fn test_prediction() {
        let config = CliffordPointNetConfig {
            num_points: 32,
            embed_dim: 4,
            num_agg_layers: 1,
            classifier_hidden: vec![8],
            num_classes: 3,
            activation: Activation::Square,
            pooling: PoolingMethod::Mean,
            use_poly_norm: true,
        };

        let model = CliffordPointNet::new(config);
        let pc = make_test_cloud(32, 1);

        let pred = model.predict(&pc);
        assert!(pred < 3);

        let proba = model.predict_proba(&pc);
        assert_eq!(proba.len(), 3);

        // Probabilities should sum to 1
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_loss() {
        let config = CliffordPointNetConfig {
            num_points: 32,
            embed_dim: 4,
            num_agg_layers: 1,
            classifier_hidden: vec![8],
            num_classes: 3,
            activation: Activation::Square,
            pooling: PoolingMethod::Mean,
            use_poly_norm: true,
        };

        let model = CliffordPointNet::new(config);
        let pc = make_test_cloud(32, 1);

        let loss = model.loss(&pc, 1);

        // Loss should be positive and finite
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_evaluation() {
        let config = CliffordPointNetConfig {
            num_points: 32,
            embed_dim: 4,
            num_agg_layers: 1,
            classifier_hidden: vec![8],
            num_classes: 3,
            activation: Activation::Square,
            pooling: PoolingMethod::Mean,
            use_poly_norm: true,
        };

        let model = CliffordPointNet::new(config);

        let samples: Vec<PointCloud> = (0..9)
            .map(|i| make_test_cloud(32, i % 3))
            .collect();

        let (accuracy, avg_loss) = model.evaluate(&samples);

        assert!(accuracy >= 0.0 && accuracy <= 1.0);
        assert!(avg_loss >= 0.0 && avg_loss.is_finite());
    }

    #[test]
    fn test_different_configs() {
        let configs = vec![
            CliffordPointNetConfig::small().with_num_classes(10),
            CliffordPointNetConfig::standard().with_num_classes(10),
        ];

        for config in configs {
            let model = CliffordPointNet::new(config.clone());
            let pc = make_test_cloud(config.num_points, 0);

            let logits = model.forward(&pc);
            assert_eq!(logits.len(), 10);
        }
    }
}
