//! GP Feature Classifier for Encrypted Inference
//!
//! A lightweight classifier that operates on geometric product features
//! extracted from Cl(3,0) multivectors. Designed for the encrypted pipeline:
//!
//! ```text
//! Points → Cl(3,0) encoding → GP self-product → mean pool → classifier → logits
//! ```
//!
//! The encoding, GP, and mean pooling run encrypted (server-side).
//! The classifier weights are public (plaintext) constants.

use crate::datasets::point_cloud::PointCloud;
use rand::Rng;

use super::serialization::GPClassifierWeights;

/// Classifier trained on mean-pooled geometric product features.
///
/// Architecture:
///   [8] → [hidden_dim] (square activation) → [num_classes] (linear)
///
/// The input is the mean of GP(point_i, point_i) over all N points,
/// where each point is encoded as a Cl(3,0) multivector.
#[derive(Debug, Clone)]
pub struct GPFeatureClassifier {
    /// Layer 1 weights: [hidden_dim][8]
    pub layer1_w: Vec<Vec<f64>>,
    pub layer1_b: Vec<f64>,

    /// Classifier weights: [num_classes][hidden_dim]
    pub classifier_w: Vec<Vec<f64>>,
    pub classifier_b: Vec<f64>,

    pub hidden_dim: usize,
    pub num_classes: usize,
}

impl GPFeatureClassifier {
    pub fn new(hidden_dim: usize, num_classes: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let scale1 = (2.0 / 8.0_f64).sqrt();
        let layer1_w: Vec<Vec<f64>> = (0..hidden_dim)
            .map(|_| (0..8).map(|_| rng.gen_range(-scale1..scale1)).collect())
            .collect();
        let layer1_b = vec![0.0; hidden_dim];

        let scale_c = (2.0 / hidden_dim as f64).sqrt();
        let classifier_w: Vec<Vec<f64>> = (0..num_classes)
            .map(|_| (0..hidden_dim).map(|_| rng.gen_range(-scale_c..scale_c)).collect())
            .collect();
        let classifier_b = vec![0.0; num_classes];

        GPFeatureClassifier {
            layer1_w, layer1_b,
            classifier_w, classifier_b,
            hidden_dim, num_classes,
        }
    }

    /// Forward pass: GP features [8] → class logits [num_classes]
    pub fn forward(&self, gp_features: &[f64; 8]) -> Vec<f64> {
        // Layer 1: linear + square activation (FHE-friendly)
        let mut h = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut sum = self.layer1_b[i];
            for j in 0..8 {
                sum += self.layer1_w[i][j] * gp_features[j];
            }
            h[i] = sum * sum; // Square activation (polynomial, FHE-compatible)
        }

        // Classifier: linear (no activation)
        let mut logits = vec![0.0; self.num_classes];
        for i in 0..self.num_classes {
            let mut sum = self.classifier_b[i];
            for j in 0..self.hidden_dim {
                sum += self.classifier_w[i][j] * h[j];
            }
            logits[i] = sum;
        }

        logits
    }

    /// Compute softmax probabilities
    pub fn softmax(logits: &[f64]) -> Vec<f64> {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Predict class from GP features
    pub fn predict(&self, gp_features: &[f64; 8]) -> usize {
        let logits = self.forward(gp_features);
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Predict with confidence
    pub fn predict_with_confidence(&self, gp_features: &[f64; 8]) -> (usize, f64) {
        let logits = self.forward(gp_features);
        let probs = Self::softmax(&logits);
        let (class, &conf) = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        (class, conf)
    }

    /// Cross-entropy loss
    pub fn loss(&self, gp_features: &[f64; 8], target: usize) -> f64 {
        let logits = self.forward(gp_features);
        let probs = Self::softmax(&logits);
        -probs[target].max(1e-10).ln()
    }

    /// Save weights to JSON
    pub fn save_weights(&self, path: &std::path::Path) -> Result<(), String> {
        let weights = GPClassifierWeights {
            layer1_w: self.layer1_w.clone(),
            layer1_b: self.layer1_b.clone(),
            classifier_w: self.classifier_w.clone(),
            classifier_b: self.classifier_b.clone(),
            input_dim: 8,
            hidden_dim: self.hidden_dim,
            num_classes: self.num_classes,
        };
        weights.save(path)
    }

    /// Load weights from JSON
    pub fn load_weights(path: &std::path::Path) -> Result<Self, String> {
        let weights = GPClassifierWeights::load(path)?;
        Ok(GPFeatureClassifier {
            layer1_w: weights.layer1_w,
            layer1_b: weights.layer1_b,
            classifier_w: weights.classifier_w,
            classifier_b: weights.classifier_b,
            hidden_dim: weights.hidden_dim,
            num_classes: weights.num_classes,
        })
    }

    pub fn num_params(&self) -> usize {
        self.hidden_dim * 8 + self.hidden_dim +
        self.num_classes * self.hidden_dim + self.num_classes
    }
}

/// Encode a point as an augmented Cl(3,0) multivector: 1 + x·e₁ + y·e₂ + z·e₃
///
/// The scalar component of 1 ensures the GP self-product produces
/// non-trivial features in all vector components.
pub fn encode_point_augmented(x: f64, y: f64, z: f64) -> [f64; 8] {
    [1.0, x, y, z, 0.0, 0.0, 0.0, 0.0]
}

/// Compute plaintext geometric product for Cl(3,0)
pub fn plaintext_geometric_product(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
    let mut result = [0.0; 8];

    result[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
              - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];
    result[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[4] - a[4]*b[2]
              + a[3]*b[5] - a[5]*b[3] - a[6]*b[7] + a[7]*b[6];
    result[2] = a[0]*b[2] + a[2]*b[0] - a[1]*b[4] + a[4]*b[1]
              + a[3]*b[6] - a[6]*b[3] - a[5]*b[7] + a[7]*b[5];
    result[3] = a[0]*b[3] + a[3]*b[0] - a[1]*b[5] + a[5]*b[1]
              - a[2]*b[6] + a[6]*b[2] - a[4]*b[7] + a[7]*b[4];
    result[4] = a[0]*b[4] + a[4]*b[0] + a[1]*b[2] - a[2]*b[1]
              + a[3]*b[7] - a[7]*b[3] + a[5]*b[6] - a[6]*b[5];
    result[5] = a[0]*b[5] + a[5]*b[0] + a[1]*b[3] - a[3]*b[1]
              - a[2]*b[7] + a[7]*b[2] - a[4]*b[6] + a[6]*b[4];
    result[6] = a[0]*b[6] + a[6]*b[0] + a[2]*b[3] - a[3]*b[2]
              + a[1]*b[7] - a[7]*b[1] + a[4]*b[5] - a[5]*b[4];
    result[7] = a[0]*b[7] + a[7]*b[0] + a[1]*b[6] - a[6]*b[1]
              - a[2]*b[5] + a[5]*b[2] + a[3]*b[4] - a[4]*b[3];

    result
}

/// Compute GP features for a point cloud
///
/// 1. Encode each point as augmented Cl(3,0) multivector
/// 2. Compute GP self-product for each point
/// 3. Mean pool over all points
pub fn compute_gp_features(pc: &PointCloud) -> [f64; 8] {
    let n = pc.points.len();
    let mut mean_gp = [0.0; 8];

    for p in &pc.points {
        let mv = encode_point_augmented(p.x, p.y, p.z);
        let gp = plaintext_geometric_product(&mv, &mv);
        for i in 0..8 {
            mean_gp[i] += gp[i];
        }
    }

    for i in 0..8 {
        mean_gp[i] /= n as f64;
    }

    mean_gp
}

/// Train GPFeatureClassifier on synthetic data
///
/// Returns (final_test_accuracy, final_test_loss)
pub fn train_gp_classifier(
    model: &mut GPFeatureClassifier,
    train_data: &[PointCloud],
    test_data: &[PointCloud],
    epochs: usize,
    lr: f64,
    print_every: usize,
) -> (f64, f64) {
    use rand::seq::SliceRandom;

    let mut rng = rand::thread_rng();
    let mut best_acc = 0.0;

    // Precompute GP features for all samples
    let train_features: Vec<([f64; 8], usize)> = train_data.iter()
        .map(|pc| (compute_gp_features(pc), pc.label.unwrap_or(0)))
        .collect();
    let test_features: Vec<([f64; 8], usize)> = test_data.iter()
        .map(|pc| (compute_gp_features(pc), pc.label.unwrap_or(0)))
        .collect();

    // Adam optimizer state
    let mut m_l1_w = vec![vec![0.0; 8]; model.hidden_dim];
    let mut v_l1_w = vec![vec![0.0; 8]; model.hidden_dim];
    let mut m_l1_b = vec![0.0; model.hidden_dim];
    let mut v_l1_b = vec![0.0; model.hidden_dim];
    let mut m_cl_w = vec![vec![0.0; model.hidden_dim]; model.num_classes];
    let mut v_cl_w = vec![vec![0.0; model.hidden_dim]; model.num_classes];
    let mut m_cl_b = vec![0.0; model.num_classes];
    let mut v_cl_b = vec![0.0; model.num_classes];
    let beta1: f64 = 0.9;
    let beta2: f64 = 0.999;
    let eps = 1e-8;
    let mut t = 0;

    for epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..train_features.len()).collect();
        indices.shuffle(&mut rng);

        // LR schedule: warmup + cosine decay
        let warmup = 5;
        let warmup_factor = if epoch < warmup { (epoch + 1) as f64 / warmup as f64 } else { 1.0 };
        let cosine_factor = if epoch >= warmup {
            0.5 * (1.0 + (std::f64::consts::PI * (epoch - warmup) as f64 / (epochs - warmup).max(1) as f64).cos())
        } else { 1.0 };
        let current_lr = lr * warmup_factor * cosine_factor;

        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0;

        for &idx in &indices {
            let (ref features, target) = train_features[idx];
            t += 1;
            let bc1 = 1.0 - beta1.powi(t as i32);
            let bc2 = 1.0 - beta2.powi(t as i32);

            // Forward pass
            let mut h_pre = vec![0.0; model.hidden_dim];
            let mut h = vec![0.0; model.hidden_dim];
            for i in 0..model.hidden_dim {
                let mut sum = model.layer1_b[i];
                for j in 0..8 {
                    sum += model.layer1_w[i][j] * features[j];
                }
                h_pre[i] = sum;
                h[i] = sum * sum; // square activation
            }

            let mut logits = vec![0.0; model.num_classes];
            for i in 0..model.num_classes {
                let mut sum = model.classifier_b[i];
                for j in 0..model.hidden_dim {
                    sum += model.classifier_w[i][j] * h[j];
                }
                logits[i] = sum;
            }

            // Loss
            let probs = GPFeatureClassifier::softmax(&logits);
            let loss = -probs[target].max(1e-10).ln();
            if !loss.is_finite() || loss > 50.0 { continue; }
            epoch_loss += loss;

            let pred = logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == target { epoch_correct += 1; }

            // Backward: softmax gradient
            let mut grad_logits = probs.clone();
            grad_logits[target] -= 1.0;

            // Grad classifier
            for i in 0..model.num_classes {
                for j in 0..model.hidden_dim {
                    let grad = (grad_logits[i] * h[j]).clamp(-1.0, 1.0);
                    m_cl_w[i][j] = beta1 * m_cl_w[i][j] + (1.0 - beta1) * grad;
                    v_cl_w[i][j] = beta2 * v_cl_w[i][j] + (1.0 - beta2) * grad * grad;
                    model.classifier_w[i][j] -= current_lr * (m_cl_w[i][j] / bc1) / ((v_cl_w[i][j] / bc2).sqrt() + eps);
                }
                let grad = grad_logits[i].clamp(-1.0, 1.0);
                m_cl_b[i] = beta1 * m_cl_b[i] + (1.0 - beta1) * grad;
                v_cl_b[i] = beta2 * v_cl_b[i] + (1.0 - beta2) * grad * grad;
                model.classifier_b[i] -= current_lr * (m_cl_b[i] / bc1) / ((v_cl_b[i] / bc2).sqrt() + eps);
            }

            // Grad through h (chain through square activation: d/dx (x²) = 2x)
            let mut grad_h_pre = vec![0.0; model.hidden_dim];
            for j in 0..model.hidden_dim {
                let mut grad_h_j = 0.0;
                for i in 0..model.num_classes {
                    grad_h_j += grad_logits[i] * model.classifier_w[i][j];
                }
                grad_h_pre[j] = grad_h_j * 2.0 * h_pre[j]; // d(x²)/dx = 2x
            }

            // Grad layer1
            for i in 0..model.hidden_dim {
                for j in 0..8 {
                    let grad = (grad_h_pre[i] * features[j]).clamp(-1.0, 1.0);
                    m_l1_w[i][j] = beta1 * m_l1_w[i][j] + (1.0 - beta1) * grad;
                    v_l1_w[i][j] = beta2 * v_l1_w[i][j] + (1.0 - beta2) * grad * grad;
                    model.layer1_w[i][j] -= current_lr * (m_l1_w[i][j] / bc1) / ((v_l1_w[i][j] / bc2).sqrt() + eps);
                }
                let grad = grad_h_pre[i].clamp(-1.0, 1.0);
                m_l1_b[i] = beta1 * m_l1_b[i] + (1.0 - beta1) * grad;
                v_l1_b[i] = beta2 * v_l1_b[i] + (1.0 - beta2) * grad * grad;
                model.layer1_b[i] -= current_lr * (m_l1_b[i] / bc1) / ((v_l1_b[i] / bc2).sqrt() + eps);
            }
        }

        if print_every > 0 && (epoch + 1) % print_every == 0 {
            let train_acc = epoch_correct as f64 / train_features.len() as f64;
            let avg_loss = epoch_loss / train_features.len() as f64;

            // Evaluate test
            let mut test_correct = 0;
            let mut test_loss = 0.0;
            for (features, target) in &test_features {
                let pred = model.predict(features);
                if pred == *target { test_correct += 1; }
                test_loss += model.loss(features, *target);
            }
            let test_acc = test_correct as f64 / test_features.len() as f64;
            let test_avg_loss = test_loss / test_features.len() as f64;

            if test_acc > best_acc { best_acc = test_acc; }

            println!("  Epoch {:3} | Train {:.1}% (loss {:.4}) | Test {:.1}% (loss {:.4}) | LR {:.5}",
                epoch + 1, train_acc * 100.0, avg_loss, test_acc * 100.0, test_avg_loss, current_lr);
        }
    }

    // Final evaluation
    let mut test_correct = 0;
    let mut test_loss = 0.0;
    for (features, target) in &test_features {
        if model.predict(features) == *target { test_correct += 1; }
        test_loss += model.loss(features, *target);
    }
    (test_correct as f64 / test_features.len() as f64, test_loss / test_features.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::point_cloud::Point3D;

    #[test]
    fn test_encode_point_augmented() {
        let mv = encode_point_augmented(1.0, 2.0, 3.0);
        assert_eq!(mv, [1.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_gp_self_product_augmented() {
        // For mv = [1, x, y, z, 0, 0, 0, 0]:
        // GP(mv, mv) should give [1+x²+y²+z², 2x, 2y, 2z, 0, 0, 0, 0]
        let mv = encode_point_augmented(1.0, 2.0, 3.0);
        let gp = plaintext_geometric_product(&mv, &mv);

        assert!((gp[0] - (1.0 + 1.0 + 4.0 + 9.0)).abs() < 1e-10); // 15.0
        assert!((gp[1] - 2.0).abs() < 1e-10); // 2*x = 2
        assert!((gp[2] - 4.0).abs() < 1e-10); // 2*y = 4
        assert!((gp[3] - 6.0).abs() < 1e-10); // 2*z = 6
        assert!(gp[4].abs() < 1e-10);
        assert!(gp[5].abs() < 1e-10);
        assert!(gp[6].abs() < 1e-10);
        assert!(gp[7].abs() < 1e-10);
    }

    #[test]
    fn test_gp_classifier_forward() {
        let model = GPFeatureClassifier::new(16, 3);
        let features = [1.0, 0.5, -0.3, 0.7, 0.0, 0.0, 0.0, 0.0];
        let logits = model.forward(&features);
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|l| l.is_finite()));
    }

    #[test]
    fn test_compute_gp_features() {
        let points = vec![
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(-1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(0.0, -1.0, 0.0),
        ];
        let pc = PointCloud::from_points(points);
        let features = compute_gp_features(&pc);

        // Mean of GP(augmented, augmented):
        // Each point has norm²=1, so scalar component = 1+1 = 2
        // Mean of 2x: mean over [2, -2, 0, 0] = 0
        // Mean of 2y: mean over [0, 0, 2, -2] = 0
        assert!((features[0] - 2.0).abs() < 1e-10); // mean(1 + norm²) = 2
        assert!(features[1].abs() < 1e-10); // mean(2x) = 0
        assert!(features[2].abs() < 1e-10); // mean(2y) = 0
        assert!(features[3].abs() < 1e-10); // mean(2z) = 0
    }
}
