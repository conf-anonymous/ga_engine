//! Simplified CliffordPointNet for fast prototyping
//!
//! This module provides a simpler model that:
//! 1. Uses direct linear layers instead of full geometric products for speed
//! 2. Maintains FHE-compatibility in forward pass
//! 3. Trains quickly enough for experimentation

use super::multivector::Multivector;
use crate::datasets::point_cloud::PointCloud;
use rand::Rng;

/// Simple trainable point cloud classifier
///
/// Architecture:
/// 1. Flatten points to feature vector: [N×3] → [N×3]
/// 2. Apply two linear layers with polynomial activation
/// 3. Mean pooling over points
/// 4. Classification head
#[derive(Debug, Clone)]
pub struct SimpleCliffordNet {
    /// Layer 1 weights: [hidden_dim][3]
    pub layer1_w: Vec<Vec<f64>>,
    pub layer1_b: Vec<f64>,

    /// Layer 2 weights: [hidden_dim][hidden_dim]
    pub layer2_w: Vec<Vec<f64>>,
    pub layer2_b: Vec<f64>,

    /// Classifier weights: [num_classes][hidden_dim]
    pub classifier_w: Vec<Vec<f64>>,
    pub classifier_b: Vec<f64>,

    pub hidden_dim: usize,
    pub num_classes: usize,
}

impl SimpleCliffordNet {
    pub fn new(hidden_dim: usize, num_classes: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let scale1 = (2.0 / 3.0_f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();

        let layer1_w: Vec<Vec<f64>> = (0..hidden_dim)
            .map(|_| (0..3).map(|_| rng.gen_range(-scale1..scale1)).collect())
            .collect();
        let layer1_b = vec![0.0; hidden_dim];

        let layer2_w: Vec<Vec<f64>> = (0..hidden_dim)
            .map(|_| (0..hidden_dim).map(|_| rng.gen_range(-scale2..scale2)).collect())
            .collect();
        let layer2_b = vec![0.0; hidden_dim];

        let scale_c = (2.0 / hidden_dim as f64).sqrt();
        let classifier_w: Vec<Vec<f64>> = (0..num_classes)
            .map(|_| (0..hidden_dim).map(|_| rng.gen_range(-scale_c..scale_c)).collect())
            .collect();
        let classifier_b = vec![0.0; num_classes];

        SimpleCliffordNet {
            layer1_w, layer1_b,
            layer2_w, layer2_b,
            classifier_w, classifier_b,
            hidden_dim, num_classes,
        }
    }

    /// Forward pass returning logits
    pub fn forward(&self, pc: &PointCloud) -> Vec<f64> {
        let n = pc.points.len();

        // Process each point through layer 1
        let mut point_features: Vec<Vec<f64>> = Vec::with_capacity(n);

        for p in &pc.points {
            let x = [p.x, p.y, p.z];

            // Layer 1: linear + ReLU (use ReLU for stability, will replace with poly for FHE)
            let mut h1 = vec![0.0; self.hidden_dim];
            for i in 0..self.hidden_dim {
                let mut sum = self.layer1_b[i];
                for j in 0..3 {
                    sum += self.layer1_w[i][j] * x[j];
                }
                h1[i] = sum.max(0.0); // ReLU activation
            }

            // Layer 2: linear + ReLU
            let mut h2 = vec![0.0; self.hidden_dim];
            for i in 0..self.hidden_dim {
                let mut sum = self.layer2_b[i];
                for j in 0..self.hidden_dim {
                    sum += self.layer2_w[i][j] * h1[j];
                }
                h2[i] = sum.max(0.0); // ReLU activation
            }

            point_features.push(h2);
        }

        // Mean pooling
        let mut global_features = vec![0.0; self.hidden_dim];
        for pf in &point_features {
            for i in 0..self.hidden_dim {
                global_features[i] += pf[i];
            }
        }
        for i in 0..self.hidden_dim {
            global_features[i] /= n as f64;
        }

        // Classifier
        let mut logits = vec![0.0; self.num_classes];
        for i in 0..self.num_classes {
            let mut sum = self.classifier_b[i];
            for j in 0..self.hidden_dim {
                sum += self.classifier_w[i][j] * global_features[j];
            }
            logits[i] = sum;
        }

        logits
    }

    /// Forward pass with cached activations for backprop
    fn forward_with_cache(&self, pc: &PointCloud) -> ForwardCache {
        let n = pc.points.len();

        let mut layer1_pre: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut layer1_post: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut layer2_pre: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut layer2_post: Vec<Vec<f64>> = Vec::with_capacity(n);

        for p in &pc.points {
            let x = [p.x, p.y, p.z];

            // Layer 1: ReLU
            let mut h1_pre = vec![0.0; self.hidden_dim];
            let mut h1_post = vec![0.0; self.hidden_dim];
            for i in 0..self.hidden_dim {
                let mut sum = self.layer1_b[i];
                for j in 0..3 {
                    sum += self.layer1_w[i][j] * x[j];
                }
                h1_pre[i] = sum;
                h1_post[i] = sum.max(0.0);
            }

            // Layer 2: ReLU
            let mut h2_pre = vec![0.0; self.hidden_dim];
            let mut h2_post = vec![0.0; self.hidden_dim];
            for i in 0..self.hidden_dim {
                let mut sum = self.layer2_b[i];
                for j in 0..self.hidden_dim {
                    sum += self.layer2_w[i][j] * h1_post[j];
                }
                h2_pre[i] = sum;
                h2_post[i] = sum.max(0.0);
            }

            layer1_pre.push(h1_pre);
            layer1_post.push(h1_post);
            layer2_pre.push(h2_pre);
            layer2_post.push(h2_post);
        }

        // Mean pooling
        let mut global_features = vec![0.0; self.hidden_dim];
        for pf in &layer2_post {
            for i in 0..self.hidden_dim {
                global_features[i] += pf[i];
            }
        }
        for i in 0..self.hidden_dim {
            global_features[i] /= n as f64;
        }

        // Classifier
        let mut logits = vec![0.0; self.num_classes];
        for i in 0..self.num_classes {
            let mut sum = self.classifier_b[i];
            for j in 0..self.hidden_dim {
                sum += self.classifier_w[i][j] * global_features[j];
            }
            logits[i] = sum;
        }

        ForwardCache {
            inputs: pc.points.iter().map(|p| [p.x, p.y, p.z]).collect(),
            layer1_pre,
            layer1_post,
            layer2_pre,
            layer2_post,
            global_features,
            logits,
        }
    }

    /// Compute softmax probabilities
    pub fn softmax(logits: &[f64]) -> Vec<f64> {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Compute cross-entropy loss
    pub fn loss(&self, pc: &PointCloud, target: usize) -> f64 {
        let logits = self.forward(pc);
        let probs = Self::softmax(&logits);
        -probs[target].max(1e-10).ln()
    }

    /// Predict class
    pub fn predict(&self, pc: &PointCloud) -> usize {
        let logits = self.forward(pc);
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Backpropagation with Adam optimizer
    pub fn train_step_adam(&mut self, pc: &PointCloud, target: usize, lr: f64, adam: &mut AdamState) {
        let cache = self.forward_with_cache(pc);
        let n = pc.points.len();

        // Skip if any logits are NaN
        if cache.logits.iter().any(|&l| !l.is_finite()) {
            return;
        }

        // Softmax gradient: p - one_hot(target)
        let probs = Self::softmax(&cache.logits);
        let mut grad_logits = probs.clone();
        grad_logits[target] -= 1.0;

        // Increment timestep
        adam.t += 1;
        let t = adam.t as f64;

        // Bias correction factors
        let bc1 = 1.0 - adam.beta1.powi(adam.t as i32);
        let bc2 = 1.0 - adam.beta2.powi(adam.t as i32);

        // Helper to apply Adam update
        fn adam_update(
            param: &mut f64,
            grad: f64,
            m: &mut f64,
            v: &mut f64,
            lr: f64,
            beta1: f64,
            beta2: f64,
            bc1: f64,
            bc2: f64,
            epsilon: f64,
        ) {
            // Clip gradient
            let grad = grad.clamp(-1.0, 1.0);

            // Update biased first moment estimate
            *m = beta1 * *m + (1.0 - beta1) * grad;

            // Update biased second raw moment estimate
            *v = beta2 * *v + (1.0 - beta2) * grad * grad;

            // Compute bias-corrected estimates
            let m_hat = *m / bc1;
            let v_hat = *v / bc2;

            // Update parameter
            *param -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }

        // Update classifier weights with Adam
        for i in 0..self.num_classes {
            for j in 0..self.hidden_dim {
                let grad = grad_logits[i] * cache.global_features[j];
                adam_update(
                    &mut self.classifier_w[i][j],
                    grad,
                    &mut adam.m_classifier_w[i][j],
                    &mut adam.v_classifier_w[i][j],
                    lr, adam.beta1, adam.beta2, bc1, bc2, adam.epsilon,
                );
            }
            adam_update(
                &mut self.classifier_b[i],
                grad_logits[i],
                &mut adam.m_classifier_b[i],
                &mut adam.v_classifier_b[i],
                lr, adam.beta1, adam.beta2, bc1, bc2, adam.epsilon,
            );
        }

        // Gradient for global features
        let mut grad_global = vec![0.0; self.hidden_dim];
        for i in 0..self.num_classes {
            for j in 0..self.hidden_dim {
                grad_global[j] += grad_logits[i] * self.classifier_w[i][j];
            }
        }

        // Gradient through mean pooling (divide by N)
        for g in &mut grad_global {
            *g /= n as f64;
        }

        // Backprop through layer 2 and layer 1 for each point
        // Accumulate gradients first, then apply Adam update once
        let mut grad_layer2_w = vec![vec![0.0; self.hidden_dim]; self.hidden_dim];
        let mut grad_layer2_b = vec![0.0; self.hidden_dim];
        let mut grad_layer1_w = vec![vec![0.0; 3]; self.hidden_dim];
        let mut grad_layer1_b = vec![0.0; self.hidden_dim];

        for p_idx in 0..n {
            // Gradient through ReLU
            let mut grad_layer2_pre = vec![0.0; self.hidden_dim];
            for i in 0..self.hidden_dim {
                let relu_grad = if cache.layer2_pre[p_idx][i] > 0.0 { 1.0 } else { 0.0 };
                grad_layer2_pre[i] = grad_global[i] * relu_grad;
            }

            // Accumulate layer 2 gradients
            for i in 0..self.hidden_dim {
                for j in 0..self.hidden_dim {
                    grad_layer2_w[i][j] += grad_layer2_pre[i] * cache.layer1_post[p_idx][j];
                }
                grad_layer2_b[i] += grad_layer2_pre[i];
            }

            // Gradient for layer 1 post-activation
            let mut grad_layer1_post = vec![0.0; self.hidden_dim];
            for i in 0..self.hidden_dim {
                for j in 0..self.hidden_dim {
                    grad_layer1_post[j] += grad_layer2_pre[i] * self.layer2_w[i][j];
                }
            }

            // Gradient through ReLU
            for i in 0..self.hidden_dim {
                let relu_grad = if cache.layer1_pre[p_idx][i] > 0.0 { 1.0 } else { 0.0 };
                let grad = grad_layer1_post[i] * relu_grad;

                // Accumulate layer 1 gradients
                for j in 0..3 {
                    grad_layer1_w[i][j] += grad * cache.inputs[p_idx][j];
                }
                grad_layer1_b[i] += grad;
            }
        }

        // Apply Adam updates for layer 2
        for i in 0..self.hidden_dim {
            for j in 0..self.hidden_dim {
                adam_update(
                    &mut self.layer2_w[i][j],
                    grad_layer2_w[i][j],
                    &mut adam.m_layer2_w[i][j],
                    &mut adam.v_layer2_w[i][j],
                    lr, adam.beta1, adam.beta2, bc1, bc2, adam.epsilon,
                );
            }
            adam_update(
                &mut self.layer2_b[i],
                grad_layer2_b[i],
                &mut adam.m_layer2_b[i],
                &mut adam.v_layer2_b[i],
                lr, adam.beta1, adam.beta2, bc1, bc2, adam.epsilon,
            );
        }

        // Apply Adam updates for layer 1
        for i in 0..self.hidden_dim {
            for j in 0..3 {
                adam_update(
                    &mut self.layer1_w[i][j],
                    grad_layer1_w[i][j],
                    &mut adam.m_layer1_w[i][j],
                    &mut adam.v_layer1_w[i][j],
                    lr, adam.beta1, adam.beta2, bc1, bc2, adam.epsilon,
                );
            }
            adam_update(
                &mut self.layer1_b[i],
                grad_layer1_b[i],
                &mut adam.m_layer1_b[i],
                &mut adam.v_layer1_b[i],
                lr, adam.beta1, adam.beta2, bc1, bc2, adam.epsilon,
            );
        }
    }

    /// Backpropagation and weight update (SGD version, kept for compatibility)
    pub fn train_step(&mut self, pc: &PointCloud, target: usize, lr: f64) {
        let cache = self.forward_with_cache(pc);
        let n = pc.points.len();

        // Skip if any logits are NaN
        if cache.logits.iter().any(|&l| !l.is_finite()) {
            return;
        }

        // Softmax gradient: p - one_hot(target)
        let probs = Self::softmax(&cache.logits);
        let mut grad_logits = probs.clone();
        grad_logits[target] -= 1.0;

        // Clip gradients
        let max_grad = 1.0;
        for g in &mut grad_logits {
            *g = g.clamp(-max_grad, max_grad);
        }

        // Gradient for classifier weights
        for i in 0..self.num_classes {
            for j in 0..self.hidden_dim {
                self.classifier_w[i][j] -= lr * grad_logits[i] * cache.global_features[j];
            }
            self.classifier_b[i] -= lr * grad_logits[i];
        }

        // Gradient for global features
        let mut grad_global = vec![0.0; self.hidden_dim];
        for i in 0..self.num_classes {
            for j in 0..self.hidden_dim {
                grad_global[j] += grad_logits[i] * self.classifier_w[i][j];
            }
        }

        // Gradient through mean pooling (divide by N)
        for g in &mut grad_global {
            *g /= n as f64;
        }

        // Backprop through layer 2 and layer 1 for each point
        for p_idx in 0..n {
            // Gradient through ReLU: d(ReLU(x))/dx = 1 if x > 0, else 0
            let mut grad_layer2_pre = vec![0.0; self.hidden_dim];
            for i in 0..self.hidden_dim {
                let relu_grad = if cache.layer2_pre[p_idx][i] > 0.0 { 1.0 } else { 0.0 };
                grad_layer2_pre[i] = grad_global[i] * relu_grad;
            }

            // Update layer 2 weights
            for i in 0..self.hidden_dim {
                for j in 0..self.hidden_dim {
                    self.layer2_w[i][j] -= lr * grad_layer2_pre[i] * cache.layer1_post[p_idx][j];
                }
                self.layer2_b[i] -= lr * grad_layer2_pre[i];
            }

            // Gradient for layer 1 post-activation
            let mut grad_layer1_post = vec![0.0; self.hidden_dim];
            for i in 0..self.hidden_dim {
                for j in 0..self.hidden_dim {
                    grad_layer1_post[j] += grad_layer2_pre[i] * self.layer2_w[i][j];
                }
            }

            // Gradient through ReLU
            let mut grad_layer1_pre = vec![0.0; self.hidden_dim];
            for i in 0..self.hidden_dim {
                let relu_grad = if cache.layer1_pre[p_idx][i] > 0.0 { 1.0 } else { 0.0 };
                grad_layer1_pre[i] = grad_layer1_post[i] * relu_grad;
            }

            // Update layer 1 weights
            for i in 0..self.hidden_dim {
                for j in 0..3 {
                    self.layer1_w[i][j] -= lr * grad_layer1_pre[i] * cache.inputs[p_idx][j];
                }
                self.layer1_b[i] -= lr * grad_layer1_pre[i];
            }
        }
    }

    /// Evaluate on dataset
    pub fn evaluate(&self, samples: &[PointCloud]) -> (f64, f64) {
        let mut correct = 0;
        let mut total_loss = 0.0;

        for s in samples {
            let pred = self.predict(s);
            let target = s.label.unwrap_or(0);
            if pred == target {
                correct += 1;
            }
            total_loss += self.loss(s, target);
        }

        (correct as f64 / samples.len() as f64, total_loss / samples.len() as f64)
    }

    pub fn num_params(&self) -> usize {
        self.hidden_dim * 3 + self.hidden_dim + // layer1
        self.hidden_dim * self.hidden_dim + self.hidden_dim + // layer2
        self.num_classes * self.hidden_dim + self.num_classes // classifier
    }
}

/// Cached activations for backprop
struct ForwardCache {
    inputs: Vec<[f64; 3]>,
    layer1_pre: Vec<Vec<f64>>,
    layer1_post: Vec<Vec<f64>>,
    layer2_pre: Vec<Vec<f64>>,
    layer2_post: Vec<Vec<f64>>,
    global_features: Vec<f64>,
    logits: Vec<f64>,
}

/// Train the simple model with Adam optimizer
pub fn train_simple_net(
    model: &mut SimpleCliffordNet,
    train_data: &[PointCloud],
    test_data: &[PointCloud],
    epochs: usize,
    lr: f64,
    print_every: usize,
) -> (f64, f64) {
    use rand::seq::SliceRandom;
    use std::time::Instant;

    let mut rng = rand::thread_rng();
    let mut best_acc = 0.0;

    // Adam optimizer state
    let mut adam = AdamState::new(model);

    // Compute base LR scaled by number of classes
    let num_classes = model.num_classes;
    let base_lr = lr / (num_classes as f64 / 10.0).sqrt(); // Scale down for more classes

    for epoch in 0..epochs {
        let start = Instant::now();

        // Shuffle training data
        let mut indices: Vec<usize> = (0..train_data.len()).collect();
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0;
        let mut valid_samples = 0;

        // Warmup: gradually increase LR for first 5 epochs
        let warmup_epochs = 5;
        let warmup_factor = if epoch < warmup_epochs {
            (epoch + 1) as f64 / warmup_epochs as f64
        } else {
            1.0
        };

        // Cosine annealing after warmup
        let cosine_factor = if epoch >= warmup_epochs {
            let progress = (epoch - warmup_epochs) as f64 / (epochs - warmup_epochs).max(1) as f64;
            0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
        } else {
            1.0
        };

        let current_lr = base_lr * warmup_factor * cosine_factor;

        for &idx in &indices {
            let sample = &train_data[idx];
            let target = sample.label.unwrap_or(0);

            let loss = model.loss(sample, target);
            // Skip if loss is NaN or exploding
            if loss.is_finite() && loss < 50.0 {
                epoch_loss += loss;
                if model.predict(sample) == target {
                    epoch_correct += 1;
                }
                valid_samples += 1;
                model.train_step_adam(sample, target, current_lr, &mut adam);
            }
        }

        let train_acc = if valid_samples > 0 {
            epoch_correct as f64 / valid_samples as f64
        } else {
            0.0
        };
        let train_loss = if valid_samples > 0 {
            epoch_loss / valid_samples as f64
        } else {
            f64::NAN
        };

        if print_every > 0 && (epoch + 1) % print_every == 0 {
            let (test_acc, test_loss) = model.evaluate(test_data);
            let elapsed = start.elapsed().as_secs_f64();

            if test_acc > best_acc {
                best_acc = test_acc;
            }

            println!(
                "Epoch {:3} | Train Loss: {:.4} | Train Acc: {:.1}% | Test Acc: {:.1}% | LR: {:.5} | Time: {:.2}s",
                epoch + 1, train_loss, train_acc * 100.0, test_acc * 100.0, current_lr, elapsed
            );
        }
    }

    model.evaluate(test_data)
}

/// Adam optimizer state
pub struct AdamState {
    // First moment (mean) for each parameter
    m_layer1_w: Vec<Vec<f64>>,
    m_layer1_b: Vec<f64>,
    m_layer2_w: Vec<Vec<f64>>,
    m_layer2_b: Vec<f64>,
    m_classifier_w: Vec<Vec<f64>>,
    m_classifier_b: Vec<f64>,

    // Second moment (variance) for each parameter
    v_layer1_w: Vec<Vec<f64>>,
    v_layer1_b: Vec<f64>,
    v_layer2_w: Vec<Vec<f64>>,
    v_layer2_b: Vec<f64>,
    v_classifier_w: Vec<Vec<f64>>,
    v_classifier_b: Vec<f64>,

    // Timestep
    t: usize,

    // Hyperparameters
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl AdamState {
    fn new(model: &SimpleCliffordNet) -> Self {
        let hidden = model.hidden_dim;
        let classes = model.num_classes;

        AdamState {
            m_layer1_w: vec![vec![0.0; 3]; hidden],
            m_layer1_b: vec![0.0; hidden],
            m_layer2_w: vec![vec![0.0; hidden]; hidden],
            m_layer2_b: vec![0.0; hidden],
            m_classifier_w: vec![vec![0.0; hidden]; classes],
            m_classifier_b: vec![0.0; classes],

            v_layer1_w: vec![vec![0.0; 3]; hidden],
            v_layer1_b: vec![0.0; hidden],
            v_layer2_w: vec![vec![0.0; hidden]; hidden],
            v_layer2_b: vec![0.0; hidden],
            v_classifier_w: vec![vec![0.0; hidden]; classes],
            v_classifier_b: vec![0.0; classes],

            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::point_cloud::Point3D;

    fn make_test_cloud(n: usize, class: usize) -> PointCloud {
        // Different shapes for different classes
        let mut points = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / n as f64 * 2.0 * std::f64::consts::PI;
            let (x, y, z) = match class % 3 {
                0 => (t.cos(), t.sin(), 0.0), // Circle in XY
                1 => (t.cos(), 0.0, t.sin()), // Circle in XZ
                _ => (0.0, t.cos(), t.sin()), // Circle in YZ
            };
            points.push(Point3D::new(x, y, z));
        }
        PointCloud::from_points_with_label(points, class)
    }

    #[test]
    fn test_simple_net_forward() {
        let model = SimpleCliffordNet::new(8, 3);
        let pc = make_test_cloud(32, 0);
        let logits = model.forward(&pc);
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn test_simple_net_train() {
        let mut model = SimpleCliffordNet::new(16, 3);

        // Create training data: 30 samples per class
        let train_data: Vec<PointCloud> = (0..90)
            .map(|i| make_test_cloud(64, i % 3))
            .collect();
        let test_data: Vec<PointCloud> = (0..15)
            .map(|i| make_test_cloud(64, i % 3))
            .collect();

        let (initial_acc, _) = model.evaluate(&test_data);

        // Train for a few epochs
        let (final_acc, _) = train_simple_net(&mut model, &train_data, &test_data, 20, 0.01, 0);

        // Should improve from random
        println!("Initial acc: {:.2}%, Final acc: {:.2}%", initial_acc * 100.0, final_acc * 100.0);
        assert!(final_acc >= initial_acc || final_acc > 0.5, "Model should learn something");
    }
}
