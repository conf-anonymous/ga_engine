//! Training infrastructure for CliffordPointNet
//!
//! Provides training with numerical gradient descent.
//! For production, consider using automatic differentiation frameworks.

use super::encoding::encode_points;
use super::model::CliffordPointNet;
use super::multivector::Multivector;
use crate::datasets::augmentation::{Augmentation, AugmentationConfig};
use crate::datasets::point_cloud::{DatasetSplit, PointCloud};
use rand::seq::SliceRandom;
use std::time::Instant;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Learning rate decay factor
    pub lr_decay: f64,
    /// Decay learning rate every N epochs
    pub lr_decay_epochs: usize,
    /// Number of epochs
    pub num_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Epsilon for numerical gradients
    pub epsilon: f64,
    /// Gradient clipping threshold (0 = no clipping)
    pub grad_clip: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Data augmentation config
    pub augmentation: AugmentationConfig,
    /// Print progress every N batches
    pub print_every: usize,
    /// Evaluate on test set every N epochs
    pub eval_every: usize,
    /// Early stopping patience (0 = disabled)
    pub early_stopping: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            learning_rate: 0.001,
            lr_decay: 0.7,
            lr_decay_epochs: 20,
            num_epochs: 100,
            batch_size: 32,
            epsilon: 1e-5,
            grad_clip: 1.0,
            weight_decay: 1e-4,
            augmentation: AugmentationConfig::modelnet40(),
            print_every: 10,
            eval_every: 5,
            early_stopping: 10,
        }
    }
}

impl TrainingConfig {
    /// Quick training config for testing
    pub fn quick() -> Self {
        TrainingConfig {
            learning_rate: 0.01,
            lr_decay: 0.9,
            lr_decay_epochs: 5,
            num_epochs: 10,
            batch_size: 16,
            epsilon: 1e-4,
            grad_clip: 1.0,
            weight_decay: 0.0,
            augmentation: AugmentationConfig::light(),
            print_every: 5,
            eval_every: 2,
            early_stopping: 0,
        }
    }

    /// Full training config
    pub fn full() -> Self {
        TrainingConfig {
            learning_rate: 0.001,
            lr_decay: 0.7,
            lr_decay_epochs: 20,
            num_epochs: 200,
            batch_size: 32,
            epsilon: 1e-5,
            grad_clip: 1.0,
            weight_decay: 1e-4,
            augmentation: AugmentationConfig::modelnet40(),
            print_every: 20,
            eval_every: 5,
            early_stopping: 15,
        }
    }
}

/// Metrics collected during training
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Training loss per epoch
    pub train_loss: Vec<f64>,
    /// Training accuracy per epoch
    pub train_accuracy: Vec<f64>,
    /// Test loss per evaluation
    pub test_loss: Vec<f64>,
    /// Test accuracy per evaluation
    pub test_accuracy: Vec<f64>,
    /// Evaluation epochs
    pub eval_epochs: Vec<usize>,
    /// Best test accuracy achieved
    pub best_accuracy: f64,
    /// Epoch of best accuracy
    pub best_epoch: usize,
    /// Training time per epoch (seconds)
    pub epoch_times: Vec<f64>,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        TrainingMetrics {
            train_loss: Vec::new(),
            train_accuracy: Vec::new(),
            test_loss: Vec::new(),
            test_accuracy: Vec::new(),
            eval_epochs: Vec::new(),
            best_accuracy: 0.0,
            best_epoch: 0,
            epoch_times: Vec::new(),
        }
    }

    pub fn print_summary(&self) {
        println!("\nTraining Summary");
        println!("================");
        println!("Epochs trained: {}", self.train_loss.len());
        println!(
            "Final train loss: {:.4}",
            self.train_loss.last().unwrap_or(&0.0)
        );
        println!(
            "Final train accuracy: {:.2}%",
            self.train_accuracy.last().unwrap_or(&0.0) * 100.0
        );
        println!(
            "Final test accuracy: {:.2}%",
            self.test_accuracy.last().unwrap_or(&0.0) * 100.0
        );
        println!("Best test accuracy: {:.2}% (epoch {})", self.best_accuracy * 100.0, self.best_epoch);
        let total_time: f64 = self.epoch_times.iter().sum();
        println!("Total training time: {:.1}s", total_time);
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Trainer for CliffordPointNet
pub struct Trainer {
    config: TrainingConfig,
    augmentation: Augmentation,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Self {
        let augmentation = Augmentation::new(config.augmentation.clone());
        Trainer {
            config,
            augmentation,
        }
    }

    /// Train the model
    pub fn train(
        &self,
        model: &mut CliffordPointNet,
        train_data: &DatasetSplit,
        test_data: &DatasetSplit,
    ) -> TrainingMetrics {
        let mut metrics = TrainingMetrics::new();
        let mut lr = self.config.learning_rate;
        let mut no_improve_count = 0;

        println!("\nStarting training...");
        println!("Train samples: {}, Test samples: {}", train_data.len(), test_data.len());
        println!("Batch size: {}, Epochs: {}", self.config.batch_size, self.config.num_epochs);
        model.summary();

        for epoch in 0..self.config.num_epochs {
            let epoch_start = Instant::now();

            // Learning rate decay
            if epoch > 0 && epoch % self.config.lr_decay_epochs == 0 {
                lr *= self.config.lr_decay;
                println!("  Learning rate decayed to: {:.6}", lr);
            }

            // Train one epoch
            let (train_loss, train_acc) = self.train_epoch(model, train_data, lr, epoch);
            metrics.train_loss.push(train_loss);
            metrics.train_accuracy.push(train_acc);

            let epoch_time = epoch_start.elapsed().as_secs_f64();
            metrics.epoch_times.push(epoch_time);

            // Evaluate on test set
            if (epoch + 1) % self.config.eval_every == 0 || epoch == self.config.num_epochs - 1 {
                let (test_acc, test_loss) = model.evaluate(&test_data.samples);
                metrics.test_loss.push(test_loss);
                metrics.test_accuracy.push(test_acc);
                metrics.eval_epochs.push(epoch);

                println!(
                    "Epoch {:3} | Train Loss: {:.4} | Train Acc: {:.2}% | Test Acc: {:.2}% | Time: {:.1}s",
                    epoch + 1,
                    train_loss,
                    train_acc * 100.0,
                    test_acc * 100.0,
                    epoch_time
                );

                // Track best accuracy
                if test_acc > metrics.best_accuracy {
                    metrics.best_accuracy = test_acc;
                    metrics.best_epoch = epoch + 1;
                    no_improve_count = 0;
                } else {
                    no_improve_count += 1;
                }

                // Early stopping
                if self.config.early_stopping > 0 && no_improve_count >= self.config.early_stopping {
                    println!("Early stopping at epoch {} (no improvement for {} evaluations)",
                             epoch + 1, no_improve_count);
                    break;
                }
            } else if (epoch + 1) % 10 == 0 {
                println!(
                    "Epoch {:3} | Train Loss: {:.4} | Train Acc: {:.2}% | Time: {:.1}s",
                    epoch + 1,
                    train_loss,
                    train_acc * 100.0,
                    epoch_time
                );
            }
        }

        metrics.print_summary();
        metrics
    }

    /// Train one epoch
    fn train_epoch(
        &self,
        model: &mut CliffordPointNet,
        data: &DatasetSplit,
        lr: f64,
        epoch: usize,
    ) -> (f64, f64) {
        let mut indices: Vec<usize> = (0..data.len()).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        // Process in batches
        for batch_start in (0..data.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(data.len());
            let batch_indices = &indices[batch_start..batch_end];

            for &idx in batch_indices {
                let sample = &data.samples[idx];
                let target = sample.label.unwrap_or(0);

                // Apply augmentation
                let augmented = self.augmentation.apply(sample);
                let encoded = encode_points(&augmented);

                // Forward pass and loss
                let loss = model.loss_encoded(&encoded, target);
                total_loss += loss;

                // Prediction for accuracy
                let pred = model.predict_encoded(&encoded);
                if pred == target {
                    correct += 1;
                }
                total += 1;

                // Compute gradients and update (simplified - update classifier only)
                self.update_classifier_weights(model, &encoded, target, lr);
            }

            // Print progress
            if self.config.print_every > 0 && (batch_start / self.config.batch_size + 1) % self.config.print_every == 0 {
                let batch_num = batch_start / self.config.batch_size + 1;
                let total_batches = (data.len() + self.config.batch_size - 1) / self.config.batch_size;
                print!("  Batch {}/{}\r", batch_num, total_batches);
            }
        }

        let avg_loss = total_loss / total as f64;
        let accuracy = correct as f64 / total as f64;

        (avg_loss, accuracy)
    }

    /// Update classifier weights using analytical gradients
    ///
    /// Uses the closed-form gradient for softmax cross-entropy loss.
    /// For layer output y = Wx + b with softmax probabilities p,
    /// the gradient is: dL/dW = (p - one_hot(target)) * x^T
    fn update_classifier_weights(
        &self,
        model: &mut CliffordPointNet,
        input: &[Multivector],
        target: usize,
        lr: f64,
    ) {
        // Forward through feature extraction to get classifier input
        let features = model.forward_features(input);

        // Forward through classifier to get all layer activations
        let (all_activations, logits) = model.classifier.forward_with_activations(&features);

        // Compute softmax probabilities
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits: Vec<f64> = logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();
        let probs: Vec<f64> = exp_logits.iter().map(|e| e / sum_exp).collect();

        // Gradient of loss w.r.t. logits: p - one_hot(target)
        let mut grad_logits: Vec<f64> = probs.clone();
        grad_logits[target] -= 1.0;

        // Backprop through all classifier layers (reverse order)
        let mut grad_input = grad_logits;

        for layer_idx in (0..model.classifier.layers.len()).rev() {
            let input_to_layer = if layer_idx == 0 {
                &all_activations[0] // Original flattened input
            } else {
                &all_activations[layer_idx]
            };

            let (ref mut weights, ref mut biases) = model.classifier.layers[layer_idx];
            let in_dim = weights[0].len();
            let out_dim = weights.len();

            // Gradient w.r.t. weights: grad_logits * input^T
            for i in 0..out_dim {
                let grad_clipped = if self.config.grad_clip > 0.0 {
                    grad_input[i].clamp(-self.config.grad_clip, self.config.grad_clip)
                } else {
                    grad_input[i]
                };

                for j in 0..in_dim {
                    let grad_w = grad_clipped * input_to_layer[j];
                    let update = lr * (grad_w + self.config.weight_decay * weights[i][j]);
                    weights[i][j] -= update;
                }

                // Gradient w.r.t. biases
                biases[i] -= lr * grad_clipped;
            }

            // Compute gradient w.r.t. input for next layer
            if layer_idx > 0 {
                let mut grad_prev = vec![0.0; in_dim];
                for i in 0..out_dim {
                    for j in 0..in_dim {
                        grad_prev[j] += grad_input[i] * weights[i][j];
                    }
                }

                // Apply ReLU derivative (ReLU' = 1 if x > 0, else 0)
                // We use the activation values to determine the mask
                for j in 0..in_dim {
                    if input_to_layer[j] <= 0.0 {
                        grad_prev[j] = 0.0;
                    }
                }

                grad_input = grad_prev;
            }
        }
    }

    /// Quick evaluation on a dataset
    pub fn evaluate(&self, model: &CliffordPointNet, data: &DatasetSplit) -> (f64, f64) {
        model.evaluate(&data.samples)
    }
}

/// Simple training without augmentation (for testing)
pub fn train_simple(
    model: &mut CliffordPointNet,
    train_data: &[PointCloud],
    test_data: &[PointCloud],
    num_epochs: usize,
    lr: f64,
) -> (f64, f64) {
    let config = TrainingConfig {
        learning_rate: lr,
        num_epochs,
        batch_size: 16,
        augmentation: AugmentationConfig::none(),
        print_every: 0,
        eval_every: num_epochs, // Only evaluate at end
        early_stopping: 0,
        ..Default::default()
    };

    let trainer = Trainer::new(config);

    let train_split = DatasetSplit::new(train_data.to_vec(), "train");
    let test_split = DatasetSplit::new(test_data.to_vec(), "test");

    let metrics = trainer.train(model, &train_split, &test_split);

    (
        metrics.test_accuracy.last().copied().unwrap_or(0.0),
        metrics.train_loss.last().copied().unwrap_or(f64::MAX),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_pointnet::model::CliffordPointNetConfig;
    use crate::datasets::modelnet40::generate_synthetic_modelnet40;

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig::quick();
        let _trainer = Trainer::new(config);
    }

    #[test]
    fn test_simple_training() {
        // Create tiny model and dataset
        let config = CliffordPointNetConfig {
            num_points: 32,
            embed_dim: 4,
            num_agg_layers: 0, // No aggregation for speed
            classifier_hidden: vec![8],
            num_classes: 3,
            ..Default::default()
        };

        let mut model = CliffordPointNet::new(config);

        // Generate synthetic data
        let (train_split, test_split) = generate_synthetic_modelnet40(10, 32, 3);

        // Quick training
        let (accuracy, _loss) = train_simple(
            &mut model,
            &train_split.samples,
            &test_split.samples,
            3,
            0.01,
        );

        // Just check it runs without error
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }

    #[test]
    fn test_metrics() {
        let metrics = TrainingMetrics::new();
        assert!(metrics.train_loss.is_empty());
        assert_eq!(metrics.best_accuracy, 0.0);
    }
}
