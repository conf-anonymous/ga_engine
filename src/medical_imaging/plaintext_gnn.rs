/// Plaintext Geometric Neural Network
///
/// Implements a feedforward neural network that operates on Cl(3,0) multivectors
/// using geometric products as the fundamental operation.
///
/// Architecture: 1 → 16 → 8 → 3
/// - Input: 1 multivector (encoded 3D point cloud)
/// - Hidden 1: 16 multivectors
/// - Hidden 2: 8 multivectors
/// - Output: 3 multivectors (class scores)

use super::clifford_encoding::Multivector3D;
use rand::Rng;

/// Geometric neural network layer
#[derive(Debug, Clone)]
pub struct GeometricLayer {
    /// Weight multivectors (input_dim × output_dim)
    pub weights: Vec<Vec<Multivector3D>>,
    /// Bias multivectors (output_dim)
    pub biases: Vec<Multivector3D>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl GeometricLayer {
    /// Create new layer with random initialization
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_dim as f64).sqrt(); // Xavier initialization

        let mut weights = Vec::new();
        for _ in 0..output_dim {
            let mut row = Vec::new();
            for _ in 0..input_dim {
                let mut components = [0.0; 8];
                for c in &mut components {
                    *c = rng.gen_range(-scale..scale);
                }
                row.push(Multivector3D::new(components));
            }
            weights.push(row);
        }

        let mut biases = Vec::new();
        for _ in 0..output_dim {
            let mut components = [0.0; 8];
            for c in &mut components {
                *c = rng.gen_range(-scale..scale);
            }
            biases.push(Multivector3D::new(components));
        }

        GeometricLayer {
            weights,
            biases,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass: y = W ⊗ x + b
    /// where ⊗ is the geometric product
    pub fn forward(&self, inputs: &[Multivector3D]) -> Vec<Multivector3D> {
        assert_eq!(inputs.len(), self.input_dim);

        let mut outputs = Vec::new();

        for i in 0..self.output_dim {
            // Start with bias
            let mut output = self.biases[i];

            // Add weighted sum: sum_j(W_ij ⊗ x_j)
            for j in 0..self.input_dim {
                let weighted = geometric_product(&self.weights[i][j], &inputs[j]);
                output = add_multivectors(&output, &weighted);
            }

            outputs.push(output);
        }

        outputs
    }
}

/// Geometric product of two multivectors (simplified version)
///
/// For now, we'll use a simplified dot-product style operation that's
/// numerically stable for training.
///
/// TODO: Implement full Cl(3,0) geometric product with structure constants
pub fn geometric_product(a: &Multivector3D, b: &Multivector3D) -> Multivector3D {
    // Use dot product of components as scalar output
    // This is a simplified operation that's numerically stable
    let mut dot_product = 0.0;
    for i in 0..8 {
        dot_product += a.components[i] * b.components[i];
    }

    // Output is a scalar multivector (only scalar component non-zero)
    let mut result = [0.0; 8];
    result[0] = dot_product;

    Multivector3D::new(result)
}

/// Add two multivectors component-wise
pub fn add_multivectors(a: &Multivector3D, b: &Multivector3D) -> Multivector3D {
    let mut result = [0.0; 8];
    for i in 0..8 {
        result[i] = a.components[i] + b.components[i];
    }
    Multivector3D::new(result)
}

/// Subtract two multivectors component-wise
pub fn subtract_multivectors(a: &Multivector3D, b: &Multivector3D) -> Multivector3D {
    let mut result = [0.0; 8];
    for i in 0..8 {
        result[i] = a.components[i] - b.components[i];
    }
    Multivector3D::new(result)
}

/// Scale multivector by scalar
pub fn scale_multivector(mv: &Multivector3D, scalar: f64) -> Multivector3D {
    let mut result = [0.0; 8];
    for i in 0..8 {
        result[i] = mv.components[i] * scalar;
    }
    Multivector3D::new(result)
}

/// ReLU activation (component-wise)
pub fn relu(mv: &Multivector3D) -> Multivector3D {
    let mut result = [0.0; 8];
    for i in 0..8 {
        result[i] = mv.components[i].max(0.0);
    }
    Multivector3D::new(result)
}

/// Softmax over multivector scalars (for classification)
pub fn softmax_scalar(outputs: &[Multivector3D]) -> Vec<f64> {
    // Take scalar component of each output multivector
    let scalars: Vec<f64> = outputs.iter().map(|mv| mv.scalar()).collect();

    // Compute softmax
    let max_val = scalars.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = scalars.iter().map(|&x| (x - max_val).exp()).sum();

    scalars.iter().map(|&x| (x - max_val).exp() / exp_sum).collect()
}

/// Full geometric neural network
#[derive(Debug, Clone)]
pub struct GeometricNeuralNetwork {
    pub layer1: GeometricLayer,  // 1 → 16
    pub layer2: GeometricLayer,  // 16 → 8
    pub layer3: GeometricLayer,  // 8 → 3
}

impl GeometricNeuralNetwork {
    /// Create new GNN with random initialization
    pub fn new() -> Self {
        GeometricNeuralNetwork {
            layer1: GeometricLayer::new(1, 16),
            layer2: GeometricLayer::new(16, 8),
            layer3: GeometricLayer::new(8, 3),
        }
    }

    /// Forward pass through entire network
    pub fn forward(&self, input: &Multivector3D) -> Vec<f64> {
        // Layer 1: 1 → 16
        let hidden1 = self.layer1.forward(&[*input]);
        let hidden1_activated: Vec<Multivector3D> = hidden1.iter().map(relu).collect();

        // Layer 2: 16 → 8
        let hidden2 = self.layer2.forward(&hidden1_activated);
        let hidden2_activated: Vec<Multivector3D> = hidden2.iter().map(relu).collect();

        // Layer 3: 8 → 3
        let output = self.layer3.forward(&hidden2_activated);

        // Softmax for class probabilities
        softmax_scalar(&output)
    }

    /// Predict class (0, 1, or 2)
    pub fn predict(&self, input: &Multivector3D) -> usize {
        let probs = self.forward(input);

        // Handle NaN by treating as -infinity
        probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                if a.is_nan() && b.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.is_nan() {
                    std::cmp::Ordering::Less
                } else if b.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    a.partial_cmp(b).unwrap()
                }
            })
            .map(|(idx, _)| idx)
            .unwrap()
    }

    /// Compute loss (cross-entropy)
    pub fn loss(&self, input: &Multivector3D, target_label: usize) -> f64 {
        let probs = self.forward(input);
        // Clamp probability to avoid log(0) = -inf and log(negative) = NaN
        let p = probs[target_label].max(1e-10).min(1.0);
        -p.ln()
    }

    /// Compute accuracy on a dataset
    pub fn accuracy(&self, inputs: &[Multivector3D], labels: &[usize]) -> f64 {
        let mut correct = 0;
        for (input, &label) in inputs.iter().zip(labels.iter()) {
            if self.predict(input) == label {
                correct += 1;
            }
        }
        correct as f64 / inputs.len() as f64
    }
}

/// Simple SGD trainer (simplified backpropagation)
pub struct Trainer {
    pub learning_rate: f64,
}

impl Trainer {
    pub fn new(learning_rate: f64) -> Self {
        Trainer { learning_rate }
    }

    /// Train for one epoch using numerical gradient approximation
    ///
    /// Note: This is a simplified trainer using finite differences for gradients.
    /// A full implementation would use automatic differentiation through
    /// the geometric product.
    pub fn train_epoch(
        &self,
        model: &mut GeometricNeuralNetwork,
        inputs: &[Multivector3D],
        labels: &[usize],
    ) -> f64 {
        let epsilon = 1e-5;
        let mut total_loss = 0.0;

        for (input, &label) in inputs.iter().zip(labels.iter()) {
            // Compute current loss
            let current_loss = model.loss(input, label);
            total_loss += current_loss;

            // Numerical gradient for layer3 weights (simplified)
            for i in 0..model.layer3.output_dim {
                for j in 0..model.layer3.input_dim {
                    for c in 0..8 {
                        // Perturb weight
                        let original = model.layer3.weights[i][j].components[c];
                        model.layer3.weights[i][j].components[c] += epsilon;

                        // Compute perturbed loss
                        let perturbed_loss = model.loss(input, label);

                        // Approximate gradient
                        let grad = (perturbed_loss - current_loss) / epsilon;

                        // Restore and update
                        model.layer3.weights[i][j].components[c] = original - self.learning_rate * grad;
                    }
                }
            }

            // Update biases similarly (simplified - just layer3 for now)
            for i in 0..model.layer3.output_dim {
                for c in 0..8 {
                    let original = model.layer3.biases[i].components[c];
                    model.layer3.biases[i].components[c] += epsilon;

                    let perturbed_loss = model.loss(input, label);
                    let grad = (perturbed_loss - current_loss) / epsilon;

                    model.layer3.biases[i].components[c] = original - self.learning_rate * grad;
                }
            }
        }

        total_loss / inputs.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometric_product() {
        let a = Multivector3D::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = Multivector3D::new([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let c = geometric_product(&a, &b);

        // Scalar * scalar = scalar (simplified version)
        assert_eq!(c.scalar(), 2.0);
    }

    #[test]
    fn test_add_multivectors() {
        let a = Multivector3D::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = Multivector3D::new([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let c = add_multivectors(&a, &b);

        assert_eq!(c.components, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_relu() {
        let a = Multivector3D::new([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
        let b = relu(&a);

        assert_eq!(b.components, [1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0]);
    }

    #[test]
    fn test_softmax() {
        let outputs = vec![
            Multivector3D::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Multivector3D::new([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Multivector3D::new([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];

        let probs = softmax_scalar(&outputs);

        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Largest input should have largest probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_layer_forward() {
        let layer = GeometricLayer::new(2, 3);
        let inputs = vec![
            Multivector3D::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Multivector3D::new([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];

        let outputs = layer.forward(&inputs);
        assert_eq!(outputs.len(), 3);
    }

    #[test]
    fn test_gnn_forward() {
        let gnn = GeometricNeuralNetwork::new();
        let input = Multivector3D::new([1.0, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.05]);

        let probs = gnn.forward(&input);

        // Should have 3 class probabilities
        assert_eq!(probs.len(), 3);

        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // All probabilities should be between 0 and 1
        for &p in &probs {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_gnn_predict() {
        let gnn = GeometricNeuralNetwork::new();
        let input = Multivector3D::new([1.0, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.05]);

        let prediction = gnn.predict(&input);

        // Should predict one of 3 classes
        assert!(prediction < 3);
    }
}
