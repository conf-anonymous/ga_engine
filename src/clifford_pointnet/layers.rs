//! CliffordPointNet neural network layers
//!
//! Implements the core layers for CliffordPointNet:
//! - CliffordEmbedding: Point-wise feature embedding
//! - CliffordAggregation: Local feature aggregation with geometric attention
//! - CliffordPooling: Global permutation-invariant pooling
//! - CliffordClassifier: Classification head

use super::multivector::Multivector;
use rand::Rng;

/// Activation function types (all polynomial for FHE compatibility)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// No activation (identity)
    None,
    /// Square: f(x) = x²
    Square,
    /// Cube: f(x) = x³
    Cube,
    /// Polynomial GELU approximation
    GeluApprox,
    /// Custom polynomial: ax² + bx + c
    Poly { a: f64, b: f64, c: f64 },
}

impl Activation {
    pub fn apply(&self, mv: &Multivector) -> Multivector {
        match self {
            Activation::None => *mv,
            Activation::Square => mv.square_activation(),
            Activation::Cube => mv.cube_activation(),
            Activation::GeluApprox => mv.gelu_approx(),
            Activation::Poly { a, b, c } => mv.poly_activation(*a, *b, *c),
        }
    }
}

/// Clifford Embedding Layer
///
/// Transforms input multivectors using learnable weight multivectors.
/// Each output channel is computed as: W_c ⊗ x + b_c
#[derive(Debug, Clone)]
pub struct CliffordEmbedding {
    /// Weight multivectors: [output_dim][input_dim]
    pub weights: Vec<Vec<Multivector>>,
    /// Bias multivectors: [output_dim]
    pub biases: Vec<Multivector>,
    pub input_dim: usize,
    pub output_dim: usize,
    pub activation: Activation,
}

impl CliffordEmbedding {
    /// Create new embedding layer with Xavier initialization
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> Self {
        let fan_in = input_dim * 8; // 8 components per multivector

        let weights: Vec<Vec<Multivector>> = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| Multivector::random_xavier(fan_in))
                    .collect()
            })
            .collect();

        let biases: Vec<Multivector> = (0..output_dim)
            .map(|_| Multivector::random_xavier(fan_in).scale(0.01))
            .collect();

        CliffordEmbedding {
            weights,
            biases,
            input_dim,
            output_dim,
            activation,
        }
    }

    /// Forward pass: apply embedding to each input
    ///
    /// Input: [batch_size][input_dim] multivectors
    /// Output: [batch_size][output_dim] multivectors
    pub fn forward(&self, inputs: &[Vec<Multivector>]) -> Vec<Vec<Multivector>> {
        inputs.iter().map(|x| self.forward_single(x)).collect()
    }

    /// Forward pass for single sample
    pub fn forward_single(&self, input: &[Multivector]) -> Vec<Multivector> {
        assert_eq!(input.len(), self.input_dim);

        let mut output = Vec::with_capacity(self.output_dim);

        for o in 0..self.output_dim {
            // Compute weighted sum: sum_i(W_oi ⊗ x_i) + b_o
            let mut sum = self.biases[o];

            for i in 0..self.input_dim {
                let weighted = self.weights[o][i].gp(&input[i]);
                sum = sum.add(&weighted);
            }

            // Apply activation
            output.push(self.activation.apply(&sum));
        }

        output
    }

    /// Get total number of parameters
    pub fn num_params(&self) -> usize {
        (self.input_dim * self.output_dim + self.output_dim) * 8
    }
}

/// Clifford Local Aggregation Layer
///
/// Aggregates features from all points using geometric attention.
/// This replaces K-NN based aggregation with algebraic attention that
/// doesn't require distance comparisons.
///
/// Attention: α_ij = softmax_j(⟨Q(h_i), K(h_j)⟩)
/// Aggregation: g_i = Σ_j α_ij · V(h_j)
///
/// For FHE, we use polynomial normalization instead of softmax.
#[derive(Debug, Clone)]
pub struct CliffordAggregation {
    /// Query projection weights
    pub query_weights: Vec<Vec<Multivector>>,
    /// Key projection weights
    pub key_weights: Vec<Vec<Multivector>>,
    /// Value projection weights
    pub value_weights: Vec<Vec<Multivector>>,
    /// Update weights after aggregation
    pub update_weights: Vec<Vec<Multivector>>,
    /// Biases for update
    pub update_biases: Vec<Multivector>,
    pub hidden_dim: usize,
    pub activation: Activation,
    /// Use polynomial normalization (FHE-friendly) vs true softmax
    pub use_poly_norm: bool,
}

impl CliffordAggregation {
    pub fn new(hidden_dim: usize, activation: Activation, use_poly_norm: bool) -> Self {
        let fan_in = hidden_dim * 8;

        let make_weights = || -> Vec<Vec<Multivector>> {
            (0..hidden_dim)
                .map(|_| {
                    (0..hidden_dim)
                        .map(|_| Multivector::random_xavier(fan_in))
                        .collect()
                })
                .collect()
        };

        CliffordAggregation {
            query_weights: make_weights(),
            key_weights: make_weights(),
            value_weights: make_weights(),
            update_weights: make_weights(),
            update_biases: (0..hidden_dim)
                .map(|_| Multivector::random_xavier(fan_in).scale(0.01))
                .collect(),
            hidden_dim,
            activation,
            use_poly_norm,
        }
    }

    /// Project input using weight matrix
    fn project(&self, input: &[Multivector], weights: &[Vec<Multivector>]) -> Vec<Multivector> {
        let dim = weights.len();
        let mut output = Vec::with_capacity(dim);

        for o in 0..dim {
            let mut sum = Multivector::zero();
            for (i, mv) in input.iter().enumerate() {
                if i < weights[o].len() {
                    sum = sum.add(&weights[o][i].gp(mv));
                }
            }
            output.push(sum);
        }

        output
    }

    /// Compute attention scores using geometric inner product
    fn compute_attention(&self, query: &Multivector, keys: &[Multivector]) -> Vec<f64> {
        let n = keys.len();
        let mut scores: Vec<f64> = keys
            .iter()
            .map(|k| {
                // Use scalar part of geometric product as attention score
                query.gp(k).scalar_part()
            })
            .collect();

        if self.use_poly_norm {
            // Polynomial normalization (FHE-friendly)
            // Normalize by L2 norm of scores
            let norm_sq: f64 = scores.iter().map(|s| s * s).sum();
            let norm = (norm_sq + 1e-10).sqrt();
            for s in &mut scores {
                *s /= norm;
            }
        } else {
            // True softmax (plaintext only)
            let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = scores.iter().map(|s| (s - max_score).exp()).sum();
            for s in &mut scores {
                *s = (*s - max_score).exp() / exp_sum;
            }
        }

        scores
    }

    /// Forward pass: aggregate features across all points
    ///
    /// Input: [num_points][hidden_dim] multivectors
    /// Output: [num_points][hidden_dim] multivectors (same shape)
    pub fn forward(&self, inputs: &[Vec<Multivector>]) -> Vec<Vec<Multivector>> {
        let num_points = inputs.len();
        if num_points == 0 {
            return vec![];
        }

        // Project all points to Q, K, V
        let queries: Vec<Vec<Multivector>> = inputs
            .iter()
            .map(|x| self.project(x, &self.query_weights))
            .collect();

        let keys: Vec<Vec<Multivector>> = inputs
            .iter()
            .map(|x| self.project(x, &self.key_weights))
            .collect();

        let values: Vec<Vec<Multivector>> = inputs
            .iter()
            .map(|x| self.project(x, &self.value_weights))
            .collect();

        // For each point, aggregate from all points
        let mut outputs = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let mut aggregated = vec![Multivector::zero(); self.hidden_dim];

            // Compute attention for each hidden dimension
            for d in 0..self.hidden_dim {
                let query = &queries[i][d];

                // Get keys for this dimension from all points
                let dim_keys: Vec<Multivector> = keys.iter().map(|k| k[d]).collect();

                // Compute attention weights
                let attention = self.compute_attention(query, &dim_keys);

                // Weighted sum of values
                let mut weighted_sum = Multivector::zero();
                for j in 0..num_points {
                    weighted_sum = weighted_sum.add(&values[j][d].scale(attention[j]));
                }
                aggregated[d] = weighted_sum;
            }

            // Residual connection + update
            let mut updated = Vec::with_capacity(self.hidden_dim);
            for d in 0..self.hidden_dim {
                // Residual: h + g
                let residual = inputs[i][d].add(&aggregated[d]);

                // Update projection
                let mut projected = self.update_biases[d];
                for (j, mv) in residual.to_array().iter().enumerate() {
                    if j < self.update_weights[d].len() {
                        // Simple scalar-weighted sum for update
                        projected = projected.add(&self.update_weights[d][j].scale(*mv));
                    }
                }

                updated.push(self.activation.apply(&projected));
            }

            outputs.push(updated);
        }

        outputs
    }

    pub fn num_params(&self) -> usize {
        let proj_params = 4 * self.hidden_dim * self.hidden_dim * 8; // Q, K, V, Update
        let bias_params = self.hidden_dim * 8;
        proj_params + bias_params
    }
}

/// Pooling method for global aggregation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolingMethod {
    /// Sum all point features
    Sum,
    /// Mean of all point features
    Mean,
    /// Max across each component (requires comparison - not FHE-friendly)
    Max,
    /// Geometric mean using product
    GeometricMean,
}

/// Clifford Global Pooling Layer
///
/// Aggregates point-wise features into a single global descriptor.
/// All methods are permutation invariant.
#[derive(Debug, Clone)]
pub struct CliffordPooling {
    /// Optional projection before pooling
    pub projection: Option<Vec<Vec<Multivector>>>,
    /// Pooling method
    pub method: PoolingMethod,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl CliffordPooling {
    pub fn new(input_dim: usize, output_dim: usize, method: PoolingMethod) -> Self {
        let projection = if input_dim != output_dim {
            let fan_in = input_dim * 8;
            Some(
                (0..output_dim)
                    .map(|_| {
                        (0..input_dim)
                            .map(|_| Multivector::random_xavier(fan_in))
                            .collect()
                    })
                    .collect(),
            )
        } else {
            None
        };

        CliffordPooling {
            projection,
            method,
            input_dim,
            output_dim,
        }
    }

    /// Simple sum/mean pooling (no projection)
    pub fn new_simple(method: PoolingMethod) -> Self {
        CliffordPooling {
            projection: None,
            method,
            input_dim: 0,
            output_dim: 0,
        }
    }

    /// Forward pass: pool all point features
    ///
    /// Input: [num_points][hidden_dim] multivectors
    /// Output: [hidden_dim] multivectors (single global descriptor)
    pub fn forward(&self, inputs: &[Vec<Multivector>]) -> Vec<Multivector> {
        if inputs.is_empty() {
            return vec![Multivector::zero()];
        }

        let hidden_dim = inputs[0].len();
        let num_points = inputs.len();

        // Pool across points for each dimension
        let mut pooled = Vec::with_capacity(hidden_dim);

        for d in 0..hidden_dim {
            let features: Vec<Multivector> = inputs.iter().map(|x| x[d]).collect();

            let result = match self.method {
                PoolingMethod::Sum => {
                    let mut sum = Multivector::zero();
                    for f in &features {
                        sum = sum.add(f);
                    }
                    sum
                }
                PoolingMethod::Mean => {
                    let mut sum = Multivector::zero();
                    for f in &features {
                        sum = sum.add(f);
                    }
                    sum.scale(1.0 / num_points as f64)
                }
                PoolingMethod::Max => {
                    // Component-wise max (not FHE-friendly!)
                    let mut max_components = [f64::NEG_INFINITY; 8];
                    for f in &features {
                        for i in 0..8 {
                            max_components[i] = max_components[i].max(f.components[i]);
                        }
                    }
                    Multivector::new(max_components)
                }
                PoolingMethod::GeometricMean => {
                    // Product of all features, then nth root (approximated)
                    let mut product = Multivector::scalar(1.0);
                    for f in &features {
                        product = product.gp(f);
                    }
                    // Approximate nth root by scaling
                    let scale = 1.0 / (num_points as f64).powf(0.5);
                    product.scale(scale)
                }
            };

            pooled.push(result);
        }

        // Apply projection if needed
        if let Some(ref proj) = self.projection {
            let mut projected = Vec::with_capacity(self.output_dim);
            for o in 0..self.output_dim {
                let mut sum = Multivector::zero();
                for (i, mv) in pooled.iter().enumerate() {
                    if i < proj[o].len() {
                        sum = sum.add(&proj[o][i].gp(mv));
                    }
                }
                projected.push(sum);
            }
            projected
        } else {
            pooled
        }
    }

    pub fn num_params(&self) -> usize {
        match &self.projection {
            Some(p) => p.len() * p[0].len() * 8,
            None => 0,
        }
    }
}

/// Clifford Classification Head
///
/// Maps global features to class logits using fully connected layers.
#[derive(Debug, Clone)]
pub struct CliffordClassifier {
    /// FC layers weights
    pub layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>, // (weights, biases)
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub num_classes: usize,
    pub activation: Activation,
}

impl CliffordClassifier {
    pub fn new(
        input_dim: usize,
        hidden_dims: Vec<usize>,
        num_classes: usize,
        activation: Activation,
    ) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = input_dim * 8; // Flatten multivectors

        // Hidden layers
        for &dim in &hidden_dims {
            let fan_in = prev_dim;
            let scale = (2.0 / fan_in as f64).sqrt();

            let weights: Vec<Vec<f64>> = (0..dim)
                .map(|_| {
                    let mut rng = rand::thread_rng();
                    (0..prev_dim).map(|_| rng.gen_range(-scale..scale)).collect()
                })
                .collect();

            let biases: Vec<f64> = (0..dim).map(|_| 0.0).collect();

            layers.push((weights, biases));
            prev_dim = dim;
        }

        // Output layer
        let fan_in = prev_dim;
        let scale = (2.0 / fan_in as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..num_classes)
            .map(|_| {
                let mut rng = rand::thread_rng();
                (0..prev_dim).map(|_| rng.gen_range(-scale..scale)).collect()
            })
            .collect();

        let biases: Vec<f64> = (0..num_classes).map(|_| 0.0).collect();
        layers.push((weights, biases));

        CliffordClassifier {
            layers,
            input_dim,
            hidden_dims,
            num_classes,
            activation,
        }
    }

    /// Forward pass: global features → class logits
    ///
    /// Input: [input_dim] multivectors (global descriptor)
    /// Output: [num_classes] logits
    pub fn forward(&self, input: &[Multivector]) -> Vec<f64> {
        // Flatten input multivectors to single vector
        let mut x: Vec<f64> = input
            .iter()
            .flat_map(|mv| mv.components.iter().copied())
            .collect();

        // Apply FC layers
        for (layer_idx, (weights, biases)) in self.layers.iter().enumerate() {
            let output_dim = weights.len();
            let mut y = vec![0.0; output_dim];

            for (i, (w, b)) in weights.iter().zip(biases.iter()).enumerate() {
                let mut sum = *b;
                for (j, &xj) in x.iter().enumerate() {
                    if j < w.len() {
                        sum += w[j] * xj;
                    }
                }
                y[i] = sum;
            }

            // Apply activation (except for last layer)
            if layer_idx < self.layers.len() - 1 {
                match self.activation {
                    Activation::Square => {
                        for yi in &mut y {
                            *yi = *yi * *yi;
                        }
                    }
                    Activation::Cube => {
                        for yi in &mut y {
                            *yi = *yi * *yi * *yi;
                        }
                    }
                    Activation::GeluApprox => {
                        for yi in &mut y {
                            *yi = 0.5 * *yi + 0.17 * *yi * *yi * *yi;
                        }
                    }
                    Activation::Poly { a, b, c } => {
                        for yi in &mut y {
                            *yi = a * *yi * *yi + b * *yi + c;
                        }
                    }
                    Activation::None => {}
                }
            }

            x = y;
        }

        x // Return logits (no softmax - applied during loss computation)
    }

    /// Forward pass with intermediate activations (for backprop)
    ///
    /// Returns: (all_activations, logits)
    /// all_activations[0] = flattened input
    /// all_activations[i] = post-activation of layer i-1 (for i > 0)
    pub fn forward_with_activations(&self, input: &[Multivector]) -> (Vec<Vec<f64>>, Vec<f64>) {
        // Flatten input multivectors to single vector
        let flattened: Vec<f64> = input
            .iter()
            .flat_map(|mv| mv.components.iter().copied())
            .collect();

        let mut activations = vec![flattened.clone()];
        let mut x = flattened;

        // Apply FC layers
        for (layer_idx, (weights, biases)) in self.layers.iter().enumerate() {
            let output_dim = weights.len();
            let mut y = vec![0.0; output_dim];

            for (i, (w, b)) in weights.iter().zip(biases.iter()).enumerate() {
                let mut sum = *b;
                for (j, &xj) in x.iter().enumerate() {
                    if j < w.len() {
                        sum += w[j] * xj;
                    }
                }
                y[i] = sum;
            }

            // Apply activation (except for last layer)
            if layer_idx < self.layers.len() - 1 {
                match self.activation {
                    Activation::Square => {
                        for yi in &mut y {
                            *yi = *yi * *yi;
                        }
                    }
                    Activation::Cube => {
                        for yi in &mut y {
                            *yi = *yi * *yi * *yi;
                        }
                    }
                    Activation::GeluApprox => {
                        for yi in &mut y {
                            *yi = 0.5 * *yi + 0.17 * *yi * *yi * *yi;
                        }
                    }
                    Activation::Poly { a, b, c } => {
                        for yi in &mut y {
                            *yi = a * *yi * *yi + b * *yi + c;
                        }
                    }
                    Activation::None => {}
                }
                activations.push(y.clone());
            }

            x = y;
        }

        (activations, x)
    }

    /// Predict class from logits
    pub fn predict(&self, input: &[Multivector]) -> usize {
        let logits = self.forward(input);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn num_params(&self) -> usize {
        self.layers
            .iter()
            .map(|(w, b)| w.len() * w[0].len() + b.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_layer() {
        let layer = CliffordEmbedding::new(1, 8, Activation::Square);

        let input = vec![Multivector::vector(1.0, 2.0, 3.0)];
        let output = layer.forward_single(&input);

        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_pooling_sum() {
        let pooling = CliffordPooling::new_simple(PoolingMethod::Sum);

        let inputs = vec![
            vec![Multivector::scalar(1.0), Multivector::scalar(2.0)],
            vec![Multivector::scalar(3.0), Multivector::scalar(4.0)],
        ];

        let output = pooling.forward(&inputs);

        assert_eq!(output.len(), 2);
        assert!((output[0].scalar_part() - 4.0).abs() < 1e-10); // 1 + 3
        assert!((output[1].scalar_part() - 6.0).abs() < 1e-10); // 2 + 4
    }

    #[test]
    fn test_pooling_mean() {
        let pooling = CliffordPooling::new_simple(PoolingMethod::Mean);

        let inputs = vec![
            vec![Multivector::scalar(2.0)],
            vec![Multivector::scalar(4.0)],
        ];

        let output = pooling.forward(&inputs);

        assert!((output[0].scalar_part() - 3.0).abs() < 1e-10); // (2 + 4) / 2
    }

    #[test]
    fn test_classifier() {
        let classifier = CliffordClassifier::new(4, vec![16], 3, Activation::Square);

        let input: Vec<Multivector> = (0..4).map(|_| Multivector::random_normal(0.1)).collect();

        let logits = classifier.forward(&input);

        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn test_activations() {
        let mv = Multivector::scalar(2.0);

        let sq = Activation::Square.apply(&mv);
        assert!((sq.scalar_part() - 4.0).abs() < 1e-10);

        let cube = Activation::Cube.apply(&mv);
        assert!((cube.scalar_part() - 8.0).abs() < 1e-10);
    }
}
