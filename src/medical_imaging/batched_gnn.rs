/// Batched Geometric Neural Network Inference
///
/// Processes 512 samples in parallel using SIMD batching.
/// This achieves 512× throughput compared to single-sample inference.

use super::clifford_encoding::Multivector3D;
use super::plaintext_gnn::GeometricNeuralNetwork;
use super::simd_batching::*;

impl GeometricNeuralNetwork {
    /// Batched forward pass for 512 samples
    ///
    /// # Arguments
    /// * `inputs` - Batch of input multivectors (up to 512)
    ///
    /// # Returns
    /// Vector of class probabilities for each sample: Vec<[f64; 3]>
    pub fn forward_batched(&self, inputs: &[Multivector3D]) -> Vec<[f64; 3]> {
        assert!(
            inputs.len() <= BatchedMultivectors::MAX_BATCH_SIZE,
            "Batch size {} exceeds maximum {}",
            inputs.len(),
            BatchedMultivectors::MAX_BATCH_SIZE
        );

        let batch_size = inputs.len();

        // Pack inputs into batched representation
        let batched_input = BatchedMultivectors::from_multivectors(inputs);

        // --- Layer 1: 1 → 16 ---
        let mut layer1_outputs = Vec::with_capacity(16);

        for neuron_idx in 0..16 {
            // Get weight for this neuron (1 input)
            let weight = &self.layer1.weights[neuron_idx][0];

            // Broadcast weight to batch
            let weight_batch =
                BatchedMultivectors::from_multivectors(&vec![weight.clone(); batch_size]);

            // Batched geometric product
            let mut output = batched_geometric_product(&batched_input, &weight_batch);

            // Add bias (broadcast to all slots)
            let bias = &self.layer1.biases[neuron_idx];
            let bias_batch =
                BatchedMultivectors::from_multivectors(&vec![bias.clone(); batch_size]);
            output = batched_add(&output, &bias_batch);

            // ReLU activation
            output = batched_relu(&output);

            layer1_outputs.push(output);
        }

        // --- Layer 2: 16 → 8 ---
        let mut layer2_outputs = Vec::with_capacity(8);

        for neuron_idx in 0..8 {
            // Accumulate weighted sum from 16 inputs
            let mut accumulator = BatchedMultivectors::zeros(batch_size);

            for input_idx in 0..16 {
                let weight = &self.layer2.weights[neuron_idx][input_idx];
                let weight_batch =
                    BatchedMultivectors::from_multivectors(&vec![weight.clone(); batch_size]);

                let prod = batched_geometric_product(&layer1_outputs[input_idx], &weight_batch);
                accumulator = batched_add(&accumulator, &prod);
            }

            // Add bias
            let bias = &self.layer2.biases[neuron_idx];
            let bias_batch =
                BatchedMultivectors::from_multivectors(&vec![bias.clone(); batch_size]);
            accumulator = batched_add(&accumulator, &bias_batch);

            // ReLU activation
            let output = batched_relu(&accumulator);
            layer2_outputs.push(output);
        }

        // --- Layer 3: 8 → 3 ---
        let mut layer3_outputs = Vec::with_capacity(3);

        for neuron_idx in 0..3 {
            // Accumulate weighted sum from 8 inputs
            let mut accumulator = BatchedMultivectors::zeros(batch_size);

            for input_idx in 0..8 {
                let weight = &self.layer3.weights[neuron_idx][input_idx];
                let weight_batch =
                    BatchedMultivectors::from_multivectors(&vec![weight.clone(); batch_size]);

                let prod = batched_geometric_product(&layer2_outputs[input_idx], &weight_batch);
                accumulator = batched_add(&accumulator, &prod);
            }

            // Add bias
            let bias = &self.layer3.biases[neuron_idx];
            let bias_batch =
                BatchedMultivectors::from_multivectors(&vec![bias.clone(); batch_size]);
            accumulator = batched_add(&accumulator, &bias_batch);

            layer3_outputs.push(accumulator);
        }

        // --- Softmax per sample ---
        // Extract scalar components for each sample
        let mut all_probs = Vec::with_capacity(batch_size);

        for sample_idx in 0..batch_size {
            // Get 3 logits for this sample
            let logits: Vec<f64> = layer3_outputs
                .iter()
                .map(|output| output.components[0][sample_idx])
                .collect();

            // Softmax
            let max_val = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = logits.iter().map(|&x| (x - max_val).exp()).sum();

            let probs: Vec<f64> = logits
                .iter()
                .map(|&x| (x - max_val).exp() / exp_sum)
                .collect();

            all_probs.push([probs[0], probs[1], probs[2]]);
        }

        all_probs
    }

    /// Batched prediction for 512 samples
    ///
    /// Returns predicted class (0, 1, or 2) for each sample
    pub fn predict_batched(&self, inputs: &[Multivector3D]) -> Vec<usize> {
        let all_probs = self.forward_batched(inputs);

        all_probs
            .iter()
            .map(|probs| {
                probs
                    .iter()
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
            })
            .collect()
    }

    /// Batched accuracy computation
    pub fn accuracy_batched(&self, inputs: &[Multivector3D], labels: &[usize]) -> f64 {
        assert_eq!(inputs.len(), labels.len());

        let predictions = self.predict_batched(inputs);

        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&pred, &label)| pred == label)
            .count();

        correct as f64 / labels.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_forward_matches_single() {
        let model = GeometricNeuralNetwork::new();

        // Test 5 random inputs
        let inputs: Vec<Multivector3D> = (0..5)
            .map(|i| {
                Multivector3D::new([
                    i as f64 * 0.1,
                    i as f64 * 0.2,
                    i as f64 * 0.3,
                    i as f64 * 0.4,
                    i as f64 * 0.5,
                    i as f64 * 0.6,
                    i as f64 * 0.7,
                    i as f64 * 0.8,
                ])
            })
            .collect();

        // Single-sample inference
        let single_results: Vec<[f64; 3]> = inputs
            .iter()
            .map(|input| {
                let probs = model.forward(input);
                [probs[0], probs[1], probs[2]]
            })
            .collect();

        // Batched inference
        let batched_results = model.forward_batched(&inputs);

        // Verify they match
        for (single, batched) in single_results.iter().zip(batched_results.iter()) {
            for i in 0..3 {
                assert!(
                    (single[i] - batched[i]).abs() < 1e-6,
                    "Mismatch: single={}, batched={}",
                    single[i],
                    batched[i]
                );
            }
        }
    }

    #[test]
    fn test_batched_predict() {
        let model = GeometricNeuralNetwork::new();

        let inputs: Vec<Multivector3D> = (0..10)
            .map(|i| Multivector3D::new([i as f64; 8]))
            .collect();

        let predictions = model.predict_batched(&inputs);
        assert_eq!(predictions.len(), 10);

        // All predictions should be valid classes
        for &pred in &predictions {
            assert!(pred < 3);
        }
    }

    #[test]
    fn test_max_batch_size() {
        let model = GeometricNeuralNetwork::new();

        // Test with 512 samples
        let inputs: Vec<Multivector3D> = (0..512)
            .map(|i| Multivector3D::new([i as f64; 8]))
            .collect();

        let predictions = model.predict_batched(&inputs);
        assert_eq!(predictions.len(), 512);
    }

    #[test]
    fn test_batched_accuracy() {
        let model = GeometricNeuralNetwork::new();

        let inputs: Vec<Multivector3D> = vec![
            Multivector3D::new([1.0; 8]),
            Multivector3D::new([2.0; 8]),
            Multivector3D::new([3.0; 8]),
            Multivector3D::new([4.0; 8]),
        ];

        let labels = vec![0, 1, 2, 0];

        let acc = model.accuracy_batched(&inputs, &labels);
        assert!(acc >= 0.0 && acc <= 1.0);
    }
}
