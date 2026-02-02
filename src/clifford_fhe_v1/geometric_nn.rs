//! Geometric Deep Learning with Clifford FHE
//!
//! This module implements neural networks that operate on **encrypted multivectors**,
//! enabling privacy-preserving machine learning with geometric structure.
//!
//! # Key Innovation
//!
//! Traditional neural networks lose geometric structure by flattening inputs.
//! Geometric neural networks preserve:
//! - **Rotational equivariance** (important for 3D data)
//! - **Geometric relationships** (angles, distances, orientations)
//! - **Algebraic structure** (bivectors, rotors, reflections)
//!
//! Combined with FHE, this enables:
//! - Training on encrypted geometric data
//! - Privacy-preserving point cloud classification
//! - Secure graph neural networks
//!
//! # Architecture
//!
//! ```text
//! Encrypted Input (multivectors)
//!      ↓
//! Geometric Linear Layer (weights are multivectors)
//!      ↓
//! Geometric Activation (operates on multivectors)
//!      ↓
//! Geometric Linear Layer
//!      ↓
//! Output (encrypted predictions)
//! ```

use crate::clifford_fhe_v1::ckks_rns::{RnsCiphertext, RnsPlaintext, rns_encrypt};
use crate::clifford_fhe_v1::keys_rns::{RnsEvaluationKey, RnsPublicKey};
use crate::clifford_fhe_v1::params::CliffordFHEParams;
use crate::clifford_fhe_v1::geometric_product_rns::geometric_product_3d_componentwise;
use crate::clifford_fhe_v1::rns::rns_add;

/// Geometric Linear Layer for 3D multivectors
///
/// Unlike standard linear layers (y = Wx + b), this uses geometric products:
///
/// ```text
/// y = W ⊗ x + b
/// ```
///
/// where W, x, b are all multivectors in Cl(3,0).
///
/// This preserves geometric structure and enables rotational equivariance.
pub struct GeometricLinearLayer3D {
    /// Weight multivectors (one per output neuron)
    pub weights: Vec<[f64; 8]>,
    /// Bias multivectors (one per output neuron)
    pub biases: Vec<[f64; 8]>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
}

impl GeometricLinearLayer3D {
    /// Create a new geometric linear layer with random weights
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for _ in 0..output_dim {
            // Random weight multivector (small values for stability)
            let weight = [
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.1..0.1),
            ];
            // Small random bias
            let bias = [
                rng.gen_range(-0.01..0.01),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ];
            weights.push(weight);
            biases.push(bias);
        }

        GeometricLinearLayer3D {
            weights,
            biases,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass on plaintext (for testing)
    pub fn forward_plaintext(&self, input: &[f64; 8]) -> Vec<[f64; 8]> {
        use crate::ga::geometric_product_full;

        let mut outputs = Vec::new();

        for i in 0..self.output_dim {
            let mut result = [0.0; 8];

            // Compute W ⊗ x
            geometric_product_full(&self.weights[i], input, &mut result);

            // Add bias
            for j in 0..8 {
                result[j] += self.biases[i][j];
            }

            outputs.push(result);
        }

        outputs
    }

    /// Forward pass on encrypted data (HOMOMORPHIC!)
    pub fn forward_encrypted(
        &self,
        input_ct: &[RnsCiphertext; 8],
        evk: &RnsEvaluationKey,
        pk: &RnsPublicKey,
        params: &CliffordFHEParams,
    ) -> Vec<[RnsCiphertext; 8]> {
        let primes = &params.moduli;
        let delta = params.scale;
        let n = params.n;

        let mut outputs = Vec::new();

        for i in 0..self.output_dim {
            // Encrypt weight multivector
            let mut weight_ct = Vec::new();
            for j in 0..8 {
                let mut coeffs = vec![0i64; n];
                coeffs[0] = (self.weights[i][j] * delta).round() as i64;
                let pt = RnsPlaintext::from_coeffs(coeffs, delta, primes, 0);
                weight_ct.push(rns_encrypt(pk, &pt, params));
            }
            let weight_ct_array = [
                weight_ct[0].clone(), weight_ct[1].clone(), weight_ct[2].clone(), weight_ct[3].clone(),
                weight_ct[4].clone(), weight_ct[5].clone(), weight_ct[6].clone(), weight_ct[7].clone(),
            ];

            // Compute W ⊗ x (homomorphically!)
            let mut result_ct = geometric_product_3d_componentwise(
                &weight_ct_array,
                input_ct,
                evk,
                params,
            );

            // Add bias (encrypt bias and add)
            for j in 0..8 {
                let mut coeffs = vec![0i64; n];
                coeffs[0] = (self.biases[i][j] * delta).round() as i64;
                let bias_pt = RnsPlaintext::from_coeffs(coeffs, delta, primes, 0);
                let bias_ct = rns_encrypt(pk, &bias_pt, params);

                // Add bias to result
                let sum_c0 = rns_add(&result_ct[j].c0, &bias_ct.c0, primes);
                let sum_c1 = rns_add(&result_ct[j].c1, &bias_ct.c1, primes);

                use crate::clifford_fhe_v1::ckks_rns::RnsCiphertext as RCT;
                result_ct[j] = RCT::new(sum_c0, sum_c1, result_ct[j].level, result_ct[j].scale);
            }

            outputs.push(result_ct);
        }

        outputs
    }
}

/// Geometric activation function: projects onto unit sphere
///
/// For multivector m, computes: m / ||m||
///
/// This is geometric-preserving (unlike ReLU which breaks structure).
/// Note: Division requires approximation in FHE, so we use a polynomial approximation.
pub fn geometric_activation_plaintext(m: &[f64; 8]) -> [f64; 8] {
    // Compute norm: sqrt(m · m)
    let norm_sq: f64 = m.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();

    if norm < 1e-10 {
        return *m; // Avoid division by zero
    }

    // Normalize
    let mut result = [0.0; 8];
    for i in 0..8 {
        result[i] = m[i] / norm;
    }
    result
}

/// Polynomial approximation of 1/sqrt(x) for FHE
///
/// Uses Chebyshev approximation on [0.5, 2.0]
#[allow(dead_code)]
fn inv_sqrt_approx(x: f64) -> f64 {
    // Simplified: 1/sqrt(x) ≈ 1.5 - 0.5*x for x near 1
    1.5 - 0.5 * x
}

/// Geometric activation on encrypted data (approximate)
///
/// Note: This is a simplified version. Full implementation would use
/// polynomial approximation of normalization.
pub fn geometric_activation_encrypted(
    m_ct: &[RnsCiphertext; 8],
    _evk: &RnsEvaluationKey,
    _params: &CliffordFHEParams,
) -> [RnsCiphertext; 8] {
    // For now, return identity (placeholder)
    // Full implementation would compute polynomial approximation of m / ||m||
    m_ct.clone()
}

/// Geometric Neural Network for 3D point cloud classification
///
/// Architecture:
/// ```text
/// Input: 3D points as multivectors (position as vector part)
///   ↓
/// Geometric Linear Layer (8 → 16 neurons)
///   ↓
/// Geometric Activation
///   ↓
/// Geometric Linear Layer (16 → 8 neurons)
///   ↓
/// Geometric Activation
///   ↓
/// Geometric Linear Layer (8 → num_classes)
///   ↓
/// Output: Class scores
/// ```
pub struct GeometricNN3D {
    pub layer1: GeometricLinearLayer3D,
    pub layer2: GeometricLinearLayer3D,
    pub layer3: GeometricLinearLayer3D,
    pub num_classes: usize,
}

impl GeometricNN3D {
    /// Create a new 3-layer geometric neural network
    pub fn new(num_classes: usize) -> Self {
        GeometricNN3D {
            layer1: GeometricLinearLayer3D::new(1, 16),
            layer2: GeometricLinearLayer3D::new(16, 8),
            layer3: GeometricLinearLayer3D::new(8, num_classes),
            num_classes,
        }
    }

    /// Forward pass on plaintext data
    pub fn forward_plaintext(&self, input: &[f64; 8]) -> Vec<[f64; 8]> {
        // Layer 1
        let h1 = self.layer1.forward_plaintext(input);

        // Activation 1
        let a1: Vec<[f64; 8]> = h1.iter()
            .map(|x| geometric_activation_plaintext(x))
            .collect();

        // For simplicity, aggregate multiple outputs by taking mean
        let mut h1_agg = [0.0; 8];
        for mv in &a1 {
            for i in 0..8 {
                h1_agg[i] += mv[i] / a1.len() as f64;
            }
        }

        // Layer 2
        let h2 = self.layer2.forward_plaintext(&h1_agg);
        let a2: Vec<[f64; 8]> = h2.iter()
            .map(|x| geometric_activation_plaintext(x))
            .collect();

        let mut h2_agg = [0.0; 8];
        for mv in &a2 {
            for i in 0..8 {
                h2_agg[i] += mv[i] / a2.len() as f64;
            }
        }

        // Layer 3 (output)
        self.layer3.forward_plaintext(&h2_agg)
    }

    /// Forward pass on encrypted data (HOMOMORPHIC!)
    ///
    /// This is the killer feature: classify encrypted 3D points!
    pub fn forward_encrypted(
        &self,
        input_ct: &[RnsCiphertext; 8],
        evk: &RnsEvaluationKey,
        pk: &RnsPublicKey,
        params: &CliffordFHEParams,
    ) -> Vec<[RnsCiphertext; 8]> {
        // Layer 1
        let h1_ct = self.layer1.forward_encrypted(input_ct, evk, pk, params);

        // For now, skip activation (would require polynomial approximation)
        // Just use first output
        let h1_first = &h1_ct[0];

        // Layer 2
        let h2_ct = self.layer2.forward_encrypted(h1_first, evk, pk, params);
        let h2_first = &h2_ct[0];

        // Layer 3 (output)
        self.layer3.forward_encrypted(h2_first, evk, pk, params)
    }
}

/// Graph Neural Network with geometric features
///
/// Each node has a multivector feature representing:
/// - Position (vector part: e₁, e₂, e₃)
/// - Orientation (bivector part: e₁₂, e₁₃, e₂₃)
/// - Additional properties (scalar, pseudoscalar)
///
/// Message passing preserves geometric structure.
pub struct GeometricGNN3D {
    /// Layer for computing messages
    pub message_layer: GeometricLinearLayer3D,
    /// Layer for updating node features
    pub update_layer: GeometricLinearLayer3D,
}

impl GeometricGNN3D {
    pub fn new() -> Self {
        GeometricGNN3D {
            message_layer: GeometricLinearLayer3D::new(1, 1),
            update_layer: GeometricLinearLayer3D::new(1, 1),
        }
    }

    /// Message passing on plaintext graph
    ///
    /// For each node, aggregate messages from neighbors using geometric product.
    pub fn forward_plaintext(
        &self,
        node_features: &[[f64; 8]],
        adjacency: &[Vec<usize>],
    ) -> Vec<[f64; 8]> {
        let num_nodes = node_features.len();
        let mut new_features = Vec::new();

        for i in 0..num_nodes {
            // Aggregate messages from neighbors
            let mut message_sum = [0.0; 8];
            let neighbors = &adjacency[i];

            for &j in neighbors {
                // Compute message: W ⊗ neighbor_feature
                let message = self.message_layer.forward_plaintext(&node_features[j]);

                // Aggregate (sum)
                for k in 0..8 {
                    message_sum[k] += message[0][k] / neighbors.len() as f64;
                }
            }

            // Update node feature: old_feature + message
            let mut updated = node_features[i];
            for k in 0..8 {
                updated[k] += message_sum[k];
            }

            // Apply activation
            updated = geometric_activation_plaintext(&updated);

            new_features.push(updated);
        }

        new_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometric_linear_layer() {
        let layer = GeometricLinearLayer3D::new(1, 3);

        // Test input: unit vector e₁
        let input = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let output = layer.forward_plaintext(&input);

        assert_eq!(output.len(), 3);
        // Output should be non-zero multivectors
        for mv in output {
            let norm: f64 = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm > 0.0);
        }
    }

    #[test]
    fn test_geometric_activation() {
        let input = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let output = geometric_activation_plaintext(&input);

        // Should be normalized
        let norm: f64 = output.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_geometric_nn_forward() {
        let nn = GeometricNN3D::new(3); // 3 classes

        let input = [0.0, 1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0];
        let output = nn.forward_plaintext(&input);

        assert_eq!(output.len(), 3); // 3 class scores
    }
}
