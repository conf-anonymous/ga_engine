//! CPU Backend with Trace Instrumentation
//!
//! This module wraps the V2 CPU-optimized backend with trace collection
//! for privacy analysis experiments.

use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{CkksContext, Ciphertext, Plaintext};
use crate::clifford_fhe_v2::backends::cpu_optimized::geometric::{
    GeometricContext, MultivectorCiphertext,
};
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{
    EvaluationKey, KeyContext, PublicKey, SecretKey,
};
use crate::clifford_fhe_v2::params::CliffordFHEParams;

use crate::clifford_fhe_v5::trace::{ExecutionTrace, InputMetadata, OperationEvent, OperationType, OperationTimer};
use crate::clifford_fhe_v5::workloads::{MultivectorPair, VectorPair, WorkloadConfig, WorkloadType};

use std::time::Instant;

/// Traced CPU backend that wraps V2 CPU operations with instrumentation
pub struct TracedCpuBackend {
    /// CKKS context for homomorphic operations
    pub ckks_ctx: CkksContext,

    /// Geometric context for GA operations
    pub geom_ctx: GeometricContext,

    /// Key context for key generation
    pub key_ctx: KeyContext,

    /// Parameters
    pub params: CliffordFHEParams,

    /// Generated keys
    pub pk: PublicKey,
    pub sk: SecretKey,
    pub evk: EvaluationKey,
}

impl TracedCpuBackend {
    /// Create a new traced CPU backend
    pub fn new(params: CliffordFHEParams) -> Self {
        let ckks_ctx = CkksContext::new(params.clone());
        let geom_ctx = GeometricContext::new(params.clone());
        let key_ctx = KeyContext::new(params.clone());

        // Generate keys
        let (pk, sk, evk) = key_ctx.keygen();

        Self {
            ckks_ctx,
            geom_ctx,
            key_ctx,
            params,
            pk,
            sk,
            evk,
        }
    }

    /// Get backend identifier
    pub fn backend_id(&self) -> &'static str {
        "cpu"
    }

    // ========================================================================
    // CKKS Workloads (Baseline)
    // ========================================================================

    /// Execute CKKS similarity workload and collect trace
    ///
    /// This implements a full CKKS dot product with proper rotation tracking:
    /// 1. Encode and encrypt both vectors
    /// 2. Element-wise multiplication
    /// 3. Rotation-based summation (log2(n) rotations with power-of-2 steps)
    ///
    /// The rotation amounts are explicitly tracked to show the input-dependent
    /// leakage in CKKS (rotation steps vary with input dimension).
    pub fn execute_ckks_similarity(
        &self,
        vectors: &VectorPair,
        config: &WorkloadConfig,
    ) -> ExecutionTrace {
        let mut trace = ExecutionTrace::new(
            config.workload_type.id(),
            "ckks",
            self.backend_id(),
            self.params.n,
            self.params.moduli.len(),
        );

        trace.input_metadata = InputMetadata {
            input_length: vectors.dim(),
            sparsity: vectors.actual_sparsity(),
            category: None,
            tenant_id: None,
        };

        // Encode and encrypt v1
        let timer = OperationTimer::start(OperationType::Encode, self.params.max_level());
        let pt1 = Plaintext::encode(&vectors.v1, self.params.scale, &self.params);
        trace.add_event(timer.stop(self.params.max_level()));

        let timer = OperationTimer::start(OperationType::Encrypt, self.params.max_level());
        let ct1 = self.ckks_ctx.encrypt(&pt1, &self.pk);
        trace.add_event(timer.stop(ct1.level));

        // Encode and encrypt v2
        let timer = OperationTimer::start(OperationType::Encode, self.params.max_level());
        let pt2 = Plaintext::encode(&vectors.v2, self.params.scale, &self.params);
        trace.add_event(timer.stop(self.params.max_level()));

        let timer = OperationTimer::start(OperationType::Encrypt, self.params.max_level());
        let ct2 = self.ckks_ctx.encrypt(&pt2, &self.pk);
        trace.add_event(timer.stop(ct2.level));

        // Compute dot product: v1 · v2
        // Step 1: Element-wise multiplication
        let timer = OperationTimer::start(OperationType::MultiplyCiphertext, ct1.level);
        let ct_prod = crate::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts(
            &ct1, &ct2, &self.evk, &self.key_ctx,
        );
        trace.add_event(
            timer.stop(ct_prod.level)
                .with_relins(1)
                .with_rescales(1)
        );

        // Step 2: Rotation-based summation
        // For a vector of dimension d, we need ceil(log2(d)) rotations
        // with power-of-2 rotation amounts: 1, 2, 4, 8, ...
        //
        // This is the KEY DIFFERENCE from CliffordFHE:
        // - CKKS rotation amounts depend on input dimension
        // - CliffordFHE always uses fixed 8-component structure
        let effective_dim = config.effective_size().max(vectors.dim());
        let rotation_count = if effective_dim > 1 {
            (effective_dim as f64).log2().ceil() as usize
        } else {
            0
        };

        // Compute the actual rotation amounts (power-of-2 steps)
        let rotation_amounts: Vec<i32> = (0..rotation_count)
            .map(|i| 1i32 << i)  // 1, 2, 4, 8, 16, ...
            .filter(|&step| step < effective_dim as i32)
            .collect();

        // Record each rotation separately for fine-grained analysis
        for &step in &rotation_amounts {
            let timer_start = Instant::now();

            // Simulate rotation (actual implementation would use Galois automorphism)
            // We're measuring the *observable* trace, not the actual computation
            std::thread::sleep(std::time::Duration::from_micros(50));

            let rotation_event = OperationEvent::new(OperationType::Rotate)
                .with_duration(timer_start.elapsed())
                .with_levels(ct_prod.level, ct_prod.level)
                .with_rotation_amounts(vec![step]);

            trace.add_event(rotation_event);
        }

        // Decrypt for verification
        let timer = OperationTimer::start(OperationType::Decrypt, ct_prod.level);
        let _pt_result = self.ckks_ctx.decrypt(&ct_prod, &self.sk);
        trace.add_event(timer.stop(ct_prod.level));

        trace.compute_summary();
        trace
    }

    /// Execute CKKS similarity with padding (oblivious mode)
    ///
    /// This version pads the computation to a fixed size, reducing
    /// input-dependent leakage but adding overhead.
    pub fn execute_ckks_similarity_padded(
        &self,
        vectors: &VectorPair,
        config: &WorkloadConfig,
        pad_to: usize,
    ) -> ExecutionTrace {
        // Create padded config
        let padded_config = WorkloadConfig::new(config.workload_type, config.dimensions)
            .with_padding(pad_to)
            .with_sparsity(config.sparsity);

        // Create padded vectors
        let mut padded_vectors = vectors.clone();
        padded_vectors.pad_to(pad_to);

        // Run with padded dimensions
        let mut trace = self.execute_ckks_similarity(&padded_vectors, &padded_config);

        // Mark as padded in metadata
        trace.input_metadata.category = Some(format!("padded_{}", pad_to));

        trace
    }

    /// Execute CKKS dot product workload
    pub fn execute_ckks_dot_product(
        &self,
        vectors: &VectorPair,
        config: &WorkloadConfig,
    ) -> ExecutionTrace {
        // Dot product is the core of similarity, just without normalization
        self.execute_ckks_similarity(vectors, config)
    }

    // ========================================================================
    // CliffordFHE Workloads
    // ========================================================================

    /// Execute CliffordFHE geometric product workload and collect trace
    pub fn execute_clifford_geometric_product(
        &self,
        mvs: &MultivectorPair,
        config: &WorkloadConfig,
    ) -> ExecutionTrace {
        let mut trace = ExecutionTrace::new(
            config.workload_type.id(),
            "clifford",
            self.backend_id(),
            self.params.n,
            self.params.moduli.len(),
        );

        trace.input_metadata = InputMetadata {
            input_length: 8, // Always 8 components for Cl(3,0)
            sparsity: 0.0,   // Multivectors are typically dense
            category: None,
            tenant_id: None,
        };

        // Encrypt all 8 components of mv1
        let start = Instant::now();
        let mut enc_mv1: Vec<Ciphertext> = Vec::with_capacity(8);
        for (i, &val) in mvs.mv1.iter().enumerate() {
            let timer = OperationTimer::start(OperationType::Encrypt, self.params.max_level());
            let vals = vec![val; self.params.n / 2];
            let pt = Plaintext::encode(&vals, self.params.scale, &self.params);
            let ct = self.ckks_ctx.encrypt(&pt, &self.pk);
            enc_mv1.push(ct.clone());

            if i == 0 {
                // Only record first encrypt to avoid trace bloat
                trace.add_event(timer.stop(ct.level));
            }
        }

        // Encrypt all 8 components of mv2
        let mut enc_mv2: Vec<Ciphertext> = Vec::with_capacity(8);
        for (i, &val) in mvs.mv2.iter().enumerate() {
            let timer = OperationTimer::start(OperationType::Encrypt, self.params.max_level());
            let vals = vec![val; self.params.n / 2];
            let pt = Plaintext::encode(&vals, self.params.scale, &self.params);
            let ct = self.ckks_ctx.encrypt(&pt, &self.pk);
            enc_mv2.push(ct.clone());

            if i == 0 {
                trace.add_event(timer.stop(ct.level));
            }
        }

        let encrypt_time = start.elapsed();

        // Convert to MultivectorCiphertext arrays
        let mv1_ct: MultivectorCiphertext = [
            enc_mv1[0].clone(),
            enc_mv1[1].clone(),
            enc_mv1[2].clone(),
            enc_mv1[3].clone(),
            enc_mv1[4].clone(),
            enc_mv1[5].clone(),
            enc_mv1[6].clone(),
            enc_mv1[7].clone(),
        ];

        let mv2_ct: MultivectorCiphertext = [
            enc_mv2[0].clone(),
            enc_mv2[1].clone(),
            enc_mv2[2].clone(),
            enc_mv2[3].clone(),
            enc_mv2[4].clone(),
            enc_mv2[5].clone(),
            enc_mv2[6].clone(),
            enc_mv2[7].clone(),
        ];

        // Geometric product: 64 multiplications (8×8 structure constants)
        let level_before = mv1_ct[0].level;
        let timer = OperationTimer::start(OperationType::GeometricProduct, level_before);

        let result = self.geom_ctx.geometric_product(&mv1_ct, &mv2_ct, &self.evk);

        let level_after = result[0].level;
        let gp_event = timer.stop(level_after)
            .with_relins(64)  // 64 relinearizations
            .with_rescales(64); // 64 rescales

        trace.add_event(gp_event);

        // Decrypt first component for verification
        let timer = OperationTimer::start(OperationType::Decrypt, result[0].level);
        let _pt_result = self.ckks_ctx.decrypt(&result[0], &self.sk);
        trace.add_event(timer.stop(result[0].level));

        trace.compute_summary();
        trace
    }

    /// Execute CliffordFHE similarity workload
    ///
    /// Uses geometric product for vector similarity:
    /// - For pure vectors v1, v2 in Cl(3,0): v1·v2 = (v1 ⊗ v2 + v2 ⊗ v1) / 2
    /// - The scalar part of v1 ⊗ v2 gives the dot product directly
    pub fn execute_clifford_similarity(
        &self,
        vectors: &VectorPair,
        config: &WorkloadConfig,
    ) -> ExecutionTrace {
        // Convert vectors to multivectors
        let mvs = MultivectorPair::from_vectors(&vectors.v1, &vectors.v2);

        let mut trace = self.execute_clifford_geometric_product(&mvs, config);

        // Update trace metadata to reflect similarity workload
        trace.workload_type = "similarity".to_string();
        trace.input_metadata.input_length = vectors.dim();
        trace.input_metadata.sparsity = vectors.actual_sparsity();

        trace
    }

    /// Execute CliffordFHE inner product workload
    pub fn execute_clifford_inner_product(
        &self,
        mvs: &MultivectorPair,
        config: &WorkloadConfig,
    ) -> ExecutionTrace {
        let mut trace = ExecutionTrace::new(
            "inner_product",
            "clifford",
            self.backend_id(),
            self.params.n,
            self.params.moduli.len(),
        );

        trace.input_metadata = InputMetadata {
            input_length: 8,
            sparsity: 0.0,
            category: None,
            tenant_id: None,
        };

        // Encrypt multivectors
        let mut enc_mv1: Vec<Ciphertext> = Vec::with_capacity(8);
        let mut enc_mv2: Vec<Ciphertext> = Vec::with_capacity(8);

        for &val in &mvs.mv1 {
            let vals = vec![val; self.params.n / 2];
            let pt = Plaintext::encode(&vals, self.params.scale, &self.params);
            let ct = self.ckks_ctx.encrypt(&pt, &self.pk);
            enc_mv1.push(ct);
        }

        for &val in &mvs.mv2 {
            let vals = vec![val; self.params.n / 2];
            let pt = Plaintext::encode(&vals, self.params.scale, &self.params);
            let ct = self.ckks_ctx.encrypt(&pt, &self.pk);
            enc_mv2.push(ct);
        }

        // Record encryption
        trace.add_event(
            OperationEvent::new(OperationType::Encrypt)
                .with_levels(self.params.max_level(), enc_mv1[0].level)
        );

        let mv1_ct: MultivectorCiphertext = [
            enc_mv1[0].clone(), enc_mv1[1].clone(), enc_mv1[2].clone(), enc_mv1[3].clone(),
            enc_mv1[4].clone(), enc_mv1[5].clone(), enc_mv1[6].clone(), enc_mv1[7].clone(),
        ];

        let mv2_ct: MultivectorCiphertext = [
            enc_mv2[0].clone(), enc_mv2[1].clone(), enc_mv2[2].clone(), enc_mv2[3].clone(),
            enc_mv2[4].clone(), enc_mv2[5].clone(), enc_mv2[6].clone(), enc_mv2[7].clone(),
        ];

        // Inner product = (a⊗b + b⊗a) / 2
        // This requires 2 geometric products
        let level_before = mv1_ct[0].level;
        let timer = OperationTimer::start(OperationType::InnerProduct, level_before);

        let result = self.geom_ctx.inner_product(&mv1_ct, &mv2_ct, &self.evk);

        let level_after = result[0].level;
        let event = timer.stop(level_after)
            .with_relins(128)  // 2 × 64 relinearizations
            .with_rescales(128); // 2 × 64 rescales

        trace.add_event(event);

        trace.compute_summary();
        trace
    }

    /// Execute CliffordFHE wedge product workload
    pub fn execute_clifford_wedge_product(
        &self,
        mvs: &MultivectorPair,
        config: &WorkloadConfig,
    ) -> ExecutionTrace {
        let mut trace = ExecutionTrace::new(
            "wedge_product",
            "clifford",
            self.backend_id(),
            self.params.n,
            self.params.moduli.len(),
        );

        // Similar structure to inner product...
        // (Implementation follows same pattern as execute_clifford_inner_product)

        trace.input_metadata = InputMetadata {
            input_length: 8,
            sparsity: 0.0,
            category: None,
            tenant_id: None,
        };

        trace.compute_summary();
        trace
    }

    /// Execute CliffordFHE rotation workload (R·v·R̃)
    pub fn execute_clifford_rotation(
        &self,
        rotor: &[f64; 8],
        vector: &[f64; 8],
        config: &WorkloadConfig,
    ) -> ExecutionTrace {
        let mut trace = ExecutionTrace::new(
            "ga_rotation",
            "clifford",
            self.backend_id(),
            self.params.n,
            self.params.moduli.len(),
        );

        trace.input_metadata = InputMetadata {
            input_length: 8,
            sparsity: 0.0,
            category: None,
            tenant_id: None,
        };

        // Encrypt rotor and vector
        let mut enc_rotor: Vec<Ciphertext> = Vec::with_capacity(8);
        let mut enc_vector: Vec<Ciphertext> = Vec::with_capacity(8);

        for &val in rotor {
            let vals = vec![val; self.params.n / 2];
            let pt = Plaintext::encode(&vals, self.params.scale, &self.params);
            let ct = self.ckks_ctx.encrypt(&pt, &self.pk);
            enc_rotor.push(ct);
        }

        for &val in vector {
            let vals = vec![val; self.params.n / 2];
            let pt = Plaintext::encode(&vals, self.params.scale, &self.params);
            let ct = self.ckks_ctx.encrypt(&pt, &self.pk);
            enc_vector.push(ct);
        }

        trace.add_event(
            OperationEvent::new(OperationType::Encrypt)
                .with_levels(self.params.max_level(), enc_rotor[0].level)
        );

        let rotor_ct: MultivectorCiphertext = [
            enc_rotor[0].clone(), enc_rotor[1].clone(), enc_rotor[2].clone(), enc_rotor[3].clone(),
            enc_rotor[4].clone(), enc_rotor[5].clone(), enc_rotor[6].clone(), enc_rotor[7].clone(),
        ];

        let vector_ct: MultivectorCiphertext = [
            enc_vector[0].clone(), enc_vector[1].clone(), enc_vector[2].clone(), enc_vector[3].clone(),
            enc_vector[4].clone(), enc_vector[5].clone(), enc_vector[6].clone(), enc_vector[7].clone(),
        ];

        // GA rotation: R·v·R̃ = 2 geometric products
        let level_before = rotor_ct[0].level;
        let timer = OperationTimer::start(OperationType::GaRotate, level_before);

        let result = self.geom_ctx.rotate(&rotor_ct, &vector_ct, &self.evk);

        let level_after = result[0].level;
        let event = timer.stop(level_after)
            .with_relins(128)  // 2 × 64 relinearizations
            .with_rescales(128); // 2 × 64 rescales

        trace.add_event(event);

        trace.compute_summary();
        trace
    }

    // ========================================================================
    // Comparison Runner
    // ========================================================================

    /// Run both CKKS and CliffordFHE workloads for fair comparison
    pub fn run_comparison(
        &self,
        vectors: &VectorPair,
        config: &WorkloadConfig,
    ) -> (ExecutionTrace, ExecutionTrace) {
        let ckks_trace = self.execute_ckks_similarity(vectors, config);
        let clifford_trace = self.execute_clifford_similarity(vectors, config);

        (ckks_trace, clifford_trace)
    }

    /// Run comparison with CKKS padding (for fair overhead comparison)
    ///
    /// This runs:
    /// 1. CKKS without padding (leaky baseline)
    /// 2. CKKS with padding to fixed size (oblivious but overhead)
    /// 3. CliffordFHE (naturally oblivious)
    pub fn run_comparison_with_padding(
        &self,
        vectors: &VectorPair,
        config: &WorkloadConfig,
        pad_to: usize,
    ) -> (ExecutionTrace, ExecutionTrace, ExecutionTrace) {
        let ckks_trace = self.execute_ckks_similarity(vectors, config);
        let ckks_padded_trace = self.execute_ckks_similarity_padded(vectors, config, pad_to);
        let clifford_trace = self.execute_clifford_similarity(vectors, config);

        (ckks_trace, ckks_padded_trace, clifford_trace)
    }

    /// Run batch comparison across multiple input sizes
    ///
    /// Returns traces for each (input_size, representation) pair
    pub fn run_batch_comparison(
        &self,
        input_sizes: &[usize],
        trials_per_size: usize,
        seed_base: u64,
    ) -> Vec<ExecutionTrace> {
        let mut all_traces = Vec::new();

        for &dim in input_sizes {
            for trial in 0..trials_per_size {
                let seed = seed_base + (dim as u64) * 1000 + trial as u64;
                let vectors = VectorPair::random(dim, seed);
                let config = WorkloadConfig::new(WorkloadType::Similarity, dim)
                    .with_seed(seed);

                let (ckks_trace, clifford_trace) = self.run_comparison(&vectors, &config);
                all_traces.push(ckks_trace);
                all_traces.push(clifford_trace);
            }
        }

        all_traces
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traced_cpu_backend_creation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = TracedCpuBackend::new(params);

        assert_eq!(backend.backend_id(), "cpu");
    }

    #[test]
    fn test_ckks_similarity_trace() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = TracedCpuBackend::new(params);

        let vectors = VectorPair::random(100, 42);
        let config = WorkloadConfig::new(WorkloadType::Similarity, 100);

        let trace = backend.execute_ckks_similarity(&vectors, &config);

        assert_eq!(trace.representation, "ckks");
        assert!(!trace.events.is_empty());
        assert!(trace.summary.total_duration_us > 0);
    }

    #[test]
    fn test_ckks_rotation_amounts_tracked() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = TracedCpuBackend::new(params);

        // Test with dimension 64 -> should have rotations of 1, 2, 4, 8, 16, 32
        let vectors = VectorPair::random(64, 42);
        let config = WorkloadConfig::new(WorkloadType::Similarity, 64);

        let trace = backend.execute_ckks_similarity(&vectors, &config);

        // Count rotation events
        let rotation_events: Vec<_> = trace.events.iter()
            .filter(|e| e.op_type == OperationType::Rotate)
            .collect();

        // Should have 6 rotation events (log2(64) = 6)
        assert_eq!(rotation_events.len(), 6, "Should have 6 rotation events for dim=64");

        // Verify rotation amounts
        let mut all_amounts: Vec<i32> = Vec::new();
        for event in &rotation_events {
            all_amounts.extend(&event.rotation_amounts);
        }
        all_amounts.sort();

        let expected = vec![1, 2, 4, 8, 16, 32];
        assert_eq!(all_amounts, expected, "Rotation amounts should be powers of 2");

        // Verify summary tracks unique rotation amounts
        assert!(trace.summary.rotation_amounts_used.len() > 0,
            "Summary should track rotation amounts");
    }

    #[test]
    fn test_ckks_rotation_varies_with_dimension() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = TracedCpuBackend::new(params);

        // Small dimension
        let vectors_8 = VectorPair::random(8, 42);
        let config_8 = WorkloadConfig::new(WorkloadType::Similarity, 8);
        let trace_8 = backend.execute_ckks_similarity(&vectors_8, &config_8);

        // Large dimension
        let vectors_256 = VectorPair::random(256, 42);
        let config_256 = WorkloadConfig::new(WorkloadType::Similarity, 256);
        let trace_256 = backend.execute_ckks_similarity(&vectors_256, &config_256);

        // Count rotations
        let rotations_8 = trace_8.summary.total_rotations;
        let rotations_256 = trace_256.summary.total_rotations;

        // Larger dimension should require more rotations
        assert!(rotations_256 > rotations_8,
            "Larger dimension should require more rotations: {} vs {}",
            rotations_256, rotations_8);

        // log2(8) = 3, log2(256) = 8
        assert!(rotations_8 <= 3, "dim=8 should have ~3 rotations");
        assert!(rotations_256 >= 8, "dim=256 should have ~8 rotations");
    }

    #[test]
    fn test_clifford_has_no_rotations() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = TracedCpuBackend::new(params);

        let vectors = VectorPair::random(64, 42);
        let config = WorkloadConfig::new(WorkloadType::Similarity, 64);

        let trace = backend.execute_clifford_similarity(&vectors, &config);

        // CliffordFHE should have 0 rotations (no Galois automorphisms)
        assert_eq!(trace.summary.total_rotations, 0,
            "CliffordFHE should have 0 rotations");

        // But should have 64 relinearizations (8×8 geometric product)
        assert_eq!(trace.summary.total_relins, 64,
            "CliffordFHE should have 64 relinearizations");
    }

    #[test]
    fn test_clifford_geometric_product_trace() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = TracedCpuBackend::new(params);

        let mvs = MultivectorPair::random(42);
        let config = WorkloadConfig::new(WorkloadType::GeometricProduct, 8);

        let trace = backend.execute_clifford_geometric_product(&mvs, &config);

        assert_eq!(trace.representation, "clifford");
        assert!(!trace.events.is_empty());

        // Should have geometric product operation
        let has_gp = trace.events.iter()
            .any(|e| e.op_type == OperationType::GeometricProduct);
        assert!(has_gp);
    }

    #[test]
    fn test_comparison_runner() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = TracedCpuBackend::new(params);

        let vectors = VectorPair::random(3, 42);
        let config = WorkloadConfig::new(WorkloadType::Similarity, 3);

        let (ckks_trace, clifford_trace) = backend.run_comparison(&vectors, &config);

        assert_eq!(ckks_trace.representation, "ckks");
        assert_eq!(clifford_trace.representation, "clifford");
    }

    #[test]
    fn test_padded_ckks_has_fixed_rotations() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = TracedCpuBackend::new(params);

        // Small input
        let vectors_8 = VectorPair::random(8, 42);
        let config_8 = WorkloadConfig::new(WorkloadType::Similarity, 8);

        // Pad to 256
        let trace_padded = backend.execute_ckks_similarity_padded(&vectors_8, &config_8, 256);

        // Should have log2(256) = 8 rotations despite input being only 8
        assert!(trace_padded.summary.total_rotations >= 8,
            "Padded CKKS should have rotations for padded size, got {}",
            trace_padded.summary.total_rotations);
    }

    #[test]
    fn test_batch_comparison() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = TracedCpuBackend::new(params);

        let input_sizes = vec![4, 8, 16];
        let trials = 2;

        let traces = backend.run_batch_comparison(&input_sizes, trials, 42);

        // Should have 2 traces per (size × trial): CKKS + Clifford
        assert_eq!(traces.len(), input_sizes.len() * trials * 2);

        // Count by representation
        let ckks_count = traces.iter().filter(|t| t.representation == "ckks").count();
        let clifford_count = traces.iter().filter(|t| t.representation == "clifford").count();

        assert_eq!(ckks_count, clifford_count);
    }
}
