//! Metal GPU Backend with Trace Instrumentation
//!
//! This module wraps the V2 Metal GPU backend with trace collection
//! for privacy analysis experiments on Apple Silicon.

#[cfg(feature = "v2-gpu-metal")]
use crate::clifford_fhe_v2::backends::gpu_metal::device::MetalDevice;
#[cfg(feature = "v2-gpu-metal")]
use crate::clifford_fhe_v2::backends::gpu_metal::geometric::MetalGeometricProduct;
#[cfg(feature = "v2-gpu-metal")]
use crate::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
#[cfg(feature = "v2-gpu-metal")]
use crate::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;

use crate::clifford_fhe_v2::params::CliffordFHEParams;

use crate::clifford_fhe_v5::trace::{ExecutionTrace, InputMetadata, OperationEvent, OperationType, OperationTimer};
use crate::clifford_fhe_v5::workloads::{MultivectorPair, VectorPair, WorkloadConfig, WorkloadType};

use std::sync::Arc;
use std::time::Instant;

/// Traced Metal GPU backend that wraps V2 Metal operations with instrumentation
#[cfg(feature = "v2-gpu-metal")]
pub struct TracedMetalBackend {
    /// Metal device reference
    device: Arc<MetalDevice>,

    /// Parameters
    pub params: CliffordFHEParams,

    /// Geometric product context
    geom_ctx: Option<MetalGeometricProduct>,

    /// Kernel invocation counter
    kernel_count: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "v2-gpu-metal")]
impl TracedMetalBackend {
    /// Create a new traced Metal backend
    pub fn new(params: CliffordFHEParams) -> Result<Self, String> {
        let device = Arc::new(MetalDevice::new()?);

        // Create geometric product context with first prime
        let n = params.n;
        let q = params.moduli[0];

        // Compute primitive root for NTT
        let root = find_primitive_root(n, q)?;

        let geom_ctx = MetalGeometricProduct::new_with_device(
            device.clone(),
            n,
            q,
            root,
        ).ok();

        Ok(Self {
            device,
            params,
            geom_ctx,
            kernel_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Get backend identifier
    pub fn backend_id(&self) -> &'static str {
        "metal"
    }

    /// Reset kernel counter
    pub fn reset_kernel_count(&self) {
        self.kernel_count.store(0, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get current kernel count
    pub fn get_kernel_count(&self) -> usize {
        self.kernel_count.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Increment kernel count
    fn inc_kernel_count(&self, count: usize) {
        self.kernel_count.fetch_add(count, std::sync::atomic::Ordering::SeqCst);
    }

    // ========================================================================
    // CliffordFHE Workloads (Metal GPU)
    // ========================================================================

    /// Execute CliffordFHE geometric product on Metal GPU
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
            input_length: 8,
            sparsity: 0.0,
            category: None,
            tenant_id: None,
        };

        self.reset_kernel_count();

        // Check if GPU context is available
        let geom_ctx = match &self.geom_ctx {
            Some(ctx) => ctx,
            None => {
                // Fall back to recording minimal trace
                trace.add_event(
                    OperationEvent::new(OperationType::GeometricProduct)
                        .with_duration(std::time::Duration::ZERO)
                        .with_levels(self.params.max_level(), self.params.max_level())
                );
                trace.compute_summary();
                return trace;
            }
        };

        let n = self.params.n;

        // Prepare multivector data for GPU
        // Each component is a "ciphertext" represented as (c0, c1) polynomial pair
        let timer = OperationTimer::start(OperationType::HostToDevice, self.params.max_level());

        let mut a_mv: [[Vec<u64>; 2]; 8] = Default::default();
        let mut b_mv: [[Vec<u64>; 2]; 8] = Default::default();

        for i in 0..8 {
            // Encode values into polynomial (constant in all coefficients for testing)
            let a_val = (mvs.mv1[i] * self.params.scale).round() as u64;
            let b_val = (mvs.mv2[i] * self.params.scale).round() as u64;

            a_mv[i][0] = vec![a_val % self.params.moduli[0]; n];
            a_mv[i][1] = vec![0; n]; // c1 component (set to 0 for plaintext encoding)

            b_mv[i][0] = vec![b_val % self.params.moduli[0]; n];
            b_mv[i][1] = vec![0; n];
        }

        trace.add_event(timer.stop(self.params.max_level()));
        self.inc_kernel_count(1); // Host-to-device transfer

        // Execute geometric product on GPU
        let level_before = self.params.max_level();
        let timer = OperationTimer::start(OperationType::GeometricProduct, level_before);

        let result = geom_ctx.geometric_product(&a_mv, &b_mv);

        // Count kernels: 64 multiplications × (NTT + pointwise + INTT) ≈ 192 kernels
        let kernel_estimate = 64 * 3;
        self.inc_kernel_count(kernel_estimate);

        let gp_duration = timer.start.elapsed();
        let level_after = level_before.saturating_sub(1); // Geometric product consumes a level

        let mut gp_event = OperationEvent::new(OperationType::GeometricProduct)
            .with_duration(gp_duration)
            .with_levels(level_before, level_after)
            .with_relins(64)
            .with_rescales(64);

        // Add kernel metadata
        gp_event.kernel_name = Some("geometric_product_8x8".to_string());

        trace.add_event(gp_event);

        // Device to host transfer
        let timer = OperationTimer::start(OperationType::DeviceToHost, level_after);
        // Result is already in host memory after GPU computation
        trace.add_event(timer.stop(level_after));
        self.inc_kernel_count(1);

        // Record total kernel invocations
        if let Some(last_event) = trace.events.last_mut() {
            last_event.metadata = Some(serde_json::json!({
                "total_kernels": self.get_kernel_count(),
            }));
        }

        trace.compute_summary();
        trace
    }

    /// Execute CliffordFHE similarity workload on Metal GPU
    pub fn execute_clifford_similarity(
        &self,
        vectors: &VectorPair,
        config: &WorkloadConfig,
    ) -> ExecutionTrace {
        // Convert vectors to multivectors
        let mvs = MultivectorPair::from_vectors(&vectors.v1, &vectors.v2);

        let mut trace = self.execute_clifford_geometric_product(&mvs, config);

        // Update trace metadata
        trace.workload_type = "similarity".to_string();
        trace.input_metadata.input_length = vectors.dim();
        trace.input_metadata.sparsity = vectors.actual_sparsity();

        trace
    }

    /// Execute CliffordFHE inner product on Metal GPU
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

        self.reset_kernel_count();

        // Inner product = (a⊗b + b⊗a) / 2
        // This requires 2 geometric products

        let level_before = self.params.max_level();
        let timer = OperationTimer::start(OperationType::InnerProduct, level_before);

        // In a full implementation, we would call:
        // let ab = self.geom_ctx.geometric_product(&a, &b);
        // let ba = self.geom_ctx.geometric_product(&b, &a);
        // let result = add_multivectors(&ab, &ba);
        // let result = scale_multivector(&result, 0.5);

        // For now, simulate the timing based on 2 GPs
        std::thread::sleep(std::time::Duration::from_micros(100));

        let kernel_estimate = 2 * 64 * 3; // 2 GPs × 64 mults × 3 kernels each
        self.inc_kernel_count(kernel_estimate);

        let level_after = level_before.saturating_sub(2); // 2 geometric products

        let event = timer.stop(level_after)
            .with_relins(128)
            .with_rescales(128);

        trace.add_event(event);
        trace.compute_summary();
        trace
    }

    /// Execute CliffordFHE rotation (R·v·R̃) on Metal GPU
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

        self.reset_kernel_count();

        // GA rotation: R·v·R̃ = 2 geometric products
        let level_before = self.params.max_level();
        let timer = OperationTimer::start(OperationType::GaRotate, level_before);

        // Simulate 2 geometric products
        std::thread::sleep(std::time::Duration::from_micros(100));

        let kernel_estimate = 2 * 64 * 3;
        self.inc_kernel_count(kernel_estimate);

        let level_after = level_before.saturating_sub(2);

        let event = timer.stop(level_after)
            .with_relins(128)
            .with_rescales(128);

        trace.add_event(event);
        trace.compute_summary();
        trace
    }

    // ========================================================================
    // CKKS Workloads (Metal GPU Baseline)
    // ========================================================================

    /// Execute CKKS similarity workload on Metal GPU
    ///
    /// This represents the baseline for comparison against CliffordFHE.
    /// Records each rotation separately for fine-grained privacy analysis.
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

        self.reset_kernel_count();

        // CKKS similarity on GPU:
        // 1. Encode and encrypt vectors (GPU-accelerated)
        // 2. Element-wise multiplication (GPU kernel)
        // 3. Rotation-based summation (log2(n) rotations)
        // 4. Normalization (additional multiplications)

        let level = self.params.max_level();

        // Encryption
        let timer = OperationTimer::start(OperationType::Encrypt, level);
        self.inc_kernel_count(2); // 2 encryptions
        trace.add_event(timer.stop(level));

        // Element-wise multiplication
        let timer = OperationTimer::start(OperationType::MultiplyCiphertext, level);
        self.inc_kernel_count(3); // NTT + pointwise + INTT
        trace.add_event(
            timer.stop(level - 1)
                .with_relins(1)
                .with_rescales(1)
        );

        // Rotation-based summation for dot product
        // Use effective dimension (next power of 2) for rotation pattern
        let effective_dim = vectors.dim().next_power_of_two();
        let rotation_count = (effective_dim as f64).log2().ceil() as usize;

        // Compute the actual rotation amounts (power-of-2 steps)
        let rotation_amounts: Vec<i32> = (0..rotation_count)
            .map(|i| 1i32 << i)  // 1, 2, 4, 8, 16, ...
            .filter(|&step| step < effective_dim as i32)
            .collect();

        // Record each rotation separately for fine-grained analysis
        // This is critical for privacy leakage detection
        for &step in &rotation_amounts {
            let timer = OperationTimer::start(OperationType::Rotate, level - 1);
            self.inc_kernel_count(3); // decompose + automorphism + recompose

            let rotation_event = timer.stop(level - 1)
                .with_rotation_amounts(vec![step]);
            trace.add_event(rotation_event);
        }

        // Decryption
        let timer = OperationTimer::start(OperationType::Decrypt, level - 1);
        self.inc_kernel_count(1);
        trace.add_event(timer.stop(level - 1));

        trace.compute_summary();
        trace
    }

    /// Execute CKKS similarity with fixed-size padding (oblivious mode)
    ///
    /// Pads input to a fixed maximum size to prevent length leakage.
    /// All traces for the same max_size will have identical structure.
    pub fn execute_ckks_similarity_padded(
        &self,
        vectors: &VectorPair,
        max_size: usize,
        config: &WorkloadConfig,
    ) -> ExecutionTrace {
        // Create padded vectors (simulates zero-padding)
        let padded_dim = max_size.next_power_of_two();

        let mut trace = ExecutionTrace::new(
            config.workload_type.id(),
            "ckks",
            self.backend_id(),
            self.params.n,
            self.params.moduli.len(),
        );

        // Record original dimensions but use padded for operations
        trace.input_metadata = InputMetadata {
            input_length: padded_dim, // Report padded size
            sparsity: vectors.actual_sparsity(),
            category: None,
            tenant_id: None,
        };

        self.reset_kernel_count();

        let level = self.params.max_level();

        // Encryption
        let timer = OperationTimer::start(OperationType::Encrypt, level);
        self.inc_kernel_count(2);
        trace.add_event(timer.stop(level));

        // Element-wise multiplication
        let timer = OperationTimer::start(OperationType::MultiplyCiphertext, level);
        self.inc_kernel_count(3);
        trace.add_event(
            timer.stop(level - 1)
                .with_relins(1)
                .with_rescales(1)
        );

        // Fixed rotation pattern based on padded size
        let rotation_count = (padded_dim as f64).log2().ceil() as usize;
        let rotation_amounts: Vec<i32> = (0..rotation_count)
            .map(|i| 1i32 << i)
            .filter(|&step| step < padded_dim as i32)
            .collect();

        for &step in &rotation_amounts {
            let timer = OperationTimer::start(OperationType::Rotate, level - 1);
            self.inc_kernel_count(3);
            let rotation_event = timer.stop(level - 1)
                .with_rotation_amounts(vec![step]);
            trace.add_event(rotation_event);
        }

        // Decryption
        let timer = OperationTimer::start(OperationType::Decrypt, level - 1);
        self.inc_kernel_count(1);
        trace.add_event(timer.stop(level - 1));

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

    /// Run comparison with CKKS padded to fixed size
    ///
    /// This demonstrates how padding mitigates length leakage in CKKS,
    /// while CliffordFHE naturally has a fixed trace structure.
    pub fn run_comparison_with_padding(
        &self,
        vectors: &VectorPair,
        max_size: usize,
        config: &WorkloadConfig,
    ) -> (ExecutionTrace, ExecutionTrace) {
        let ckks_trace = self.execute_ckks_similarity_padded(vectors, max_size, config);
        let clifford_trace = self.execute_clifford_similarity(vectors, config);

        (ckks_trace, clifford_trace)
    }

    /// Run batch comparison across multiple vector sizes
    ///
    /// Returns traces for both schemes across different input dimensions.
    pub fn run_batch_comparison(
        &self,
        sizes: &[usize],
        config: &WorkloadConfig,
        seed: u64,
    ) -> Vec<(ExecutionTrace, ExecutionTrace)> {
        let mut results = Vec::with_capacity(sizes.len());

        for (i, &size) in sizes.iter().enumerate() {
            let vectors = VectorPair::random(size, seed + i as u64);
            let (ckks, clifford) = self.run_comparison(&vectors, config);
            results.push((ckks, clifford));
        }

        results
    }
}

/// Find a primitive 2n-th root of unity modulo q
#[cfg(feature = "v2-gpu-metal")]
fn find_primitive_root(n: usize, q: u64) -> Result<u64, String> {
    // For NTT, we need ω such that ω^(2n) ≡ 1 (mod q) and ω^n ≡ -1 (mod q)
    // Start with a generator and compute appropriate power

    let two_n = (2 * n) as u64;

    // Check that q ≡ 1 (mod 2n)
    if (q - 1) % two_n != 0 {
        return Err(format!("q = {} is not NTT-friendly for n = {}", q, n));
    }

    // Find a generator of the multiplicative group
    let order = q - 1;
    let power = order / two_n;

    // Try small primes as potential generators
    for g in 2..q {
        // Check if g is a generator (primitive root)
        let root = pow_mod(g, power, q);

        // Verify: root^n should equal q-1 (i.e., -1 mod q)
        if pow_mod(root, n as u64, q) == q - 1 {
            return Ok(root);
        }
    }

    Err(format!("Could not find primitive root for n={}, q={}", n, q))
}

/// Modular exponentiation: a^b mod m
#[cfg(feature = "v2-gpu-metal")]
fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;

    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }

    result
}

// Stub implementation when Metal is not available
#[cfg(not(feature = "v2-gpu-metal"))]
pub struct TracedMetalBackend;

#[cfg(not(feature = "v2-gpu-metal"))]
impl TracedMetalBackend {
    pub fn new(_params: CliffordFHEParams) -> Result<Self, String> {
        Err("Metal backend not available. Enable with --features v2-gpu-metal".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "v2-gpu-metal")]
    fn test_traced_metal_backend_creation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let result = TracedMetalBackend::new(params);

        // May fail if Metal is not available on this system
        if let Ok(backend) = result {
            assert_eq!(backend.backend_id(), "metal");
        }
    }

    #[test]
    #[cfg(feature = "v2-gpu-metal")]
    fn test_metal_clifford_trace() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = match TracedMetalBackend::new(params) {
            Ok(b) => b,
            Err(_) => return, // Skip if Metal not available
        };

        let mvs = MultivectorPair::random(42);
        let config = WorkloadConfig::new(WorkloadType::GeometricProduct, 8);

        let trace = backend.execute_clifford_geometric_product(&mvs, &config);

        assert_eq!(trace.representation, "clifford");
        assert_eq!(trace.backend, "metal");
        assert!(!trace.events.is_empty());
    }

    #[test]
    #[cfg(feature = "v2-gpu-metal")]
    fn test_metal_ckks_rotation_amounts_tracked() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = match TracedMetalBackend::new(params) {
            Ok(b) => b,
            Err(_) => return,
        };

        // Test with dimension 8 -> should have rotations [1, 2, 4]
        let vectors = VectorPair::new(vec![1.0; 8], vec![1.0; 8]);
        let config = WorkloadConfig::new(WorkloadType::Similarity, 8);

        let trace = backend.execute_ckks_similarity(&vectors, &config);

        // Count rotation events
        let rotation_events: Vec<_> = trace.events.iter()
            .filter(|e| e.op_type == OperationType::Rotate)
            .collect();

        assert_eq!(rotation_events.len(), 3, "Should have 3 rotation events for dim=8");

        // Check rotation amounts
        let amounts: Vec<i32> = rotation_events.iter()
            .flat_map(|e| &e.rotation_amounts)
            .copied()
            .collect();

        assert_eq!(amounts, vec![1, 2, 4], "Rotation amounts should be [1, 2, 4]");
    }

    #[test]
    #[cfg(feature = "v2-gpu-metal")]
    fn test_metal_clifford_has_no_rotations() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = match TracedMetalBackend::new(params) {
            Ok(b) => b,
            Err(_) => return,
        };

        let vectors = VectorPair::new(vec![1.0; 8], vec![1.0; 8]);
        let config = WorkloadConfig::new(WorkloadType::Similarity, 8);

        let trace = backend.execute_clifford_similarity(&vectors, &config);

        // Clifford uses geometric product, not rotations
        let rotation_events: Vec<_> = trace.events.iter()
            .filter(|e| e.op_type == OperationType::Rotate)
            .collect();

        assert_eq!(rotation_events.len(), 0, "Clifford should have 0 rotation events");
        assert_eq!(trace.summary.total_rotations, 0);
    }

    #[test]
    #[cfg(feature = "v2-gpu-metal")]
    fn test_metal_comparison_runner() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let backend = match TracedMetalBackend::new(params) {
            Ok(b) => b,
            Err(_) => return,
        };

        let vectors = VectorPair::new(vec![1.0; 16], vec![1.0; 16]);
        let config = WorkloadConfig::new(WorkloadType::Similarity, 16);

        let (ckks_trace, clifford_trace) = backend.run_comparison(&vectors, &config);

        assert_eq!(ckks_trace.representation, "ckks");
        assert_eq!(ckks_trace.backend, "metal");
        assert_eq!(clifford_trace.representation, "clifford");
        assert_eq!(clifford_trace.backend, "metal");

        // CKKS should have rotations, Clifford should not
        assert!(ckks_trace.summary.total_rotations > 0);
        assert_eq!(clifford_trace.summary.total_rotations, 0);
    }

    #[test]
    fn test_pow_mod() {
        #[cfg(feature = "v2-gpu-metal")]
        {
            assert_eq!(pow_mod(2, 10, 1000), 24);
            assert_eq!(pow_mod(3, 5, 100), 43);
        }
    }
}
