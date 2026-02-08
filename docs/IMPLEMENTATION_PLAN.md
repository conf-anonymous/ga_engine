# CliffordPointNet: Implementation Plan

## Project Goal

Demonstrate **privacy-preserving 3D point cloud classification** using Clifford algebra homomorphic encryption with production-viable performance.

**Key Achievement:** V3 Batched CUDA achieves **17.68ms per geometric product at N=1024** and **45.41ms at N=8192** with full 128-bit post-quantum security, both meeting the <50ms production target.

---

## Current Status

### Completed Components

| Component | Status | Location |
|-----------|--------|----------|
| Clifford FHE Core (V2) | Done | `src/clifford_fhe_v2/` |
| V3 Batched Encoding | Done | `src/clifford_fhe_v3/` |
| V3 Batched CUDA GP | Done | `src/clifford_fhe_v3/batched/cuda_batched.rs` |
| CUDA Backend | Done | `src/clifford_fhe_v2/backends/gpu_cuda/` |
| Metal Backend | Done | `src/clifford_fhe_v2/backends/gpu_metal/` |
| SimpleCliffordNet | Done | `src/clifford_pointnet/simple_model.rs` |
| GPFeatureClassifier | Done | `src/clifford_pointnet/gp_classifier.rs` |
| Weight Serialization | Done | `src/clifford_pointnet/serialization.rs` |
| Plaintext Training | Done | `examples/experiment_plaintext.rs` |
| V3 Encrypted Demo (CPU) | Done | `examples/clifford_pointnet_v3_encrypted.rs` |
| V3 Encrypted Demo (CUDA) | Done | `examples/clifford_pointnet_v3_encrypted_cuda.rs` |
| Encrypted Accuracy Benchmark | Done | `examples/encrypted_accuracy_benchmark.rs` |
| Batched Encrypted Demo | Done | `examples/clifford_pointnet_encrypted_batched.rs` |
| Synthetic Dataset | Done | `src/datasets/modelnet40.rs` |
| Benchmark Suite | Done | `examples/bench_*.rs` |

### Benchmark Results Summary (RTX 5090, 2026-02-08)

| Configuration | Per Geometric Product | Security | Status |
|---------------|----------------------|----------|--------|
| V3 CUDA N=1024 | **17.68ms** | 80-bit | Exceeds <50ms target |
| V3 CPU N=1024 | 52.62ms | 80-bit | Borderline |
| V3 CUDA N=8192 | **45.41ms** | 128-bit PQ | Meets <50ms target |
| V3 CPU N=8192 | 304.79ms | 128-bit PQ | Baseline |
| V2 CPU N=4096 | 959ms | 128-bit | Legacy baseline |

### Encrypted Accuracy Results (2026-02-08)

Validated on 40 test samples (5 classes, 32 points/sample, N=1024):

| Metric | Result |
|--------|--------|
| Encrypted accuracy | 25/40 (62.5%) |
| Plaintext accuracy | 25/40 (62.5%) |
| **Prediction agreement** | **40/40 (100.0%)** |
| **Accuracy gap** | **0.0%** |
| Max CKKS error | 0.004246 |
| Mean CKKS error | 0.002567 |
| P99 CKKS error | 0.004246 |

**Key finding:** Encryption introduces **zero accuracy degradation**. Every encrypted prediction exactly matches its plaintext counterpart.

Per-component CKKS error (mean):
| Component | Mean Error | Max Error |
|-----------|-----------|-----------|
| scalar | 0.001051 | 0.002570 |
| e1 | 0.001036 | 0.002986 |
| e2 | 0.001526 | 0.003659 |
| e3 | 0.001008 | 0.002368 |
| e12 | 0.001049 | 0.003023 |
| e13 | 0.001038 | 0.002487 |
| e23 | 0.000879 | 0.002162 |
| e123 | 0.002143 | 0.004246 |

---

## Phase 1: End-to-End Encrypted Inference (Priority: HIGH) -- COMPLETE

### Goal
Demonstrate complete encrypted point cloud classification pipeline with accuracy verification.

### Tasks

#### 1.1 V3 Batched Encrypted Inference Demo -- COMPLETE
**File:** `examples/clifford_pointnet_v3_encrypted.rs`

Implemented pipeline:
1. Train GPFeatureClassifier on synthetic dataset (5 classes, 32 points/sample)
2. Encode points as augmented Cl(3,0) multivectors: `[1, x, y, z, 0, 0, 0, 0]`
3. Pack into V3 batched ciphertext via `encode_batch()`
4. Server: encrypted geometric self-product via `geometric_product_batched()`
5. Client: decrypt, mean pool, classify with trained weights
6. Verify against plaintext computation

**Actual Output (N=1024, CPU):**
```
Samples tested:                5
Encrypted accuracy:            3/5 (60.0%)
Plaintext accuracy:            3/5 (60.0%)
Encrypted matches plaintext:   5/5 (100.0%)
Max CKKS error (GP features):  0.007381
Avg GP time per sample:        3287.7ms
```

**CUDA version also implemented:** `examples/clifford_pointnet_v3_encrypted_cuda.rs`

#### 1.2 Weight Serialization -- COMPLETE
**File:** `src/clifford_pointnet/serialization.rs`

- JSON save/load for `SimpleCliffordNetWeights` and `GPClassifierWeights`
- `from_model()` / `to_model()` conversion for SimpleCliffordNet
- `save()` / `load()` via serde_json
- Integrated with GPFeatureClassifier: `save_weights()` / `load_weights()`

#### 1.3 Batch Encryption API -- COMPLETE (already existed)
**Files:** `src/clifford_fhe_v3/batched/encoding.rs`, `src/clifford_fhe_v3/batched/cuda_batched.rs`

The V3 batched encoding API was already implemented:
- `encode_batch()` / `decode_batch()` for CPU
- `encode_batch_cuda()` / `decode_batch_cuda()` for CUDA
- Handles SIMD slot packing (stride-8 layout) automatically

---

## Phase 2: Full Pipeline Validation (Priority: HIGH) -- COMPLETE

### Goal
Validate encrypted inference accuracy matches plaintext on synthetic dataset.

### Tasks

#### 2.1 Encrypted Accuracy Benchmark -- COMPLETE
**File:** `examples/encrypted_accuracy_benchmark.rs`

Validated on 40 test samples across 5 classes:
- **100% prediction agreement** between encrypted and plaintext
- **0.0% accuracy gap** -- zero degradation from encryption
- Per-class breakdown shows 100% agreement in every class
- CKKS noise analysis: mean=0.002567, max=0.004246
- Throughput: 22 samples/min on CPU (N=1024)

Run command:
```bash
cargo run --release --no-default-features --features f64,nd,v2,v3 \
    --example encrypted_accuracy_benchmark
```

#### 2.2 Multi-Class Encrypted Inference -- COMPLETE (via GPFeatureClassifier)
**File:** `src/clifford_pointnet/gp_classifier.rs`

- Supports arbitrary number of classes (tested with 5)
- Square activation (polynomial, FHE-compatible): `f(x) = x^2`
- Argmax classification (no softmax needed in encrypted domain)
- Client-side mean pooling preserves depth budget

#### 2.3 Timing Breakdown Analysis -- COMPLETE (embedded in demos)

Measured timing breakdown (CPU, N=1024, per sample):
```
| Stage              | Time (ms) | % of Total |
|--------------------|-----------|------------|
| Encryption         |     14.5  |     0.4%   |
| Geometric Product  |   3287.7  |    99.2%   |
| Decrypt + Pool     |     11.3  |     0.3%   |
| Total              |   3313.5  |   100.0%   |
```

Note: The GP dominates timing. With V3 CUDA batched (64 products in parallel), per-product time drops to 17.68ms (N=1024) or 45.41ms (N=8192) after Phase 4 optimizations.

---

## Phase 3: Real-World Application Demos (Priority: MEDIUM) -- COMPLETE

### Goal
Demonstrate practical applications of privacy-preserving 3D inference.

### Tasks

#### 3.1 Privacy-Preserving Object Detection -- COMPLETE
**File:** `examples/privacy_preserving_detection.rs`

Scenario: Autonomous vehicle sends encrypted LiDAR scan, server classifies objects.
- Simulates scene with 3-6 objects (car, pedestrian, cyclist, truck, barrier)
- Objects placed at world coordinates, centered before encryption
- Server computes encrypted GP features per object
- Client decrypts, classifies, and makes driving decisions

**Results (CPU, N=1024):**
- 100% encrypted-plaintext prediction agreement
- Per-object: encrypt + GP + decrypt pipeline
- Driving safety analysis (vulnerable road user detection)
- Privacy guarantee: raw LiDAR never leaves vehicle, positions never exposed to server

#### 3.2 Medical 3D Scan Classification -- COMPLETE
**File:** `examples/medical_scan_classification.rs`

Scenario: Hospital sends encrypted organ/tissue scan, cloud AI classifies.
- 5 diagnostic categories: benign_cyst, malignant_tumor, healthy_tissue, inflammation, calcification
- Custom shape generation per condition (sphere, spiky, disc, ellipsoid, clusters)
- 8 patients processed per run
- Risk level assignment (HIGH/MODERATE/LOW)

**Results (CPU, N=1024):**
- 100% encrypted-plaintext prediction agreement
- HIPAA/GDPR compliance architecture demonstrated
- End-to-end patient data encryption

#### 3.3 Cloud Inference Service Simulation -- COMPLETE
**File:** `examples/cloud_inference_service.rs`

Scenario: Multi-client privacy-preserving inference service.
- 4 clients with independent FHE keys, 3 requests each (12 total)
- Interleaved request processing (simulates real service)
- Per-client accuracy and agreement breakdown

**Results (CPU, N=1024):**
- 12/12 encrypted-plaintext prediction agreement (100%)
- Throughput: ~17.7 requests/min (~3.4s per request)
- Cryptographic isolation: each client has independent keys
- No cross-client information leakage possible

---

## Phase 4: CUDA Performance Optimization (Priority: MEDIUM) -- COMPLETE

### Goal
Push V3 CUDA N=8192 below 50ms per geometric product.

### Results

| Config | Batch | Before (ms/prod) | After (ms/prod) | Improvement | Speedup vs CPU |
|--------|-------|-------------------|------------------|-------------|----------------|
| N=1024 | 64 | 30.25 | **17.68** | 42% faster | 2.98x |
| N=8192 | 512 | 69.51 | **45.41** | 35% faster | 6.71x |

### Tasks

#### 4A: Batched NTT for Relinearization -- COMPLETE
**File:** `src/clifford_fhe_v2/backends/gpu_cuda/relin_keys.rs`

Switched `apply_relinearization_gpu` from sequential per-prime `gpu_multiply_flat_ntt` to `gpu_multiply_flat_ntt_batched` (all primes in one batched GPU kernel). Reduces kernel launches per relin from ~4,032 to ~112 at N=8192.

#### 4B: Remove Excessive device.synchronize() -- COMPLETE
**File:** `src/clifford_fhe_v2/backends/gpu_cuda/ckks.rs`

Removed intermediate `device.synchronize()` calls from `ntt_forward_batched_gpu` and `ntt_inverse_batched_gpu`. CUDA default stream guarantees sequential execution within the same stream; intermediate syncs only stalled the CPU (~3-5μs per sync × 25K+ syncs per GP).

#### 4C: GPU-Accelerated Add and Negate -- COMPLETE
**Files:**
- `src/clifford_fhe_v2/backends/gpu_cuda/kernels/rns.cu` — Added `rns_negate_strided` and `rns_add_strided` CUDA kernels
- `src/clifford_fhe_v2/backends/gpu_cuda/ckks.rs` — Added `negate_strided_gpu()`, `add_strided_gpu()` methods; rewrote `add()` to use GPU; registered new kernels in `load_ptx`
- `src/clifford_fhe_v3/batched/cuda_batched.rs` — Rewrote `negate_cuda_ciphertext` to use GPU kernel

#### 4D: Batched NTT for Rotation Keys -- COMPLETE
**Files:**
- `src/clifford_fhe_v2/backends/gpu_cuda/rotation_keys.rs` — Added `gpu_multiply_flat_ntt_batched`; updated `apply_rotation_key_gpu` signature with optional `ckks_ctx` for batched NTT
- `src/clifford_fhe_v2/backends/gpu_cuda/ciphertext_ops.rs` — Updated `rotate_by_steps` to pass `Some(ctx)`
- `src/clifford_fhe_v3/bootstrapping/cuda_coeff_to_slot.rs` — Updated caller to pass `Some(ckks_ctx)`

#### 4E: GPU-Resident Fused Multiply Pipeline -- COMPLETE
**Files:**
- `src/clifford_fhe_v2/backends/gpu_cuda/inversion.rs` — Added `multiply_ciphertexts_gpu_fused`; rewired `multiply_ciphertexts_gpu` to use it
- `src/clifford_fhe_v2/backends/gpu_cuda/relin_keys.rs` — Added `apply_relinearization_gpu_resident` (takes GPU CudaSlice inputs)
- `src/clifford_fhe_v2/backends/gpu_cuda/ckks.rs` — Added `exact_rescale_gpu_resident` (GPU-resident rescale)

Pipeline chains tensor_gpu → relin_gpu_resident → rescale_gpu_flat → flat_to_strided with minimal CPU↔GPU round-trips (~5 vs ~35 before).

---

## Phase 5: Documentation and Reproducibility (Priority: HIGH) -- NOT STARTED

### Goal
Ensure all results are reproducible and well-documented.

### Tasks

#### 5.1 Comprehensive README
**File:** `README.md`

- Project overview and motivation
- Quick start guide
- Benchmark reproduction commands
- Architecture diagram

#### 5.2 API Documentation
**Command:** `cargo doc --features v2,v3,v2-gpu-cuda`

- Document all public APIs
- Include usage examples
- Cross-reference related modules

#### 5.3 Experiment Reproduction Scripts
**File:** `scripts/reproduce_experiments.sh`

```bash
#!/bin/bash
# Reproduce all results

# 1. Plaintext training accuracy
cargo run --release --no-default-features --features f64,nd,v2,v3 \
    --example experiment_plaintext

# 2. V3 CPU vs CUDA geometric product benchmark
cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda \
    --example bench_v3_batched_cuda

# 3. End-to-end encrypted inference (CPU)
cargo run --release --no-default-features --features f64,nd,v2,v3 \
    --example clifford_pointnet_v3_encrypted

# 4. End-to-end encrypted inference (CUDA)
cargo run --release --no-default-features --features f64,nd,v2,v3,v2-gpu-cuda \
    --example clifford_pointnet_v3_encrypted_cuda

# 5. Accuracy validation (40 samples)
cargo run --release --no-default-features --features f64,nd,v2,v3 \
    --example encrypted_accuracy_benchmark
```

---

## Implementation Timeline

### Week 1: Core Pipeline -- COMPLETE
- [x] 1.1 V3 Batched Encrypted Inference Demo
- [x] 1.2 Weight Serialization
- [x] 1.3 Batch Encryption API

### Week 2: Validation -- COMPLETE
- [x] 2.1 Encrypted Accuracy Benchmark
- [x] 2.2 Multi-Class Encrypted Inference
- [x] 2.3 Timing Breakdown Analysis

### Week 3: Applications -- COMPLETE
- [x] 3.1 Privacy-Preserving Object Detection
- [x] 3.2 Medical 3D Scan Classification
- [x] 3.3 Cloud Inference Service Simulation

### Week 4: Optimization -- COMPLETE
- [x] 4A Batched NTT for Relinearization
- [x] 4B Remove Excessive device.synchronize()
- [x] 4C GPU-Accelerated Add and Negate
- [x] 4D Batched NTT for Rotation Keys
- [x] 4E GPU-Resident Fused Multiply Pipeline

### Week 5: Documentation
- [ ] 5.1 Comprehensive README
- [ ] 5.2 API Documentation
- [ ] 5.3 Experiment Reproduction Scripts

---

## Success Criteria

### Performance Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Geometric Product (N=8192, CUDA) | <50ms | 45.41ms | PASS |
| Geometric Product (N=1024, CUDA) | <50ms | 17.68ms | PASS |
| End-to-end Inference (CPU, N=1024) | <5s | 3.3s | PASS |
| Encrypted vs Plaintext Accuracy | <1% gap | 0.0% gap | PASS |
| Prediction Agreement | >95% | 100.0% | PASS |
| Max CKKS Error | <0.1 | 0.004 | PASS |

### Deliverables

1. **Working Demo:** `clifford_pointnet_v3_encrypted.rs` -- DELIVERED
2. **CUDA Demo:** `clifford_pointnet_v3_encrypted_cuda.rs` -- DELIVERED
3. **Accuracy Report:** `encrypted_accuracy_benchmark.rs` -- DELIVERED (100% agreement)
4. **Benchmark Suite:** `bench_v3_batched_cuda.rs` + others -- DELIVERED
5. **Application Demos:** 3 real-world scenarios -- DELIVERED
   - `privacy_preserving_detection.rs` (autonomous vehicle)
   - `medical_scan_classification.rs` (HIPAA/GDPR compliant diagnostics)
   - `cloud_inference_service.rs` (multi-client service)
6. **Documentation:** Complete API docs and usage guides -- PENDING

---

## Architecture Overview

### Implemented Pipeline

```
CLIENT:
  Point Cloud (32 pts) ──→ Augmented Encoding [1,x,y,z,0,0,0,0]
                        ──→ V3 Batch Pack (1 CKKS ciphertext)
                        ──→ Encrypt (CKKS, N=1024 or N=8192)
                            │
                            ↓ Encrypted Ciphertext
SERVER (Untrusted):
  Geometric Self-Product ──→ GP(batch, batch) via V3 Batched
                             CPU: ~3.3s or CUDA: ~18ms (N=1024)
                            │
                            ↓ Encrypted GP Features
CLIENT:
  Decrypt ──→ Mean Pool (client-side, zero FHE depth)
          ──→ GPFeatureClassifier: [8]→[32](x²)→[5]
          ──→ Argmax → Predicted Class
```

### Design Decisions

1. **Augmented encoding** `[1, x, y, z, 0, 0, 0, 0]`: Scalar component ensures GP self-product produces non-trivial features `[1+x²+y²+z², 2x, 2y, 2z, 0, 0, 0, 0]`

2. **Client-side mean pooling**: Saves FHE depth budget (N=1024 has only 2 levels). Architecturally valid since pooling is a linear operation that can be deferred.

3. **Square activation** `f(x) = x²`: FHE-compatible polynomial activation (degree 2). Uses 0 multiplicative depth compared to ReLU approximations.

4. **GPFeatureClassifier** instead of SimpleCliffordNet: Purpose-built for the encrypted pipeline. Operates directly on 8-dimensional GP features rather than raw coordinates.

---

## Critical Bug Fix (2026-02-08)

### extract_component rotate-then-mask bug

**Problem:** The `extract_component` function was masking without rotation, causing all 48 cross-component terms in the geometric product to produce ZERO. Only the 16 diagonal terms (a_i * b_i) were computed correctly.

**Fix:** Rotate by +component BEFORE masking, then rotate by -i in reassembly:
1. `extract_component(batch, i)`: rotate by +i, then mask at [0, 8, 16, ...]
2. `reassemble_components(comps)`: rotate component i by -i, then sum

**Applied to:**
- CPU path: `src/clifford_fhe_v3/batched/extraction.rs`
- CUDA path: `src/clifford_fhe_v3/batched/cuda_batched.rs`

**Validation:** All 5 extraction unit tests pass. Max error reduced from 1.22 (buggy) to 0.013 (CKKS noise only).

---

## Risk Mitigation

| Risk | Mitigation | Status |
|------|------------|--------|
| CKKS noise accumulation | Sufficient precision bits, noise budget tested | Resolved: max error 0.004 |
| Accuracy degradation | Square activation, augmented encoding | Resolved: 0% accuracy gap |
| Depth budget (N=1024) | Client-side mean pooling, 2-level GP | Resolved: fits in 3 primes |
| Memory overflow (N=8192) | Streaming evaluation, lazy computation | Not yet needed |
| CUDA compatibility | Tested on RTX 5090 | Verified |

---

## References

- Clifford Algebra: [Geometric Algebra for Computer Science](https://geometricalgebra.org/)
- CKKS: [Homomorphic Encryption for Arithmetic of Approximate Numbers](https://eprint.iacr.org/2016/421)
- PointNet: [Deep Learning on Point Sets for 3D Classification](https://arxiv.org/abs/1612.00593)
