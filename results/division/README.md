# Homomorphic Division Benchmark Results

This directory contains benchmark results specifically for **homomorphic division operations** and related algorithms in the CKKS/Clifford FHE scheme.

## Context

Homomorphic division is a fundamental operation that enables:
- Vector normalization (computing `v / ||v||`)
- Rational arithmetic in encrypted domain
- Machine learning inference (softmax, layer normalization)
- Scientific computing on encrypted data

Division in CKKS is implemented using **Newton-Raphson iteration** to approximate `1/x`, then multiplying: `a/b = a * (1/b)`.

## Benchmark Categories

### 1. Core Division Performance
- `bench_division_cuda_gpu` - Basic CUDA division timing
- `bench_division_comprehensive` - Full test matrix (iterations, ring dimensions, test cases)

### 2. Division + Bootstrapping
- `bench_division_with_bootstrap` - Division chains with noise refresh
- Demonstrates unlimited-depth division capability

### 3. Application Pipelines
- `bench_vector_normalization_pipeline` - Complete geometric algorithm
- End-to-end: encode -> encrypt -> normalize -> decrypt -> verify

## Key Metrics

| Metric | Description | Target (RTX 4090) |
|--------|-------------|-------------------|
| Division Time | Single division latency | < 1000ms (N=4096) |
| Iterations | Newton-Raphson iterations | 2-4 |
| Depth Consumed | Multiplicative levels used | 2k+1 for k iterations |
| Relative Error | Accuracy vs plaintext | < 10^-3 |

## Directory Structure

```
division/
├── README.md                    # This file
├── RESULTS_SUMMARY.md          # Consolidated findings (updated after runs)
└── run_YYYYMMDD_HHMMSS/        # Timestamped benchmark runs
    ├── system_info.txt         # Hardware/software config
    ├── validation.txt          # Initial validation tests
    ├── division_basic.txt      # Basic CUDA division
    ├── division_comprehensive_n*.txt  # By ring dimension
    ├── division_bootstrap.txt  # Division + bootstrap chains
    ├── vector_normalization.txt # Application pipeline
    └── benchmark.log           # Execution log
```

## Running Benchmarks

```bash
./scripts/run_division_benchmarks.sh
```

## Hardware Configuration

All benchmarks in this directory are run on:
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **Platform**: RunPod cloud instance
- **CUDA**: 12.x
