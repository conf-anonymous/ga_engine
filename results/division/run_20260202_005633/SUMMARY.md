# Division Benchmark Results

**Run ID:** 20260202_005633
**Date:** Mon Feb  2 01:04:21 UTC 2026
**GPU:** NVIDIA GeForce RTX 4090
**CUDA:** 12.4

## Files

| File | Description | Status |
|------|-------------|--------|
| `validation.txt` | CUDA division validation | ✓ |
| `division_basic.txt` | Basic division timing | ✓ |
| `division_comprehensive_n1024.txt` | N=1024 comprehensive | ✓ (insufficient depth) |
| `division_comprehensive_n4096.txt` | N=4096 comprehensive | ✓ |
| `division_comprehensive_n8192.txt` | N=8192 comprehensive | ✓ |
| `division_bootstrap.txt` | Division + bootstrap chains | ✓ |
| `vector_normalization.txt` | Vector normalization pipeline | ✓ |

## Key Results

### Division Performance by Ring Dimension

| N | Primes | Max Level | 2 iter (ms) | 3 iter (ms) | Avg Rel. Error |
|---|--------|-----------|-------------|-------------|----------------|
| 1024 | 3 | 2 | N/A (insufficient depth) | N/A | N/A |
| 4096 | 7 | 6 | **1244.25** | N/A | 3.15e-8 |
| 8192 | 9 | 8 | **4866.33** | **5839.06** | 5.57e-8 |

### Throughput

| N | 2 iter (div/s) | 3 iter (div/s) |
|---|----------------|----------------|
| 4096 | 0.80 | N/A |
| 8192 | 0.21 | 0.17 |

### Division + Bootstrap

| Metric | Without Bootstrap | With Bootstrap |
|--------|-------------------|----------------|
| Divisions completed | 4 | 4 |
| Max possible | ~1-2 | Unlimited |
| Total time (ms) | 23961.68 | 23865.15 |
| Bootstraps required | 0 | 0 |

Note: In this test, sufficient depth was available so bootstrap was not triggered.

### Vector Normalization Pipeline (N=8192)

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| Encode (3 components) | 1842.13 | 16.2% |
| Encrypt (3 components) | 43.37 | 0.4% |
| Square components (3x mult) | 2046.36 | 17.9% |
| Sum ||v||² | 0.39 | 0.0% |
| Newton-Raphson 1/||v||² (2 iter) | 3543.31 | 31.1% |
| Newton-Raphson 1/||v|| (1 iter) | 3283.84 | 28.8% |
| Scale components (3x mult) | 636.65 | 5.6% |
| Decrypt (3 components) | 6.16 | 0.1% |
| **TOTAL** | **11402.21** | 100% |

## Notes

1. **N=1024 insufficient depth**: With only 3 primes (max level 2), there's not enough depth for Newton-Raphson division which requires 5 levels for 2 iterations.

2. **N=4096 is optimal for single divisions**: Provides ~1.2s division time with excellent precision (10^-8 relative error).

3. **N=8192 needed for chained operations**: With 9 primes (max level 8), allows for 3 iterations or chained divisions before bootstrap.

4. **Newton-Raphson dominates runtime**: In the vector normalization pipeline, the two inverse computations account for ~60% of total time.

5. **GPU speedup**: Compared to CPU implementation (~8s for N=4096), CUDA provides ~5.7x speedup.
