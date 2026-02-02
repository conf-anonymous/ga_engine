# Division Benchmark Results

**Run ID:** 20260202_015044
**Date:** Mon Feb  2 01:58:15 UTC 2026
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
| `division_bootstrap.txt` | Division + bootstrap chains | ✓ (scale tracking issue) |
| `vector_normalization.txt` | Vector normalization pipeline | ✓ |

## Key Results

### Division Performance by Ring Dimension

| N | Primes | Max Level | 2 iter (ms) | 3 iter (ms) | Rel. Error |
|---|--------|-----------|-------------|-------------|------------|
| 1024 | 3 | 2 | N/A | N/A | Insufficient depth |
| 4096 | 7 | 6 | **1241** | N/A | ~1.7e-8 |
| 8192 | 9 | 8 | **4866** | **5839** | ~5.6e-8 |

### Inverse Square Root (Newton-Raphson)

| Metric | Value |
|--------|-------|
| Time (1 iter, N=8192) | ~3460 ms |
| Precision | ~10^-9 |
| Depth consumed | 4 levels |

### Vector Normalization Pipeline (N=8192)

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| Encode (3 components) | 1885 | 16.3% |
| Encrypt (3 components) | 45 | 0.4% |
| Square components (3x mult) | 2044 | 17.6% |
| Sum ||v||² | 0.4 | 0.0% |
| Newton-Raphson 1/||v||² (2 iter) | 3520 | 30.4% |
| Newton-Raphson 1/||v|| (1 iter) | 3464 | 29.9% |
| Scale components (3x mult) | 629 | 5.4% |
| Decrypt (3 components) | 6 | 0.1% |
| **TOTAL** | **11593** | 100% |

### Accuracy Results

| Operation | Test Case | Expected | Computed | Error |
|-----------|-----------|----------|----------|-------|
| Vector norm | (3,4,0) | (0.6, 0.8, 0.0) | (0.6, 0.8, 0.0) | 9.89e-9 |
| Vector norm | (1,2,2) | (0.33, 0.67, 0.67) | (0.33, 0.67, 0.67) | 4.33e-9 |
| Vector norm | (2,3,6) | (0.286, 0.429, 0.857) | (0.286, 0.429, 0.857) | 4.20e-9 |
| 1/||v|| | ||v||=5 | 0.2 | 0.2 | 1.69e-9 |
| 1/||v||² | ||v||²=25 | 0.04 | 0.04 | 7.73e-10 |

## Notes

1. **N=1024 insufficient depth**: With only 3 primes (max level 2), there's not enough depth for Newton-Raphson division which requires 5 levels for 2 iterations. Benchmark gracefully skips this configuration.

2. **N=4096 optimal for single divisions**: Provides ~1.2s division time with excellent precision (10^-8 relative error).

3. **N=8192 needed for complex pipelines**: With 9 primes (max level 8), allows for vector normalization which requires both 1/x and 1/sqrt(x) operations.

4. **Inverse square root algorithm**: Uses reformulated Newton-Raphson: y_{n+1} = y * (1.5 - 0.5*x*y²). This avoids multiply_plain which had issues, using only ciphertext-ciphertext multiplication.

5. **Newton-Raphson dominates runtime**: In the vector normalization pipeline, the two inverse computations account for ~60% of total time.

6. **GPU acceleration**: CUDA provides ~5-6x speedup compared to optimized CPU implementation.

7. **Known issue - chained divisions**: Scale tracking in chained division operations has a bug where the tracked scale grows exponentially. Single divisions work correctly. This requires further investigation in the rescaling logic.
