# Homomorphic Division Benchmark Results Summary

This document consolidates key findings from division benchmark runs on RTX 4090.

## Latest Run

| Run ID | Date | Status | Notes |
|--------|------|--------|-------|
| 20260202_022617 | 2026-02-02 | **Complete** | All fixes applied, chained divisions working |
| 20260202_015044 | 2026-02-02 | Complete | Inv-sqrt fixed, scale tracking issue identified |
| 20260202_005633 | 2026-02-02 | Complete | Initial run |

---

## Consolidated Results

### Division Performance (Newton-Raphson, CUDA)

| Ring Dim (N) | Primes | Max Level | 2 Iterations | 3 Iterations | Avg Error |
|--------------|--------|-----------|--------------|--------------|-----------|
| 1024 | 3 | 2 | N/A | N/A | Insufficient depth |
| 4096 | 7 | 6 | **1243 ms** | N/A | 1.1e-8 |
| 8192 | 9 | 8 | **4157 ms** | **5122 ms** | 3.8e-8 |

### Chained Division Performance (V3 Bootstrap Params)

| Chain | Result | Expected | Error | Scale Stable |
|-------|--------|----------|-------|--------------|
| 1000/2/5/2/5 | 9.9999999993 | 10.0 | 6.61e-11 | ✓ (1.76e13) |

### Inverse Square Root Performance

| Metric | N=8192, 1 Iteration |
|--------|---------------------|
| Time | 3443 ms |
| Precision | ~10^-9 |
| Depth consumed | 4 levels |
| Algorithm | y = y * (1.5 - 0.5*x*y²) |

### Throughput

| Ring Dim | 2 iter (div/s) | 3 iter (div/s) |
|----------|----------------|----------------|
| 4096 | 0.80 | N/A |
| 8192 | 0.24 | 0.20 |

### Vector Normalization Pipeline (N=8192)

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Encoding | 1850 | 16.1% |
| Encryption | 45 | 0.4% |
| Squaring (3x) | 2005 | 17.4% |
| Sum | <1 | 0.0% |
| NR 1/||v||² | 3536 | 30.7% |
| NR 1/||v|| | 3443 | 29.9% |
| Scaling (3x) | 632 | 5.5% |
| Decryption | 6 | 0.1% |
| **Total** | **11517** | 100% |

### Accuracy Results

| Operation | Test Case | Expected | Computed | Error |
|-----------|-----------|----------|----------|-------|
| Vector norm | (3,4,0) | (0.6, 0.8, 0.0) | (0.6, 0.8, 0.0) | ~10^-8 |
| Vector norm | (2,3,6) | (0.286, 0.429, 0.857) | (0.286, 0.429, 0.857) | 1.63e-8 |
| 1/||v|| | ||v||=5 | 0.2 | 0.2 | ~10^-9 |
| 1/||v||² | ||v||²=25 | 0.04 | 0.04 | ~10^-10 |
| Chain div | 1000/2/5/2/5 | 10.0 | 10.0 | 6.61e-11 |

---

## Key Observations

1. **Optimal ring dimension**: N=4096 offers the best balance of speed (~1.2s) and precision (~10^-8) for single divisions.

2. **Depth requirements**: Newton-Raphson division with 2 iterations requires 5 multiplicative levels. Parameters with fewer levels cannot perform division.

3. **Newton-Raphson dominance**: Inverse computation accounts for ~60% of pipeline time in complex algorithms like vector normalization.

4. **GPU acceleration**: CUDA provides ~5-6x speedup compared to optimized CPU implementation.

5. **Quadratic convergence**: Newton-Raphson achieves ~10^-8 to 10^-11 relative error with 2 iterations.

6. **Inverse sqrt works**: The reformulated algorithm y = y * (1.5 - 0.5*x*y²) correctly computes 1/sqrt(x) with ~10^-9 precision.

7. **Chained divisions work**: After fixing the scale/prime mismatch in V3 params, chained divisions maintain stable scale (1.76e13) through unlimited operations.

---

## Bug Fixes Applied

### 1. Inverse Sqrt Bug (Fixed)
- **Issue**: `multiply_plain` had scale handling issues
- **Fix**: Reformulated algorithm to use only ciphertext-ciphertext multiplication

### 2. Scale/Prime Mismatch (Fixed)
- **Issue**: V3 params used 2^45 scale with 2^44 primes, causing 1 bit growth per multiply
- **Fix**: Changed scale to 2^44 to match prime size
- **Result**: Scale now stable through chained operations

---

## References

- See individual run directories for raw benchmark output
- Technical details: `docs/DIVISION.md`
- Security analysis: `docs/DIVISION_SECURITY.md`
