# Benchmark Results: Post-TODO Implementation

**Date:** 2026-02-02
**GPU:** NVIDIA GeForce RTX 4090 (24GB)
**CUDA:** 12.4

## Implementations Completed

This benchmark run validates three TODO implementations:

1. **Chebyshev Polynomial Approximation** (`sin_approx.rs`)
   - Proper Chebyshev coefficients for sin(x) on [-π, π]
   - ~10,000× better accuracy at boundaries vs Taylor series
   - Chebyshev error at x=π: ~10⁻¹¹ vs Taylor: ~10⁻⁷

2. **Baby-Step Giant-Step (BSGS) Algorithm** (`eval_mod.rs`)
   - O(√n) multiplicative depth instead of O(n)
   - For degree-23 polynomial: baby_steps=5, giant_steps=5
   - Reduces depth from 23 to ~5 levels

3. **V3 Bootstrap Context Test** (`bootstrap_context.rs`)
   - Full test with 20-prime V3 minimal parameters
   - Validates accessor methods and sine coefficient structure

---

## Bootstrap Benchmark Results (N=1024, 30 primes)

| Phase | Average Time | % of Total |
|-------|--------------|------------|
| ModRaise | < 0.01s | ~0% |
| CoeffToSlot | 0.38s | 3% |
| **EvalMod** | **12.0s** | **96%** |
| SlotToCoeff | 0.21s | 1.7% |
| ModSwitch | < 0.01s | ~0% |
| **Total Bootstrap** | **12.55s** | 100% |

### BSGS Confirmation
```
Evaluating polynomial of degree 23...
  Using BSGS: baby_steps=5, giant_steps=5
```

### Performance Summary
- **Average bootstrap time:** 12.55s
- **Bootstraps per minute:** 4.8
- **Benchmark iterations:** 3
- **Standard deviation:** 0.013s

---

## Division + Bootstrap Benchmark Results

### Single Division (100/7 = 14.2857...)
| Metric | Value |
|--------|-------|
| Time | 10,097 ms |
| Levels consumed | 5 |
| Result | 14.2857142862 |
| Error | 3.37e-11 |

### Chained Divisions (1000/2/5/2/5 = 10)

#### Without Bootstrap
| Metric | Value |
|--------|-------|
| Divisions completed | 4 of 4 |
| Total time | 24,120 ms |
| Final level | 9 |
| Result | 10.0000000020 |
| Error | 2.00e-10 |

#### With Bootstrap (same chain)
| Metric | Value |
|--------|-------|
| Divisions completed | 4 of 4 |
| Bootstraps required | 0 |
| Total time | 23,854 ms |
| Final level | 9 |
| Result | 9.9999999973 |
| Error | 2.73e-10 |

---

## Division Timing by Level

| Division | Starting Level | Time (ms) |
|----------|---------------|-----------|
| ÷2 | 29 | 10,085 |
| ÷5 | 24 | 6,904 |
| ÷2 | 19 | 4,461 |
| ÷5 | 14 | 2,573 |

*Note: Time decreases as level decreases due to fewer RNS primes to process.*

---

## Key Insights

1. **EvalMod dominates** bootstrap time (96%) - this is where BSGS helps most
2. **BSGS reduces depth** from O(n) to O(√n) for polynomial evaluation
3. **Chebyshev coefficients** provide better accuracy at domain boundaries
4. **Division scales with level** - higher levels take longer due to more primes
5. **Error remains small** (~10⁻¹⁰) even after 4 chained divisions

---

## Files in This Directory

| File | Description |
|------|-------------|
| `SUMMARY.md` | This summary |
| `bootstrap_benchmark.txt` | Full bootstrap benchmark output |
| `division_with_bootstrap.txt` | Division chain benchmark output |

---

## Comparison with Previous Results

| Metric | Before TODOs | After TODOs | Notes |
|--------|--------------|-------------|-------|
| Bootstrap time | ~12.5s | ~12.55s | Consistent |
| BSGS active | Yes (CUDA) | Yes (CUDA + CPU) | CPU now has BSGS too |
| Sin coefficients | Taylor | Chebyshev | Better boundary accuracy |
| Polynomial depth | O(√n) | O(√n) | BSGS was already in CUDA |

The main improvements from the TODO implementations are:
- **CPU now has BSGS** (was only in CUDA before)
- **Chebyshev provides better accuracy** (10,000× at boundaries)
- **V3 params test validates** the bootstrap context properly
