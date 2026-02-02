# CUDA Benchmark Results: 20260112_034830

## Environment
- **GPU**: NVIDIA A40
- **VRAM**: 46068 MiB
- **CUDA**: V12.4.131
- **Mode**: full

## Benchmarks Run
1. ✓ CUDA Integration Verification
2. ✓ All Homomorphic Operations (Encode/Encrypt/Add/Mult/Rotate/etc.)
3. ✓ V4 Geometric Product (Quick)
4. ✓ V4 Geometric Product (Full)
5. ✓ V4 Packing
6. ✓ Division
7. ✓ Bootstrap (V3 CKKS)

## Key Results
Extract key timing from log files:
```
grep -E "geometric product|Geometric Product|ms|µs|speedup|Avg|Bootstrap" results/cuda_benchmarks_20260112_034830/*.log
```
