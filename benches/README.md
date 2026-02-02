# Benchmark Suite

This directory contains modular benchmarks for V1 and V2 implementations.

## Quick Reference

### Run All Benchmarks

```bash
# Full comparison (V1 + V2, all operations) - WARNING: ~70 minutes!
cargo bench --bench v1_vs_v2_benchmark --features v1,v2
```

### Run V1 Only

```bash
# V1 core operations only (~2 minutes)
cargo bench --bench v1_core_ops --features v1

# V1 geometric operations only (~55 minutes)
cargo bench --bench v1_geometric_ops --features v1

# All V1 benchmarks (~57 minutes)
cargo bench --features v1
```

### Run V2 Only

```bash
# V2 core operations only (~2 minutes)
cargo bench --bench v2_core_ops --features v2

# V2 geometric operations only (~10 minutes)
cargo bench --bench v2_geometric_ops --features v2

# All V2 benchmarks (~12 minutes)
cargo bench --features v2
```

### Run Specific Operations

#### V1 Core Operations
```bash
# Just key generation
cargo bench --bench v1_core_ops --features v1 keygen

# Just encryption
cargo bench --bench v1_core_ops --features v1 encrypt

# Just decryption
cargo bench --bench v1_core_ops --features v1 decrypt

# Just multiplication
cargo bench --bench v1_core_ops --features v1 multiply
```

#### V1 Geometric Operations
```bash
# Just reverse (~1 minute)
cargo bench --bench v1_geometric_ops --features v1 reverse

# Just geometric product (~12 minutes)
cargo bench --bench v1_geometric_ops --features v1 geometric_product

# Just wedge product (~23 minutes)
cargo bench --bench v1_geometric_ops --features v1 wedge_product

# Just inner product (~23 minutes)
cargo bench --bench v1_geometric_ops --features v1 inner_product
```

#### V2 Core Operations
```bash
# Just key generation
cargo bench --bench v2_core_ops --features v2 keygen

# Just encryption
cargo bench --bench v2_core_ops --features v2 encrypt

# Just decryption
cargo bench --bench v2_core_ops --features v2 decrypt

# Just multiplication
cargo bench --bench v2_core_ops --features v2 multiply
```

#### V2 Geometric Operations
```bash
# Just reverse (~1 minute)
cargo bench --bench v2_geometric_ops --features v2 reverse

# Just geometric product (~3 minutes)
cargo bench --bench v2_geometric_ops --features v2 geometric_product

# Just wedge product (~5 minutes)
cargo bench --bench v2_geometric_ops --features v2 wedge_product

# Just inner product (~5 minutes)
cargo bench --bench v2_geometric_ops --features v2 inner_product
```

## Benchmark Files

| File | Description | Runtime | Command |
|------|-------------|---------|---------|
| `v1_core_ops.rs` | V1 key generation, encrypt, decrypt, multiply | ~2 min | `cargo bench --bench v1_core_ops --features v1` |
| `v1_geometric_ops.rs` | V1 reverse, geometric product, wedge, inner | ~55 min | `cargo bench --bench v1_geometric_ops --features v1` |
| `v2_core_ops.rs` | V2 key generation, encrypt, decrypt, multiply | ~2 min | `cargo bench --bench v2_core_ops --features v2` |
| `v2_geometric_ops.rs` | V2 reverse, geometric product, wedge, inner | ~10 min | `cargo bench --bench v2_geometric_ops --features v2` |
| `v1_vs_v2_benchmark.rs` | Full V1 vs V2 comparison (all operations) | ~70 min | `cargo bench --bench v1_vs_v2_benchmark --features v1,v2` |

## Reducing Sample Size

For faster testing, you can reduce the number of samples:

```bash
# Run with fewer samples (10 instead of 50)
cargo bench --bench v1_geometric_ops --features v1 -- --sample-size 10

# This reduces V1 geometric operations from ~55 minutes to ~11 minutes
```

## Configuration

All benchmarks use:
- **Measurement time**: 10 seconds per benchmark
- **Sample size**: 100 for core operations, 50 for geometric operations
- **Criterion framework**: Statistical analysis with outlier detection

## Expected Runtimes (with default sample sizes)

### V1 Operations (Actual on Apple M3 Max)
- Key Generation: ~10s (100 samples × 48ms)
- Encryption: ~10s (100 samples × 11ms)
- Decryption: ~10s (100 samples × 5ms)
- Multiplication: ~10s (100 samples × 110ms)
- Reverse: ~1 min (50 samples × 1ms)
- Geometric Product: ~10 min (50 samples × 11.42s)
- Wedge Product: ~19 min (50 samples × 22.8s)
- Inner Product: ~19 min (50 samples × 22.8s)

**Total V1: ~50 minutes**

### V2 Operations
- Key Generation: ~10s (100 samples × 13ms)
- Encryption: ~10s (100 samples × 2.3ms)
- Decryption: ~10s (100 samples × 1.1ms)
- Multiplication: ~10s (100 samples × 34ms)
- Reverse: ~1 min (50 samples × 0.8ms)
- Geometric Product: ~3 min (50 samples × 2.1s)
- Wedge Product: ~5 min (50 samples × 4.2s)
- Inner Product: ~5 min (50 samples × 4.1s)

**Total V2: ~12 minutes**

## Tips

1. **Start with V2 only** - It's much faster and shows the optimized performance
2. **Use specific operation benchmarks** - Don't run everything if you only need one metric
3. **Reduce sample size for testing** - Use `--sample-size 10` for quick validation
4. **Run overnight for V1** - V1 geometric operations take ~55 minutes
5. **Check Criterion reports** - Results are saved in `target/criterion/` with HTML reports

## Example Workflow

```bash
# 1. Quick check: Run V2 core ops (2 minutes)
cargo bench --bench v2_core_ops --features v2

# 2. Full V2 benchmarks (12 minutes)
cargo bench --features v2

# 3. Compare: Run V1 core ops (2 minutes)
cargo bench --bench v1_core_ops --features v1

# 4. Full comparison: Run V1 geometric product only (12 minutes)
cargo bench --bench v1_geometric_ops --features v1 geometric_product

# 5. Get full V1 data overnight (~55 minutes)
cargo bench --bench v1_geometric_ops --features v1
```

## Output Location

Benchmark results are saved in:
- **HTML reports**: `target/criterion/*/report/index.html`
- **Raw data**: `target/criterion/*/base/`
- **Comparisons**: Shown in terminal and saved to `target/criterion/*/change/`
