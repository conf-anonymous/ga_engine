# Makefile

.PHONY: coverage bench-rotate bench-classical bench-matrix_naive bench-matrixmultiply bench-ndarray bench-all

# Coverage
coverage:
	@echo "⏳ Running coverage…"
	cargo llvm-cov \
	  --json \
	  --summary-only \
	  --output-path cov.json
	cargo run --bin coverage_summary

# 1) Rotate benchmarks (from benches/bench.rs)
bench-rotate:
	@echo "🔬 Running rotate benchmarks"
	cargo build --release --bench bench
	./target/release/deps/bench-*

# 2) All‐in‐one classical + BLAS + nalgebra (from benches/classical.rs)
bench-classical:
	@echo "🔬 Running classical/matrixmultiply/nalgebra benchmarks"
	cargo bench --bench classical

# 3) Stand-alone naive vs nalgebra (from benches/matrix_naive.rs)
bench-matrix_naive:
	@echo "🔬 Running naive vs nalgebra DMatrix benchmarks"
	cargo build --release --bench matrix_naive
	./target/release/deps/matrix_naive-*

# 4) matrixmultiply::dgemm @ 128×128 (from benches/matrix_matrixmultiply.rs)
bench-matrixmultiply:
	@echo "🔬 Running matrixmultiply dgemm (128×128) benchmarks"
	cargo bench --bench matrix_matrixmultiply

# 5) ndarray + BLAS (from benches/matrix_ndarray.rs)
bench-ndarray:
	@echo "🔬 Running ndarray + BLAS (128×128) benchmarks"
	cargo bench --bench matrix_ndarray

# Run them all in sequence
bench-all: bench-rotate bench-classical bench-matrix_naive bench-matrixmultiply bench-ndarray
	@echo "✅ All benchmarks complete"