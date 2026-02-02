#!/bin/bash

# Quick Test Suite for GA Engine
# Tests core functionality before committing

set -e  # Exit on first error

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         GA Engine - Quick Test Suite                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Test 1: V2 Core Tests (fastest, most critical)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1/5: V2 Core Tests (127 tests, ~2 seconds)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cargo test --lib --features f64,nd,v2 --no-default-features --quiet
echo "✓ V2 tests passed"
echo ""

# Test 2: V3 Core Tests
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2/5: V3 Core Tests (52 tests, ~3 seconds)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cargo test --lib --features v2,v3 clifford_fhe_v3 --quiet
echo "✓ V3 tests passed"
echo ""

# Test 3: V5 Privacy Tests
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3/5: V5 Privacy Tests (~10 tests, ~2 seconds)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cargo test --lib --features f64,nd,v5 --no-default-features clifford_fhe_v5 --quiet
echo "✓ V5 tests passed"
echo ""

# Test 4: V1 Core Tests
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4/5: V1 Core Tests (31 tests, ~1 second)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cargo test --lib --features f64,nd,v1 --no-default-features --quiet
echo "✓ V1 tests passed"
echo ""

# Test 5: Build All Versions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5/5: Build All Versions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cargo build --release --features f64,nd,v1,v2,v3,v5 --no-default-features --quiet
echo "✓ Build successful"
echo ""

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              ✓ ALL QUICK TESTS PASSED                           ║"
echo "║              Total: 223+ tests in ~8 seconds                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  ./TEST_COMPREHENSIVE.sh"
