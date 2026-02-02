#!/bin/bash
# Disable parallel_lift and revert to stub crates
#
# This script updates Cargo.toml to use the stub crates instead of real parallel_lift.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Reverting Cargo.toml to use stub crates..."

# Check if already using stubs
if grep -q 'path = "crates/parallel_lift_stubs' Cargo.toml; then
    echo "Cargo.toml already configured for stub crates"
    exit 0
fi

# Replace real paths with stub paths
sed -i 's|path = "../parallel_lift/rust/crates/parallel_lift_core"|path = "crates/parallel_lift_stubs/parallel_lift_core"|g' Cargo.toml
sed -i 's|path = "../parallel_lift/rust/crates/parallel_lift_cuda"|path = "crates/parallel_lift_stubs/parallel_lift_cuda"|g' Cargo.toml

echo "Done! Cargo.toml reverted to stub crates."
echo ""
echo "The project will now compile without the parallel_lift repository."
