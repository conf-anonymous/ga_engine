#!/bin/bash
# Enable real parallel_lift GPU acceleration
#
# This script checks for the parallel_lift repository and updates Cargo.toml
# to use the real implementations instead of the stub crates.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PARALLEL_LIFT_PATH="../parallel_lift"

cd "$PROJECT_ROOT"

echo "Checking for parallel_lift repository..."

if [ ! -d "$PARALLEL_LIFT_PATH/rust/crates/parallel_lift_core" ]; then
    echo "ERROR: parallel_lift repository not found at $PARALLEL_LIFT_PATH"
    echo ""
    echo "Please clone the parallel_lift repository:"
    echo "  cd $(dirname "$PROJECT_ROOT")"
    echo "  git clone <parallel_lift_url> parallel_lift"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "Found parallel_lift at $PARALLEL_LIFT_PATH"

# Check if already configured
if grep -q 'path = "../parallel_lift' Cargo.toml; then
    echo "Cargo.toml already configured for real parallel_lift"
    exit 0
fi

echo "Updating Cargo.toml to use real parallel_lift..."

# Backup Cargo.toml
cp Cargo.toml Cargo.toml.backup

# Replace stub paths with real paths
sed -i 's|path = "crates/parallel_lift_stubs/parallel_lift_core"|path = "../parallel_lift/rust/crates/parallel_lift_core"|g' Cargo.toml
sed -i 's|path = "crates/parallel_lift_stubs/parallel_lift_cuda"|path = "../parallel_lift/rust/crates/parallel_lift_cuda"|g' Cargo.toml

echo "Done! Cargo.toml updated."
echo ""
echo "Backup saved to Cargo.toml.backup"
echo ""
echo "You can now build with parallel_lift:"
echo "  cargo build --features v6-cuda"
echo ""
echo "To revert to stub crates, run:"
echo "  ./scripts/disable_parallel_lift.sh"
