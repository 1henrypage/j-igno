#!/bin/bash
# Build the container on DAIC (run on login node, not in SLURM job)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use /tmp for cache - bulk storage doesn't support the chmod operations apptainer needs
export APPTAINER_CACHEDIR="/tmp/apptainer_cache_${USER}"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp_${USER}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "Building container..."
echo "Cache dir: $APPTAINER_CACHEDIR"
echo "Output: ${SCRIPT_DIR}/jigno.sif"

apptainer build "${SCRIPT_DIR}/jigno.sif" "${SCRIPT_DIR}/jigno.def"

# Clean up cache
rm -rf "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo ""
echo "Done: ${SCRIPT_DIR}/jigno.sif"
