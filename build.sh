#!/bin/bash
# Build the container on DAIC (run on login node, not in SLURM job)
# Everything stays within the project directory on bulk storage
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use project directory for cache to avoid home quota issues
export APPTAINER_CACHEDIR="${SCRIPT_DIR}/.apptainer_cache"
mkdir -p "$APPTAINER_CACHEDIR"

echo "Building container in: $SCRIPT_DIR"
echo "Cache dir: $APPTAINER_CACHEDIR"

apptainer build jigno.sif jigno.def

echo ""
echo "Done: ${SCRIPT_DIR}/jigno.sif"
echo "Add to .gitignore: jigno.sif, .apptainer_cache/"