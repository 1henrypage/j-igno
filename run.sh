#!/bin/bash
#SBATCH --job-name=jigno
#SBATCH --partition=general,insy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=slurm_logs/job_%j.out
#SBATCH --error=slurm_logs/job_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=h.page@student.tudelft.nl
# =============================================================================
# Usage:
#   sbatch --qos=short --time=04:00:00 run.sh train configs/exp.yaml
#   sbatch --qos=medium --time=48:00:00 run.sh evaluate configs/eval.yaml
# =============================================================================

set -euo pipefail

MODE="${1:?Usage: run.sh <train|evaluate> <config> [args...]}"
CONFIG="${2:?Usage: run.sh <train|evaluate> <config> [args...]}"
shift 2

# Get project directory (where this script lives)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Project dir: $PROJECT_DIR"
echo "Mode: $MODE"
echo "Config: $CONFIG"
nvidia-smi

# Validate
[[ "$MODE" != "train" && "$MODE" != "evaluate" ]] && { echo "ERROR: mode must be train or evaluate"; exit 1; }
[[ ! -f "$CONFIG" ]] && { echo "ERROR: config not found: $CONFIG"; exit 1; }

CONTAINER="${PROJECT_DIR}/jigno.sif"
[[ ! -f "$CONTAINER" ]] && { echo "ERROR: container not found. Run ./build.sh first"; exit 1; }

case "$MODE" in
    train)    SCRIPT="training.py" ;;
    evaluate) SCRIPT="evaluate.py" ;;
esac

# -C isolates container filesystem
# --nv enables GPU
# --bind mounts project dir to /workspace
srun apptainer exec --nv -C \
    --bind "${PROJECT_DIR}:/workspace" \
    --pwd /workspace \
    "$CONTAINER" \
    uv run python "$SCRIPT" --config "$CONFIG" "$@"
