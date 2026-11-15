#!/bin/bash
#SBATCH --job-name=mace_epscan
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out

# Usage:
#   sbatch --array=0-40 scripts/submit_ep_scan.sh WEIGHT QUANTITY LABEL [compute_matrix.py flags...]
#
# Computes the full table at production orders plus the given flags, for
# temperature index = array task id, into output/ep_scan/LABEL/. Compare the
# variants against output/tables with scripts/scan_ep_window.py, e.g.
#   sbatch --array=0-40 scripts/submit_ep_scan.sh wien sigma k10ep48 \
#       --ep-k-cut 10 --ep-edge-order 48 --ep-interior-order 48

USAGE="Usage: sbatch --array=... scripts/submit_ep_scan.sh WEIGHT QUANTITY LABEL [FLAGS...]"
WEIGHT="${1:?$USAGE}"
QUANTITY="${2:?$USAGE}"
LABEL="${3:?$USAGE}"
shift 3
TIDX="${SLURM_ARRAY_TASK_ID:?Submit as an array job}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT/logs"

"$REPO_ROOT/external/ComptonMatrixExact/.venv/bin/python3" "$REPO_ROOT/scripts/compute_matrix.py" \
    --weight "$WEIGHT" \
    --quantity "$QUANTITY" \
    --temperature-index "$TIDX" \
    --out-root "$REPO_ROOT/output/ep_scan/$LABEL" \
    --skip-existing \
    "$@"
