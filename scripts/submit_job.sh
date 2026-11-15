#!/bin/bash
#SBATCH --job-name=mace_table
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out

# Usage:
#   sbatch --array=0-63 scripts/submit_job.sh WEIGHT QUANTITY
#   sbatch scripts/submit_job.sh WEIGHT QUANTITY TEMPERATURE_INDEX
#
# As an array job, task id t computes temperature index 63 - t so that the
# slow hot temperatures are scheduled first.

WEIGHT="${1:?Usage: sbatch scripts/submit_job.sh WEIGHT QUANTITY [TEMPERATURE_INDEX]}"
QUANTITY="${2:?Usage: sbatch scripts/submit_job.sh WEIGHT QUANTITY [TEMPERATURE_INDEX]}"
if [ -n "${3:-}" ]; then
    TIDX="$3"
else
    TIDX=$((63 - ${SLURM_ARRAY_TASK_ID:?Pass TEMPERATURE_INDEX or submit as an array}))
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT/logs"

"$REPO_ROOT/external/ComptonMatrixExact/.venv/bin/python3" "$REPO_ROOT/scripts/compute_matrix.py" \
    --weight "$WEIGHT" \
    --quantity "$QUANTITY" \
    --temperature-index "$TIDX" \
    --skip-existing
