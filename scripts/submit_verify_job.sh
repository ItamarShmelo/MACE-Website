#!/bin/bash
#SBATCH --job-name=mace_verify
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out

# Usage: sbatch scripts/submit_verify_job.sh SCRIPT [ARGS...]
# Runs scripts/SCRIPT with the MACE venv python, e.g.
#   sbatch scripts/submit_verify_job.sh verify_derivative_fd.py --weight wien --temperature-index 10

SCRIPT="${1:?Usage: sbatch scripts/submit_verify_job.sh SCRIPT [ARGS...]}"
shift

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT/logs"

"$REPO_ROOT/external/ComptonMatrixExact/.venv/bin/python3" "$REPO_ROOT/scripts/$SCRIPT" "$@"
