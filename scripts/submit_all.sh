#!/bin/bash
set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"
mkdir -p logs

# Time limits per quantity. dsigma/dT has no cutoff and needs three passes
# for Planck/Wien, so it gets a longer limit.
SIGMA_TIME="${SIGMA_TIME:-7-00:00:00}"
DERIV_TIME="${DERIV_TIME:-7-00:00:00}"

# Optional gate, e.g. AFTER_OK="$(paste -sd: logs/pilot_fd_jobs.txt)": the
# arrays start only if all those jobs succeed.
# ARRAY selects array task ids (task t computes T index 63 - t), e.g. to leave
# out temperatures that are already being computed elsewhere.
ARRAY="${ARRAY:-0-63}"

DEP=()
if [ -n "${AFTER_OK:-}" ]; then
    DEP=(--dependency="afterok:${AFTER_OK}" --kill-on-invalid-dep=yes)
fi

echo "Submitting 6 x 64 MACE table jobs (existing outputs are skipped) ..."

for weight in uniform wien planck; do
    sbatch "${DEP[@]}" --array="$ARRAY" --time="$SIGMA_TIME" --job-name="mace_${weight}_sigma" \
        scripts/submit_job.sh "$weight" sigma
    sbatch "${DEP[@]}" --array="$ARRAY" --time="$DERIV_TIME" --job-name="mace_${weight}_dsdT" \
        scripts/submit_job.sh "$weight" dsigma_dT
done

echo "All arrays submitted."
