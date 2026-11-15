#!/bin/bash
set -euo pipefail

# Pilot run before the full submission: production sigma and dsigma/dT jobs
# at a few temperatures for every weight function (outputs are reused by
# submit_all.sh), plus the finite-difference derivative check and the direct
# MACE collapse check, each starting once its tables exist.
#
#   scripts/submit_pilot.sh                       # T010, T032, T054
#   TEMPS="0" COLLAPSE_TEMPS="0" scripts/submit_pilot.sh   # quick smoke test
#   CONV_TEMPS="" scripts/submit_pilot.sh         # without convergence runs
#
# Tables use the production quadrature orders (compute_matrix.py). The IDs of
# the finite-difference jobs, which exit non-zero when the check fails, are
# written to logs/pilot_fd_jobs.txt for use as a gate (see submit_all.sh).

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"
mkdir -p logs

TEMPS="${TEMPS:-10 32 54}"
COLLAPSE_TEMPS="${COLLAPSE_TEMPS:-10 32}"
SIGMA_TIME="${SIGMA_TIME:-3-00:00:00}"
DERIV_TIME="${DERIV_TIME:-7-00:00:00}"
: > logs/pilot_fd_jobs.txt

for weight in uniform wien planck; do
    for t in $TEMPS; do
        tag=$(printf "%s_T%03d" "$weight" "$t")
        jid_s=$(sbatch --parsable --output=logs/slurm_%x_%j.out --time="$SIGMA_TIME" --job-name="pilot_${tag}_sigma" \
            scripts/submit_job.sh "$weight" sigma "$t")
        jid_d=$(sbatch --parsable --output=logs/slurm_%x_%j.out --time="$DERIV_TIME" --job-name="pilot_${tag}_dsdT" \
            scripts/submit_job.sh "$weight" dsigma_dT "$t")
        jid_fd=$(sbatch --parsable --dependency="afterok:${jid_s}:${jid_d}" \
            --job-name="pilot_${tag}_fd" \
            scripts/submit_verify_job.sh verify_derivative_fd.py --weight "$weight" --temperature-index "$t")
        echo "$jid_fd" >> logs/pilot_fd_jobs.txt
        line="$tag: sigma=$jid_s dsdT=$jid_d fd=$jid_fd"
        if [[ " $COLLAPSE_TEMPS " == *" $t "* ]]; then
            jid_c=$(sbatch --parsable --dependency="afterok:${jid_s}:${jid_d}" \
                --job-name="pilot_${tag}_collapse" \
                scripts/submit_verify_job.sh verify_collapse_direct.py --weight "$weight" --temperature-index "$t")
            line="$line collapse=$jid_c"
        fi
        echo "$line"
    done
done

# Quadrature convergence of the production tables: uniform sigma with a
# higher E-panel Gauss-Legendre order, compared against the production run.
CONV_ORDER="${CONV_ORDER:-64}"
for t in ${CONV_TEMPS-$TEMPS}; do
    jid=$(sbatch --parsable --output=logs/slurm_%x_%j.out --time="$SIGMA_TIME" \
        --job-name="$(printf "conv_uniform_T%03d_ep%s" "$t" "$CONV_ORDER")" \
        scripts/submit_verify_job.sh compute_matrix.py --weight uniform --quantity sigma \
        --temperature-index "$t" --e-panel-order "$CONV_ORDER" --skip-existing \
        --out-root "$PWD/output/convergence/ep$CONV_ORDER")
    echo "convergence T$t ep$CONV_ORDER: $jid"
done
