#!/usr/bin/env bash
# Historical protocol only: the active model no longer implements Lag.
echo "Lag training has been retired. See docs/lag-ablation-results.md and its saved artifacts." >&2
exit 2

# Four matched training runs on one GPU, followed by community evaluation.
# Usage: bash scripts/run_lag_ablation.sh [output_dir] [seed] [device] [variant]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
output_root="${1:-results/lag_ablation_20260907}"
seed="${2:-42}"
device="${3:-cuda:0}"
variant_filter="${4:-all}"
if [[ "$variant_filter" != all && "$variant_filter" != with_lag && "$variant_filter" != no_lag ]]; then
    echo "variant must be all, with_lag, or no_lag" >&2
    exit 1
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_variant() {
    local dataset="$1" variant="$2"
    local slices start_t test_start_t
    if [[ "$dataset" == email ]]; then
        slices=data/email-Eu-core-temporal/time_slices/step_302400_window_604800
        start_t=105
        test_start_t=127
    else
        slices=data/mooc/time_slices/step_43200_window_86400
        start_t=52
        test_start_t=50
    fi
    local output_dir="$output_root/$dataset"
    local extra_args=(--core-lookback 5)
    if [[ "$variant" == no_lag ]]; then extra_args+=(--no-lag); fi
    mkdir -p "$output_dir"
    if [[ -e "$output_dir/$variant.pt" ]]; then
        echo "Refusing to overwrite checkpoint: $output_dir/$variant.pt" >&2
        return 1
    fi
    python -u train_coreness.py "$slices" \
        --device "$device" --seed "$seed" --epochs 30 --patience 5 \
        --output "$output_dir/$variant.pt" "${extra_args[@]}" \
        > "$output_dir/${variant}_train.log" 2>&1
    # Historical evaluation scope, kept for comparison with previous reports.
    python -u hybrid_community.py eval \
        --slices-dir "$slices" --checkpoint "$output_dir/$variant.pt" \
        --device "$device" --start-t "$start_t" \
        --output "$output_dir/${variant}_community.json" \
        > "$output_dir/${variant}_community.log" 2>&1
    # Strict holdout: t+1 belongs to the training pipeline's test split.
    python -u hybrid_community.py eval \
        --slices-dir "$slices" --checkpoint "$output_dir/$variant.pt" \
        --device "$device" --start-t "$test_start_t" \
        --output "$output_dir/${variant}_community_test.json" \
        > "$output_dir/${variant}_community_test.log" 2>&1
}

job_pids=()
for dataset in email mooc; do
    for variant in with_lag no_lag; do
        if [[ "$variant_filter" != all && "$variant_filter" != "$variant" ]]; then
            continue
        fi
        run_variant "$dataset" "$variant" &
        job_pids+=("$!")
    done
done
status=0
for job_pid in "${job_pids[@]}"; do
    wait "$job_pid" || status=1
done
exit "$status"
