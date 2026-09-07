#!/usr/bin/env bash
# Usage: bash scripts/run_history_encoder_ablation.sh [output_dir] [seed] [device]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
output_root="${1:-results/history_encoder_ablation_20260907}"
seed="${2:-42}"
device="${3:-cuda:0}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_variant() {
    local dataset="$1" encoder="$2" slices test_start_t
    if [[ "$dataset" == email ]]; then
        slices=data/email-Eu-core-temporal/time_slices/step_302400_window_604800
        test_start_t=127
    else
        slices=data/mooc/time_slices/step_43200_window_86400
        test_start_t=50
    fi
    local output_dir="$output_root/$dataset"
    mkdir -p "$output_dir"
    if [[ -e "$output_dir/$encoder.pt" ]]; then
        echo "Refusing to overwrite checkpoint: $output_dir/$encoder.pt" >&2
        return 1
    fi
    python -u train_coreness.py "$slices" \
        --history-encoder "$encoder" --core-lookback 5 \
        --output-head tied \
        --history-transformer-layers 2 --history-attention-heads 4 \
        --device "$device" --seed "$seed" --epochs 30 --patience 5 \
        --output "$output_dir/$encoder.pt" \
        > "$output_dir/${encoder}_train.log" 2>&1
    python -u hybrid_community.py eval \
        --slices-dir "$slices" --checkpoint "$output_dir/$encoder.pt" \
        --device "$device" --start-t "$test_start_t" \
        --output "$output_dir/${encoder}_community_test.json" \
        > "$output_dir/${encoder}_community_test.log" 2>&1
}

job_pids=()
for dataset in email mooc; do
    for encoder in gru transformer; do
        run_variant "$dataset" "$encoder" &
        job_pids+=("$!")
    done
done
status=0
for job_pid in "${job_pids[@]}"; do
    wait "$job_pid" || status=1
done
exit "$status"
