#!/usr/bin/env bash
# Train the email Zebra baseline, then evaluate three candidate windows.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
zebra_root="$(dirname "$project_root")/Zebra"
run_dir="$project_root/results/zebra_rho_20260908/email"
checkpoint="$zebra_root/saved_checkpoints/email-snapshot-50-0.0001-streaming-[0.1, 0.1]-[0.5, 0.95]-20.pth"
mkdir -p "$run_dir"
if [[ -e "$checkpoint" ]]; then
  echo "Refusing to overwrite existing checkpoint: $checkpoint" >&2
  exit 1
fi
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd "$zebra_root"
python -u train.py \
  --data email-snapshot --n_epoch 50 --bs 200 --lr 0.0001 \
  --patience 5 --n_runs 1 --n_degree 10 --n_layer 2 --n_head 2 \
  --node_dim 100 --time_dim 100 --memory_dim 100 --drop_out 0.3 \
  --message_function identity --memory_updater gru --aggregator last \
  --tppr_strategy streaming --topk 20 --alpha_list 0.1 0.1 \
  --beta_list 0.5 0.95 --gpu 0 --save_best > "$run_dir/train.log" 2>&1
cd "$project_root"
for rho in 0.1 0.2 0.3; do
  python -u zebra_community_rho_backup.py eval \
    --slices-dir "$project_root/data/email-Eu-core-temporal/time_slices/step_302400_window_604800" \
    --zebra-root "$zebra_root" --zebra-dataset email-snapshot \
    --checkpoint "$checkpoint" --device cuda:0 \
    --candidate-window-rho "$rho" --cache-dir "$run_dir/cache_$rho" \
    --output "$run_dir/rho_$rho.json" > "$run_dir/rho_$rho.log" 2>&1
done
