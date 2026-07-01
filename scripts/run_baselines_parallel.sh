#!/usr/bin/env bash
# Launch all 10 baseline validation experiments across GPUs 3-6 in parallel
# Each baseline: 100 steps on GSM8K, K=4 for quick validation
# Output: results/phase3d_validate/

set -euo pipefail

CONDA_BASE="/data/jackey_workspace/miniconda3"
CODE_DIR="/data/jackey_workspace/egspo_ca/code"
RESULTS_DIR="$CODE_DIR/results/phase3d_validate"
DATA="$CODE_DIR/data/gsm8k.jsonl"
MODEL="Qwen/Qwen2.5-7B-Instruct"

# Common args for quick validation
COMMON_ARGS="--model $MODEL --data $DATA --K 4 --max_new_tokens 256 --steps 100 --max_problems 500"

mkdir -p "$RESULTS_DIR/logs"

# Ensure data exists
if [ ! -f "$DATA" ]; then
    echo "ERROR: $DATA not found!"
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate egspo
export TMPDIR=/dev/shm TMP=/dev/shm TEMP=/dev/shm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE_DIR"

echo "============================================"
echo "Launching 10 baseline validation experiments"
echo "GPUs: 3,4,5,6 | Steps: 100 each | K=4"
echo "============================================"

# GPU 3: EGSPO, Dr.GRPO, GTPO (simple, fast)
CUDA_VISIBLE_DEVICES=3 nohup bash -c "
  python3 -u baselines/run_baseline.py --method egspo --output $RESULTS_DIR $COMMON_ARGS 2>&1
  python3 -u baselines/run_baseline.py --method drgrpo --output $RESULTS_DIR $COMMON_ARGS 2>&1
  python3 -u baselines/run_baseline.py --method gtpo --output $RESULTS_DIR $COMMON_ARGS 2>&1
  echo 'GPU 3: ALL DONE' 
" > "$RESULTS_DIR/logs/gpu3_egspo_drgrpo_gtpo.log" 2>&1 &
echo "  GPU 3: egspo → drgrpo → gtpo (PID $!)"

sleep 5

# GPU 4: DAPO, TEMPO, DelTA
CUDA_VISIBLE_DEVICES=4 nohup bash -c "
  python3 -u baselines/run_baseline.py --method dapo --output $RESULTS_DIR $COMMON_ARGS 2>&1
  python3 -u baselines/run_baseline.py --method tempo --output $RESULTS_DIR $COMMON_ARGS 2>&1
  python3 -u baselines/run_baseline.py --method delta --output $RESULTS_DIR $COMMON_ARGS 2>&1
  echo 'GPU 4: ALL DONE'
" > "$RESULTS_DIR/logs/gpu4_dapo_tempo_delta.log" 2>&1 &
echo "  GPU 4: dapo → tempo → delta (PID $!)"

sleep 5

# GPU 5: SPO, HAPO (complex, more memory)
CUDA_VISIBLE_DEVICES=5 nohup bash -c "
  python3 -u baselines/run_baseline.py --method spo --output $RESULTS_DIR $COMMON_ARGS 2>&1
  python3 -u baselines/run_baseline.py --method hapo --output $RESULTS_DIR $COMMON_ARGS 2>&1
  echo 'GPU 5: ALL DONE'
" > "$RESULTS_DIR/logs/gpu5_spo_hapo.log" 2>&1 &
echo "  GPU 5: spo → hapo (PID $!)"

sleep 5

# GPU 6: CAPO, CF Credit (complex, more memory)
CUDA_VISIBLE_DEVICES=6 nohup bash -c "
  python3 -u baselines/run_baseline.py --method capo --output $RESULTS_DIR $COMMON_ARGS 2>&1
  python3 -u baselines/run_baseline.py --method cfcredit --output $RESULTS_DIR $COMMON_ARGS 2>&1
  echo 'GPU 6: ALL DONE'
" > "$RESULTS_DIR/logs/gpu6_capo_cfcredit.log" 2>&1 &
echo "  GPU 6: capo → cfcredit (PID $!)"

echo "============================================"
echo "All baselines launched. Monitor with:"
echo "  tail -f $RESULTS_DIR/logs/gpu3_egspo_drgrpo_gtpo.log"
echo "  tail -f $RESULTS_DIR/logs/gpu4_dapo_tempo_delta.log"
echo "============================================"
