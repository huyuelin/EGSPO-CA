#!/usr/bin/env bash
# Launch GRPO + EGSPO-CA on GSM8K across all 8 GPUs
# Seeds: 42, 123, 1337 (reproducible via random seeding)
# K=6, max_new_tokens=512, 2000 steps

set -euo pipefail

CONDA_BASE="/data/jackey_workspace/miniconda3"
CODE_DIR="/data/jackey_workspace/egspo_ca/code"
RESULTS="$CODE_DIR/results/phase4d_gsm8k"
MODEL="Qwen/Qwen2.5-7B-Instruct"
STEPS=2000

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate egspo
export TMPDIR=/dev/shm TMP=/dev/shm TEMP=/dev/shm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE_DIR"

mkdir -p "$RESULTS"

echo "========================================="
echo "Launching GSM8K training on ALL 8 GPUs"
echo "K=6, Steps=$STEPS, Model=Qwen2.5-7B"
echo "========================================="

# GPU 0: GRPO seed 42
CUDA_VISIBLE_DEVICES=0 nohup python3 -u scripts/run_minimal_train.py \
    --method grpo --model "$MODEL" --data data/gsm8k.jsonl \
    --output "$RESULTS/grpo_s42" --K 6 --max_new_tokens 512 --steps "$STEPS" \
    > "$RESULTS/grpo_s42.log" 2>&1 &
echo "GPU 0: GRPO seed=42 PID=$!"

# GPU 1: EGSPO-CA seed 42
CUDA_VISIBLE_DEVICES=1 nohup python3 -u scripts/run_minimal_train.py \
    --method egspo_ca --model "$MODEL" --data data/gsm8k.jsonl \
    --output "$RESULTS/egspo_ca_s42" --K 6 --max_new_tokens 512 --steps "$STEPS" \
    > "$RESULTS/egspo_ca_s42.log" 2>&1 &
echo "GPU 1: EGSPO-CA seed=42 PID=$!"

# GPU 2: GRPO seed 123
CUDA_VISIBLE_DEVICES=2 nohup python3 -u scripts/run_minimal_train.py \
    --method grpo --model "$MODEL" --data data/gsm8k.jsonl \
    --output "$RESULTS/grpo_s123" --K 6 --max_new_tokens 512 --steps "$STEPS" \
    > "$RESULTS/grpo_s123.log" 2>&1 &
echo "GPU 2: GRPO seed=123 PID=$!"

# GPU 3: EGSPO-CA seed 123
CUDA_VISIBLE_DEVICES=3 nohup python3 -u scripts/run_minimal_train.py \
    --method egspo_ca --model "$MODEL" --data data/gsm8k.jsonl \
    --output "$RESULTS/egspo_ca_s123" --K 6 --max_new_tokens 512 --steps "$STEPS" \
    > "$RESULTS/egspo_ca_s123.log" 2>&1 &
echo "GPU 3: EGSPO-CA seed=123 PID=$!"

# GPU 4: GRPO seed 1337
CUDA_VISIBLE_DEVICES=4 nohup python3 -u scripts/run_minimal_train.py \
    --method grpo --model "$MODEL" --data data/gsm8k.jsonl \
    --output "$RESULTS/grpo_s1337" --K 6 --max_new_tokens 512 --steps "$STEPS" \
    > "$RESULTS/grpo_s1337.log" 2>&1 &
echo "GPU 4: GRPO seed=1337 PID=$!"

# GPU 5: EGSPO-CA seed 1337
CUDA_VISIBLE_DEVICES=5 nohup python3 -u scripts/run_minimal_train.py \
    --method egspo_ca --model "$MODEL" --data data/gsm8k.jsonl \
    --output "$RESULTS/egspo_ca_s1337" --K 6 --max_new_tokens 512 --steps "$STEPS" \
    > "$RESULTS/egspo_ca_s1337.log" 2>&1 &
echo "GPU 5: EGSPO-CA seed=1337 PID=$!"

# GPU 6: GRPO ablation - no credit (baseline comparison)
CUDA_VISIBLE_DEVICES=6 nohup python3 -u scripts/run_minimal_train.py \
    --method grpo --model "$MODEL" --data data/gsm8k.jsonl \
    --output "$RESULTS/grpo_baseline" --K 6 --max_new_tokens 512 --steps "$STEPS" \
    > "$RESULTS/grpo_baseline.log" 2>&1 &
echo "GPU 6: GRPO baseline PID=$!"

# GPU 7: EGSPO-CA ablation - longer training (3000 steps)
CUDA_VISIBLE_DEVICES=7 nohup python3 -u scripts/run_minimal_train.py \
    --method egspo_ca --model "$MODEL" --data data/gsm8k.jsonl \
    --output "$RESULTS/egspo_ca_long" --K 6 --max_new_tokens 512 --steps 3000 \
    > "$RESULTS/egspo_ca_long.log" 2>&1 &
echo "GPU 7: EGSPO-CA 3000 steps PID=$!"

echo "========================================="
echo "ALL 8 GPUs launched!"
echo "Monitor: tail -f $RESULTS/*.log"
echo "========================================="
