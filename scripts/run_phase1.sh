#!/usr/bin/env bash
# run_phase1.sh — Phase 1 launch script for EGSPO-CA v2
# Usage: bash scripts/run_phase1.sh [grpo|egspo_ca]

set -u  # No `set -e` (waiting loop needs to survive sub-command failures)

# ── Config ──────────────────────────────────────────
MODEL="Qwen/Qwen2.5-7B-Instruct"
DATA_DIR="../data"
OUTPUT_DIR="../results/phase1"
SEEDS=(42 123)
# SEEDS=(42 123 1337 2024 31415)  # Full seeds (5)

# ── GPU Wait ────────────────────────────────────────
MEM_THRESH=40000   # MiB (A800 80GB: empty ~4 MiB, threshold 40GB)
UTIL_THRESH=20     # %

free_gpus() {
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v m="$MEM_THRESH" -v u="$UTIL_THRESH" \
        '$2 < m && $3 < u {print $1}'
}

wait_for_n_gpus() {
    local needed=$1
    echo "[$(date '+%H:%M:%S')] Waiting for $needed GPU(s)..."
    while true; do
        mapfile -t available < <(free_gpus)
        [[ ${#available[@]} -ge $needed ]] && return 0
        echo "[$(date '+%H:%M:%S')] ${#available[@]}/$needed free. Sleeping 120s..."
        sleep 120
    done
}

# ── Launch GRPO Baseline ───────────────────────────
launch_grpo() {
    local seed=$1
    echo "[$(date '+%H:%M:%S')] Launching GRPO (seed=$seed)..."

    wait_for_n_gpus 1
    GPU=$(free_gpus | head -1)

    CUDA_VISIBLE_DEVICES=$GPU python3 training/grpo_trainer.py \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR/grpo/seed_$seed" \
        --seed "$seed" \
        --data_dir "$DATA_DIR" \
        --K 8 \
        --learning_rate 5.0e-6 \
        --num_train_epochs 3 \
        > "../logs/grpo_seed${seed}_gpu${GPU}.log" 2>&1 &

    local pid=$!
    echo "[$(date '+%H:%M:%S')] GRPO launched (PID=$pid, GPU=$GPU)"
    echo "$pid" > "../logs/grpo_seed${seed}.pid"
}

# ── Launch EGSPO-CA v2 ────────────────────────────
launch_egspo_ca() {
    local seed=$1
    echo "[$(date '+%H:%M:%S')] Launching EGSPO-CA v2 (seed=$seed)..."

    wait_for_n_gpus 1
    GPU=$(free_gpus | head -1)

    CUDA_VISIBLE_DEVICES=$GPU python3 training/egspo_ca_trainer.py \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR/egspo_ca/seed_$seed" \
        --seed "$seed" \
        --data_dir "$DATA_DIR" \
        --K 8 \
        --delta 8 \
        --eta 0.5 \
        --beta 0.6 \
        --gamma 0.1 \
        --learning_rate 5.0e-6 \
        --num_train_epochs 3 \
        > "../logs/egspo_ca_seed${seed}_gpu${GPU}.log" 2>&1 &

    local pid=$!
    echo "[$(date '+%H:%M:%S')] EGSPO-CA v2 launched (PID=$pid, GPU=$GPU)"
    echo "$pid" > "../logs/egspo_ca_seed${seed}.pid"
}

# ── Main ────────────────────────────────────────────
main() {
    local mode="${1:-all}"

    # Create log directory
    mkdir -p "../logs"
    mkdir -p "$OUTPUT_DIR"

    echo "[$(date '+%H:%M:%S')] Starting Phase 1 (mode=$mode)..."

    case "$mode" in
        "grpo")
            for seed in "${SEEDS[@]}"; do
                launch_grpo "$seed"
                sleep 5  # Stagger launches
            done
            ;;
        "egspo_ca")
            for seed in "${SEEDS[@]}"; do
                launch_egspo_ca "$seed"
                sleep 5
            done
            ;;
        "all")
            for seed in "${SEEDS[@]}"; do
                launch_grpo "$seed"
                sleep 5
            done
            for seed in "${SEEDS[@]}"; do
                launch_egspo_ca "$seed"
                sleep 5
            done
            ;;
        *)
            echo "Unknown mode: $mode"
            echo "Usage: $0 [grpo|egspo_ca|all]"
            exit 1
            ;;
    esac

    echo "[$(date '+%H:%M:%S')] All jobs launched. Monitor with: tail -f ../logs/*.log"
}

main "$@"
