#!/usr/bin/env bash
# gpu_wait.sh — GPU 等待脚本（双条件判断）
# 用法: source gpu_wait.sh && wait_for_n_gpus 2

# 不要 set -e，让等待循环中的子命令失败不杀脚本

MEM_THRESH=40000   # MiB，根据机器实际空卡内存调整
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

# 启动实验（CUDA_VISIBLE_DEVICES 映射到 cuda:0）
launch_exp() {
    local gpu_id=$1
    local script=$2
    shift 2
    local extra_args="$@"

    echo "[$(date '+%H:%M:%S')] Launching $script on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python $script $extra_args \
        > logs/$(basename $script .py)_gpu${gpu_id}.log 2>&1 &
    local pid=$!
    echo "[$(date '+%H:%M:%S')] Launched PID $pid"
}

echo "[gpu_wait] Loaded. Use: wait_for_n_gpus N"
