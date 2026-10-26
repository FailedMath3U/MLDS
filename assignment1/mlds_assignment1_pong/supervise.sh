#!/usr/bin/env bash
# Invoked by nohup/setsid. Only one physical GPU is visible to the Python process.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=GPU-cef65388-88b0-f39a-6bac-c5316eb78540
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PONG_PYTHON="$HOME/mlds_assignment1_pong_env/bin/python"
PONG_CHILD_PID=''
stop_child() {
    if [[ -n "$PONG_CHILD_PID" ]] && kill -0 "$PONG_CHILD_PID" 2>/dev/null; then
        kill -TERM "$PONG_CHILD_PID"
        wait "$PONG_CHILD_PID" || true
    fi
    exit 0
}
trap stop_child TERM INT
for PONG_ATTEMPT in 1 2 3; do
    echo "$(date -Is) Starting/resuming Pong, attempt $PONG_ATTEMPT (one GPU)"
    "$PONG_PYTHON" -u run_pong.py train &
    PONG_CHILD_PID=$!
    echo "$PONG_CHILD_PID" > trainer.pid
    wait "$PONG_CHILD_PID"
    PONG_EXIT=$?
    PONG_CHILD_PID=''
    if [[ "$PONG_EXIT" == 0 ]]; then
        echo "$(date -Is) Pong process exited normally (complete or deliberately paused)."
        "$PONG_PYTHON" plot_results.py || echo "Plot generation failed; saved numeric results are intact."
        exit 0
    fi
    echo "$(date -Is) Pong exited $PONG_EXIT; failure.json contains the traceback."
    if [[ "$PONG_ATTEMPT" -lt 3 ]]; then
        echo 'Will retry from the latest atomic checkpoint in 30 seconds.'
        sleep 30
    fi
done
echo "$(date -Is) Stopped after three failures. Checkpoints and previous checkpoint retained."
exit 1
