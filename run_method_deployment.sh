#!/usr/bin/env bash
# Manual git pull first. Keep the successful method-control environment/weights.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$PWD"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-${GPU_ID:-auto}}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false USE_TF=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p results/transfer
RUN_DIR="$(mktemp -d "$REPO_ROOT/results/transfer/method_deployment_run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
LOG_FILE="$RUN_DIR/launcher.log"
ARCHIVE="${RUN_DIR}.tar.gz"
finish() {
    local status=$?
    trap - EXIT
    set +e
    printf '\nLauncher exit code: %s\n' "$status" >> "$LOG_FILE"
    local tar_args=(-C "$REPO_ROOT/results/transfer" "$(basename "$RUN_DIR")")
    if [[ -d "$REPO_ROOT/results/followup/method_deployment" ]]; then
        tar_args+=(-C "$REPO_ROOT/results/followup" method_deployment)
    fi
    tar --exclude='*.pt' --exclude='*.tmp' -czf "${ARCHIVE}.tmp" "${tar_args[@]}"
    local packed=$?
    if [[ "$packed" == 0 ]]; then mv -- "${ARCHIVE}.tmp" "$ARCHIVE"; packed=$?; fi
    if [[ "$packed" == 0 ]]; then
        printf '\n[TRANSFER] %s\n[LOG] %s\n' "$ARCHIVE" "$LOG_FILE"
        printf '[RETURN] Download this archive; raw data/cache/frozen weights remain on the server.\n'
    else
        printf '[ERROR] Packaging failed; preserve %s and results/followup/method_deployment/\n' "$RUN_DIR" >&2
        if [[ "$status" == 0 ]]; then status=1; fi
    fi
    if [[ "$status" != 0 ]]; then printf '[FAILED] Preserve and return evidence; no overwrite/resume.\n' >&2; fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_step() { "$@" 2>&1 | tee -a "$LOG_FILE"; }
{
    printf 'Repository: %s\nPython: %s\nGPUs: %s\n' "$REPO_ROOT" "$PYTHON_BIN" "$GPUS"
    printf 'Git HEAD: '; git rev-parse HEAD
    printf 'All target RGB images; four frozen arms; original AOSS/score/grid unchanged.\n'
} | tee -a "$LOG_FILE"
if [[ "$GPUS" != auto && ! "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf '[ERROR] Activate the successful method-control environment and choose visible CUDA indices.\n' | tee -a "$LOG_FILE"
    exit 2
fi
if [[ -d results/followup/method_deployment ]] && [[ -n "$(ls -A results/followup/method_deployment)" ]]; then
    if run_step "$PYTHON_BIN" -m src.analyze_method_deployment; then
        cp -a -- results/analysis/method_deployment "$RUN_DIR/analysis"
        printf '[SUCCESS] Existing RGB evaluation verified; no GPU workers ran.\n' | tee -a "$LOG_FILE"
        exit 0
    fi
    printf '[ERROR] Existing RGB evidence is incomplete/invalid; refuse overwrite/resume.\n' | tee -a "$LOG_FILE"
    exit 2
fi
run_step "$PYTHON_BIN" -m src.method_deployment --gpus "$GPUS"
run_step "$PYTHON_BIN" -m src.analyze_method_deployment
cp -a -- results/analysis/method_deployment "$RUN_DIR/analysis"
printf '[SUCCESS] Full-target RGB evaluation and complete-detector cost receipts ready.\n' | tee -a "$LOG_FILE"
