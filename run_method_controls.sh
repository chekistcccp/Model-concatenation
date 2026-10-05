#!/usr/bin/env bash
# Manual git pull first; activate the environment of the successful paired audit.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$PWD"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-${GPU_ID:-auto}}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p results/transfer
RUN_DIR="$(mktemp -d "$REPO_ROOT/results/transfer/method_controls_run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
LOG_FILE="$RUN_DIR/launcher.log"
ARCHIVE="${RUN_DIR}.tar.gz"
finish() {
    local status=$?
    trap - EXIT
    set +e
    printf '\nLauncher exit code: %s\n' "$status" >> "$LOG_FILE"
    local tar_args=(-C "$REPO_ROOT/results/transfer" "$(basename "$RUN_DIR")")
    if [[ -d "$REPO_ROOT/results/followup/method_controls" ]]; then
        tar_args+=(-C "$REPO_ROOT/results/followup" method_controls)
    fi
    # Keep all newly trained weights and self-contained bundles on the server.
    tar --exclude='*.pt' --exclude='*.tmp' -czf "${ARCHIVE}.tmp" "${tar_args[@]}"
    local packed=$?
    if [[ "$packed" == 0 ]]; then mv -- "${ARCHIVE}.tmp" "$ARCHIVE"; packed=$?; fi
    if [[ "$packed" == 0 ]]; then
        printf '\n[TRANSFER] %s\n[LOG] %s\n' "$ARCHIVE" "$LOG_FILE"
        printf '[RETURN] Download this method_controls_run_*.tar.gz, including method_controls/run_manifest.json.\n'
    else
        printf '\n[ERROR] Packaging failed; retain %s and results/followup/method_controls/\n' "$RUN_DIR" >&2
        if [[ "$status" == 0 ]]; then status=1; fi
    fi
    if [[ "$status" != 0 ]]; then
        printf '[FAILED] Return the evidence; preserve existing outputs and original experiments.\n' >&2
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_step() { "$@" 2>&1 | tee -a "$LOG_FILE"; }
{
    printf 'Repository: %s\nPython: %s\nGPUs: %s\n' "$REPO_ROOT" "$PYTHON_BIN" "$GPUS"
    printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'Git HEAD: '; git rev-parse HEAD
    printf 'Method controls; normal source training; original AOSS/score/grid preserved.\n'
} | tee -a "$LOG_FILE"
if [[ "$GPUS" != auto && ! "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf '[ERROR] Activate the successful paired-response environment; GPUS must be auto or indices such as 0,1,2,3.\n' | tee -a "$LOG_FILE"
    exit 2
fi
if [[ -d results/followup/method_controls ]] && [[ -n "$(ls -A results/followup/method_controls)" ]]; then
    printf '[CHECK] Existing method outputs: validate before repackaging; no experiment workers will run.\n' | tee -a "$LOG_FILE"
    if run_step "$PYTHON_BIN" -m src.analyze_method_controls; then
        cp -a -- "$REPO_ROOT/results/analysis/method_controls" "$RUN_DIR/analysis"
        printf '[SUCCESS] Existing complete method evidence verified; download the new TRANSFER archive.\n' | tee -a "$LOG_FILE"
        exit 0
    else
        printf '[ERROR] Existing method evidence failed validation; package unchanged; refuse overwrite/resume.\n' | tee -a "$LOG_FILE"
        exit 2
    fi
fi
run_step "$PYTHON_BIN" --version
run_step "$PYTHON_BIN" -m src.method_controls --gpus "$GPUS" --check-only
run_step "$PYTHON_BIN" -m src.method_controls --gpus "$GPUS"
run_step "$PYTHON_BIN" -m src.analyze_method_controls
cp -a -- "$REPO_ROOT/results/analysis/method_controls" "$RUN_DIR/analysis"
printf '[SUCCESS] Fixed method controls complete; download the TRANSFER archive.\n' | tee -a "$LOG_FILE"
