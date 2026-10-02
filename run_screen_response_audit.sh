#!/usr/bin/env bash
# Manual git pull first; use the environment of the completed paired-response audit.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$PWD"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p results/transfer
RUN_DIR="$(mktemp -d "$REPO_ROOT/results/transfer/screen_response_run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
LOG_FILE="$RUN_DIR/launcher.log"
ARCHIVE="${RUN_DIR}.tar.gz"
finish() {
    local status=$?
    trap - EXIT
    set +e
    printf '\nLauncher exit code: %s\n' "$status" >> "$LOG_FILE"
    local tar_args=(-C "$REPO_ROOT/results/transfer" "$(basename "$RUN_DIR")")
    if [[ -d "$REPO_ROOT/results/followup/screen_response_audit" ]]; then
        tar_args+=(-C "$REPO_ROOT/results/followup" screen_response_audit)
    fi
    tar -czf "${ARCHIVE}.tmp" "${tar_args[@]}"
    local pack_status=$?
    if [[ "$pack_status" == 0 ]]; then mv -- "${ARCHIVE}.tmp" "$ARCHIVE"; pack_status=$?; fi
    if [[ "$pack_status" == 0 ]]; then
        printf '\n[TRANSFER] %s\n[LOG] %s\n' "$ARCHIVE" "$LOG_FILE"
    else
        printf '\n[ERROR] Packaging failed; keep %s and results/followup/screen_response_audit/\n' "$RUN_DIR" >&2
        if [[ "$status" == 0 ]]; then status=1; fi
    fi
    if [[ "$status" != 0 ]]; then
        printf '[FAILED] Return the evidence; do not delete outputs, retrain, or relax replay tolerance.\n' >&2
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_step() { "$@" 2>&1 | tee -a "$LOG_FILE"; }
{
    printf 'Repository: %s\nPython: %s\nGPU: %s\n' "$REPO_ROOT" "$PYTHON_BIN" "$GPU_ID"
    printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'Git HEAD: '; git rev-parse HEAD
    printf 'Original screening checkpoints only; no training/reselection/cache rebuild.\n'
} | tee -a "$LOG_FILE"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    printf '[ERROR] GPU_ID must be a nonnegative visible GPU index.\n' | tee -a "$LOG_FILE"; exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf '[ERROR] Activate the paired-response environment; missing Python.\n' | tee -a "$LOG_FILE"; exit 2
fi
if [[ -f results/followup/screen_response_audit/run_manifest.json ]]; then
    printf '[ERROR] Existing audit will be packaged unchanged; no overwrite/resume.\n' | tee -a "$LOG_FILE"; exit 2
fi
run_step "$PYTHON_BIN" --version
run_step "$PYTHON_BIN" -m src.screen_response_audit --check-only
run_step "$PYTHON_BIN" -m src.screen_response_audit --gpu "$GPU_ID"
run_step "$PYTHON_BIN" -c '
import hashlib, json
from pathlib import Path
p = Path("results/followup/screen_response_audit")
m = json.loads((p / "run_manifest.json").read_text())
assert m["status"] == "complete" and m["original_inputs_unchanged"]
assert m["no_training"] and m["no_reselection"] and not m["target_test_read"] and not m["target_valid_read"]
assert len(m["jobs"]) == 36 and m["n_source_rows"] == 221184
assert len(m["output_sha256"]) == 219
for name, digest in m["output_sha256"].items():
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == digest, name
print("[SUCCESS] Original screen response replay complete; hashes verified.")
'
run_step "$PYTHON_BIN" -m src.analyze_screen_response
cp -a -- "$REPO_ROOT/results/analysis/screen_response_audit" "$RUN_DIR/analysis"
