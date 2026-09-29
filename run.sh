#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

CONFIG="${CONFIG:-configs/experiment.yaml}"
STAGE="${1:-all}"

mkdir -p data model cache results logs

python - <<'PY'
import importlib.util, sys
required = ["torch", "torchvision", "transformers", "modelscope", "numpy", "sklearn", "yaml", "PIL"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print("[ERROR] Missing Python packages:", ", ".join(missing), file=sys.stderr)
    print("Install dependencies with: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(2)
PY

python -m src.pipeline --config "$CONFIG" --stage "$STAGE"
