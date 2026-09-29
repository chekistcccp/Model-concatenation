from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def configure_torch(allow_tf32: bool = True) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def read_jsonl(path: str | Path) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def gpu_ids(max_gpus: int = 4) -> List[int]:
    env = os.environ.get("GPUS", "").strip()
    if env:
        ids = [int(x) for x in env.split(",") if x.strip()]
        return ids[:max_gpus]
    return list(range(min(torch.cuda.device_count(), max_gpus)))


def stage_key(stage: int) -> str:
    if stage not in (1, 2, 3):
        raise ValueError(f"source stage must be 1, 2, or 3, got {stage}")
    return f"s{stage}"


def config_name(source_name: str, target_name: str, stage: int, block: int) -> str:
    return f"{source_name}_to_{target_name}_s{stage}_b{block}"


def pair_name(source_name: str, target_name: str) -> str:
    return f"{source_name}_to_{target_name}"


def parse_config_name(name: str) -> tuple[str, str, int, int]:
    left, stage_part, block_part = name.rsplit("_", 2)
    source_name, target_name = left.split("_to_", 1)
    return source_name, target_name, int(stage_part[1:]), int(block_part[1:])
