from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from .common import ensure_dir, write_json, write_jsonl


DATASET_CANON = {
    # canonical / legacy names
    "brain": "Brain",
    "liver": "liver",
    "resc": "RESC",
    "oct2017": "OCT2017",
    "rsna": "RSNA",
    "chest": "RSNA",
    "chest_rsna": "RSNA",
    "camelyon16": "camelyon16",
    "camelyon16_256": "camelyon16",
    # BMAD official Google-Drive archive/folder names after removing _AD
    "retina_resc": "RESC",
    "retina_oct2017": "OCT2017",
    "histopathology": "camelyon16",
}


def _normalized_dataset_name(name: str) -> str:
    """Normalize BMAD raw archive/extraction directory names.

    Examples:
      3918150265_Brain_AD      -> brain
      3b257ef473_Liver_AD      -> liver
      Retina_OCT2017_AD        -> retina_oct2017
      Retina_RESC_AD           -> retina_resc
      Chest-AD                 -> chest
      Histopathology_AD        -> histopathology
    """
    low = name.strip().lower().replace("-", "_").replace(" ", "_")
    low = re.sub(r"^[0-9a-f]{10}_", "", low)
    low = re.sub(r"_+", "_", low).strip("_")
    if low.endswith("_ad"):
        low = low[:-3]
    return low
NORMAL_KEYS = {"good", "normal", "healthy"}
ABNORMAL_KEYS = {"ungood", "bad", "anomaly", "anomalous", "abnormal", "disease", "diseased"}
MASK_KEYS = {"anomaly_mask", "mask", "masks", "label", "labels", "ground_truth", "gt"}
SPLIT_KEYS = {"train": "train", "valid": "valid", "val": "valid", "validation": "valid", "test": "test"}


def _is_archive(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2"))


def _archive_stem(p: Path) -> str:
    name = p.name
    for suffix in (".tar.bz2", ".tar.gz", ".tbz2", ".tgz", ".zip", ".tar"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return p.stem


def _extract_one_archive(arc: Path, target: Path) -> None:
    ensure_dir(target)
    try:
        shutil.unpack_archive(str(arc), str(target))
    except (shutil.ReadError, ValueError):
        if arc.suffix.lower() == ".zip":
            with zipfile.ZipFile(arc) as zf:
                zf.extractall(target)
        else:
            with tarfile.open(arc) as tf:
                tf.extractall(target)


def auto_extract_archives(data_dir: Path) -> dict:
    """Recursively extract raw Google-Drive archives dropped in data/.

    Supports both a single outer Google-Drive zip and nested dataset archives.
    The user never needs to inspect or manually unpack archive contents.
    """
    out_root = ensure_dir(data_dir / "_extracted")
    top_archives = sorted(
        p for p in data_dir.iterdir() if p.is_file() and _is_archive(p)
    )
    queue = list(top_archives)
    seen = set()
    extracted = []

    while queue:
        arc = queue.pop(0)
        key = str(arc.resolve())
        if key in seen:
            continue
        seen.add(key)

        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        target = ensure_dir(out_root / f"{digest}_{_archive_stem(arc)}")
        marker = target / ".extract_complete"
        signature = f"{arc.stat().st_size}:{int(arc.stat().st_mtime)}"
        current = marker.read_text(encoding="utf-8").strip() if marker.exists() else None

        if current != signature:
            print(f"[data] extracting archive {arc} -> {target}")
            for child in list(target.iterdir()):
                if child.name == ".extract_complete":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            _extract_one_archive(arc, target)
            marker.write_text(signature + "\n", encoding="utf-8")

        extracted.append(str(target))

        # Google Drive outer zips may contain dataset archives. Unpack those too.
        nested = sorted(
            p for p in target.rglob("*")
            if p.is_file() and _is_archive(p)
        )
        queue.extend(nested)

    return {
        "archives": [str(p) for p in top_archives],
        "extracted": extracted,
    }


def _dataset_roots(data_dir: Path, preferred_subdir: str = "BMAD") -> Dict[str, Path]:
    # Prefer the documented canonical layout data/BMAD/<dataset>/.
    preferred = data_dir / preferred_subdir
    search_roots = [preferred, data_dir] if preferred.exists() else [data_dir]
    candidates: Dict[str, List[Path]] = {v: [] for v in DATASET_CANON.values()}
    seen = set()
    for base in search_roots:
        for p in base.rglob("*"):
            if not p.is_dir():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            normalized = _normalized_dataset_name(p.name)
            if normalized in DATASET_CANON:
                candidates[DATASET_CANON[normalized]].append(p)
    roots: Dict[str, Path] = {}
    for canon, items in candidates.items():
        if not items:
            continue
        roots[canon] = sorted(
            items,
            key=lambda x: (
                0 if preferred in x.parents else 1,
                len(x.parts),
                str(x),
            ),
        )[0]
    return roots


def data_status(cfg: dict) -> dict:
    data_dir = Path(cfg["paths"]["data_dir"]).resolve()
    preferred_subdir = str(cfg["data"].get("preferred_subdir", "BMAD"))
    archives = [str(p) for p in data_dir.iterdir() if p.is_file() and _is_archive(p)] if data_dir.exists() else []
    roots = _dataset_roots(data_dir, preferred_subdir)
    expected = cfg["data"]["datasets"]
    return {
        "data_dir": str(data_dir),
        "raw_archives": archives,
        "automatic_extract_root": str(data_dir / "_extracted"),
        "recommended_root": str(data_dir / preferred_subdir),
        "datasets": {
            d: {
                "found": d in roots,
                "path": str(roots[d]) if d in roots else None,
            }
            for d in expected
        },
        "missing": [d for d in expected if d not in roots],
        "ready": all(d in roots for d in expected),
    }


def _find_split_dir(root: Path, split: str) -> Optional[Path]:
    aliases = {k for k, v in SPLIT_KEYS.items() if v == split}

    # Prefer the normal case first.
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in aliases:
            return child

    # Raw Google-Drive archives occasionally introduce one or more wrapper
    # directories. Search recursively and select the nearest matching split.
    matches = [
        p for p in root.rglob("*")
        if p.is_dir() and p.name.lower() in aliases
    ]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda p: (len(p.relative_to(root).parts), str(p)),
    )[0]


def _image_files(root: Path, extensions: set[str]) -> List[Path]:
    out = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue
        parts = {x.lower() for x in p.parts}
        if parts & MASK_KEYS:
            continue
        out.append(p)
    return sorted(out)


def _mask_map(root: Path, extensions: set[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue
        parts = {x.lower() for x in p.parts}
        if not (parts & MASK_KEYS):
            continue
        out.setdefault(p.stem, p)
    return out


def _label_from_path(p: Path) -> Optional[int]:
    parts = [x.lower() for x in p.parts]
    if any(x in ABNORMAL_KEYS for x in parts):
        return 1
    if any(x in NORMAL_KEYS for x in parts):
        return 0
    return None


def prepare_manifest(cfg: dict) -> dict:
    data_dir = Path(cfg["paths"]["data_dir"]).resolve()
    ensure_dir(data_dir)
    extraction = {}
    if cfg["data"].get("auto_extract_archives", True):
        extraction = auto_extract_archives(data_dir)

    status = data_status(cfg)
    status["extraction"] = extraction
    preferred_subdir = str(cfg["data"].get("preferred_subdir", "BMAD"))
    roots = _dataset_roots(data_dir, preferred_subdir)
    expected = cfg["data"]["datasets"]
    missing = status["missing"]
    if missing:
        raise FileNotFoundError(
            "BMAD datasets not found after automatic archive extraction: "
            + ", ".join(missing)
            + ". Put the original Google-Drive BMAD .zip/.tar.* file(s) directly "
              "under data/; do not rename or manually extract them."
        )

    extensions = {x.lower() for x in cfg["data"]["extensions"]}
    modality_map = cfg["data"]["modality_map"]
    rows: List[dict] = []
    audit: Dict[str, dict] = {}

    for dataset in expected:
        root = roots[dataset]
        audit[dataset] = {"root": str(root), "splits": {}}
        for split in ("train", "valid", "test"):
            split_dir = _find_split_dir(root, split)
            if split_dir is None:
                continue
            masks = _mask_map(split_dir, extensions)
            counts = {"normal": 0, "abnormal": 0, "with_mask": 0}
            for img in _image_files(split_dir, extensions):
                label = _label_from_path(img)
                if label is None:
                    continue
                mask = masks.get(img.stem)
                rec = {
                    "dataset": dataset,
                    "modality": modality_map[dataset],
                    "split": split,
                    "label": int(label),
                    "image": str(img),
                    "mask": str(mask) if mask is not None else None,
                }
                rows.append(rec)
                counts["abnormal" if label else "normal"] += 1
                counts["with_mask"] += int(mask is not None)
            audit[dataset]["splits"][split] = counts

    if not rows:
        raise RuntimeError("No BMAD images were discovered. Check directory structure under data/.")

    manifest_path = Path(cfg["paths"]["cache_dir"]) / "manifest.jsonl"
    write_jsonl(manifest_path, rows)
    write_json(Path(cfg["paths"]["cache_dir"]) / "data_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit
