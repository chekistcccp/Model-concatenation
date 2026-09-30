"""CPU-only server check of cross-split filename candidates, without training.

Run in the server repo: python -m src.verify_manifest_duplicates
Only reads manifest/images; writes a separate JSON diagnostic. Pixel hashes use
decoded RGB values and image dimensions, not resized/normalized model inputs.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

from PIL import Image


def hashes(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixel = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
    return dict(file_sha256=digest.hexdigest(), rgb_pixel_sha256=pixel)


def verify(records, datasets):
    groups = defaultdict(list)
    for record in records:
        if record["dataset"] in datasets:
            groups[(record["dataset"], Path(record["image"]).name)].append(record)
    pairs, errors, memo = [], [], {}
    for (dataset, name), group in groups.items():
        if len({r["split"] for r in group}) < 2:
            continue
        for record in group:
            path = record["image"]
            if path not in memo:
                try:
                    memo[path] = hashes(path)
                except Exception as exc:
                    memo[path] = None
                    errors.append(dict(image=path, error=str(exc)))
        for a, b in itertools.combinations(group, 2):
            if a["split"] == b["split"]:
                continue
            ha, hb = memo[a["image"]], memo[b["image"]]
            pairs.append(dict(dataset=dataset, filename=name, split_a=a["split"], split_b=b["split"], image_a=a["image"], image_b=b["image"],
                              hashes_a=ha, hashes_b=hb, same_bytes=ha["file_sha256"] == hb["file_sha256"] if ha and hb else None,
                              same_rgb_pixels=ha["rgb_pixel_sha256"] == hb["rgb_pixel_sha256"] if ha and hb else None))
    return dict(candidate_pairs=len(pairs), verified_same_bytes=sum(r["same_bytes"] is True for r in pairs),
                verified_same_rgb_pixels=sum(r["same_rgb_pixels"] is True for r in pairs), errors=errors, pairs=pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("cache/manifest.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("results/analysis/server_duplicate_checks.json"))
    ap.add_argument("--datasets", nargs="+", default=["Brain", "OCT2017"])
    args = ap.parse_args()
    if args.output.resolve() == args.manifest.resolve():
        raise ValueError("Output must not overwrite input manifest")
    with args.manifest.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if args.output.resolve() in {Path(r["image"]).resolve() for r in records}:
        raise ValueError("Output must not overwrite an image")
    result = verify(records, set(args.datasets))
    result["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"pairs", "errors"}}, indent=2))
    print(f"Unreadable files: {len(result['errors'])}; output: {args.output}")
    if result["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
