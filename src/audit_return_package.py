"""Read-only tar integrity/stage audit; this does not validate research protocols."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tarfile


def digest(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(chunk)
    return value.hexdigest()


def safe_name(name):
    """Normalize harmless './' aliases while rejecting paths outside a stage."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Invalid archive/output path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or PureWindowsPath(name).drive or ".." in path.parts:
        raise ValueError(f"Unsafe archive/output path: {name!r}")
    return path.as_posix()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest JSON key: {key!r}")
        result[key] = value
    return result


def audit(archive, required_stages=()):
    """Return a JSON-compatible receipt without extracting or changing inputs."""
    archive = Path(archive)
    required_stages = sorted(set(required_stages))
    report = dict(archive=str(archive.resolve()), archive_sha256=None,
                  scope="Archive paths, manifest status, and declared output SHA256 only; "
                        "not scientific protocol, metric, or model validity.",
                  required_stages=required_stages, missing_required_stages=[],
                  member_count=0, regular_file_count=0, stages=[], errors=[], ok=False)
    try:
        with archive.open("rb") as stream:
            report["archive_sha256"] = digest(stream)
        with tarfile.open(archive, "r:*") as package:
            members = {}
            for member in package.getmembers():
                report["member_count"] += 1
                name = safe_name(member.name)
                if name in members:
                    raise ValueError(f"Duplicate archive member: {name}")
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Link/special archive member rejected: {name}")
                if name == "." and member.isfile():
                    raise ValueError("Archive root cannot be a regular file")
                members[name] = member
                report["regular_file_count"] += int(member.isfile())

            for name, member in members.items():
                if not member.isfile() or PurePosixPath(name).name != "run_manifest.json":
                    continue
                parent = PurePosixPath(name).parent
                stage = dict(stage=parent.name or ".", manifest=name, status=None,
                             git_head=None, versions=None, python=None,
                             declared_outputs=0, verified_outputs=0, missing_outputs=[],
                             hash_mismatches=[], errors=[], complete=False)
                report["stages"].append(stage)
                try:
                    with package.extractfile(member) as stream:
                        manifest = json.load(stream, object_pairs_hook=unique_keys)
                    if not isinstance(manifest, dict):
                        raise ValueError("Manifest must be a JSON object")
                    for key in ["status", "git_head", "versions", "python"]:
                        stage[key] = manifest.get(key)
                    if stage["status"] != "complete":
                        stage["errors"].append(f"Stage status is not complete: {stage['status']!r}")
                    outputs = manifest.get("output_sha256")
                    if not isinstance(outputs, dict) or not outputs:
                        raise ValueError("Manifest lacks a nonempty output_sha256 object")
                    stage["declared_outputs"] = len(outputs)
                    seen_outputs = set()
                    for output, expected in outputs.items():
                        relative = safe_name(output)
                        if relative == "." or relative in seen_outputs:
                            raise ValueError(f"Invalid/duplicate declared output: {output}")
                        seen_outputs.add(relative)
                        target = (parent / relative).as_posix()
                        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                            raise ValueError(f"Invalid SHA256 for output: {output}")
                        item = members.get(target)
                        if item is None or not item.isfile():
                            stage["missing_outputs"].append(output)
                            continue
                        with package.extractfile(item) as stream:
                            actual = digest(stream)
                        if actual != expected.lower():
                            stage["hash_mismatches"].append(dict(path=output, expected=expected, actual=actual))
                        else:
                            stage["verified_outputs"] += 1
                except (ValueError, TypeError, UnicodeError, OSError, tarfile.TarError) as exc:
                    stage["errors"].append(str(exc))
                stage["complete"] = not (stage["errors"] or stage["missing_outputs"] or stage["hash_mismatches"])
                if not stage["complete"]:
                    report["errors"].append(f"Stage integrity/status failed: {name}")
            if not report["stages"]:
                report["errors"].append("No run_manifest.json found in archive")
    except (ValueError, TypeError, UnicodeError, OSError, tarfile.TarError) as exc:
        report["errors"].append(str(exc))
    present = {stage["stage"] for stage in report["stages"]}
    report["missing_required_stages"] = sorted(set(required_stages) - present)
    if report["missing_required_stages"]:
        report["errors"].append("Missing required stages: " + ", ".join(report["missing_required_stages"]))
    report["ok"] = not report["errors"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--require-stage", action="append", default=[], help="Repeat for multiple required stages")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.archive.resolve() == args.output.resolve():
        parser.error("The audit output must not overwrite the archive")
    report = audit(args.archive, args.require_stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[return-audit] {'PASS' if report['ok'] else 'FAIL'}; report={args.output}")
    for error in report["errors"]:
        print(f"[return-audit] {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
