"""Package the completed focused_search fold0 study without retraining."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import joblib

from .compound_experiment import read_json, sha
from .focused_runtime import PROFILE, FocusedMatcher
from .hangul_expansion import ExpansionConfig, HangulExpansion
from .ranking import VesselRanker
from .ranker_commands import RankedMatcher
from .submission_package import _json_bytes, _read_file, _verified_index_hash
from .tonnage_repair_experiment import check


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULES = (
    "__init__.py", "data.py", "features.py", "feature_selection.py", "text.py", "memory.py",
    "retrieval.py", "pipeline.py", "robust_retrieval.py", "ranking.py", "name_features.py",
    "name_feature_cache.py", "hangul_expansion.py", "focused_runtime.py",
)


def protected_snapshot(root=ROOT):
    """Hash existing models and submission artifacts; never change these files."""
    return {str(p.resolve()): sha(p) for folder in ("artifacts", "submissions")
            for p in sorted((root / folder).rglob("*")) if p.is_file()}


def check_snapshot(before):
    for path, expected in before.items():
        if sha(path) != expected:
            raise ValueError(f"Existing model or submission changed: {path}")


def write_archive(output, payload):
    """Reproducible paths/timestamps; publish only after a complete ZIP check."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new submission ZIP; existing files are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".focused-", suffix=".zip", dir=output.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name, data in sorted(payload.items()):
                    entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    entry.create_system = 3
                    entry.external_attr = 0o100644 << 16
                    archive.writestr(entry, data, compresslevel=6)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive CRC validation failed")
        if os.name == "nt":
            os.rename(temporary, output)  # Windows rename refuses an existing destination.
        else:
            os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def build(study, output):
    study, output = Path(study).resolve(), Path(output).resolve()
    report_path = output.parent / "package_report.json"
    if output.suffix.lower() != ".zip" or output.exists() or report_path.exists():
        raise FileExistsError("Choose a new .zip destination and a fresh package report directory")
    before = protected_snapshot()
    frozen = check(study)
    for path, expected in frozen["protected_files"].items():
        if path in before and before[path] != expected:
            raise ValueError(f"Preserved artifact differs from the frozen study: {path}")
    info = frozen["baselines"]["0"]
    baseline_directory = Path(info["directory"])
    for name, expected in info["files"].items():
        if sha(baseline_directory / name) != expected:
            raise ValueError(f"Baseline asset changed: {name}")
    state = read_json(study / "fold0/status.json")
    if state.get("complete") is not True or state.get("fold") != 0:
        raise ValueError("Fold0 must be complete")
    saved_hashes = {name.replace("\\", "/"): digest for name, digest in state["files"].items()}
    for relative in ("focused/model.txt", "focused/ranker.json", "focused/repair_model.json", "predictions.csv"):
        if sha(study / "fold0" / relative) != saved_hashes[relative]:
            raise ValueError(f"Completed study asset changed: {relative}")
    repair = read_json(study / "fold0/focused/repair_model.json")
    if (repair.get("complete") is not True or repair.get("branch") != "focused" or repair.get("fold") != 0
            or Path(repair["protocol_path"]).resolve() != study / "protocol.json"):
        raise ValueError("Wrong focused model provenance")
    summary = read_json(baseline_directory / "training_summary.json")
    index = _read_file(baseline_directory / "index.joblib")
    _verified_index_hash(summary, index, Path(info["cache"]))
    baseline = RankedMatcher(baseline_directory)
    ranker = VesselRanker.load(study / "fold0/focused")
    if (ranker.feature_names != baseline.ranker.feature_names or ranker.config != baseline.ranker.config
            or repair["feature_names"] != ranker.feature_names):
        raise ValueError("Focused model schema/configuration changed")
    # Fixed official catalog only. Persist this index before seeing any test input.
    expansion = HangulExpansion(baseline.pipeline.catalog)
    buffer = io.BytesIO()
    joblib.dump(expansion, buffer, compress=3, protocol=4)
    assets = {
        "index.joblib": index,
        "hangul_expansion.joblib": buffer.getvalue(),
        "ranker/model.txt": _read_file(study / "fold0/focused/model.txt"),
        "ranker/ranker.json": _read_file(study / "fold0/focused/ranker.json"),
    }
    dataset = summary["dataset"]
    provenance = {
        "study_version": frozen["version"], "study_protocol_sha256": sha(study / "protocol.json"),
        "fold_status_sha256": sha(study / "fold0/status.json"),
        "repair_model_metadata_sha256": sha(study / "fold0/focused/repair_model.json"),
        "reference_predictions_sha256": sha(study / "fold0/predictions.csv"),
        "baseline_summary_sha256": sha(baseline_directory / "training_summary.json"),
        "official_data_sha256": {name: record["sha256"] for name, record in frozen["data"].items()},
    }
    runtime = {
        "schema_version": 1, "complete": True, "profile": PROFILE, "fold": 0,
        "assets": {name: hashlib.sha256(data).hexdigest() for name, data in assets.items()},
        "feature_selection": summary["feature_selection"], "feature_transform": dataset["feature_transform"],
        "feature_code_sha256": dataset["provenance"]["feature_code_sha256"],
        "retrieval_metadata": {k: dataset[k] for k in ("retrieval", "retrieval_extension")},
        "expansion_config": asdict(ExpansionConfig()), "provenance": provenance,
        "training": {k: repair[k] for k in ("branch", "fold", "groups", "pairs", "retained_base_groups",
                                             "retained_joint_groups", "repair_groups")},
    }
    payload = {"model/" + name: data for name, data in assets.items()}
    payload["model/runtime.json"] = _json_bytes(runtime)
    payload["script.py"] = _read_file(ROOT / "submission_focused/script.py")
    payload["requirements.txt"] = _read_file(ROOT / "submission/requirements.txt")
    for name in RUNTIME_MODULES:
        payload["model/portmatch/" + name] = _read_file(ROOT / "portmatch" / name)
    manifest = {
        "schema_version": 1, "profile": PROFILE, "competition_id": 236771, "feature_count": 86,
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                  for name, data in sorted(payload.items())},
    }
    payload["model/package_manifest.json"] = _json_bytes(manifest)
    # Exercise deserialization and schema checks before publishing a ZIP.
    with tempfile.TemporaryDirectory(prefix="focused-restore-", dir=ROOT / "work") as scratch:
        directory = Path(scratch)
        for name, data in payload.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        restored = FocusedMatcher(directory / "model")
        if restored.ranker.feature_names != ranker.feature_names:
            raise ValueError("Restored feature schema changed")
    check_snapshot(before)
    write_archive(output, payload)
    check_snapshot(before)
    report = {
        "complete": True, "profile": PROFILE, "zip_path": str(output), "zip_sha256": sha(output),
        "zip_bytes": output.stat().st_size, "expanded_bytes": sum(map(len, payload.values())),
        "file_count": len(payload), "runtime_modules": list(RUNTIME_MODULES),
        "model_restoration_passed": True, "source_model_sha256": runtime["assets"]["ranker/model.txt"],
        "preserved_files": before, "preserved_file_count": len(before), "existing_artifacts_unchanged": True,
        "provenance": provenance,
    }
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return {k: v for k, v in report.items() if k not in {"preserved_files", "provenance", "runtime_modules"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="runs/tonnage_repair_v1_20261031", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.study, args.output), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
