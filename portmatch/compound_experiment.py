"""Separate experiment runner; existing caches, bundles and ZIPs are read-only.

python -m portmatch.compound_experiment freeze --output runs/compound_v1
python -m portmatch.compound_experiment run --output runs/compound_v1 --fold 0
python -m portmatch.compound_experiment report --output runs/compound_v1
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .compound_protocol import (EVAL_RECIPES, VERSION, evaluation_variant, input_key,
                                protocol, training_variant)
from .compound_mining import expanded_masses, mined_positions, teacher_features
from .data import QUERY_COLUMNS, read_catalog, read_queries
from .feature_transforms import transform_features
from .ranking import VesselRanker, ranking_metrics
from .ranker_commands import RankedMatcher, score_order
from .retrieval_comparison import _load_model, _validate_partition, _initialize_name_features
from .robust_retrieval import RobustMatchingPipeline, RobustRetrievalConfig
from .sampling import SamplingConfig, sample_training_candidates
from .text import has_hangul
from .training_data import iter_shards, _ShardWriter

MODEL_NAMES = ("baseline", "expanded", "expanded_mined")
NEW_MODULES = ("compound_protocol.py", "compound_mining.py", "compound_experiment.py")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze(root, workspace=Path.cwd()):
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    if root.exists():
        raise FileExistsError("A frozen experiment must use a new directory")
    baselines = {}
    for fold in range(5):
        directory = workspace / f"artifacts/ranker_robust_gt_missing_token_phonetic_fold{fold}"
        summary = read_json(directory / "training_summary.json")
        cache = Path(summary["cache_directory"]).resolve()
        baselines[str(fold)] = {"directory": str(directory), "cache": str(cache),
                               "files": {name: sha(directory / name) for name in
                                         ("model.txt", "ranker.json", "index.joblib", "training_summary.json", "validation_predictions.csv")},
                               "cache_metadata_sha256": sha(cache / "metadata.json")}
        if baselines[str(fold)]["cache_metadata_sha256"] != summary["cache_metadata_sha256"]:
            raise ValueError("Baseline cache provenance differs")
    protected = {}
    for folder in ("artifacts", "submissions", "submission"):
        for path in sorted((workspace / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                protected[str(path)] = sha(path)
    # Preserve the bytes responsible for every previous cache/model version.
    for path in (workspace / "portmatch").glob("*.py"):
        if path.name not in NEW_MODULES:
            protected[str(path)] = sha(path)
    data = {name: {"path": str(workspace / "work/data" / name),
                   "sha256": sha(workspace / "work/data" / name)}
            for name in ("train.csv", "vessel_catalog.csv")}
    frozen = {"version": VERSION, "protocol": protocol(), "baselines": baselines,
              "data": data, "protected_files": protected,
              "implementation": {name: sha(Path(__file__).with_name(name)) for name in NEW_MODULES},
              "frozen_before_training": True, "planned_folds": [0, 1, 2, 3, 4]}
    root.mkdir(parents=True)
    write_json(root / "protocol.json", frozen)
    print(f"Frozen protocol; {len(protected)} existing files protected", flush=True)
    return frozen


def check_protocol(root, *, protection=False):
    frozen = read_json(Path(root) / "protocol.json")
    if frozen["version"] != VERSION or frozen["protocol"] != protocol():
        raise ValueError("Experiment protocol changed")
    for name, expected in frozen["implementation"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Implementation changed after protocol freeze: {name}")
    for asset in frozen["data"].values():
        if sha(asset["path"]) != asset["sha256"]:
            raise ValueError("Official data changed")
    if protection:
        for path, expected in frozen["protected_files"].items():
            if sha(path) != expected:
                raise ValueError(f"Protected asset changed: {path}")
    return frozen


def progress(phase, **values):
    print(json.dumps({"phase": phase, **values}, ensure_ascii=True), flush=True)


def _group_meta(cache):
    metadata = read_json(cache / "metadata.json")
    rows = []
    for relative in metadata["train_shards"]:
        with np.load(cache / relative, allow_pickle=False) as z:
            rows.extend(zip(z["sample_id"].tolist(), z["query_id"].tolist(),
                            z["augmentation_variant"].tolist(), z["query_mass"].tolist(),
                            z["group"].tolist(), z["has_hangul"].tolist()))
    return pd.DataFrame(rows, columns=["source", "query", "variant", "mass", "group", "language"])


@contextmanager
def heavy_training_lock(root):
    """Serialize large matrix allocation/fitting across independent fold workers."""
    with (Path(root) / "training.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        import os
        if os.name == "nt":
            import msvcrt
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(1)
            try:
                yield
            finally:
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare_teacher_pool(root):
    """One shared static pool, then filter BOTH outer and inner vessels at fit.

    The pool is not a fitted model. Its retrieval sees the fixed catalog only;
    no training aliases, call signs, category counts, or labels enter indexing.
    """
    root = Path(root).resolve()
    frozen = check_protocol(root)
    output = root / "teacher_pool"
    if output.exists():
        raise FileExistsError("Teacher pool already exists")
    output.mkdir()
    train = read_queries(frozen["data"]["train.csv"]["path"], labeled=True)
    catalog = read_catalog(frozen["data"]["vessel_catalog.csv"]["path"])
    baseline = RankedMatcher(frozen["baselines"]["0"]["directory"])
    data = read_json(Path(frozen["baselines"]["0"]["directory"]) / "training_summary.json")["dataset"]
    pipeline = RobustMatchingPipeline(train.iloc[:0], catalog, RobustRetrievalConfig(**data["retrieval"]))
    assert pipeline.memory.training_rows == 0
    assert not pipeline.memory.training_vessels
    _initialize_name_features(type("Holder", (), {"pipeline": pipeline})())
    sources = train.copy()
    sources["_language"] = sources.vessel_name.map(has_hangul)
    sources["_order"] = sources.sample_id.map(lambda s: hashlib.sha256(str(s).encode()).hexdigest())
    sources = sources.sort_values(["_language", "_order"], ascending=[False, True]).drop_duplicates("target_vessel_id")
    names = teacher_features(baseline.ranker.feature_names)
    writer = _ShardWriter(output, "train", 64)
    id_to_index = dict(zip(catalog.vessel_id, range(len(catalog))))
    skipped = Counter()
    for number, source in enumerate(sources.to_dict("records"), 1):
        for variant in ("original", "both_gt_missing"):
            query = evaluation_variant(source, variant)
            candidates = pipeline.retrieve(query)
            features = pipeline.features(query, candidates)
            features = transform_features(baseline.feature_transform, pipeline, query, candidates, features)
            target = id_to_index[source["target_vessel_id"]]
            qid = source["sample_id"] + "::catalog_only::" + variant
            selected = sample_training_candidates(candidates, features, target, qid, SamplingConfig(**data["sampling"]))
            if not len(selected):
                skipped[variant] += 1
                continue
            writer.add(features[names].iloc[selected], candidates.iloc[selected],
                       sample_id=source["sample_id"], query_id=qid, language=has_hangul(source["vessel_name"]), target=target)
        if number % 250 == 0:
            progress("teacher_pool", done=number, total=len(sources))
    writer.flush()
    write_json(output / "manifest.json", {"complete": True, "fitted_training_rows": 0,
        "feature_names": names, "files": {f: sha(output / f) for f in writer.files},
        "source_count": len(sources), "groups": writer.groups, "pairs": writer.pairs,
        "skipped_missing_target": dict(skipped)})
    progress("teacher_pool_complete", sources=len(sources), groups=writer.groups)


def train_teachers(cache, outer_train, partitions, baseline, output):
    """Cross-fit labels by vessel; teacher features have no fitted lookup state.

    Shared pool candidates also come from a catalog-only index. Neither the
    selected pair features nor candidate selection uses held-out fitted state.
    """
    output.mkdir()
    teacher_names = teacher_features(baseline.ranker.feature_names)
    if len(teacher_names) != 54:
        raise ValueError("Teacher static feature whitelist must have 54 columns")
    partition = partitions.set_index("sample_id")
    source_ids = set(outer_train.sample_id)
    pool = output.parent.parent / "teacher_pool"
    pool_manifest = read_json(pool / "manifest.json")
    if (not pool_manifest.get("complete") or pool_manifest["feature_names"] != teacher_names
            or pool_manifest["fitted_training_rows"] != 0):
        raise ValueError("Invalid catalog-only teacher pool")
    chunks, labels, groups, langs, group_sources, folds = [], [], [], [], [], []
    for relative, expected in pool_manifest["files"].items():
        if sha(pool / relative) != expected:
            raise ValueError("Teacher pool changed")
        with np.load(pool / relative, allow_pickle=False) as z:
            shard = {key: z[key] for key in z.files}
        offset = 0
        for i, count in enumerate(shard["group"]):
            count = int(count)
            source = str(shard["sample_id"][i])
            if source in source_ids:
                chunks.append(shard["X"][offset:offset+count].copy())
                labels.append(shard["y"][offset:offset+count].copy())
                groups.append(count); langs.append(bool(shard["has_hangul"][i])); group_sources.append(source)
                folds.append(int(partition.loc[source, "inner_vessel"]))
            offset += count
    matrix, y = np.concatenate(chunks), np.concatenate(labels)
    del chunks, labels
    groups, langs, folds = np.array(groups), np.array(langs), np.array(folds)
    teachers = []
    for inner in range(3):
        keep = folds != inner
        pair_keep = np.repeat(keep, groups)
        fitting_sources = np.asarray(group_sources)[keep]
        counts = Counter(fitting_sources)
        masses = np.array([1 / counts[s] for s in fitting_sources])
        targets = set(outer_train.loc[outer_train.sample_id.isin(fitting_sources), "target_vessel_id"])
        held_ids = set(partitions.loc[partitions.inner_vessel.eq(inner) & partitions.role.eq("train"), "sample_id"])
        held_targets = set(outer_train.loc[outer_train.sample_id.isin(held_ids), "target_vessel_id"])
        assert not targets & held_targets
        assert not set(fitting_sources) & held_ids
        teacher = VesselRanker(replace(baseline.ranker.config, n_estimators=100))
        teacher.fit(matrix[pair_keep], y[pair_keep], groups[keep], langs[keep], teacher_names, query_mass=masses)
        directory = output / str(inner)
        teacher.save(directory)
        write_json(directory / "crossfit.json", {
            "inner_vessel_fold": inner, "training_sample_ids": sorted(set(fitting_sources)),
            "training_vessel_ids": sorted(targets), "excluded_vessel_ids": sorted(held_targets),
            "label_and_vessel_disjoint": True, "feature_names": teacher_names,
            "fitted_retrieval_or_category_features": False,
            "groups": int(keep.sum()), "pairs": int(groups[keep].sum()),
        })
        teachers.append(teacher)
        progress("teacher_complete", inner_fold=inner, pairs=int(groups[keep].sum()))
    return teachers


class ExtraWriter:
    def __init__(self, output, names):
        self.output, self.names = output, names
        output.mkdir()
        self.pending, self.files, self.rows = [], [], []

    def add(self, features, candidates, target, heuristic, mined, row):
        union = np.union1d(heuristic, mined)
        ids = candidates.catalog_index.to_numpy(dtype=np.int64)[union]
        self.pending.append({"X": features[union], "y": (ids == target).astype(np.uint8),
                             "catalog_index": ids, "heuristic": np.isin(union, heuristic),
                             "mined": np.isin(union, mined), **row})
        self.rows.append({**row, "group": len(heuristic), "mining_changed": not np.array_equal(heuristic, mined)})
        if len(self.pending) >= 64:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        relative = f"part_{len(self.files):05d}.npz"
        arrays = {key: np.concatenate([r[key] for r in self.pending])
                  for key in ("X", "y", "catalog_index", "heuristic", "mined")}
        arrays["union_group"] = np.array([len(r["y"]) for r in self.pending], dtype=np.int32)
        for key in ("source", "query", "recipe", "language", "teacher_fold"):
            arrays[key] = np.array([r[key] for r in self.pending])
        np.savez_compressed(self.output / relative, **arrays)
        self.files.append(relative)
        self.pending.clear()


def prepare_extra(outer_train, catalog, partitions, baseline, dataset, teachers, output):
    writer = ExtraWriter(output / "extra", baseline.ranker.feature_names)
    teacher_positions = [baseline.ranker.feature_names.index(n) for n in teachers[0].feature_names]
    partition = partitions.set_index("sample_id")
    id_to_index = dict(zip(catalog.vessel_id, range(len(catalog))))
    sampling = SamplingConfig(**dataset["sampling"])
    rows, pipeline_proofs = [], []
    outer_valid_targets = set(partitions.loc[partitions.role.eq("valid"), "target_vessel_id"])
    for view in ("vessel", "row"):
        for inner in range(3):
            mapping = outer_train.sample_id.map(partition["inner_" + view])
            fitting, held = outer_train.loc[mapping.ne(inner)], outer_train.loc[mapping.eq(inner)]
            assert not set(fitting.sample_id) & set(held.sample_id)
            assert not set(fitting.target_vessel_id) & outer_valid_targets
            if view == "vessel":
                assert not set(fitting.target_vessel_id) & set(held.target_vessel_id)
            pipeline_proofs.append({"view": view, "inner": inner,
                                    "fitting_sample_ids": sorted(fitting.sample_id.tolist()),
                                    "held_sample_ids": sorted(held.sample_id.tolist()),
                                    "outer_validation_excluded": True})
            pipeline = RobustMatchingPipeline(fitting, catalog, RobustRetrievalConfig(**dataset["retrieval"]))
            _initialize_name_features(type("Holder", (), {"pipeline": pipeline})())
            held = held.copy(); held["_input_key"] = held.apply(input_key, axis=1)
            held = held.sort_values(["_input_key", "sample_id"])
            last_key, last_result = None, None
            for number, source in enumerate(held.to_dict("records"), 1):
                recipe, query = training_variant(source, view)
                teacher_fold = int(partition.loc[source["sample_id"], "inner_vessel"])
                query_id = str(source["sample_id"]) + "::" + view + "::compound_" + recipe
                record = {"source": source["sample_id"], "query": query_id, "recipe": recipe,
                          "language": bool(has_hangul(source["vessel_name"])), "teacher_fold": teacher_fold}
                key = input_key(query)
                status = "stored"
                if key == input_key(source):
                    status = "skipped_noop"
                else:
                    if key != last_key:
                        candidates = pipeline.retrieve(query)
                        features = pipeline.features(query, candidates)
                        features = transform_features(baseline.feature_transform, pipeline, query, candidates, features)
                        matrix = baseline.selection.apply(features.to_numpy(dtype=np.float32), features.columns.tolist())
                        last_key, last_result = key, (candidates, features, matrix)
                    candidates, features, matrix = last_result
                    target = id_to_index[source["target_vessel_id"]]
                    heuristic = sample_training_candidates(candidates, features, target, query_id, sampling)
                    if not len(heuristic):
                        status = "skipped_missing_target"
                    else:
                        scores = teachers[teacher_fold].predict(matrix[:, teacher_positions], teachers[teacher_fold].feature_names)
                        mined = mined_positions(candidates.catalog_index.to_numpy(), target, heuristic, scores)
                        writer.add(matrix, candidates, target, heuristic, mined, record)
                rows.append({**record, "status": status, "view": view, "inner_fold": inner})
                if number % 250 == 0:
                    progress("prepare_extra", view=view, inner=inner, done=number, total=len(held))
            progress("prepare_extra_complete", view=view, inner=inner, queries=len(held))
            del pipeline, last_result
            gc.collect()
    writer.flush()
    meta = pd.DataFrame(writer.rows)
    if meta.empty:
        raise ValueError("No usable added augmentation groups")
    meta.to_csv(output / "extra_groups.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(output / "augmentation_log.csv", index=False, encoding="utf-8-sig")
    write_json(output / "inner_partitions.json", pipeline_proofs)
    manifest = {"complete": True, "feature_names": writer.names,
                "files": {f: sha(writer.output / f) for f in writer.files},
                "groups": len(meta), "pairs_per_branch": int(meta.group.sum()),
                "statuses": dict(Counter(r["status"] for r in rows)),
                "recipes": dict(Counter(meta.recipe)), "mining_changed_groups": int(meta.mining_changed.sum())}
    write_json(output / "extra_manifest.json", manifest)
    return meta, manifest


def fit_branch(cache, base_groups, extra_groups, extra_manifest, baseline, branch, output):
    with heavy_training_lock(output.parent):
        return _fit_branch(cache, base_groups, extra_groups, extra_manifest, baseline, branch, output)


def _fit_branch(cache, base_groups, extra_groups, extra_manifest, baseline, branch, output):
    base_mass, extra_mass = expanded_masses(base_groups.source, base_groups.variant, base_groups.mass,
                                           extra_groups.source)
    nbase = int(base_groups.group.sum())
    matrix = np.empty((nbase + int(extra_groups.group.sum()), len(baseline.ranker.feature_names)), dtype=np.float32)
    labels = np.empty(len(matrix), dtype=np.uint8)
    offset = 0
    names = read_json(cache / "metadata.json")["feature_names"]
    positions = [names.index(n) for n in baseline.ranker.feature_names]
    for shard in iter_shards(cache, "train"):
        n = len(shard["y"])
        matrix[offset:offset+n] = shard["X"][:, positions]
        labels[offset:offset+n] = shard["y"]
        offset += n
    assert offset == nbase
    selection = "heuristic" if branch == "expanded" else "mined"
    observed_groups = []
    for relative, expected_hash in extra_manifest["files"].items():
        path = output / "extra" / relative
        if sha(path) != expected_hash:
            raise ValueError("Added training shard changed")
        with np.load(path, allow_pickle=False) as z:
            chosen = z[selection]
            n = int(chosen.sum())
            matrix[offset:offset+n] = z["X"][chosen]
            labels[offset:offset+n] = z["y"][chosen]
            starts = np.r_[0, np.cumsum(z["union_group"])[:-1]]
            observed_groups.extend(np.add.reduceat(chosen.astype(int), starts).tolist())
            offset += n
    assert offset == len(matrix)
    assert observed_groups == extra_groups.group.tolist()
    groups = np.r_[base_groups.group.to_numpy(), extra_groups.group.to_numpy()]
    languages = np.r_[base_groups.language.to_numpy(dtype=bool), extra_groups.language.to_numpy(dtype=bool)]
    masses = np.r_[base_mass, extra_mass]
    totals = pd.DataFrame({"source": np.r_[base_groups.source, extra_groups.source], "mass": masses}).groupby("source").mass.sum()
    assert np.allclose(totals, 1, atol=1e-10, rtol=0)
    ranker = VesselRanker(baseline.ranker.config)
    started = time.perf_counter()
    ranker.fit(matrix, labels, groups, languages, baseline.ranker.feature_names, query_mass=masses)
    directory = output / branch
    ranker.save(directory)
    summary = {"complete": True, "branch": branch, "groups": len(groups), "pairs": len(matrix),
               "sources": len(totals), "source_mass_min": float(totals.min()), "source_mass_max": float(totals.max()),
               "base_candidate_features_unchanged": True, "same_added_groups_and_mass_across_branches": True,
               "train_seconds": time.perf_counter()-started, "feature_names": ranker.feature_names,
               "base_cache_metadata_sha256": sha(cache / "metadata.json"),
               "protocol_path": str(output.parent / "protocol.json"), "fold": int(output.name.removeprefix("fold"))}
    write_json(directory / "compound_model.json", summary)
    ranker.feature_importance().to_csv(directory / "feature_importance.csv", index=False)
    del matrix, labels
    gc.collect()
    progress("model_complete", branch=branch, seconds=summary["train_seconds"])
    return ranker


def evaluate(baseline, models, heldout, output, baseline_directory):
    rankers = {"baseline": baseline.ranker, **models}
    assert tuple(rankers) == MODEL_NAMES
    for ranker in rankers.values():
        assert ranker.feature_names == baseline.ranker.feature_names
    saved = pd.read_csv(baseline_directory / "validation_predictions.csv").set_index("sample_id")
    _initialize_name_features(baseline)
    records, timings, effect_counts = [], [], Counter()
    memo = OrderedDict()
    ids = baseline.catalog_ids
    heldout = heldout.copy(); heldout["_input_key"] = heldout.apply(input_key, axis=1)
    for number, source in enumerate(heldout.sort_values(["_input_key", "sample_id"]).to_dict("records"), 1):
        for variant in EVAL_RECIPES:
            query = evaluation_variant(source, variant)
            key = input_key(query)
            effect_counts[variant] += key != input_key(source)
            if key not in memo:
                start = time.perf_counter()
                candidates = baseline.pipeline.retrieve(query)
                retrieved = time.perf_counter()
                features = baseline.pipeline.features(query, candidates)
                features = transform_features(baseline.feature_transform, baseline.pipeline, query, candidates, features)
                matrix = baseline.selection.apply(features.to_numpy(dtype=np.float32), features.columns.tolist())
                featured = time.perf_counter()
                indices = candidates.catalog_index.to_numpy(dtype=np.int64)
                outcomes = {}
                for name, ranker in rankers.items():
                    started = time.perf_counter()
                    scores = ranker.predict(matrix, ranker.feature_names)
                    order = score_order(scores, indices, ids)
                    outcomes[name] = (ids[indices[order]], scores[order])
                    timings.append({"stage": "rank", "model": name, "seconds": time.perf_counter()-started})
                timings.append({"stage": "shared_retrieval_features", "seconds": featured-start,
                                "retrieval_seconds": retrieved-start, "feature_seconds": featured-retrieved})
                memo[key] = outcomes
                if len(memo) > 64:
                    memo.popitem(last=False)
            else:
                memo.move_to_end(key)
            for name, (ordered_ids, scores) in memo[key].items():
                location = np.flatnonzero(ordered_ids == source["target_vessel_id"])
                record = {"variant": variant, "model": name, "sample_id": source["sample_id"],
                          "target_vessel_id": source["target_vessel_id"], "has_hangul": bool(has_hangul(source["vessel_name"])),
                          "predicted_vessel_id": str(ordered_ids[0]), "candidate_found": bool(len(location)),
                          "target_rank": int(location[0]+1) if len(location) else 0,
                          "candidate_count": len(ordered_ids), "top_score": float(scores[0]),
                          "score_margin": float(scores[0]-scores[1]) if len(scores)>1 else None}
                if name == "baseline" and variant == "original":
                    old = saved.loc[source["sample_id"]]
                    for field in ("predicted_vessel_id", "candidate_found", "target_rank", "candidate_count"):
                        assert record[field] == old[field], (source["sample_id"], field)
                    assert np.isclose(record["top_score"], old.top_score, atol=1e-10, rtol=0)
                records.append(record)
        if number % 100 == 0:
            progress("evaluate", done=number, total=len(heldout), predictions=len(records))
    frame = pd.DataFrame(records)
    assert len(frame) == len(heldout) * len(EVAL_RECIPES) * len(MODEL_NAMES)
    frame.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    metrics = {variant: {name: ranking_metrics(frame[frame.variant.eq(variant) & frame.model.eq(name)])
                        for name in MODEL_NAMES} for variant in EVAL_RECIPES}
    write_json(output / "evaluation.json", {"complete": True, "queries": len(heldout), "predictions": len(frame),
               "baseline_original_parity": True, "same_candidate_and_feature_arrays": True,
               "unique_inputs_computed": sum(t["stage"] == "shared_retrieval_features" for t in timings),
               "changed_inputs_per_variant": dict(effect_counts), "metrics": metrics,
               "shared_comparison_seconds": sum(t["seconds"] for t in timings),
               "timing_note": "Comparison shares retrieval/features and memoizes identical active inputs; not a submission runtime benchmark"})
    return metrics


class CompoundMatcher:
    """Experimental model bound to a preserved baseline index and schema."""
    def __init__(self, directory):
        directory = Path(directory).resolve()
        summary = read_json(directory / "compound_model.json")
        if not summary.get("complete"):
            raise ValueError("Incomplete experimental model")
        root = Path(summary["protocol_path"]).parent
        frozen = check_protocol(root)
        baseline = frozen["baselines"][str(summary["fold"])]
        for name, expected in baseline["files"].items():
            if sha(Path(baseline["directory"]) / name) != expected:
                raise ValueError("Parent model/index changed")
        self.matcher = RankedMatcher(baseline["directory"])
        ranker = VesselRanker.load(directory)
        if ranker.feature_names != self.matcher.ranker.feature_names:
            raise ValueError("Experimental feature schema differs")
        self.matcher.ranker = ranker

    def predict(self, row):
        return self.matcher.predict({key: row.get(key) for key in QUERY_COLUMNS})


def run_fold(root, fold):
    root = Path(root).resolve()
    frozen = check_protocol(root)
    baseline_info = frozen["baselines"][str(fold)]
    baseline_directory, cache = Path(baseline_info["directory"]), Path(baseline_info["cache"])
    for name, expected in baseline_info["files"].items():
        if sha(baseline_directory / name) != expected:
            raise ValueError("Baseline asset changed")
    output = root / f"fold{fold}"
    if output.exists():
        raise FileExistsError("Fold output already exists; no experiment is overwritten")
    output.mkdir()
    started = time.perf_counter()
    write_json(output / "status.json", {"complete": False, "phase": "initializing"})
    train = read_queries(frozen["data"]["train.csv"]["path"], labeled=True)
    catalog = read_catalog(frozen["data"]["vessel_catalog.csv"]["path"])
    baseline, summary, proof = _load_model(baseline_directory, frozen["data"]["train.csv"]["sha256"],
                                          frozen["data"]["vessel_catalog.csv"]["sha256"])
    if summary["dataset"]["outer_fold"] != fold or summary["dataset"]["inner_folds"] != 3:
        raise ValueError("Requested fold or inner-fold count differs from baseline")
    heldout = _validate_partition(baseline, summary, train, catalog)
    partitions = pd.read_csv(cache / "partitions.csv")
    assert set(partitions.loc[partitions.role.eq("valid"), "sample_id"]) == set(heldout.sample_id)
    outer_train = train.loc[~train.sample_id.isin(heldout.sample_id)].copy()
    base_groups = _group_meta(cache)
    teachers = train_teachers(cache, outer_train, partitions, baseline, output / "teachers")
    extra_groups, extra_manifest = prepare_extra(outer_train, catalog, partitions, baseline, summary["dataset"], teachers, output)
    del teachers
    gc.collect()
    models = {branch: fit_branch(cache, base_groups, extra_groups, extra_manifest, baseline, branch, output)
              for branch in MODEL_NAMES[1:]}
    metrics = evaluate(baseline, models, heldout, output, baseline_directory)
    # Native save/reload and singleton/reversed-input parity on both languages.
    independence = []
    sample = pd.concat([heldout.loc[heldout.vessel_name.map(has_hangul).eq(lang)].head(2)
                        for lang in (True, False)])
    for branch in MODEL_NAMES[1:]:
        restored = CompoundMatcher(output / branch)
        inputs = [evaluation_variant(r, v) for r in sample.to_dict("records")
                  for v in ("original", "name_delete_both_gt_missing", "unseen_swap_dual_gt_x0_70")]
        direct = [restored.predict(r) for r in inputs]
        reverse = [restored.predict(r) for r in reversed(inputs)][::-1]
        expected = pd.read_csv(output / "predictions.csv")
        expected = expected.loc[expected.model.eq(branch)].set_index(["sample_id", "variant"])
        reference = [expected.loc[(r["sample_id"], v), "predicted_vessel_id"]
                     for r in sample.to_dict("records") for v in
                     ("original", "name_delete_both_gt_missing", "unseen_swap_dual_gt_x0_70")]
        assert direct == reverse == reference
        independence.append({"branch": branch, "queries": len(inputs), "native_reload_and_order_parity": True})
    write_json(output / "status.json", {"complete": True, "fold": fold, "seconds": time.perf_counter()-started,
                                        "baseline": proof, "independence_checks": independence})
    progress("fold_complete", fold=fold, seconds=time.perf_counter()-started)
    return metrics


def report(root):
    root = Path(root).resolve()
    frozen = check_protocol(root, protection=True)
    frames, folds = [], []
    for fold in frozen["planned_folds"]:
        path = root / f"fold{fold}"
        if (path / "status.json").exists() and read_json(path / "status.json").get("complete"):
            frames.append(pd.read_csv(path / "predictions.csv").assign(fold=fold))
            folds.append(fold)
    if not frames:
        raise ValueError("No complete folds")
    predictions = pd.concat(frames, ignore_index=True)
    metrics = {v: {m: ranking_metrics(predictions[predictions.variant.eq(v) & predictions.model.eq(m)])
                   for m in MODEL_NAMES} for v in EVAL_RECIPES}
    changes = []
    for variant in EVAL_RECIPES:
        part = predictions[predictions.variant.eq(variant)]
        wide = part.pivot(index=["fold", "sample_id", "target_vessel_id", "has_hangul"],
                          columns="model", values="predicted_vessel_id").reset_index()
        for reference, experiment in (("baseline", "expanded"), ("baseline", "expanded_mined"), ("expanded", "expanded_mined")):
            before, after = wide[reference].eq(wide.target_vessel_id), wide[experiment].eq(wide.target_vessel_id)
            fixed, regressed = ~before & after, before & ~after
            changes.append({"variant": variant, "reference": reference, "experiment": experiment,
                            "corrected": int(fixed.sum()), "regressed": int(regressed.sum()),
                            "corrected_vessels": int(wide.loc[fixed, "target_vessel_id"].nunique()),
                            "regressed_vessels": int(wide.loc[regressed, "target_vessel_id"].nunique()),
                            "weighted_delta_pp": 100*(metrics[variant][experiment]["weighted_top1"]-metrics[variant][reference]["weighted_top1"])})
    result = {"complete_for_folds": folds, "full_five_fold": folds == [0,1,2,3,4],
              "queries": predictions.sample_id.nunique(), "predictions": len(predictions),
              "metrics": metrics, "paired_changes": changes,
              "protected_files_verified": len(frozen["protected_files"]),
              "public_submission_performed": False, "automatic_promotion": False}
    write_json(root / "comparison.json", result)
    pd.DataFrame(changes).to_csv(root / "paired_changes.csv", index=False, encoding="utf-8-sig")
    progress("report_complete", folds=folds, queries=result["queries"],
             protected_files_verified=result["protected_files_verified"],
             weighted_top1={v: {m: metrics[v][m]["weighted_top1"] for m in MODEL_NAMES} for v in EVAL_RECIPES})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "prepare-teachers", "run", "report"):
        p = sub.add_parser(command); p.add_argument("--output", required=True)
        if command == "run":
            p.add_argument("--fold", type=int, required=True, choices=range(5))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.output)
    elif args.command == "prepare-teachers":
        prepare_teacher_pool(args.output)
    elif args.command == "run":
        run_fold(args.output, args.fold)
    else:
        report(args.output)


if __name__ == "__main__":
    main()
