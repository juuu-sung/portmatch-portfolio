"""Audit saved compound results, report paired changes and vessel uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portmatch.compound_experiment import (MODEL_NAMES, _group_meta, check_protocol, read_json, sha, write_json)
from portmatch.compound_protocol import EVAL_RECIPES
from portmatch.compound_mining import expanded_masses
from portmatch.ranking import ranking_metrics

PAIRS = (("baseline", "expanded"), ("baseline", "expanded_mined"), ("expanded", "expanded_mined"))


def analyze(root, replicates=5000):
    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    root = Path(root).resolve()
    frozen = check_protocol(root, protection=True)
    comparison = read_json(root / "comparison.json")
    folds = comparison["complete_for_folds"]
    train = pd.read_csv(frozen["data"]["train.csv"]["path"])
    catalog = pd.read_csv(frozen["data"]["vessel_catalog.csv"]["path"])
    catalog_names = catalog.set_index("vessel_id").vessel_name.to_dict()
    query_names = train.set_index("sample_id").vessel_name.to_dict()
    targets = train.set_index("sample_id").target_vessel_id.to_dict()
    records, audit = [], []
    for fold in folds:
        output = root / f"fold{fold}"
        if not read_json(output / "status.json")["complete"]:
            raise ValueError("Incomplete fold")
        cache = Path(frozen["baselines"][str(fold)]["cache"])
        partitions = pd.read_csv(cache / "partitions.csv").set_index("sample_id")
        training_sources = set(partitions.index[partitions.role.eq("train")])
        validation_sources = set(partitions.index[partitions.role.eq("valid")])
        validation_vessels = set(partitions.loc[list(validation_sources), "target_vessel_id"])
        assert not training_sources & validation_sources
        assert not {targets[s] for s in training_sources} & validation_vessels
        inner_proofs = read_json(output / "inner_partitions.json")
        assert {(p["view"], p["inner"]) for p in inner_proofs} == {(v, i) for v in ("vessel", "row") for i in range(3)}
        assert len(inner_proofs) == 6
        for proof in inner_proofs:
            expected_held = set(partitions.index[partitions.role.eq("train") & partitions["inner_"+proof["view"]].eq(proof["inner"])])
            fitting, held = set(proof["fitting_sample_ids"]), set(proof["held_sample_ids"])
            assert held == expected_held
            assert fitting == training_sources - expected_held
            if proof["view"] == "vessel":
                assert not {targets[s] for s in fitting} & {targets[s] for s in held}
        predictions = pd.read_csv(output / "predictions.csv")
        assert set(predictions.sample_id) == validation_sources
        assert set(predictions.model) == set(MODEL_NAMES)
        assert set(predictions.variant) == set(EVAL_RECIPES)
        assert not predictions.duplicated(["sample_id", "variant", "model"]).any()
        assert len(predictions) == len(validation_sources)*len(MODEL_NAMES)*len(EVAL_RECIPES)
        assert predictions.target_vessel_id.equals(predictions.sample_id.map(targets))
        assert predictions.predicted_vessel_id.isin(catalog.vessel_id).all()
        for _, same_query in predictions.groupby(["sample_id", "variant"]):
            assert same_query.candidate_count.nunique() == 1
            assert same_query.candidate_found.nunique() == 1
        extra = pd.read_csv(output / "extra_groups.csv")
        assert set(extra.source).issubset(training_sources)
        assert extra["query"].is_unique
        assert np.array_equal(extra.teacher_fold.to_numpy(), extra.source.map(partitions.inner_vessel).to_numpy())
        base_groups = _group_meta(cache)
        assert set(base_groups.source).issubset(training_sources)
        base_mass, extra_mass = expanded_masses(base_groups.source, base_groups.variant, base_groups.mass, extra.source)
        totals = pd.DataFrame({"source": np.r_[base_groups.source, extra.source], "mass": np.r_[base_mass, extra_mass]}).groupby("source").mass.sum()
        assert np.allclose(totals, 1, atol=1e-10, rtol=0)
        for model in MODEL_NAMES[1:]:
            summary = read_json(output / model / "compound_model.json")
            assert summary["sources"] == len(totals)
            assert summary["pairs"] == int(base_groups.group.sum() + extra.group.sum())
            assert summary["groups"] == len(base_groups) + len(extra)
            assert summary["base_cache_metadata_sha256"] == sha(cache / "metadata.json")
            model_state = read_json(output / model / "ranker.json")
            baseline_state = read_json(Path(frozen["baselines"][str(fold)]["directory"]) / "ranker.json")
            for field in ("artifact_version", "lightgbm_version", "config", "feature_names"):
                assert model_state[field] == baseline_state[field]
            assert model_state["model_sha256"] == sha(output / model / "model.txt")
        checks = []
        for inner in range(3):
            proof = read_json(output / f"teachers/{inner}/crossfit.json")
            fitting_ids = set(proof["training_sample_ids"])
            fitting_vessels = set(proof["training_vessel_ids"])
            excluded = set(proof["excluded_vessel_ids"])
            assert fitting_ids.issubset(training_sources)
            assert not fitting_vessels & (excluded | validation_vessels)
            assert fitting_vessels == {targets[s] for s in fitting_ids}
            assert len(proof["feature_names"]) == 54
            assert not any(n.startswith(("flag_", "type_", "retrieval_")) for n in proof["feature_names"])
            checks.append({"inner": inner, "fitting_vessels": len(fitting_vessels), "excluded_vessels": len(excluded)})
        manifest = read_json(output / "extra_manifest.json")
        seen = []
        for relative, expected in manifest["files"].items():
            assert sha(output / "extra" / relative) == expected
            with np.load(output / "extra" / relative, allow_pickle=False) as z:
                offset = 0
                for i, n in enumerate(z["union_group"]):
                    n = int(n); source = str(z["source"][i]); section = slice(offset, offset+n)
                    ids = z["catalog_index"][section]; labels = z["y"][section]
                    assert len(set(ids.tolist())) == len(ids)
                    assert np.isin(labels, [0,1]).all()
                    assert np.array_equal(labels, catalog.iloc[ids].vessel_id.eq(targets[source]).to_numpy().astype(np.uint8))
                    assert labels.sum() == 1
                    counts = []
                    for selection in ("heuristic", "mined"):
                        mask = z[selection][section]
                        assert labels[mask].sum() == 1
                        counts.append(int(mask.sum()))
                    assert counts[0] == counts[1]
                    seen.append((source, str(z["query"][i]), counts[0]))
                    offset += n
        assert seen == list(zip(extra.source, extra["query"], extra.group))
        records.append(predictions.assign(fold=fold))
        audit.append({"fold": fold, "validation_queries": len(validation_sources),
                      "extra_groups_checked": len(seen), "crossfit": checks,
                      "inner_partitions_checked": len(inner_proofs), "source_mass_totals_checked": len(totals),
                      "native_model_hashes": {name: sha(output / name / "model.txt") for name in MODEL_NAMES[1:]},
                      "predictions_sha256": sha(output / "predictions.csv")})

    predictions = pd.concat(records, ignore_index=True)
    assert predictions[predictions.model.eq("baseline") & predictions.variant.eq("original")].sample_id.is_unique
    baseline_queries = predictions[predictions.model.eq("baseline") & predictions.variant.eq("original")]
    # Each vessel is one resampling unit, including vessels with both languages.
    vessel_ids = sorted(baseline_queries.target_vessel_id.unique())
    denominator = baseline_queries.assign(h=baseline_queries.has_hangul.astype(int),
                                         e=(~baseline_queries.has_hangul).astype(int))
    counts = denominator.groupby("target_vessel_id")[["h", "e"]].sum().reindex(vessel_ids).to_numpy(float)
    deltas, changes, metrics_rows, comparisons = [], [], [], []
    for variant in EVAL_RECIPES:
        part = predictions[predictions.variant.eq(variant)]
        for model in MODEL_NAMES:
            m = ranking_metrics(part[part.model.eq(model)])
            assert m == comparison["metrics"][variant][model]
            metrics_rows.append({"variant": variant, "model": model, "weighted_top1": m["weighted_top1"],
                                 "weighted_candidate_recall": m["weighted_candidate_recall"],
                                 "hangul_correct": m["hangul"]["correct_top1"], "hangul_queries": m["hangul"]["queries"],
                                 "non_hangul_correct": m["non_hangul"]["correct_top1"], "non_hangul_queries": m["non_hangul"]["queries"]})
        wide = part.pivot(index=["fold", "sample_id", "target_vessel_id", "has_hangul"], columns="model", values="predicted_vessel_id").reset_index()
        for reference, experiment in PAIRS:
            before, after = wide[reference].eq(wide.target_vessel_id), wide[experiment].eq(wide.target_vessel_id)
            diff = after.astype(int)-before.astype(int)
            d = wide.assign(hd=diff*wide.has_hangul, ed=diff*(~wide.has_hangul))
            deltas.append(d.groupby("target_vessel_id")[["hd", "ed"]].sum().reindex(vessel_ids, fill_value=0).to_numpy(float))
            comparisons.append((variant, reference, experiment))
            for _, row in wide.loc[wide[reference].ne(wide[experiment])].iterrows():
                b, a = row[reference] == row.target_vessel_id, row[experiment] == row.target_vessel_id
                changes.append({"fold": int(row.fold), "variant": variant, "reference": reference, "experiment": experiment,
                    "sample_id": row.sample_id, "target_vessel_id": row.target_vessel_id,
                    "query_name": query_names[row.sample_id], "target_name": catalog_names[row.target_vessel_id],
                    "has_hangul": bool(row.has_hangul), "reference_prediction": catalog_names[row[reference]],
                    "experiment_prediction": catalog_names[row[experiment]],
                    "status": "corrected" if not b and a else "regressed" if b and not a else "still_wrong"})
    matrix = np.concatenate([counts, *deltas], axis=1)
    samples = []
    rng = np.random.default_rng(20261030)
    accepted = 0
    while accepted < replicates:
        value = rng.poisson(1, size=(min(100, replicates-accepted), len(vessel_ids))) @ matrix
        value = value[(value[:,0] > 0) & (value[:,1] > 0)]
        if not len(value):
            continue
        samples.append(100*(.7*value[:,2::2]/value[:,0,None]+.3*value[:,3::2]/value[:,1,None]))
        accepted += len(value)
    samples = np.concatenate(samples)
    intervals = []
    for index, (variant, reference, experiment) in enumerate(comparisons):
        intervals.append({"variant": variant, "reference": reference, "experiment": experiment,
            "delta_pp": 100*(comparison["metrics"][variant][experiment]["weighted_top1"]-comparison["metrics"][variant][reference]["weighted_top1"]),
            "vessel_bootstrap_95pct_pp": np.quantile(samples[:,index], [.025,.975]).tolist()})
    pd.DataFrame(metrics_rows).to_csv(root / "metrics.csv", index=False, encoding="utf-8-sig")
    changed = pd.DataFrame(changes)
    changed.to_csv(root / "changed_rows.csv", index=False, encoding="utf-8-sig")
    if len(changed):
        changed.groupby(["variant", "reference", "experiment", "target_vessel_id", "target_name", "status"]).size().reset_index(name="rows").to_csv(root / "vessel_changes.csv", index=False, encoding="utf-8-sig")
    write_json(root / "confidence_intervals.json", {"replicates": replicates, "method": "Paired Poisson(1) vessel-cluster bootstrap", "seed": 20261030,
        "caveat": "Descriptive development-set uncertainty, not correction for model selection or hidden-test distribution shift", "comparisons": intervals})
    write_json(root / "analysis_audit.json", {"complete": True, "folds": folds, "per_fold": audit,
        "protected_files_verified": len(frozen["protected_files"]), "predictions": len(predictions), "vessels": len(vessel_ids),
        "analysis_code_sha256": sha(Path(__file__)), "old_models_and_submissions_unchanged": True})
    print(json.dumps({"audit_complete": True, "folds": folds, "predictions": len(predictions), "vessels": len(vessel_ids)}, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    analyze(args.root, args.replicates)
