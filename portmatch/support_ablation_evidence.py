"""All-candidate input invariance and bounded evidence for 86-to-82 ablation."""
from __future__ import annotations

import csv
import json

import numpy as np

from .index_factorial_comparison import _cell
from .support_ablation_report import CONTRAST, EXPERIMENT, REFERENCE, RUNNERS, SUPPORT_FEATURES
from .text import has_hangul


def _schema(schema):
    names = {key: list(schema[key]) for key in
        ("baseline_features", "experiment_features", "excluded_features")}
    baseline, experiment, excluded = (names[key] for key in names)
    if (len(baseline) != 86 or len(set(baseline)) != 86
            or len(experiment) != 82 or len(set(experiment)) != 82
            or len(excluded) != 4 or set(excluded) != set(SUPPORT_FEATURES)
            or experiment != [name for name in baseline if name not in excluded]):
        raise ValueError("Feature schema must be 86 inputs minus exactly four support logs, preserving order")
    return names


def validate_shared_features(pair, schema):
    """Check every candidate, not only exported winners/targets, including NaNs."""
    names = _schema(schema)
    if set(pair) != set(RUNNERS):
        raise ValueError("Exactly two support-ablation runners are required")
    left, right = (pair[name][1] for name in RUNNERS)
    if (not left.index.is_unique or not right.index.is_unique
            or set(left.index) != set(right.index)):
        raise ValueError("Support-feature ablation altered the candidate set")
    for frame, key in ((left, "baseline_features"), (right, "experiment_features")):
        if not frame.columns.is_unique or not set(names[key]).issubset(frame.columns):
            raise ValueError("Actual model-input columns do not match the feature schema")
    if set(names["excluded_features"]) & set(right.columns):
        raise ValueError("The 82-feature evidence must contain only actual model inputs, not removed support logs")
    columns = names["experiment_features"]
    if not np.array_equal(left[columns].to_numpy(), right.loc[left.index, columns].to_numpy(), equal_nan=True):
        raise ValueError("Support-feature ablation altered shared 82 model inputs")
    return {"same_search_pairs": 1, "shared_feature_candidate_pairs": len(left)}


class SupportFeatureEvidenceWriter:
    """Stream target/two-winner evidence; explicit metadata distinguishes omitted inputs."""
    def __init__(self, output, schema):
        self.schema, self.streams = _schema(schema), []
        self.counts = {"candidate_feature_rows": 0, "candidate_comparison_rows": 0,
            "selected_candidate_groups": 0}
        identity = ["variant", "sample_id", "target_vessel_id", "vessel_name", "has_hangul",
            "candidate_vessel_id", "candidate_catalog_name", "is_target"]
        self.feature_writer = self._open(output / "candidate_features.csv", identity +
            ["model", "present", "is_top1", "score", "candidate_rank", "feature_count", "omitted_features",
             *self.schema["baseline_features"]])
        self.comparison_writer = self._open(output / "candidate_comparisons.csv", identity +
            ["contrast", "reference_runner", "experiment_runner", "reference_present", "experiment_present",
             "reference_score", "experiment_score", "score_delta", "reference_rank", "experiment_rank",
             "shared_feature_count", "shared_changed_features", "omitted_feature_count"])

    def _open(self, path, columns):
        stream = path.open("w", encoding="utf-8-sig", newline="")
        self.streams.append(stream)
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        return writer

    def close(self):
        for stream in self.streams:
            stream.close()

    def write(self, pair, row, variant):
        validate_shared_features(pair, self.schema)
        selected = sorted({str(row["target_vessel_id"]),
            *(str(record["predicted_vessel_id"]) for record, _ in pair.values())})
        for vessel in selected:
            self.counts["selected_candidate_groups"] += 1
            known_names = [details.at[vessel, "candidate_catalog_name"] for _, details in pair.values()
                if vessel in details.index]
            identity = {"variant": variant, "sample_id": str(row["sample_id"]),
                "target_vessel_id": str(row["target_vessel_id"]),
                "vessel_name": row["vessel_name"] if isinstance(row["vessel_name"], str) else "",
                "has_hangul": has_hangul(row["vessel_name"]), "candidate_vessel_id": vessel,
                "candidate_catalog_name": known_names[0] if known_names else None,
                "is_target": vessel == str(row["target_vessel_id"])}
            for runner, key in zip(RUNNERS, ("baseline_features", "experiment_features")):
                record, details = pair[runner]
                present = vessel in details.index
                value = details.loc[vessel] if present else None
                names = self.schema[key]
                omitted = self.schema["excluded_features"] if runner == EXPERIMENT else []
                self.feature_writer.writerow({**identity, "model": runner, "present": present,
                    "is_top1": vessel == str(record["predicted_vessel_id"]),
                    "score": _cell(value["score"]) if present else None,
                    "candidate_rank": int(value["candidate_rank"]) if present else None,
                    "feature_count": len(names), "omitted_features": json.dumps(omitted),
                    **{name: _cell(value[name]) if present and name in names else None
                       for name in self.schema["baseline_features"]}})
                self.counts["candidate_feature_rows"] += 1
            left, right = pair[REFERENCE][1], pair[EXPERIMENT][1]
            lp, rp = vessel in left.index, vessel in right.index
            self.comparison_writer.writerow({**identity, "contrast": CONTRAST,
                "reference_runner": REFERENCE, "experiment_runner": EXPERIMENT,
                "reference_present": lp, "experiment_present": rp,
                "reference_score": _cell(left.at[vessel, "score"]) if lp else None,
                "experiment_score": _cell(right.at[vessel, "score"]) if rp else None,
                "score_delta": _cell(right.at[vessel, "score"] - left.at[vessel, "score"]) if lp and rp else None,
                "reference_rank": int(left.at[vessel, "candidate_rank"]) if lp else None,
                "experiment_rank": int(right.at[vessel, "candidate_rank"]) if rp else None,
                "shared_feature_count": len(self.schema["experiment_features"]),
                "shared_changed_features": 0 if lp and rp else None,
                "omitted_feature_count": len(self.schema["excluded_features"])})
            self.counts["candidate_comparison_rows"] += 1
