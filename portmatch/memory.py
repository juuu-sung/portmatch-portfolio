"""Lookup state derived exclusively from the training partition supplied to fit."""

from collections import Counter, defaultdict

import pandas as pd

from .text import normalize_name


LOOKUP_COLUMNS = ["catalog_index", "call_match", "alias_match", "lookup_support", "lookup_purity"]


def call_key(value: object) -> str:
    # Anonymous call signs support equality, not fuzzy or substring matching.
    return value.strip() if isinstance(value, str) else ""


class TrainingMemory:
    def __init__(self):
        self.call_signs = {}
        self.names = {}
        self.aliases = pd.DataFrame(columns=["vessel_name", "catalog_index"])
        self.training_rows = 0
        self.training_vessels = frozenset()

    def fit(self, train: pd.DataFrame, catalog: pd.DataFrame):
        required = {"call_sign", "vessel_name", "target_vessel_id"}
        if not required.issubset(train.columns):
            raise ValueError(f"Training columns missing: {sorted(required - set(train.columns))}")
        if catalog.vessel_id.isna().any() or not catalog.vessel_id.is_unique:
            raise ValueError("Catalog vessel_id must be present and unique")
        id_to_index = {str(v): i for i, v in enumerate(catalog.vessel_id)}
        unknown = set(train.target_vessel_id.astype(str)) - set(id_to_index)
        if unknown:
            raise ValueError(f"Training has {len(unknown)} targets absent from the catalog")
        calls, names = defaultdict(Counter), defaultdict(Counter)
        aliases = []
        for row in train.to_dict("records"):
            index = id_to_index[str(row["target_vessel_id"])]
            call, name = call_key(row["call_sign"]), normalize_name(row["vessel_name"])
            if call:
                calls[call][index] += 1
            if name:
                names[name][index] += 1
                aliases.append((row["vessel_name"], index))
        self.call_signs, self.names = dict(calls), dict(names)
        self.aliases = pd.DataFrame(aliases, columns=["vessel_name", "catalog_index"]).drop_duplicates().reset_index(drop=True)
        self.training_rows = len(train)
        self.training_vessels = frozenset(train.target_vessel_id.astype(str))
        return self

    def lookup(self, row: dict) -> pd.DataFrame:
        call = self.call_signs.get(call_key(row.get("call_sign")), {})
        alias = self.names.get(normalize_name(row.get("vessel_name")), {})
        call_total, alias_total = sum(call.values()), sum(alias.values())
        records = []
        for index in sorted(set(call) | set(alias)):
            records.append({
                "catalog_index": index,
                "call_match": float(index in call),
                "alias_match": float(index in alias),
                # Max avoids double-counting the same supporting training rows.
                "lookup_support": max(call.get(index, 0), alias.get(index, 0)),
                "lookup_purity": max(call.get(index, 0) / call_total if call_total else 0,
                                     alias.get(index, 0) / alias_total if alias_total else 0),
            })
        return pd.DataFrame(records, columns=LOOKUP_COLUMNS)
