from pathlib import Path

import pandas as pd


QUERY_COLUMNS = ["sample_id", "vessel_name", "call_sign", "gross_tonnage", "international_gross_tonnage", "flag_name", "vessel_type", "arrival_year"]
CATALOG_COLUMNS = ["vessel_id", "vessel_name", "vessel_type", "size", "unit", "deadweight_tonnage", "gross_tonnage", "flag_name"]


def read_catalog(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"vessel_id": str, "vessel_name": str})
    missing = set(CATALOG_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Catalog columns missing: {sorted(missing)}")
    if df.vessel_id.isna().any() or not df.vessel_id.is_unique:
        raise ValueError("Catalog vessel_id must be non-null and unique")
    return df.reset_index(drop=True)


def read_queries(path: str | Path, labeled=False) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"sample_id": str, "vessel_name": str, "call_sign": str, "target_vessel_id": str})
    required = set(QUERY_COLUMNS) | ({"target_vessel_id"} if labeled else set())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Query columns missing: {sorted(missing)}")
    if df.sample_id.isna().any() or not df.sample_id.is_unique:
        raise ValueError("sample_id must be non-null and unique")
    if labeled and df.target_vessel_id.isna().any():
        raise ValueError("Training target_vessel_id must be non-null")
    return df.reset_index(drop=True)
