"""Small synthetic retrieval regressions; no competition-wide evaluation."""
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from portmatch.memory import TrainingMemory
from portmatch.retrieval import CandidateRetriever, RetrievalConfig


def make_catalog(names, tonnages=None, ids=None):
    return pd.DataFrame({
        "vessel_id": ids if ids is not None else [f"V{i:04}" for i in range(len(names))],
        "vessel_name": names,
        "gross_tonnage": tonnages if tonnages is not None else [np.nan] * len(names),
    })


def empty_memory(catalog):
    train = pd.DataFrame(columns=["vessel_name", "call_sign", "target_vessel_id"])
    return TrainingMemory().fit(train, catalog)


class RetrievalTests(unittest.TestCase):
    def test_broad_tonnage_window_retains_target_beyond_nearest_twelve(self):
        # All 301 candidates are in the measured noise window. The true target
        # is 8% away and would disappear under the old nearest-12 strategy.
        catalog = make_catalog(
            [f"DECOY {i}" for i in range(300)] + ["TRUE TARGET"],
            list(np.linspace(995, 1005, 300)) + [1080],
        )
        retriever = CandidateRetriever(catalog, empty_memory(catalog))
        result = retriever.retrieve({"vessel_name": "", "international_gross_tonnage": 1000})
        self.assertEqual(len(result), 301)
        target = result.set_index("catalog_index").loc[300]
        self.assertEqual(target.tonnage_hit, 1)
        self.assertGreater(target.tonnage_rank, 12)
        self.assertEqual(len(result.catalog_index.unique()), len(result))

    def test_window_boundary_and_gross_fallback(self):
        catalog = make_catalog(["A", "B", "C"], [1000, 2000, 0])
        retriever = CandidateRetriever(catalog, empty_memory(catalog))
        boundary = retriever.retrieve({"vessel_name": "", "international_gross_tonnage": 1100})
        self.assertEqual(boundary.catalog_index.tolist(), [0])
        fallback = retriever.retrieve({
            "vessel_name": "", "international_gross_tonnage": 1_000_000,
            "gross_tonnage": 2000,
        })
        self.assertEqual(fallback.catalog_index.tolist(), [1])
        self.assertEqual(fallback.tonnage_hit.tolist(), [1])

    def test_numeric_fallback_and_missing_values_are_finite(self):
        catalog = make_catalog(["A", "B", "C"], [0, np.nan, 1000])
        retriever = CandidateRetriever(catalog, empty_memory(catalog))
        far = retriever.retrieve({"vessel_name": None, "international_gross_tonnage": 10_000})
        self.assertEqual(far.catalog_index.tolist(), [2])
        missing = retriever.retrieve({"vessel_name": None, "gross_tonnage": -1})
        self.assertTrue(np.isfinite(missing.to_numpy()).all())
        self.assertEqual(missing.tonnage_hit.sum(), 0)

    def test_training_alias_is_indexed_only_when_supplied(self):
        catalog = make_catalog(["ZZZZZZ", "BBBBBB"])
        train = pd.DataFrame([{
            "vessel_name": "ALPHA DREAM", "call_sign": "TRAIN CALL",
            "target_vessel_id": "V0000",
        }])
        config = RetrievalConfig(edit_top_k=0, phonetic_top_k=0)
        with_alias = CandidateRetriever(catalog, TrainingMemory().fit(train, catalog), config)
        without_alias = CandidateRetriever(catalog, empty_memory(catalog), config)
        # A typo is not an exact memory hit; it must come through the alias index.
        query = {"vessel_name": "ALPHA DREAN", "call_sign": "NEW CALL"}
        hit = with_alias.retrieve(query).set_index("catalog_index").loc[0]
        self.assertEqual(hit.name_hit, 1)
        self.assertEqual(hit.alias_match, 0)
        held_out = without_alias.retrieve(query)
        self.assertEqual(held_out.name_hit.sum(), 0)
        self.assertEqual(held_out.alias_match.sum(), 0)
        self.assertTrue(without_alias.memory.aliases.empty)

    def test_romanized_korean_query_has_phonetic_candidate(self):
        catalog = make_catalog(["HANBIT", "OCEAN STAR"])
        config = RetrievalConfig(name_top_k=0, phonetic_top_k=1, edit_top_k=0)
        retriever = CandidateRetriever(catalog, empty_memory(catalog), config)
        result = retriever.retrieve({"vessel_name": "한빛"})
        self.assertEqual(result.catalog_index.tolist(), [0])
        self.assertEqual(result.phonetic_hit.tolist(), [1])
        self.assertGreater(result.phonetic_score.iloc[0], 0)

    def test_edit_distance_recovers_no_bigram_match(self):
        catalog = make_catalog(["AB", "ZZZZ"])
        config = RetrievalConfig(name_top_k=0, phonetic_top_k=0, edit_top_k=1)
        retriever = CandidateRetriever(catalog, empty_memory(catalog), config)
        result = retriever.retrieve({"vessel_name": "AC"})
        self.assertEqual(result.catalog_index.tolist(), [0])
        self.assertEqual(result.edit_hit.tolist(), [1])
        self.assertAlmostEqual(result.edit_score.iloc[0], 0.5)

    def test_ties_and_query_execution_order_are_deterministic(self):
        catalog = make_catalog(["ALPHAZ", "ALPHAY"], [1000, 1000], ["Z", "A"])
        config = RetrievalConfig(name_top_k=1, phonetic_top_k=0, edit_top_k=0)
        retriever = CandidateRetriever(catalog, empty_memory(catalog), config)
        query = {"vessel_name": "ALPHAX", "international_gross_tonnage": 1000}
        first = retriever.retrieve(query)
        retriever.retrieve({"vessel_name": "UNRELATED", "international_gross_tonnage": 50})
        second = retriever.retrieve(query)
        assert_frame_equal(first, second, check_exact=True)
        ordered = first.set_index("catalog_index")
        self.assertEqual(ordered.loc[1, "name_rank"], 1)
        self.assertEqual(ordered.loc[1, "tonnage_rank"], 1)
        self.assertEqual(ordered.loc[0, "tonnage_rank"], 2)

    def test_explicit_cap_never_discards_exact_lookup_candidates(self):
        catalog = make_catalog(["A", "B", "C"], [1000, 1000, 1000])
        train = pd.DataFrame([
            {"vessel_name": "ALIAS ONE", "call_sign": "SHARED", "target_vessel_id": "V0000"},
            {"vessel_name": "ALIAS TWO", "call_sign": "SHARED", "target_vessel_id": "V0001"},
        ])
        retriever = CandidateRetriever(catalog, TrainingMemory().fit(train, catalog),
                                       RetrievalConfig(max_total=1))
        result = retriever.retrieve({"call_sign": "SHARED", "international_gross_tonnage": 1000})
        self.assertEqual(result.catalog_index.tolist(), [0, 1])
        self.assertEqual(result.call_match.tolist(), [1, 1])
        self.assertEqual(result.lookup_purity.tolist(), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
