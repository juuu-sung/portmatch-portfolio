"""Synthetic checks for retrieval expansion, independence and provenance."""
from dataclasses import asdict
import io
import unittest

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from portmatch.memory import TrainingMemory
from portmatch.pipeline import MatchingPipeline
from portmatch.retrieval import CandidateRetriever, RetrievalConfig, RESULT_COLUMNS
from portmatch.robust_retrieval import (
    RobustCandidateRetriever, RobustMatchingPipeline, RobustRetrievalConfig,
    build_pipeline, check_retrieval_metadata, retrieval_spec,
)


def catalog(names, tonnage=None, ids=None):
    return pd.DataFrame({
        "vessel_id": ids or [f"V{i:04}" for i in range(len(names))],
        "vessel_name": names,
        "gross_tonnage": tonnage if tonnage is not None else [np.nan] * len(names),
        "flag_name": [""] * len(names), "vessel_type": [""] * len(names),
    })


def training(rows=()):
    return pd.DataFrame(rows, columns=["vessel_name", "call_sign", "target_vessel_id",
                                       "flag_name", "vessel_type"])


def retriever(data, config=None, train=None):
    memory = TrainingMemory().fit(training() if train is None else train, data)
    return RobustCandidateRetriever(data, memory, config)


def by_id(result, data):
    result = result.copy()
    result["vessel_id"] = data.vessel_id.iloc[result.catalog_index].to_numpy()
    return result.drop(columns="catalog_index").set_index("vessel_id").sort_index()


class RobustRetrievalTests(unittest.TestCase):
    def test_context_recovers_hangul_without_tonnage_or_legacy_searches(self):
        data = catalog(["CHAMPIONS", "CHAMPION", "DREAM"])
        config = RobustRetrievalConfig(profile="hangul", name_top_k=0,
            phonetic_top_k=0, edit_top_k=0, context_top_k=1,
            vowel_top_k=0, context_edit_top_k=0, numeric_fallback_k=1)
        result = retriever(data, config).retrieve({"vessel_name": "챔피언"})
        target = result.set_index("catalog_index").loc[1]
        self.assertEqual(target.phonetic_hit, 1)
        self.assertAlmostEqual(target.phonetic_score, 1.0, places=6)
        self.assertEqual(target.tonnage_hit, 0)
        self.assertEqual(target.phonetic_rank, 1)

    def test_vowel_search_adds_candidate_without_tonnage(self):
        data = catalog(["ZZZZ", "AB"])
        config = RobustRetrievalConfig(profile="hangul", name_top_k=0,
            phonetic_top_k=0, edit_top_k=0, context_top_k=0,
            vowel_top_k=1, context_edit_top_k=0, numeric_fallback_k=1)
        result = retriever(data, config).retrieve({"vessel_name": "아비"})
        self.assertEqual(result.set_index("catalog_index").loc[1, "phonetic_hit"], 1)
        self.assertEqual(result.tonnage_hit.sum(), 0)

    def test_edit_search_recovers_short_view_without_shared_bigram(self):
        data = catalog(["QQQQ", "ACE"])
        config = RobustRetrievalConfig(profile="hangul", name_top_k=0,
            phonetic_top_k=0, edit_top_k=0, context_top_k=0,
            vowel_top_k=0, context_edit_top_k=1, numeric_fallback_k=1)
        result = retriever(data, config).retrieve({"vessel_name": "에이스"})
        self.assertEqual(result.set_index("catalog_index").loc[1, "phonetic_score"], 1)

    def test_both_tonnage_windows_are_kept_when_international_already_hits(self):
        data = catalog(["ONE", "TWO", "THREE"], [1000, 2000, 3000])
        config = RobustRetrievalConfig(profile="dual-gt", name_top_k=0,
                                       phonetic_top_k=0, edit_top_k=0)
        row = {"international_gross_tonnage": 1000, "gross_tonnage": 2000}
        result = retriever(data, config).retrieve(row)
        self.assertEqual(result.catalog_index.tolist(), [0, 1])
        self.assertEqual(result.tonnage_hit.tolist(), [1, 1])
        swapped = {"international_gross_tonnage": 2000, "gross_tonnage": 1000}
        assert_frame_equal(result, retriever(data, config).retrieve(swapped), check_exact=True)

    def test_numeric_fallback_union_keeps_legacy_gross_window(self):
        data = catalog(["ONE", "TWO", "THREE"], [1000, 2000, 3000])
        config = RobustRetrievalConfig(profile="dual-gt", name_top_k=0,
            phonetic_top_k=0, edit_top_k=0, numeric_fallback_k=1)
        result = retriever(data, config).retrieve({
            "international_gross_tonnage": 1_000_000, "gross_tonnage": 1000,
        })
        self.assertEqual(result.catalog_index.tolist(), [0, 2])
        self.assertEqual(result.tonnage_hit.tolist(), [1, 1])

    def test_missing_invalid_or_equal_second_tonnage_is_exact_legacy(self):
        data = catalog(["ALPHA", "BETA", "GAMMA"], [1000, 1000, 3000])
        legacy = CandidateRetriever(data, TrainingMemory().fit(training(), data))
        extended = retriever(data, RobustRetrievalConfig(profile="dual-gt"))
        for gross in (None, np.nan, -10, 0, float("inf"), "", 1000):
            row = {"vessel_name": "ALPA", "international_gross_tonnage": 1000,
                   "gross_tonnage": gross}
            assert_frame_equal(legacy.retrieve(row), extended.retrieve(row), check_exact=True)

    def test_disabled_extensions_and_nonhangul_profile_are_exact_legacy(self):
        data = catalog(["CHAMPION", "CHAMPIONS"], [1000, 2000])
        memory = TrainingMemory().fit(training(), data)
        legacy = CandidateRetriever(data, memory)
        disabled = retriever(data, RobustRetrievalConfig(profile="hangul",
            context_top_k=0, vowel_top_k=0, context_edit_top_k=0))
        for name in ("챔피언", "CHAMPION", "", None):
            row = {"vessel_name": name, "gross_tonnage": 2000,
                   "international_gross_tonnage": 1000}
            assert_frame_equal(legacy.retrieve(row), disabled.retrieve(row), check_exact=True)
        row = {"vessel_name": "CHAMPION", "gross_tonnage": 2000,
               "international_gross_tonnage": 1000}
        assert_frame_equal(legacy.retrieve(row), retriever(data,
            RobustRetrievalConfig(profile="hangul")).retrieve(row), check_exact=True)

    def test_legacy_candidate_union_and_schema_survive_expansion(self):
        data = catalog(["CHAMPION", "ALPHA", "BETA", "CHAMPIONS"],
                       [1000, 2000, 3000, 4000])
        memory = TrainingMemory().fit(training(), data)
        legacy = CandidateRetriever(data, memory, RetrievalConfig(
            name_top_k=1, phonetic_top_k=1, edit_top_k=1))
        extended = retriever(data, RobustRetrievalConfig(
            name_top_k=1, phonetic_top_k=1, edit_top_k=1))
        row = {"vessel_name": "챔피언", "gross_tonnage": 3000,
               "international_gross_tonnage": 2000}
        before, after = legacy.retrieve(row), extended.retrieve(row)
        self.assertTrue(set(before.catalog_index).issubset(set(after.catalog_index)))
        self.assertEqual(after.columns.tolist(), RESULT_COLUMNS)
        self.assertFalse(after.catalog_index.duplicated().any())
        self.assertTrue(np.isfinite(after.to_numpy()).all())
        self.assertTrue(after.phonetic_score.between(0, 1).all())

    def test_alias_index_only_uses_provided_training_and_never_query_labels(self):
        data = catalog(["ZZZZ", "CHAMPIONS"])
        train = training([("CHAMPION", "TRAIN CALL", "V0000", "", "")])
        config = RobustRetrievalConfig(profile="hangul", name_top_k=0,
            phonetic_top_k=0, edit_top_k=0, context_top_k=1,
            vowel_top_k=0, context_edit_top_k=0, numeric_fallback_k=1)
        row = {"vessel_name": "챔피언", "call_sign": "UNSEEN"}
        supplied = retriever(data, config, train)
        withheld = retriever(data, config)
        self.assertEqual(supplied.retrieve(row).set_index("catalog_index").loc[0, "phonetic_hit"], 1)
        self.assertNotIn(0, withheld.retrieve(row).query("phonetic_hit == 1").catalog_index.tolist())
        before = withheld.retrieve(row)
        withheld.retrieve({**row, "target_vessel_id": "V0000", "sample_id": "LEAK"})
        assert_frame_equal(before, withheld.retrieve(row), check_exact=True)
        self.assertTrue(withheld.memory.aliases.empty)

    def test_catalog_query_order_and_joblib_roundtrip_are_independent(self):
        data = catalog(["CHAMPIONS", "CHAMPION", "CHAMPION", "DREAM"],
                       [2000, 1000, 1000, 3000], ["Z", "B", "A", "X"])
        config = RobustRetrievalConfig(context_top_k=1, vowel_top_k=1,
                                       context_edit_top_k=1)
        original = retriever(data, config)
        query = {"vessel_name": "챔피언", "international_gross_tonnage": 1000,
                 "gross_tonnage": 2000}
        expected = original.retrieve(query)
        original.retrieve({"vessel_name": "기타", "gross_tonnage": 5000})
        assert_frame_equal(expected, original.retrieve(query), check_exact=True)
        reordered = data.iloc[[3, 2, 0, 1]].reset_index(drop=True)
        assert_frame_equal(by_id(expected, data),
                           by_id(retriever(reordered, config).retrieve(query), reordered),
                           check_exact=True)
        stream = io.BytesIO()
        joblib.dump(original, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        assert_frame_equal(expected, restored.retrieve(query), check_exact=True)

    def test_explicit_cap_preserves_exact_memory_candidates(self):
        data = catalog(["CHAMPION", "CHAMPIONS", "DREAM"], [1000, 2000, 3000])
        train = training([("ONE", "SAME", "V0000", "", ""),
                          ("TWO", "SAME", "V0001", "", "")])
        result = retriever(data, RobustRetrievalConfig(max_total=1), train).retrieve({
            "vessel_name": "챔피언", "call_sign": "SAME", "gross_tonnage": 3000,
        })
        self.assertEqual(result.catalog_index.tolist(), [0, 1])
        self.assertEqual(result.call_match.tolist(), [1, 1])

    def test_metadata_checks_and_pipeline_factory(self):
        data = catalog(["CHAMPION", "DREAM"])
        train = training()
        config = RobustRetrievalConfig(profile="hangul")
        pipeline = build_pipeline(train, data, config)
        self.assertIsInstance(pipeline, RobustMatchingPipeline)
        metadata = {"retrieval": asdict(config), "retrieval_extension": retrieval_spec(config)}
        check_retrieval_metadata(metadata, pipeline=pipeline)
        check_retrieval_metadata({}, pipeline=MatchingPipeline(train, data))
        self.assertIsNone(retrieval_spec(RetrievalConfig()))
        self.assertIsInstance(build_pipeline(train, data), MatchingPipeline)
        self.assertIsInstance(build_pipeline(train, data, extension=retrieval_spec(config)),
                              RobustMatchingPipeline)
        for invalid in ({}, {"retrieval": asdict(config)},
                        {**metadata, "retrieval_extension": {"name": "wrong"}},
                        {**metadata, "retrieval": {**asdict(config), "context_top_k": 1}}):
            with self.assertRaises(ValueError):
                check_retrieval_metadata(invalid, pipeline=pipeline)
        with self.assertRaises(ValueError):
            check_retrieval_metadata(metadata, pipeline=MatchingPipeline(train, data))
        with self.assertRaises(ValueError):
            build_pipeline(train, data, RetrievalConfig(name_top_k=1),
                           extension=retrieval_spec(config))

    def test_new_configuration_rejects_invalid_profiles_and_limits(self):
        for kwargs in ({"profile": "unknown"}, {"context_top_k": -1},
                       {"vowel_top_k": 1.5}, {"context_edit_top_k": True}):
            with self.assertRaises(ValueError):
                RobustRetrievalConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
