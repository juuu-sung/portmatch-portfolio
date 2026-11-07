"""Source-preserving weights for original and GT-masked ranking queries."""
import importlib.util
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from portmatch.ranking import RankerConfig, VesselRanker, query_sample_weights


def _original_weights(groups, languages, hangul_weight=.7):
    """Frozen calculation used before optional source masses were introduced."""
    groups = np.asarray(groups, dtype=np.int32)
    languages = np.asarray(languages, dtype=bool)
    query_weight = np.zeros(len(groups), dtype=np.float64)
    for language, mass in ((True, hangul_weight), (False, 1 - hangul_weight)):
        selected = languages == language
        count = int(selected.sum())
        if count:
            query_weight[selected] = mass / count
    query_weight /= query_weight.sum()
    weights = np.repeat(query_weight / groups, groups)
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def _source_layout(view_mask_counts):
    """Equal OOF view masses, with 0.5 original + 0.5 masked within each view."""
    groups, sources, languages, masses, originals = [], [], [], [], []
    for source, views in enumerate(view_mask_counts):
        for mask_count in views:
            groups.append(2 + (len(groups) * 7) % 23)
            sources.append(source)
            languages.append(source < 2)
            masses.append((.5 if mask_count else 1.) / len(views))
            originals.append(True)
            for _ in range(mask_count):
                groups.append(2 + (len(groups) * 7) % 23)
                sources.append(source)
                languages.append(source < 2)
                masses.append(.5 / len(views) / mask_count)
                originals.append(False)
    return tuple(np.asarray(values) for values in (groups, sources, languages, masses, originals))


class AugmentedWeightTests(unittest.TestCase):
    def test_omitting_mass_preserves_previous_float32_bytes(self):
        rng = np.random.default_rng(43)
        for languages in (np.array([True] * 17), np.array([False] * 17),
                          np.array([True, False] * 8 + [False])):
            groups = rng.integers(2, 300, len(languages), dtype=np.int32)
            expected = _original_weights(groups, languages)
            self.assertEqual(query_sample_weights(groups, languages).tobytes(), expected.tobytes())
            self.assertEqual(query_sample_weights(groups, languages, query_mass=None).tobytes(),
                             expected.tobytes())

    def test_source_mass_is_invariant_to_masks_views_and_candidate_counts(self):
        expected_source_share = np.array([.35, .35, .15, .15])
        layouts = (((1,), (3,), (0,), (3,)),
                   ((1, 0, 3), (3, 1), (0, 0, 0), (1, 3, 3)))
        for layout in layouts:
            groups, sources, languages, masses, originals = _source_layout(layout)
            np.testing.assert_allclose(np.bincount(sources, weights=masses), 1.)
            weights = query_sample_weights(groups, languages, query_mass=masses)
            query_weights = np.add.reduceat(weights, np.r_[0, np.cumsum(groups)[:-1]])
            fractions = query_weights / weights.sum()
            np.testing.assert_allclose(np.bincount(sources, weights=fractions),
                                       expected_source_share, rtol=2e-7)
            self.assertAlmostEqual(float(query_weights[languages].sum() / weights.sum()), .7,
                                   places=6)
            self.assertAlmostEqual(float(weights.mean()), 1., places=6)
            for source in range(4):
                selected = sources == source
                original_fraction = fractions[selected & originals].sum() / fractions[selected].sum()
                expected_original = sum(.5 if masks else 1. for masks in layout[source]) / len(layout[source])
                self.assertAlmostEqual(float(original_fraction), expected_original,
                                       places=6)

    def test_single_language_keeps_source_proportions_and_mass_scale_invariance(self):
        groups = np.array([2, 7, 3])
        languages = np.array([True, True, True])
        masses = np.array([.5, .25, .25])
        weights = query_sample_weights(groups, languages, query_mass=masses)
        fractions = np.add.reduceat(weights, [0, 2, 9]) / weights.sum()
        np.testing.assert_allclose(fractions, masses, rtol=1e-7)
        np.testing.assert_array_equal(weights,
            query_sample_weights(groups, languages, query_mass=masses * 1e308))
        self.assertTrue(np.isfinite(weights).all())

    def test_invalid_query_mass_is_rejected(self):
        for masses in ([], [1.], [[1., 2.]], [0., 1.], [-1., 1.], [np.nan, 1.],
                       [np.inf, 1.], [-np.inf, 1.], [None, 1.], ["invalid", 1.],
                       np.array([1. + 2.j, 1. + 0.j])):
            with self.subTest(masses=masses), self.assertRaisesRegex(ValueError, "query_mass"):
                query_sample_weights(np.array([2, 3]), np.array([True, False]), query_mass=masses)


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is not installed")
class AugmentedNativeRankerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = np.array([2, 3, 4, 2, 3, 4], dtype=np.int32)
        cls.languages = np.array([True, True, True, False, False, False])
        cls.masses = np.array([.5, .25, .25, .5, .25, .25])
        cls.X = np.concatenate([np.column_stack((np.r_[1., np.zeros(size - 1)],
                                                  np.linspace(0., 1., size)))
                                for size in cls.groups]).astype(np.float32)
        cls.y = np.concatenate([np.r_[1, np.zeros(size - 1)] for size in cls.groups]).astype(np.uint8)
        cls.names = ["name_ratio", "international_relative_gap"]
        cls.config = RankerConfig(n_estimators=5, min_child_samples=1, num_leaves=4, n_jobs=1)

    def test_fit_passes_mass_derived_row_weights_to_lightgbm(self):
        with patch("lightgbm.LGBMRanker") as estimator:
            VesselRanker(self.config).fit(self.X, self.y, self.groups, self.languages, self.names,
                                          query_mass=self.masses)
        passed = estimator.return_value.fit.call_args.kwargs
        np.testing.assert_array_equal(passed["sample_weight"], query_sample_weights(
            self.groups, self.languages, query_mass=self.masses))
        np.testing.assert_array_equal(passed["group"], self.groups)

    def test_weighted_native_model_save_restore_preserves_prediction(self):
        model = VesselRanker(self.config).fit(self.X, self.y, self.groups, self.languages,
                                              self.names, query_mass=self.masses)
        predictions = model.predict(self.X, self.names)
        for scores in np.split(predictions, np.cumsum(self.groups)[:-1]):
            self.assertEqual(int(scores.argmax()), 0)
        with TemporaryDirectory() as root:
            model.save(root)
            restored = VesselRanker.load(root)
            np.testing.assert_array_equal(predictions, restored.predict(self.X, self.names))

    def test_fit_rejects_bad_mass_before_constructing_estimator(self):
        with patch("lightgbm.LGBMRanker") as estimator:
            with self.assertRaisesRegex(ValueError, "query_mass"):
                VesselRanker(self.config).fit(self.X, self.y, self.groups, self.languages,
                                              self.names, query_mass=np.ones(len(self.groups) - 1))
            estimator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
