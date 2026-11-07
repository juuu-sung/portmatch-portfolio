"""Select cached feature columns without changing retrieval or sampled pairs."""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase

import numpy as np


def _names(values, description):
    if isinstance(values, str):
        raise ValueError(f"{description} must be a sequence of feature names")
    names = tuple(values)
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError(f"{description} must be nonempty, unique feature names")
    return names


@dataclass(frozen=True)
class FeatureSelection:
    source_feature_names: tuple[str, ...]
    selected_features: tuple[str, ...]
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()

    def __post_init__(self):
        source = _names(self.source_feature_names, "Source schema")
        selected = _names(self.selected_features, "Selected schema")
        unknown = sorted(set(selected) - set(source))
        if unknown:
            raise ValueError(f"Unknown selected features: {unknown}")
        if tuple(name for name in source if name in set(selected)) != selected:
            raise ValueError("Selected features must preserve the original cache column order")
        object.__setattr__(self, "source_feature_names", source)
        object.__setattr__(self, "selected_features", selected)

    @classmethod
    def resolve(cls, source_feature_names, *, include=None, exclude=None):
        source = _names(source_feature_names, "Source schema")
        if include is not None and exclude is not None:
            raise ValueError("Use either include-features or exclude-features, not both")
        patterns = include if include is not None else exclude
        matched = set()
        if patterns is not None:
            if isinstance(patterns, str) or not patterns:
                raise ValueError("Feature selection needs one or more names or patterns")
            for pattern in patterns:
                if not isinstance(pattern, str) or not pattern:
                    raise ValueError("Feature patterns must be nonempty strings")
                found = {name for name in source if fnmatchcase(name, pattern)}
                if not found:
                    raise ValueError(f"Feature pattern matched no cached columns: {pattern!r}")
                matched.update(found)
        selected = tuple(name for name in source
                         if (name in matched if include is not None else name not in matched))
        return cls(source, selected, tuple(include or ()), tuple(exclude or ()))

    @classmethod
    def from_names(cls, source_feature_names, selected_features=None):
        source = _names(source_feature_names, "Source schema")
        return cls(source, source if selected_features is None else tuple(selected_features))

    @property
    def indices(self):
        positions = {name: index for index, name in enumerate(self.source_feature_names)}
        return [positions[name] for name in self.selected_features]

    def apply(self, X, feature_names):
        if tuple(feature_names) != self.source_feature_names:
            raise ValueError("Source feature names or order differ from the saved cache schema")
        matrix = np.asarray(X)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.source_feature_names):
            raise ValueError("Feature matrix does not match the source feature schema")
        if self.selected_features == self.source_feature_names:
            return matrix
        return matrix[:, self.indices]

    def to_dict(self):
        return {"schema_version": 1,
                "source_feature_names": list(self.source_feature_names),
                "selected_features": list(self.selected_features),
                "excluded_features": [name for name in self.source_feature_names
                                      if name not in self.selected_features],
                "include_patterns": list(self.include_patterns),
                "exclude_patterns": list(self.exclude_patterns)}

    @classmethod
    def from_dict(cls, state):
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported feature selection schema")
        selection = cls(state["source_feature_names"], state["selected_features"],
                        tuple(state.get("include_patterns", ())), tuple(state.get("exclude_patterns", ())))
        if selection.include_patterns or selection.exclude_patterns:
            resolved = cls.resolve(selection.source_feature_names,
                                   include=selection.include_patterns or None,
                                   exclude=selection.exclude_patterns or None)
            if resolved.selected_features != selection.selected_features:
                raise ValueError("Saved feature selection does not match its requested patterns")
        if selection.to_dict() != state:
            raise ValueError("Saved feature selection is inconsistent with its patterns or source schema")
        return selection
