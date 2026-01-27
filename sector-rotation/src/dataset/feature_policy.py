from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any
import pandas as pd

_NON_FEATURE_COLS = {
    "Date",
    "Sector",
    "label_excess",
    "y_gate",
    "rel_rank",
    "rank_target",
    "cost_bps",
    "horizon",
    "label_contract",
}

_FORBIDDEN_SUBSTRINGS = ["label", "target", "forward", "fwd"]

@dataclass(frozen=True)
class FeaturePolicy:
    forbidden_substrings: tuple[str, ...] = tuple(_FORBIDDEN_SUBSTRINGS)
    blocked_patterns: tuple[str, ...] = ()
    blocked_level_suffixes: tuple[str, ...] = ()

def _has_forbidden_substring(name: str, forbidden: tuple[str, ...]) -> bool:
    s = name.lower()
    return any(f in s for f in forbidden)

def _matches_any_pattern(name: str, patterns: tuple[str, ...]) -> bool:
    for p in patterns:
        if not p:
            continue
        if p in name:
            return True
        try:
            if re.search(p, name):
                return True
        except re.error:
            continue
    return False

def select_feature_columns(df: pd.DataFrame, policy: FeaturePolicy | None = None) -> tuple[list[str], dict[str, Any]]:

    pol = policy or FeaturePolicy()

    cols = [str(c) for c in df.columns]

    forbidden_found = [c for c in cols if (c not in _NON_FEATURE_COLS and _has_forbidden_substring(c, pol.forbidden_substrings))]
    if forbidden_found:
        forbidden_found_sorted = sorted(forbidden_found)
        raise AssertionError(f"Forbidden substrings found in feature-like columns: {forbidden_found_sorted[:20]}")

    features: list[str] = []
    blocked: list[str] = []

    for c in cols:
        if c in _NON_FEATURE_COLS:
            continue
        if _matches_any_pattern(c, pol.blocked_patterns) or any(c.endswith(suf) for suf in pol.blocked_level_suffixes):
            blocked.append(c)
            continue
        if _has_forbidden_substring(c, pol.forbidden_substrings):
            blocked.append(c)
            continue
        features.append(c)

    features_sorted = sorted(features)

    report = {
        "forbidden_substrings": list(pol.forbidden_substrings),
        "blocked_patterns": list(pol.blocked_patterns),
        "blocked_level_suffixes": list(pol.blocked_level_suffixes),
        "blocked_count": int(len(blocked)),
        "n_features": int(len(features_sorted)),
    }

    return features_sorted, report