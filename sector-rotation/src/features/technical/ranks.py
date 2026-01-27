from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal
import pandas as pd
from data.pit.policies import SECTORS

@dataclass(frozen=True)
class RankConfig:

    method: Literal["average", "first"] = "first"

def _as_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if isinstance(panel.index, pd.MultiIndex) and list(panel.index.names) == ["Date", "Sector"]:
        return panel

    if "Date" in panel.columns and "Sector" in panel.columns:
        out = panel.copy()
        out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
        out["Sector"] = out["Sector"].astype(str)
        out = out.set_index(["Date", "Sector"])
        return out

    raise ValueError("panel must have MultiIndex (Date, Sector) or columns ['Date','Sector']")

def _sort_panel(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    idx = df.index
    if not isinstance(idx, pd.MultiIndex) or list(idx.names) != ["Date", "Sector"]:
        raise ValueError("Expected MultiIndex (Date, Sector)")

    _ = idx.get_level_values("Date")
    _ = idx.get_level_values("Sector")
    df = df.reset_index()
    df["Sector"] = pd.Categorical(df["Sector"].astype(str), categories=list(SECTORS), ordered=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values(["Date", "Sector"], kind="mergesort")
    df = df.set_index(["Date", "Sector"])
    return df

def assert_group_size(panel: pd.DataFrame, *, expected: int = 9) -> None:
    df = _as_panel(panel)
    counts = df.reset_index().groupby("Date")["Sector"].nunique()
    bad = counts[counts != expected]
    if len(bad) > 0:
        first = bad.index[0]
        raise AssertionError(f"Group size != {expected} at {pd.Timestamp(first).date()}: got {int(bad.iloc[0])}")

def add_rank_columns(
    panel: pd.DataFrame,
    *,
    feature_cols: Iterable[str],
    cfg: RankConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or RankConfig()
    df = _sort_panel(_as_panel(panel))

    assert_group_size(df, expected=9)

    cols = [c for c in feature_cols]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing rank feature columns: {missing}")

    for c in cols:
        r = df.groupby(level=0, sort=False)[c].rank(pct=True, method=cfg.method)
        df[f"rank_{c}"] = r

    assert_group_size(df, expected=9)
    return df