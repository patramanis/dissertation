from __future__ import annotations
from itertools import combinations
from typing import Any, Iterable
import numpy as np
import pandas as pd
from data.pit.policies import SECTORS
from features.common.returns import safe_log_returns

DRIVERS_REQUIRED = ["rates", "oil", "usd", "bonds", "vix", "hy", "gold"]

def _reindex_prices_to_calendar(spdr_prices: pd.DataFrame, trading_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    spdr = spdr_prices.copy()
    spdr.index = pd.to_datetime(spdr.index).normalize()
    cal = pd.DatetimeIndex(pd.to_datetime(trading_calendar).normalize(), name="Date")
    spdr = spdr.reindex(cal)
    return spdr.reindex(columns=list(SECTORS)).astype("float64")

def _reindex_and_ffill_macro(
    macro_drivers: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    *,
    ffill_limits: dict[str, int] | None,
    default_limit: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    md = macro_drivers.copy()
    md.index = pd.to_datetime(md.index).normalize()
    cal = pd.DatetimeIndex(pd.to_datetime(trading_calendar).normalize(), name="Date")
    md = md.reindex(cal)

    used: dict[str, int] = {}
    limits = ffill_limits or {}
    for c in md.columns:
        if c not in limits:
            raise ValueError(
                f"Missing ffill_limits for macro driver '{c}'. "
                "Provide correlations.ffill_limits per driver to avoid implicit leakage/bias."
            )
        lim = int(limits[c])
        used[str(c)] = lim
        md[c] = md[c].astype("float64").ffill(limit=lim)

    return md.astype("float64"), used

def _rolling_var0(x: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    ex = x.rolling(window=window, min_periods=min_periods).mean()
    ex2 = (x * x).rolling(window=window, min_periods=min_periods).mean()
    return (ex2 - (ex * ex)).clip(lower=0.0)

def _rolling_cov0(x: pd.Series, y: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    ex = x.rolling(window=window, min_periods=min_periods).mean()
    ey = y.rolling(window=window, min_periods=min_periods).mean()
    exy = (x * y).rolling(window=window, min_periods=min_periods).mean()
    return exy - (ex * ey)

def _rolling_corr0(
    x: pd.Series[Any],
    y: pd.Series[Any],
    *,
    window: int,
    min_periods: int,
    eps: float = 1e-12,
) -> pd.Series[float]:
    cov0 = _rolling_cov0(x, y, window=window, min_periods=min_periods)
    vx = _rolling_var0(x, window=window, min_periods=min_periods)
    vy = _rolling_var0(y, window=window, min_periods=min_periods)
    denom = pd.Series(np.sqrt(vx) * np.sqrt(vy))
    denom = denom.where(denom > eps)
    out = cov0 / denom
    return out.clip(lower=-1.0, upper=1.0)

def _base_panel(trading_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DatetimeIndex(pd.to_datetime(trading_calendar).normalize(), name="Date")
    idx = pd.MultiIndex.from_product([cal, list(SECTORS)], names=["Date", "Sector"])
    return pd.DataFrame(index=idx)

def _wide_to_panel(wide: pd.DataFrame, *, name: str, base_index: pd.MultiIndex) -> pd.Series[Any]:
    w = wide.reindex(columns=list(SECTORS))
    try:
        s = w.stack(future_stack=True)
    except TypeError:
        s = w.stack(dropna=False)
    s.index = pd.MultiIndex.from_arrays(
        [s.index.get_level_values(0), s.index.get_level_values(1)],
        names=["Date", "Sector"],
    )
    result = s.reindex(base_index)
    result.name = name
    if isinstance(result, pd.DataFrame):
        return result.iloc[:, 0]
    return result

def _series_to_panel(s: pd.Series[float], *, name: str, base_index: pd.MultiIndex) -> pd.Series[float]:
    cal = pd.DatetimeIndex(base_index.get_level_values(0).unique(), name="Date")
    vals = s.reindex(cal).to_numpy(dtype="float64")
    out: pd.Series[float] = pd.Series(np.repeat(vals, len(SECTORS)), index=base_index, dtype="float64", name=name)
    return out

def compute_corr_features(
    *,
    spdr_prices: pd.DataFrame,
    macro_drivers: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    windows: Iterable[int],
    min_periods_by_window: dict[int, int] | None = None,
    ffill_limits: dict[str, int] | None = None,
    default_macro_ffill_limit: int = 5,
    shift_by_1: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cal = pd.DatetimeIndex(pd.to_datetime(trading_calendar).normalize(), name="Date")
    spdr = _reindex_prices_to_calendar(spdr_prices, cal)

    missing = [c for c in SECTORS if c not in spdr.columns]
    if missing:
        raise ValueError(f"SPDR missing sectors: {missing}")

    macro, ffill_used = _reindex_and_ffill_macro(
        macro_drivers,
        cal,
        ffill_limits=ffill_limits,
        default_limit=default_macro_ffill_limit,
    )

    for d in DRIVERS_REQUIRED:
        if d not in macro.columns:
            raise ValueError(f"Missing required macro driver column: {d}")

    sector_rets = safe_log_returns(spdr, name="spdr_prices")
    assert isinstance(sector_rets, pd.DataFrame), "sector_rets must be DataFrame"
    sector_rets = sector_rets.reindex(cal).astype("float64")

    panel = _base_panel(cal)
    panel_idx = panel.index
    assert isinstance(panel_idx, pd.MultiIndex), "panel.index must be MultiIndex"

    mp = min_periods_by_window or {}
    ws = sorted({int(w) for w in windows})
    if not ws:
        raise ValueError("windows must be non-empty")

    new_cols: dict[str, pd.Series[Any]] = {}

    for w in ws:
        minp = int(mp.get(w, w))
        for d in DRIVERS_REQUIRED:
            wide = pd.DataFrame(index=cal, columns=list(SECTORS), dtype="float64")
            for s in SECTORS:
                sector_col = sector_rets[s]
                macro_col = macro[d]
                assert isinstance(sector_col, pd.Series) and isinstance(macro_col, pd.Series)
                wide[s] = _rolling_corr0(sector_col, macro_col, window=w, min_periods=minp)
            col = f"corr_{d}_{w}"
            new_cols[col] = _wide_to_panel(wide, name=col, base_index=panel_idx)

    for w in ws:
        minp = int(mp.get(w, w))
        for a, b in combinations(DRIVERS_REQUIRED, 2):
            macro_a = macro[a]
            macro_b = macro[b]
            assert isinstance(macro_a, pd.Series) and isinstance(macro_b, pd.Series)
            s = _rolling_corr0(macro_a, macro_b, window=w, min_periods=minp)
            col = f"corr_macro_{a}_{b}_{w}"
            new_cols[col] = _series_to_panel(s, name=col, base_index=panel_idx)

    sector_mean = sector_rets.mean(axis="columns")
    assert isinstance(sector_mean, pd.Series), "sector_mean must be Series"
    for w in ws:
        minp = int(mp.get(w, w))
        wide_mean = pd.DataFrame(index=cal, columns=list(SECTORS), dtype="float64")
        for s in SECTORS:
            sector_col = sector_rets[s]
            assert isinstance(sector_col, pd.Series)
            wide_mean[s] = _rolling_corr0(sector_col, sector_mean, window=w, min_periods=minp)
        col = f"corr_with_sector_mean_{w}"
        new_cols[col] = _wide_to_panel(wide_mean, name=col, base_index=panel_idx)

        pair_cols: dict[tuple[str, str], pd.Series[Any]] = {}
        for a, b in combinations(SECTORS, 2):
            col_a = sector_rets[a]
            col_b = sector_rets[b]
            assert isinstance(col_a, pd.Series) and isinstance(col_b, pd.Series)
            pair_cols[(a, b)] = _rolling_corr0(col_a, col_b, window=w, min_periods=minp)

        mean_out = pd.DataFrame(index=cal, columns=list(SECTORS), dtype="float64")
        max_out = pd.DataFrame(index=cal, columns=list(SECTORS), dtype="float64")
        for s in SECTORS:
            others: list[pd.Series] = []
            for a, b in pair_cols.keys():
                if s == a:
                    others.append(pair_cols[(a, b)])
                elif s == b:
                    others.append(pair_cols[(a, b)])
            if not others:
                mean_out[s] = np.nan
                max_out[s] = np.nan
            else:
                m = pd.concat(others, axis=1)
                mean_out[s] = m.mean(axis=1)
                max_out[s] = m.max(axis=1)

        new_cols[f"mean_corr_others_{w}"] = _wide_to_panel(mean_out, name=f"mean_corr_others_{w}", base_index=panel_idx)
        new_cols[f"max_corr_others_{w}"] = _wide_to_panel(max_out, name=f"max_corr_others_{w}", base_index=panel_idx)

    if new_cols:
        new_df = pd.concat(new_cols, axis=1)
        new_df.columns = list(new_cols.keys())
        panel = pd.concat([panel, new_df], axis=1)
        panel = panel.copy()

    panel = panel.reset_index()
    panel["Date"] = pd.to_datetime(panel["Date"]).dt.normalize()
    panel["Sector"] = pd.Categorical(panel["Sector"].astype(str), categories=list(SECTORS), ordered=True)
    panel = panel.sort_values(["Date", "Sector"], kind="mergesort")
    panel["Sector"] = panel["Sector"].astype(str)
    value_cols = [c for c in panel.columns if c not in {"Date", "Sector"}]
    panel = panel[["Date", "Sector", *sorted(value_cols)]]

    if shift_by_1:
        feat_cols = [c for c in panel.columns if c not in {"Date", "Sector"}]
        panel[feat_cols] = panel.groupby("Sector", sort=False)[feat_cols].shift(1)

    corr_cols = [c for c in panel.columns if c not in {"Date", "Sector"}]
    for c in corr_cols:
        s = panel[c]
        non_nan = s.dropna()
        if not ((non_nan >= -1.0) & (non_nan <= 1.0)).all():
            raise AssertionError(f"Correlation out of bounds in {c}")

    return panel, ffill_used