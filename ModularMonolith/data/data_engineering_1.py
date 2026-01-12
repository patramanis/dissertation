from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ModularMonolith.data.train_window import TRAIN_DATE_END, TRAIN_DATE_START

try:
    from ModularMonolith import build_id

    BUILD_ID = build_id(__file__)
except Exception:
    BUILD_ID = str(Path(__file__).resolve())


MM_ROOT = Path(__file__).resolve().parents[1]
RAW1_DIR = MM_ROOT / "data" / "raw_data_1"
RAW2_DIR = MM_ROOT / "data" / "raw_data_2"
OUT_DIR = MM_ROOT / "data" / "processed_data_1"

SPY_CSV = RAW1_DIR / "SPY.csv"
SPY_RAW2 = RAW2_DIR / "SPY.parquet"

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLY", "XLV", "XLU"]


@dataclass(frozen=True)
class HorizonSpec:
    h: int
    relmom_lookbacks: tuple[int, ...]
    trend_mas: tuple[int, int]
    beta_windows: tuple[int, ...]
    idio_window: int
    vol_windows: tuple[int, ...]
    mdd_window: int
    semi_windows: tuple[int, ...]
    ratio_z_windows: tuple[int, ...]


HORIZONS: dict[int, HorizonSpec] = {
    5: HorizonSpec(
        h=5,
        relmom_lookbacks=(5, 10, 21),
        trend_mas=(5, 21),
        beta_windows=(21, 63),
        idio_window=21,
        vol_windows=(5, 21, 63),
        mdd_window=21,
        semi_windows=(5, 21),
        ratio_z_windows=(21, 63),
    ),
    21: HorizonSpec(
        h=21,
        relmom_lookbacks=(21, 42, 63),
        trend_mas=(21, 63),
        beta_windows=(63, 126),
        idio_window=63,
        vol_windows=(21, 63, 126),
        mdd_window=63,
        semi_windows=(21, 63),
        ratio_z_windows=(63, 126),
    ),
    63: HorizonSpec(
        h=63,
        relmom_lookbacks=(63, 126, 252),
        trend_mas=(63, 252),
        beta_windows=(126, 252),
        idio_window=126,
        vol_windows=(63, 126, 252),
        mdd_window=252,
        semi_windows=(63, 126),
        ratio_z_windows=(126, 252),
    ),
}


def _check_duplicates(df: pd.DataFrame, path: Path | str) -> None:
    norm = df["Date"].dt.normalize()
    dup_mask = norm.duplicated(keep=False)
    if bool(dup_mask.any()):
        dup_dates = norm.loc[dup_mask].value_counts().sort_values(ascending=False).head(10)
        examples = ", ".join([f"{d.date()}(x{int(c)})" for d, c in dup_dates.items()])
        raise ValueError(
            f"Duplicate Date rows in {path}: n_dup={int(dup_mask.sum())} examples=[{examples}]"
        )


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date")
    _check_duplicates(df, path)
    return df


def _load_spy_asof(trading_dates: pd.DatetimeIndex) -> pd.Series:
    if not SPY_RAW2.exists():
        raise FileNotFoundError(
            f"Missing PIT-aligned SPY data at {SPY_RAW2}. "
            "Run data_optimization_1.py first to generate raw_data_2/*.parquet files."
        )
    
    df = _read_parquet(SPY_RAW2)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize().dt.tz_localize(None)
    s = pd.to_numeric(df.set_index("Date")["SPY"], errors="coerce").reindex(trading_dates)
    s.name = "SPY"
    return s


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=window).mean()
    v = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - m) / v


def _rolling_max_drawdown(prices: np.ndarray) -> float:
    if prices.size == 0:
        return np.nan
    if np.isnan(prices).any():
        return np.nan
    peak = np.maximum.accumulate(prices)
    dd = prices / peak - 1.0
    return float(np.min(dd))


def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff(1)


def _beta_and_corr(r_s: pd.DataFrame, r_m: pd.Series, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps = 1e-12
    m_s = r_s.rolling(window, min_periods=window).mean()
    m_m = r_m.rolling(window, min_periods=window).mean()

    cov = (r_s.mul(r_m, axis=0)).rolling(window, min_periods=window).mean() - m_s.mul(m_m, axis=0)
    var_m = r_m.rolling(window, min_periods=window).var(ddof=0)

    var_m = var_m.mask(var_m.abs() < eps)

    beta = cov.div(var_m, axis=0)

    std_s = r_s.rolling(window, min_periods=window).std(ddof=0)
    std_m = r_m.rolling(window, min_periods=window).std(ddof=0)

    denom = std_s.mul(std_m, axis=0)
    denom = denom.mask(denom.abs() < eps)
    corr = cov.div(denom)

    return beta, corr


def _idio_vol(r_s: pd.DataFrame, r_m: pd.Series, window: int) -> pd.DataFrame:
    _, corr = _beta_and_corr(r_s, r_m, window)
    vol_s = r_s.rolling(window, min_periods=window).std(ddof=0)
    return vol_s * np.sqrt(1 - corr ** 2)


def _cross_sectional_ranks(df_long: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df_long.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        out[f"rank_{col}"] = out.groupby("Date")[col].rank(pct=True, method="average")
    return out


def build_processed_for_horizon(
    spdr_asof: pd.DataFrame,
    spy_asof: pd.Series,
    spec: HorizonSpec,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(spdr_asof.index)

    prices = spdr_asof[SECTORS].copy()
    spy_px = spy_asof.reindex(dates)

    logret_s = _compute_returns(prices)
    logret_m = _compute_returns(spy_px.to_frame())["SPY"]

    feats_wide: dict[str, pd.DataFrame] = {}

    logret_1d = logret_s.copy()
    logret_1d.columns = [f"{c}_logret1d" for c in SECTORS]
    feats_wide["logret_1d"] = logret_1d

    excess = logret_s.sub(logret_m, axis=0)
    excess_1d = excess.copy()
    excess_1d.columns = [f"{c}_excess1d" for c in SECTORS]
    feats_wide["excess_1d"] = excess_1d

    excess_5d = excess.rolling(5, min_periods=5).sum()
    excess_5d.columns = [f"{c}_excess5d" for c in SECTORS]
    feats_wide["excess_5d"] = excess_5d

    ratio = np.log(prices.div(spy_px, axis=0))

    for w in spec.ratio_z_windows:
        z = ratio.apply(lambda s: _rolling_zscore(s, w))
        z.columns = [f"{c}_ratio_z{w}" for c in SECTORS]
        feats_wide[f"ratio_z{w}"] = z

    for L in spec.relmom_lookbacks:
        m = excess.rolling(L, min_periods=L).sum()
        m.columns = [f"{c}_relmom{L}" for c in SECTORS]
        feats_wide[f"relmom{L}"] = m

    fast, slow = spec.trend_mas
    spread = ratio.rolling(fast, min_periods=fast).mean() - ratio.rolling(slow, min_periods=slow).mean()
    spread.columns = [f"{c}_ratio_ma{fast}_{slow}" for c in SECTORS]
    feats_wide[f"ratio_ma{fast}_{slow}"] = spread

    for w in spec.beta_windows:
        beta, corr = _beta_and_corr(logret_s, logret_m, w)
        beta.columns = [f"{c}_beta{w}" for c in SECTORS]
        corr.columns = [f"{c}_corr{w}" for c in SECTORS]
        feats_wide[f"beta{w}"] = beta
        feats_wide[f"corr{w}"] = corr

    idio = _idio_vol(logret_s, logret_m, spec.idio_window)
    idio.columns = [f"{c}_idio_vol{spec.idio_window}" for c in SECTORS]
    feats_wide[f"idio{spec.idio_window}"] = idio

    vols: dict[int, pd.DataFrame] = {}
    for w in spec.vol_windows:
        v = logret_s.rolling(w, min_periods=w).std(ddof=0)
        v.columns = [f"{c}_vol{w}" for c in SECTORS]
        vols[w] = v
        feats_wide[f"vol{w}"] = v

    vol_ratio_map = {5: (5, 63), 21: (21, 126)}
    if spec.h in vol_ratio_map:
        short_w, long_w = vol_ratio_map[spec.h]
        if short_w in vols and long_w in vols:
            vr = vols[short_w] / vols[long_w]
            vr.columns = [f"{c.replace(f'_vol{short_w}', f'_vol{short_w}_over_{long_w}')}" for c in vr.columns]
            feats_wide["vol_ratio"] = vr

    mdd = prices.rolling(spec.mdd_window, min_periods=spec.mdd_window).apply(_rolling_max_drawdown, raw=True)
    mdd.columns = [f"{c}_mdd{spec.mdd_window}" for c in SECTORS]
    feats_wide[f"mdd{spec.mdd_window}"] = mdd

    for w in spec.semi_windows:
        semi = np.sqrt((logret_s.clip(upper=0) ** 2).rolling(w, min_periods=w).mean())
        semi.columns = [f"{c}_semi{w}" for c in SECTORS]
        feats_wide[f"semi{w}"] = semi

    all_wide = pd.concat(feats_wide.values(), axis=1)
    all_wide.index = dates
    all_wide.index.name = "Date"

    feat_long = all_wide.reset_index().melt(id_vars=["Date"], var_name="key", value_name="value")
    feat_long[["Sector", "Feature"]] = feat_long["key"].str.split("_", n=1, expand=True)
    feat_long = feat_long.pivot_table(
        index=["Date", "Sector"], columns="Feature", values="value", aggfunc="last"
    ).reset_index()

    rank_suffixes = (
        tuple(f"relmom{L}" for L in spec.relmom_lookbacks)
        + (f"mdd{spec.mdd_window}", f"beta{spec.beta_windows[0]}", f"idio_vol{spec.idio_window}")
    )
    rank_candidates = [
        c for c in feat_long.columns
        if c.endswith(rank_suffixes) or ("ratio_z" in c and any(str(w) in c for w in spec.ratio_z_windows))
    ]
    feat_long = _cross_sectional_ranks(feat_long, sorted(set(rank_candidates)))
    return feat_long.sort_values(["Date", "Sector"]).reset_index(drop=True)


def main() -> None:
    print(f"[data_engineering_1] BUILD_ID={BUILD_ID}")
    spdr2_path = RAW2_DIR / "SPDR.parquet"
    if not spdr2_path.exists():
        raise FileNotFoundError(spdr2_path)

    spdr2 = _read_parquet(spdr2_path)
    spdr2["Date"] = pd.to_datetime(spdr2["Date"]).dt.tz_localize(None)
    spdr_asof = spdr2.set_index("Date")

    for c in SECTORS:
        if c not in spdr_asof.columns:
            raise ValueError(f"Missing sector {c} in raw_data_2/SPDR.parquet")

    trading_dates = pd.DatetimeIndex(spdr_asof.index)
    spy_asof = _load_spy_asof(trading_dates)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for h, spec in HORIZONS.items():
        df_out = build_processed_for_horizon(spdr_asof, spy_asof, spec)

        df_out["Date"] = pd.to_datetime(df_out["Date"], errors="raise").dt.tz_localize(None)
        df_out = df_out[(df_out["Date"] >= TRAIN_DATE_START) & (df_out["Date"] <= TRAIN_DATE_END)].copy()

        out_path = OUT_DIR / f"features_h{h}.parquet"
        df_out.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"Wrote {out_path} shape={df_out.shape}")


if __name__ == "__main__":
    main()
