from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MM_ROOT = Path(__file__).resolve().parents[1]
RAW1_DIR = MM_ROOT / "data" / "raw_data_1"
RAW2_DIR = MM_ROOT / "data" / "raw_data_2"
OUT_DIR = MM_ROOT / "data" / "processed_data_1"
LABELS_DIR = MM_ROOT / "data" / "labels"

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


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    return df


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    return df


def _read_panel_parquet(path: Path, key_cols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    for c in key_cols:
        if c not in df.columns:
            raise ValueError(f"Missing {c} column in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values(key_cols)
    df = df.drop_duplicates(subset=key_cols, keep="last")
    return df


def _download_spy(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        "SPY",
        start=str(start.date()),
        end=str((end + pd.Timedelta(days=1)).date()),
        auto_adjust=False,
        progress=False,
        actions=False,
        group_by="column",
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty data for SPY")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", "SPY") not in df.columns:
            raise RuntimeError("SPY download missing ('Adj Close','SPY')")
        s = df[("Adj Close", "SPY")].rename("SPY")
    else:
        if "Adj Close" not in df.columns:
            raise RuntimeError("SPY download missing 'Adj Close'")
        s = df["Adj Close"].rename("SPY")

    out = s.to_frame()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.reset_index()
    out = out.loc[:, ~out.columns.duplicated()]
    first = out.columns[0]
    if first != "Date":
        out = out.rename(columns={first: "Date"})
    out["Date"] = pd.to_datetime(out["Date"], errors="raise").dt.tz_localize(None)
    out = out.sort_values("Date").drop_duplicates("Date", keep="last")
    return out


def _load_spy_asof(trading_dates: pd.DatetimeIndex) -> pd.Series:
    if SPY_RAW2.exists():
        df = _read_parquet(SPY_RAW2)
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize().dt.tz_localize(None)
        s = pd.to_numeric(df.set_index("Date")["SPY"], errors="coerce").reindex(trading_dates)
        s.name = "SPY"
        return s

    start = pd.Timestamp(trading_dates.min())
    end = pd.Timestamp(trading_dates.max())

    if SPY_CSV.exists():
        spy = _read_csv(SPY_CSV)
    else:
        spy = _download_spy(start, end)
        spy.to_csv(SPY_CSV, index=False)

    spy["Date"] = pd.to_datetime(spy["Date"]).dt.normalize().dt.tz_localize(None)
    spy = spy.dropna(subset=["SPY"])
    s = pd.to_numeric(spy.set_index("Date")["SPY"], errors="coerce")

    s = s.reindex(trading_dates)
    s = s.ffill(limit=5)
    s = s.shift(1)
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


def _rolling_downside_semivol(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    if np.isnan(x).any():
        return np.nan
    neg = x[x < 0]
    if neg.size == 0:
        return 0.0
    return float(np.std(neg, ddof=0))


def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff(1)


def _beta_and_corr(r_s: pd.DataFrame, r_m: pd.Series, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    m_s = r_s.rolling(window, min_periods=window).mean()
    m_m = r_m.rolling(window, min_periods=window).mean()

    cov = (r_s.mul(r_m, axis=0)).rolling(window, min_periods=window).mean() - m_s.mul(m_m, axis=0)
    var_m = r_m.rolling(window, min_periods=window).var(ddof=0)

    beta = cov.div(var_m, axis=0)

    std_s = r_s.rolling(window, min_periods=window).std(ddof=0)
    std_m = r_m.rolling(window, min_periods=window).std(ddof=0)
    corr = cov.div(std_s.mul(std_m, axis=0))

    return beta, corr


def _idio_vol(r_s: pd.DataFrame, r_m: pd.Series, window: int) -> pd.DataFrame:
    beta, _ = _beta_and_corr(r_s, r_m, window)
    m_s = r_s.rolling(window, min_periods=window).mean()
    m_m = r_m.rolling(window, min_periods=window).mean()
    alpha = m_s.sub(beta.mul(m_m, axis=0))

    fitted = alpha.add(beta.mul(r_m, axis=0))
    resid = r_s - fitted

    return resid.rolling(window, min_periods=window).std(ddof=0)


def _melt_features(df_wide: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df_wide.copy()
    out.index.name = "Date"
    out = out.reset_index().melt(id_vars=["Date"], var_name="Sector", value_name=prefix)
    return out


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
    labels: pd.DataFrame,
    spec: HorizonSpec,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(spdr_asof.index)

    prices = spdr_asof[SECTORS].copy()
    spy_px = spy_asof.reindex(dates)

    logret_s = _compute_returns(prices)
    logret_m = _compute_returns(spy_px.to_frame())["SPY"]

    logret_1d = logret_s.copy()
    logret_1d.columns = [f"{c}_logret1d" for c in logret_1d.columns]
    feats_wide: dict[str, pd.DataFrame] = {"logret_1d": logret_1d}

    excess_1d = logret_s.sub(logret_m, axis=0)
    excess_1d.columns = [f"{c}_excess1d" for c in excess_1d.columns]
    feats_wide["excess_1d"] = excess_1d

    excess_5d = logret_s.sub(logret_m, axis=0).rolling(5, min_periods=5).sum()
    excess_5d.columns = [f"{c}_excess5d" for c in excess_5d.columns]
    feats_wide["excess_5d"] = excess_5d

    ratio = np.log(prices.div(spy_px, axis=0))

    ratio_level = ratio.copy()
    ratio_level.columns = [f"{c}_ratio_level" for c in ratio_level.columns]
    feats_wide["ratio_level"] = ratio_level

    for w in spec.ratio_z_windows:
        z = ratio.apply(lambda s: _rolling_zscore(s, w))
        z.columns = [f"{c}_ratio_z{w}" for c in z.columns]
        feats_wide[f"ratio_z{w}"] = z

    rel = logret_s.sub(logret_m, axis=0)
    for L in spec.relmom_lookbacks:
        m = rel.rolling(L, min_periods=L).sum()
        m.columns = [f"{c}_relmom{L}" for c in m.columns]
        feats_wide[f"relmom{L}"] = m

    fast, slow = spec.trend_mas
    ma_fast = ratio.rolling(fast, min_periods=fast).mean()
    ma_slow = ratio.rolling(slow, min_periods=slow).mean()
    spread = ma_fast - ma_slow
    spread.columns = [f"{c}_ratio_ma{fast}_{slow}" for c in spread.columns]
    feats_wide[f"ratio_ma{fast}_{slow}"] = spread

    for w in spec.beta_windows:
        beta, corr = _beta_and_corr(logret_s, logret_m, w)
        beta.columns = [f"{c}_beta{w}" for c in beta.columns]
        corr.columns = [f"{c}_corr{w}" for c in corr.columns]
        feats_wide[f"beta{w}"] = beta
        feats_wide[f"corr{w}"] = corr

    idio = _idio_vol(logret_s, logret_m, spec.idio_window)
    idio.columns = [f"{c}_idio_vol{spec.idio_window}" for c in idio.columns]
    feats_wide[f"idio{spec.idio_window}"] = idio

    vols: dict[int, pd.DataFrame] = {}
    for w in spec.vol_windows:
        v = logret_s.rolling(w, min_periods=w).std(ddof=0)
        v.columns = [f"{c}_vol{w}" for c in v.columns]
        vols[w] = v
        feats_wide[f"vol{w}"] = v

    if spec.h == 5:
        if 5 in vols and 63 in vols:
            vr = vols[5].to_numpy() / vols[63].to_numpy()
            vr = pd.DataFrame(vr, index=vols[5].index, columns=[c.replace("_vol5", "_vol5_over_63") for c in vols[5].columns])
            feats_wide["vol_ratio"] = vr
    if spec.h == 21:
        if 21 in vols and 126 in vols:
            vr = vols[21].to_numpy() / vols[126].to_numpy()
            vr = pd.DataFrame(vr, index=vols[21].index, columns=[c.replace("_vol21", "_vol21_over_126") for c in vols[21].columns])
            feats_wide["vol_ratio"] = vr

    for w in set([spec.mdd_window]):
        mdd = prices.rolling(w, min_periods=w).apply(_rolling_max_drawdown, raw=True)
        mdd.columns = [f"{c}_mdd{w}" for c in mdd.columns]
        feats_wide[f"mdd{w}"] = mdd

    for w in spec.semi_windows:
        semi = logret_s.rolling(w, min_periods=w).apply(_rolling_downside_semivol, raw=True)
        semi.columns = [f"{c}_semi{w}" for c in semi.columns]
        feats_wide[f"semi{w}"] = semi

    features = []
    for group_df in feats_wide.values():
        features.append(group_df)

    all_wide = pd.concat(features, axis=1)

    all_wide.index = dates
    all_wide.index.name = "Date"

    feat_long = all_wide.reset_index().melt(id_vars=["Date"], var_name="key", value_name="value")
    feat_long[["Sector", "Feature"]] = feat_long["key"].str.split("_", n=1, expand=True)
    feat_long = feat_long.drop(columns=["key"])
    feat_long = feat_long.pivot_table(index=["Date", "Sector"], columns="Feature", values="value", aggfunc="last")
    feat_long = feat_long.reset_index()

    label_col = f"label_excess_{spec.h}d"
    lab = labels[["Date", "Sector", label_col]].copy()

    out = lab.merge(feat_long, on=["Date", "Sector"], how="left")

    rank_candidates = [
        c
        for c in out.columns
        if any(
            c.endswith(s)
            for s in (
                f"relmom{spec.relmom_lookbacks[0]}",
                f"relmom{spec.relmom_lookbacks[1]}",
                f"relmom{spec.relmom_lookbacks[2]}",
                f"mdd{spec.mdd_window}",
                f"beta{spec.beta_windows[0]}",
                f"idio_vol{spec.idio_window}",
            )
        )
        or ("ratio_z" in c and any(str(w) in c for w in spec.ratio_z_windows))
    ]
    out = _cross_sectional_ranks(out, sorted(set(rank_candidates)))

    out = out.sort_values(["Date", "Sector"]).reset_index(drop=True)
    return out


def main() -> None:
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
        labels_path = LABELS_DIR / f"h{h}.parquet"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing {labels_path}; run build_labels.py first")

        labels = _read_panel_parquet(labels_path, key_cols=["Date", "Sector"])
        labels["Date"] = pd.to_datetime(labels["Date"]).dt.tz_localize(None)

        df_out = build_processed_for_horizon(spdr_asof, spy_asof, labels, spec)
        out_path = OUT_DIR / f"features_h{h}.parquet"
        df_out.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"Wrote {out_path} shape={df_out.shape}")


if __name__ == "__main__":
    main()
