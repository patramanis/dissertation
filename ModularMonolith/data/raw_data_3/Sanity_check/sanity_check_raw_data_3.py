from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RAW1_DIR = Path("ModularMonolith") / "data" / "raw_data_1"
RAW3_DIR = Path("ModularMonolith") / "data" / "raw_data_3"


LEVEL_SUFFIXES = ("_asinh", "_log1p", "_level")
CHANGE_SUFFIXES = ("_diff", "_logdiff", "_logret", "_dlog1p")


@dataclass(frozen=True)
class SeriesStats:
    n: int
    nan: int
    inf: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    q01: float
    q99: float


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


def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    s.index = pd.DatetimeIndex(df["Date"]).tz_localize(None)
    return s


def _stats(s: pd.Series) -> SeriesStats:
    x = pd.to_numeric(s, errors="coerce").astype("float64")
    n = int(len(x))
    nan = int(x.isna().sum())
    inf = int(np.isinf(x.to_numpy(dtype="float64", na_value=np.nan)).sum())
    finite = x.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return SeriesStats(n=n, nan=nan, inf=inf, mean=np.nan, median=np.nan, std=np.nan, min=np.nan, max=np.nan, q01=np.nan, q99=np.nan)

    return SeriesStats(
        n=n,
        nan=nan,
        inf=inf,
        mean=float(finite.mean()),
        median=float(finite.median()),
        std=float(finite.std(ddof=0)),
        min=float(finite.min()),
        max=float(finite.max()),
        q01=float(finite.quantile(0.01)),
        q99=float(finite.quantile(0.99)),
    )


def _pick_cols(df: pd.DataFrame, suffixes: tuple[str, ...]) -> list[str]:
    cols = [c for c in df.columns if c != "Date" and any(str(c).endswith(s) for s in suffixes)]
    return sorted(cols)


def _forward_log_return(px: pd.Series, horizon: int) -> pd.Series:
    px = pd.to_numeric(px, errors="coerce").astype("float64")
    out = np.log(px.shift(-horizon) / px)
    return out


def _corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").astype("float64")
    y = pd.to_numeric(b, errors="coerce").astype("float64")
    ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 200:
        return float("nan")
    return float(np.corrcoef(x.loc[ok], y.loc[ok])[0, 1])


def check_a_compare_same_kinds(df_fut3: pd.DataFrame) -> None:
    change_cols = _pick_cols(df_fut3, CHANGE_SUFFIXES)
    level_cols = _pick_cols(df_fut3, LEVEL_SUFFIXES)

    print(f"A) Feature kinds: change={len(change_cols)} level={len(level_cols)}")

    cl_change = [c for c in change_cols if c.startswith("CL=F")]
    cl_level = [c for c in level_cols if c.startswith("CL=F")]

    if cl_change:
        s = _as_series(df_fut3, cl_change[0])
        st = _stats(s)
        print(f"   Example change ({cl_change[0]}): mean={st.mean:.4f} median={st.median:.4f} std={st.std:.4f}")

    if cl_level:
        s = _as_series(df_fut3, cl_level[0])
        st = _stats(s)
        print(f"   Example level ({cl_level[0]}): median={st.median:.4f} (not expected ~0)")


def check_b_inverse_transform_oil(df_fut1: pd.DataFrame, df_fut3: pd.DataFrame) -> None:
    if "CL=F_asinh" not in df_fut3.columns:
        print("B) Inverse-transform: missing CL=F_asinh")
        return

    oil_asinh = _as_series(df_fut3, "CL=F_asinh").replace([np.inf, -np.inf], np.nan)
    asinh_med = float(oil_asinh.dropna().median())
    implied_level = float(np.sinh(asinh_med))

    if "CL=F" not in df_fut1.columns:
        print("B) Inverse-transform: missing CL=F in raw_data_1 Futures")
        return

    oil_level = pd.to_numeric(df_fut1["CL=F"], errors="coerce").astype("float64")
    level_med = float(oil_level.dropna().median())

    print(
        f"B) CL=F_asinh median={asinh_med:.4f} -> sinh(median)~={implied_level:.2f}; raw CL=F median~={level_med:.2f}"
    )


def check_c_nonfinite_and_spikes(df_fut1: pd.DataFrame, df_fut3: pd.DataFrame) -> None:
    cols = [c for c in ("CL=F", "CL=F_asinh", "CL=F_diff") if c in df_fut3.columns]
    if not cols:
        print("C) Non-finite/spikes: missing oil columns")
        return

    for col in cols:
        s = _as_series(df_fut3, col)
        st = _stats(s)
        print(
            f"C) {col}: nan={st.nan} inf={st.inf} min={st.min:.4f} max={st.max:.4f} q01={st.q01:.4f} q99={st.q99:.4f}"
        )

    if "CL=F" in df_fut1.columns:
        oil = pd.to_numeric(df_fut1["CL=F"], errors="coerce").astype("float64")
        n_nonpos = int((oil <= 0).sum())
        min_oil = float(oil.min()) if oil.notna().any() else float("nan")
        print(f"   raw CL=F: count<=0={n_nonpos} min={min_oil:.4f}")


def check_d_lead_lag_corr(df_spdr1: pd.DataFrame, df_spdr3: pd.DataFrame, df_fut3: pd.DataFrame) -> None:
    if "XLK" not in df_spdr1.columns:
        print("D) Lead/lag: missing XLK in raw_data_1 SPDR")
        return

    trading_dates = pd.DatetimeIndex(df_spdr3["Date"]).tz_localize(None)
    spdr1 = df_spdr1.copy()
    spdr1["Date"] = pd.to_datetime(spdr1["Date"]).dt.tz_localize(None)
    spdr1 = spdr1.set_index("Date")

    px = pd.to_numeric(spdr1["XLK"], errors="coerce").reindex(trading_dates)
    target_5d = _forward_log_return(px, horizon=5)

    candidates = [
        c
        for c in ("CL=F_diff", "CL=F_asinh", "CL=F")
        if c in df_fut3.columns
    ]
    if not candidates:
        print("D) Lead/lag: missing CL features")
        return

    fut3 = df_fut3.copy()
    fut3["Date"] = pd.to_datetime(fut3["Date"]).dt.tz_localize(None)
    fut3 = fut3.set_index("Date").reindex(trading_dates)

    lags = [-2, -1, 0, 1, 2]

    print("D) Corr(feature_{t+lag}, XLK_fwd_5d_t)  (best lag should not be +1/+2)")
    for col in candidates:
        vals = []
        for lag in lags:
            r = _corr(fut3[col].shift(lag), target_5d)
            vals.append(r)
        best_lag = lags[int(np.nanargmax(np.abs(vals)))] if any(np.isfinite(vals)) else None
        formatted = " ".join(
            f"{lag:+d}:{(v if np.isfinite(v) else np.nan): .4f}" for lag, v in zip(lags, vals, strict=False)
        )
        print(f"   {col}: {formatted}   best_abs_lag={best_lag}")


def check_e_level_usage(df_fut3: pd.DataFrame) -> None:
    oil_log_cols = [c for c in df_fut3.columns if c.startswith("CL=F_") and "log" in c]
    if oil_log_cols:
        raise AssertionError(f"Forbidden oil log-like columns exist: {oil_log_cols}")

    print("E) Oil level log usage: OK (no CL=F_*log* columns)")


def check_f_spy_timing() -> None:
    spy1_path = RAW1_DIR / "SPY.csv"
    spy3_path = RAW3_DIR / "SPY.parquet"
    spdr3_path = RAW3_DIR / "SPDR.parquet"

    for p in (spy1_path, spy3_path, spdr3_path):
        if not p.exists():
            raise FileNotFoundError(p)

    df_spy1 = _read_csv(spy1_path)
    df_spy3 = _read_parquet(spy3_path)
    df_spdr3 = _read_parquet(spdr3_path)

    trading_dates = pd.DatetimeIndex(df_spdr3["Date"]).tz_localize(None)

    spy1 = df_spy1.copy()
    spy1["Date"] = pd.to_datetime(spy1["Date"]).dt.tz_localize(None)
    spy1 = spy1.set_index("Date")
    px = pd.to_numeric(spy1["SPY"], errors="coerce").reindex(trading_dates)
    r_1d = np.log(px).diff(1)

    spy3 = df_spy3.copy()
    spy3["Date"] = pd.to_datetime(spy3["Date"]).dt.tz_localize(None)
    spy3 = spy3.set_index("Date").reindex(trading_dates)

    if "SPY_logret" not in spy3.columns:
        raise ValueError("Missing SPY_logret in raw_data_3/SPY.parquet")

    feat = pd.to_numeric(spy3["SPY_logret"], errors="coerce")
    corr_expected = _corr(feat, r_1d.shift(1))
    corr_double_shift = _corr(feat, r_1d.shift(2))
    st = _stats(feat)

    print(f"F) SPY timing: corr(logret, r_1d[t-1])={corr_expected:.4f} corr(logret, r_1d[t-2])={corr_double_shift:.4f}")
    print(f"   SPY_logret stats: mean={st.mean:.4f} median={st.median:.4f} std={st.std:.4f}")


def main() -> None:
    fut1_path = RAW1_DIR / "Futures.csv"
    spdr1_path = RAW1_DIR / "SPDR.csv"

    fut3_path = RAW3_DIR / "Futures.parquet"
    spdr3_path = RAW3_DIR / "SPDR.parquet"

    for p in (fut1_path, spdr1_path, fut3_path, spdr3_path):
        if not p.exists():
            raise FileNotFoundError(p)

    df_fut1 = _read_csv(fut1_path)
    df_spdr1 = _read_csv(spdr1_path)

    df_fut3 = _read_parquet(fut3_path)
    df_spdr3 = _read_parquet(spdr3_path)

    check_a_compare_same_kinds(df_fut3)
    check_b_inverse_transform_oil(df_fut1, df_fut3)
    check_c_nonfinite_and_spikes(df_fut1, df_fut3)
    check_d_lead_lag_corr(df_spdr1, df_spdr3, df_fut3)
    check_e_level_usage(df_fut3)
    check_f_spy_timing()


if __name__ == "__main__":
    main()
