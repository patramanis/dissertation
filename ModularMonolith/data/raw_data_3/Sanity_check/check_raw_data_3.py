from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW1_DIR = Path("ModularMonolith") / "data" / "raw_data_1"
RAW2_DIR = Path("ModularMonolith") / "data" / "raw_data_2"
RAW3_DIR = Path("ModularMonolith") / "data" / "raw_data_3"


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    return df


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    return df


def _corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 50:
        return float("nan")
    return float(np.corrcoef(x.loc[ok], y.loc[ok])[0, 1])


def no_double_shift_spy_check() -> None:
    spy_raw1 = _read_csv(RAW1_DIR / "SPY.csv").set_index("Date")
    spy_raw1.index = spy_raw1.index.normalize()

    spy_raw2 = _read_parquet(RAW2_DIR / "SPY.parquet").set_index("Date")
    spy_raw2.index = spy_raw2.index.normalize()

    spy_raw3 = _read_parquet(RAW3_DIR / "SPY.parquet").set_index("Date")
    spy_raw3.index = spy_raw3.index.normalize()

    if "SPY" not in spy_raw1.columns or "SPY" not in spy_raw2.columns:
        raise ValueError("SPY column not found in SPY datasets")
    if "SPY_logret" not in spy_raw3.columns:
        raise ValueError("SPY_logret not found in raw_data_3 SPY")

    trading_dates = spy_raw2.index

    px = pd.to_numeric(spy_raw1["SPY"], errors="coerce").reindex(trading_dates)
    r_1d = np.log(px).diff(1)

    feat = pd.to_numeric(spy_raw3["SPY_logret"], errors="coerce").reindex(trading_dates)

    corr_expected = _corr(feat, r_1d.shift(1))
    corr_double_shift = _corr(feat, r_1d.shift(2))

    print(
        f"No-double-shift check (SPY): corr(feature, r_1d[t-1])={corr_expected:.4f} corr(feature, r_1d[t-2])={corr_double_shift:.4f}"
    )


def no_double_shift_check() -> None:
    spdr_raw1 = _read_csv(RAW1_DIR / "SPDR.csv").set_index("Date")
    spdr_raw1.index = spdr_raw1.index.normalize()

    spdr_raw2 = _read_parquet(RAW2_DIR / "SPDR.parquet").set_index("Date")
    spdr_raw2.index = spdr_raw2.index.normalize()

    spdr_raw3 = _read_parquet(RAW3_DIR / "SPDR.parquet").set_index("Date")
    spdr_raw3.index = spdr_raw3.index.normalize()

    if "XLK" not in spdr_raw1.columns or "XLK" not in spdr_raw2.columns:
        raise ValueError("XLK not found in SPDR datasets")
    if "XLK_logret" not in spdr_raw3.columns:
        raise ValueError("XLK_logret not found in raw_data_3 SPDR")

    trading_dates = spdr_raw2.index

    px = pd.to_numeric(spdr_raw1["XLK"], errors="coerce").reindex(trading_dates)
    r_1d = np.log(px).diff(1)

    feat = pd.to_numeric(spdr_raw3["XLK_logret"], errors="coerce").reindex(trading_dates)

    corr_expected = _corr(feat, r_1d.shift(1))
    corr_double_shift = _corr(feat, r_1d.shift(2))

    print(f"No-double-shift check (XLK): corr(feature, r_1d[t-1])={corr_expected:.4f} corr(feature, r_1d[t-2])={corr_double_shift:.4f}")

    label_5d = np.log(px.shift(-5) / px)
    label_21d = np.log(px.shift(-21) / px)
    label_63d = np.log(px.shift(-63) / px)

    if label_5d.index.equals(trading_dates) is False:
        raise AssertionError("Label index mismatch")

    sample_date = trading_dates[200] if len(trading_dates) > 300 else trading_dates[len(trading_dates) // 2]
    print(
        f"Example definitions @ {sample_date.date()}: feature uses prices up to t-1; label_5d is log(P[t+5]/P[t])"
    )


def oil_robustness_check() -> None:
    fut3_path = RAW3_DIR / "Futures.parquet"
    if not fut3_path.exists():
        raise FileNotFoundError(f"Missing {fut3_path}")

    df = _read_parquet(fut3_path).set_index("Date")
    df.index = df.index.normalize()

    if "CL=F" not in df.columns:
        raise ValueError("CL=F not present in raw_data_3 Futures")
    if "CL=F_asinh" not in df.columns or "CL=F_diff" not in df.columns:
        raise ValueError("Missing CL=F_asinh and/or CL=F_diff in raw_data_3 Futures")

    oil = pd.to_numeric(df["CL=F"], errors="coerce")
    n_nonpos = int((oil <= 0).sum())
    min_oil = float(oil.min()) if oil.notna().any() else float("nan")

    asinh = pd.to_numeric(df["CL=F_asinh"], errors="coerce")
    diff = pd.to_numeric(df["CL=F_diff"], errors="coerce")

    present = oil.notna()
    ok_asinh = np.isfinite(asinh[present])
    both_present = present & present.shift(1, fill_value=False)
    ok_diff = np.isfinite(diff[both_present])

    if not bool(ok_asinh.all()):
        raise AssertionError("Non-finite values found in CL=F_asinh where CL=F is present")
    if not bool(ok_diff.all()):
        raise AssertionError("Non-finite values found in CL=F_diff where both CL=F[t] and CL=F[t-1] are present")

    bad_cols = [c for c in df.columns if c.startswith("CL=F_") and "log" in c]
    if bad_cols:
        raise AssertionError(f"Found forbidden log-type transform on oil level: {bad_cols}")

    print(f"Oil robustness check: CL=F<=0 count={n_nonpos} min={min_oil:.4f} (asinh/diff finite, no log(level) used)")


def main() -> None:
    if not RAW3_DIR.exists():
        raise FileNotFoundError(f"Missing {RAW3_DIR}; build raw_data_3 first")

    no_double_shift_check()
    no_double_shift_spy_check()
    oil_robustness_check()


if __name__ == "__main__":
    main()
