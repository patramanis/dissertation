from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW2_DIR = Path("ModularMonolith") / "data" / "raw_data_2"
OUT_PATH = Path("ModularMonolith") / "data" / "labels" / "labels.parquet"
SPDR_PARQUET = RAW2_DIR / "SPDR.parquet"
SPY_PARQUET = RAW2_DIR / "SPY.parquet"


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.tz_localize(None)
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    return df


def _undo_shift_1(level: pd.Series) -> pd.Series:
    return level.shift(-1)


def main() -> None:
    for p in (SPDR_PARQUET, SPY_PARQUET):
        if not p.exists():
            raise FileNotFoundError(p)

    spdr = _read_parquet(SPDR_PARQUET)
    spy = _read_parquet(SPY_PARQUET)

    spdr["Date"] = pd.to_datetime(spdr["Date"]).dt.normalize().dt.tz_localize(None)
    spy["Date"] = pd.to_datetime(spy["Date"]).dt.normalize().dt.tz_localize(None)

    if "SPY" not in spy.columns:
        raise ValueError("Missing SPY column in raw_data_2/SPY.parquet")

    df = spdr.merge(spy[["Date", "SPY"]], on="Date", how="left")
    sectors = [c for c in spdr.columns if c != "Date"]
    horizons = [5, 21, 63]

    out_rows: list[pd.DataFrame] = []

    spy_px = _undo_shift_1(pd.to_numeric(df["SPY"], errors="coerce"))
    for h in horizons:
        spy_ret = (spy_px.shift(-h) / spy_px) - 1.0
        df[f"SPY_ret_{h}d"] = spy_ret

    for sector in sectors:
        px = _undo_shift_1(pd.to_numeric(df[sector], errors="coerce"))
        data = {"Date": df["Date"], "Sector": sector}
        for h in horizons:
            sec_ret = (px.shift(-h) / px) - 1.0
            excess = sec_ret - df[f"SPY_ret_{h}d"]
            data[f"label_excess_{h}d"] = excess
        out_rows.append(pd.DataFrame(data))

    out = pd.concat(out_rows, ignore_index=True)
    out = out.sort_values(["Date", "Sector"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False, engine="pyarrow")
    print(f"Wrote {OUT_PATH} shape={out.shape}")


if __name__ == "__main__":
    main()
