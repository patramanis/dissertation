from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABELS_DIR = Path(__file__).resolve().parent
DATA_DIR = LABELS_DIR.parent
RAW2_DIR = DATA_DIR / "raw_data_2"
OUT_DIR = LABELS_DIR
SPDR_PARQUET = RAW2_DIR / "SPDR.parquet"
SPY_PARQUET = RAW2_DIR / "SPY.parquet"


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.tz_localize(None)
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    return df


def _print_timing_diagnostic(
    *,
    df: pd.DataFrame,
    sectors: list[str],
    horizons: list[int],
    sector_for_check: str | None,
    n_samples: int,
    seed: int,
) -> None:

    if "Date" not in df.columns or "SPY" not in df.columns:
        raise ValueError("diagnostic expects df to contain Date and SPY")
    if not sectors:
        raise ValueError("diagnostic expects non-empty sectors")

    sec_name = sector_for_check if sector_for_check is not None else sectors[0]
    if sec_name not in df.columns:
        raise ValueError(f"diagnostic sector '{sec_name}' not found in SPDR columns")

    px_spy = pd.to_numeric(df["SPY"], errors="coerce")
    px_sec = pd.to_numeric(df[sec_name], errors="coerce")
    dates = pd.to_datetime(df["Date"], errors="raise")

    print("\nTIMING DIAGNOSTIC (raw_data_2 already has P_{t-1} at Date=T)")
    print(f"Sector check: {sec_name} | samples={n_samples} | seed={seed}")

    for h in horizons:
        denom_spy = px_spy
        numer_spy = px_spy.shift(-h)
        denom_sec = px_sec
        numer_sec = px_sec.shift(-h)

        spy_ret = (numer_spy / denom_spy) - 1.0
        sec_ret = (numer_sec / denom_sec) - 1.0
        excess = sec_ret - spy_ret

        ok = denom_spy.notna() & numer_spy.notna() & denom_sec.notna() & numer_sec.notna()
        ok_idx = ok[ok].index
        if len(ok_idx) == 0:
            print(f"\n[h={h}] No valid rows for diagnostic")
            continue

        sample_idx = ok_idx.to_series().sample(n=min(n_samples, len(ok_idx)), random_state=seed).sort_values()
        print(f"\n[h={h}] Showing Date=T with denom=P_(t-1), numer=P_(t+h-1)")
        for i in sample_idx.tolist():
            dt = dates.iloc[i].date().isoformat()
            d_spy = float(denom_spy.iloc[i])
            n_spy = float(numer_spy.iloc[i])
            r_spy = float(spy_ret.iloc[i])
            d_sec = float(denom_sec.iloc[i])
            n_sec = float(numer_sec.iloc[i])
            r_sec = float(sec_ret.iloc[i])
            ex = float(excess.iloc[i])
            print(
                f"  Date={dt} | SPY denom={d_spy:.4f} numer={n_spy:.4f} ret={r_spy:+.6f}"
                f" | {sec_name} denom={d_sec:.4f} numer={n_sec:.4f} ret={r_sec:+.6f}"
                f" | excess={ex:+.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sector excess-return labels from raw_data_2 (already shifted).")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Print timing diagnostic examples to verify denom=P_{t-1} and numer=P_{t+h-1}.",
    )
    parser.add_argument("--diagnostic-n", type=int, default=5, help="Number of sample rows per horizon.")
    parser.add_argument("--diagnostic-seed", type=int, default=7, help="Random seed for sampling diagnostic rows.")
    parser.add_argument(
        "--diagnostic-sector",
        type=str,
        default=None,
        help="Optional sector column name to use for the sector diagnostic (default: first sector column).",
    )
    args = parser.parse_args()

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

    spy_px = pd.to_numeric(df["SPY"], errors="coerce")
    for h in horizons:
        spy_ret = (spy_px.shift(-h) / spy_px) - 1.0
        df[f"SPY_ret_{h}d"] = spy_ret

    if args.diagnostic:
        _print_timing_diagnostic(
            df=df,
            sectors=sectors,
            horizons=horizons,
            sector_for_check=args.diagnostic_sector,
            n_samples=int(args.diagnostic_n),
            seed=int(args.diagnostic_seed),
        )

    for sector in sectors:
        px = pd.to_numeric(df[sector], errors="coerce")
        data = {"Date": df["Date"], "Sector": sector}
        for h in horizons:
            sec_ret = (px.shift(-h) / px) - 1.0
            excess = sec_ret - df[f"SPY_ret_{h}d"]
            data[f"label_excess_{h}d"] = excess
        out_rows.append(pd.DataFrame(data))

    out = pd.concat(out_rows, ignore_index=True)
    out = out.sort_values(["Date", "Sector"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for h in horizons:
        col = f"label_excess_{h}d"
        out_h = out[["Date", "Sector", col]].copy()
        out_path = OUT_DIR / f"h{h}.parquet"
        out_h.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"Wrote {out_path} shape={out_h.shape}")


if __name__ == "__main__":
    main()
