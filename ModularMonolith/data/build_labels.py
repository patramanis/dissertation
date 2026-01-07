from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABELS_DIR = Path(__file__).resolve().parent
DATA_DIR = LABELS_DIR
RAW2_DIR = DATA_DIR / "raw_data_2"
OUT_DIR = LABELS_DIR / "labels"
SPDR_PARQUET = RAW2_DIR / "SPDR.parquet"
SPY_PARQUET = RAW2_DIR / "SPY.parquet"


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.tz_localize(None)
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    return df


def build_excess_labels(
    spdr_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    horizons: list[int],
    *,
    cost_bps: float,
) -> dict[int, pd.DataFrame]:
    if "Date" not in spdr_df.columns:
        raise ValueError("spdr_df must contain Date")
    if "Date" not in spy_df.columns or "SPY" not in spy_df.columns:
        raise ValueError("spy_df must contain Date and SPY")
    if not horizons:
        raise ValueError("horizons must be non-empty")

    sectors = [c for c in spdr_df.columns if c != "Date"]
    if not sectors:
        raise ValueError("SPDR sector set is empty")

    spdr = spdr_df.copy()
    spy = spy_df[["Date", "SPY"]].copy()

    spdr["Date"] = pd.to_datetime(spdr["Date"], errors="raise").dt.normalize().dt.tz_localize(None)
    spy["Date"] = pd.to_datetime(spy["Date"], errors="raise").dt.normalize().dt.tz_localize(None)

    df = spdr.merge(spy, on="Date", how="left")
    df = df.sort_values("Date").reset_index(drop=True)

    spy_px = pd.to_numeric(df["SPY"], errors="coerce")
    spy_rets: dict[int, pd.Series] = {}
    for h in horizons:
        spy_rets[h] = (spy_px.shift(-h) / spy_px) - 1.0

    cost = float(cost_bps) / 10_000.0
    out_by_h: dict[int, pd.DataFrame] = {}
    n_dates = int(df.shape[0])

    def _rel_rank_for_date(excess: pd.Series) -> pd.Series:
        mask = excess.notna()
        if int(mask.sum()) < 2:
            return pd.Series(pd.NA, index=excess.index, dtype="Int8")

        valid = excess[mask]
        q60 = float(valid.quantile(0.60))
        q80 = float(valid.quantile(0.80))

        rel = pd.Series(pd.NA, index=excess.index, dtype="Int8")
        rel.loc[mask & (excess <= cost)] = 0
        rel.loc[mask & (excess > cost) & (excess <= q60)] = 1
        rel.loc[mask & (excess > q60) & (excess <= q80)] = 2
        rel.loc[mask & (excess > q80)] = 3
        return rel

    for h in horizons:
        out_rows: list[pd.DataFrame] = []
        spy_ret = spy_rets[h]

        for sector in sectors:
            px = pd.to_numeric(df[sector], errors="coerce")
            sec_ret = (px.shift(-h) / px) - 1.0
            label_excess = sec_ret - spy_ret

            y_gate = pd.Series(pd.NA, index=label_excess.index, dtype="Int8")
            notna = label_excess.notna()
            y_gate.loc[notna] = (label_excess.loc[notna] > cost).astype("int8")

            out_rows.append(
                pd.DataFrame(
                    {
                        "Date": df["Date"],
                        "Sector": sector,
                        "label_excess": label_excess,
                        "y_gate": y_gate,
                        "cost_bps": float(cost_bps),
                        "horizon": int(h),
                    }
                )
            )

        out_h = pd.concat(out_rows, ignore_index=True)
        out_h = out_h.sort_values(["Date", "Sector"]).reset_index(drop=True)

        out_h["rel_rank"] = (
            out_h.groupby("Date", sort=False)["label_excess"]
            .apply(_rel_rank_for_date)
            .reset_index(level=0, drop=True)
            .astype("Int8")
        )

        out_by_h[int(h)] = out_h

    return out_by_h


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
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[5, 21, 63],
        help="Horizons in trading days (default: 5 21 63).",
    )
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Cost threshold in bps for y_gate (default: 10).")
    args = parser.parse_args()

    for p in (SPDR_PARQUET, SPY_PARQUET):
        if not p.exists():
            raise FileNotFoundError(p)

    spdr = _read_parquet(SPDR_PARQUET)
    spy = _read_parquet(SPY_PARQUET)

    spdr = spdr.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    spy = spy.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    assert (
        spdr["Date"].is_monotonic_increasing
        and spdr["Date"].is_unique
        and spy["Date"].is_monotonic_increasing
        and spy["Date"].is_unique
    )

    spdr["Date"] = pd.to_datetime(spdr["Date"]).dt.normalize().dt.tz_localize(None)
    spy["Date"] = pd.to_datetime(spy["Date"]).dt.normalize().dt.tz_localize(None)

    if "SPY" not in spy.columns:
        raise ValueError("Missing SPY column in raw_data_2/SPY.parquet")

    sectors = [c for c in spdr.columns if c != "Date"]
    assert len(sectors) > 0

    horizons = [int(h) for h in args.horizons]

    df_diag = spdr.merge(spy[["Date", "SPY"]], on="Date", how="left")
    if args.diagnostic:
        spy_nan_frac = float(pd.to_numeric(df_diag["SPY"], errors="coerce").isna().mean())
        print(f"SPY NaN fraction after merge: {spy_nan_frac:.6f}")
        sec_nan_frac = float(pd.to_numeric(df_diag[sectors], errors="coerce").isna().mean().mean())
        print(f"Sector NaN fraction (avg across sectors): {sec_nan_frac:.6f}")

    spy_px = pd.to_numeric(df_diag["SPY"], errors="coerce")
    for h in horizons:
        df_diag[f"SPY_ret_{h}d"] = (spy_px.shift(-h) / spy_px) - 1.0

    if args.diagnostic:
        _print_timing_diagnostic(
            df=df_diag,
            sectors=sectors,
            horizons=horizons,
            sector_for_check=args.diagnostic_sector,
            n_samples=int(args.diagnostic_n),
            seed=int(args.diagnostic_seed),
        )

    out_by_h = build_excess_labels(spdr, spy, horizons, cost_bps=float(args.cost_bps))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expected_cols = {"Date", "Sector", "label_excess", "y_gate", "rel_rank", "cost_bps", "horizon"}
    assert all(set(out_h.columns) == expected_cols for out_h in out_by_h.values())
    assert all(int(out_h.shape[0]) == int(spdr.shape[0]) * len(sectors) for out_h in out_by_h.values())

    for h in horizons:
        out_h = out_by_h[int(h)]
        # Single canonical format expected by data_engineering_1.py / dataset_shaping.py.
        # Keep helpful columns (y_gate/rel_rank/metadata) while using horizon-specific target name.
        out_one = out_h.rename(columns={"label_excess": f"label_excess_{int(h)}d"})
        out_path = OUT_DIR / f"h{int(h)}.parquet"
        out_one.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
