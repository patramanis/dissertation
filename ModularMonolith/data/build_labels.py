from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ModularMonolith.data.train_window import TRAIN_DATE_END, TRAIN_DATE_START

try:
    from ModularMonolith import build_id

    BUILD_ID = build_id(__file__)
except Exception:
    BUILD_ID = str(Path(__file__).resolve())


LABELS_DIR = Path(__file__).resolve().parent
DATA_DIR = LABELS_DIR
OUT_DIR = LABELS_DIR / "labels"

UNSHIFTED_LABEL_INPUTS_DIR = OUT_DIR / "unshifted_lagged_raw_data"

LABEL_DATE_START = TRAIN_DATE_START
LABEL_DATE_END = TRAIN_DATE_END

EXPECTED_SECTORS: tuple[str, ...] = (
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLY",
    "XLV",
    "XLU",
)


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.tz_localize(None)
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    return df


def _filter_label_date_range(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise ValueError("Expected Date column for label date filtering")
    dt = pd.to_datetime(df["Date"], errors="raise")
    mask = (dt >= LABEL_DATE_START) & (dt <= LABEL_DATE_END)
    return df.loc[mask].copy()


def _load_unshifted_prices_for_horizon(h: int) -> pd.DataFrame:
    path = UNSHIFTED_LABEL_INPUTS_DIR / f"prices_h{int(h)}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing unshifted label input prices for horizon={int(h)} at {path}. "
            "Run data_optimization_1.py to generate labels/unshifted_lagged_raw_data." 
        )

    df = _read_parquet(path)

    required = {"Date", "SPY", *EXPECTED_SECTORS}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Unshifted label prices missing columns in {path}: {missing}")

    df = df[["Date", "SPY", *EXPECTED_SECTORS]].copy()

    value_cols = ["SPY", *EXPECTED_SECTORS]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[value_cols] = df[value_cols].ffill(limit=5)

    df = _filter_label_date_range(df)
    return df


def build_excess_labels(
    prices_df: pd.DataFrame,
    horizons: list[int],
    *,
    cost_bps: float,
) -> dict[int, pd.DataFrame]:
    if "Date" not in prices_df.columns:
        raise ValueError("prices_df must contain Date")
    if "SPY" not in prices_df.columns:
        raise ValueError("prices_df must contain SPY")
    if not horizons:
        raise ValueError("horizons must be non-empty")

    sectors = [s for s in EXPECTED_SECTORS if s in prices_df.columns]
    if len(sectors) != len(EXPECTED_SECTORS):
        raise ValueError(f"prices_df missing expected sectors: have={sectors} expected={list(EXPECTED_SECTORS)}")

    df = prices_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.normalize().dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)

    spy_px = pd.to_numeric(df["SPY"], errors="coerce")
    spy_px_asof = spy_px.shift(1)
    spy_rets: dict[int, pd.Series] = {}
    for h in horizons:
        spy_rets[h] = (spy_px_asof.shift(-h) / spy_px_asof) - 1.0

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
            px_asof = px.shift(1)
            sec_ret = (px_asof.shift(-h) / px_asof) - 1.0
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

    print("\nTIMING DIAGNOSTIC (unshifted label inputs)")
    print(f"Sector check: {sec_name} | samples={n_samples} | seed={seed}")

    for h in horizons:
        denom_spy = px_spy.shift(1)
        numer_spy = denom_spy.shift(-h)
        denom_sec = px_sec.shift(1)
        numer_sec = denom_sec.shift(-h)

        spy_ret = (numer_spy / denom_spy) - 1.0
        sec_ret = (numer_sec / denom_sec) - 1.0
        excess = sec_ret - spy_ret

        ok = denom_spy.notna() & numer_spy.notna() & denom_sec.notna() & numer_sec.notna()
        ok_idx = ok[ok].index
        if len(ok_idx) == 0:
            print(f"\n[h={h}] No valid rows for diagnostic")
            continue

        sample_idx = ok_idx.to_series().sample(n=min(n_samples, len(ok_idx)), random_state=seed).sort_values()
        print(f"\n[h={h}] Showing Date=T with denom=Price(T-1), numer=Price(T+h-1)")
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

    for h in horizons:
        denom_spy = px_spy.shift(1)
        denom_sec = px_sec.shift(1)
        spy_ret = (denom_spy.shift(-h) / denom_spy) - 1.0
        sec_ret = (denom_sec.shift(-h) / denom_sec) - 1.0
        excess = sec_ret - spy_ret
        tail = excess.tail(h)
        non_nan = int(pd.to_numeric(tail, errors="coerce").notna().sum())
        if non_nan != 0:
            raise RuntimeError(
                f"Tail forward targets should be all-NaN for h={h}, but found non-NaN count={non_nan}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build sector excess-return labels from unshifted label input prices, "
            "using as-of (Close(T-1)) timing so Date=T labels align with PIT features."
        )
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Print timing diagnostic examples to verify denom=Price(T-1) and numer=Price(T+h-1).",
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

    print(f"[build_labels] BUILD_ID={BUILD_ID}")

    horizons = [int(h) for h in args.horizons]

    if args.diagnostic:
        prices_diag = _load_unshifted_prices_for_horizon(horizons[0])
        sectors = [s for s in EXPECTED_SECTORS]
        spy_nan_frac = float(pd.to_numeric(prices_diag["SPY"], errors="coerce").isna().mean())
        print(f"SPY NaN fraction in unshifted label inputs: {spy_nan_frac:.6f}")
        sec_nan_frac = float(prices_diag[sectors].apply(pd.to_numeric, errors="coerce").isna().mean().mean())
        print(f"Sector NaN fraction (avg across sectors) in unshifted label inputs: {sec_nan_frac:.6f}")

        _print_timing_diagnostic(
            df=prices_diag,
            sectors=sectors,
            horizons=horizons,
            sector_for_check=args.diagnostic_sector,
            n_samples=int(args.diagnostic_n),
            seed=int(args.diagnostic_seed),
        )

    out_by_h: dict[int, pd.DataFrame] = {}
    for h in horizons:
        prices = _load_unshifted_prices_for_horizon(int(h))
        out_one = build_excess_labels(prices, [int(h)], cost_bps=float(args.cost_bps))[int(h)]
        out_by_h[int(h)] = out_one

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expected_cols = {"Date", "Sector", "label_excess", "y_gate", "rel_rank", "cost_bps", "horizon"}
    assert all(set(out_h.columns) == expected_cols for out_h in out_by_h.values())
    assert all(int(out_h["Sector"].nunique()) == len(EXPECTED_SECTORS) for out_h in out_by_h.values())

    for h in horizons:
        out_h = out_by_h[int(h)]
        out_one = out_h
        out_path = OUT_DIR / f"h{int(h)}.parquet"
        out_one.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
