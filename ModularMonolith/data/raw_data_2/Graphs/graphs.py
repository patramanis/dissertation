from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


RAW2_DIR = Path("ModularMonolith") / "data" / "raw_data_2"
GRAPHS_DIR = RAW2_DIR / "Graphs"


def load_parquet_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="raise")
        df = df.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
    else:
        df.index = pd.to_datetime(df.index, errors="raise")
        df = df.sort_index()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def pick_primary_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    if not cols:
        raise ValueError("No columns found")
    for preferred in ("XLK", "SPY"):
        if preferred in cols:
            return preferred
    return cols[0]


def save_zoom_in_line(df: pd.DataFrame, col: str, out_path: Path, window_days: int = 15) -> None:
    s = pd.to_numeric(df[col], errors="coerce")
    if s.dropna().empty:
        return

    last_valid = s.dropna().index.max()
    df_win = df.loc[:last_valid].tail(window_days)

    feat = pd.to_numeric(df_win[col], errors="coerce")
    targ = feat.shift(-1)

    plt.figure(figsize=(10, 4))
    plt.plot(df_win.index, feat, label=f"Feature {col} (t)")
    plt.plot(df_win.index, targ, label=f"Target {col} (t+1)")
    plt.title(f"Zoom-In Leakage Check ({col})")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_scatter_t_t1(df: pd.DataFrame, col: str, out_path: Path) -> None:
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.dropna()
    if len(s) < 3:
        return

    x = s.iloc[:-1].to_numpy()
    y = s.iloc[1:].to_numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=6, alpha=0.35)

    # 45-degree reference line (based on data range)
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.7)

    plt.title(f"Scatter: {col}(t) vs {col}(t+1)")
    plt.xlabel(f"{col} at t")
    plt.ylabel(f"{col} at t+1")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_corr_heatmap(df: pd.DataFrame, primary: str, out_path: Path, max_cols: int = 20) -> None:
    cols = list(df.columns)
    if not cols:
        return
    if len(cols) > max_cols:
        keep = cols[:max_cols]
        if primary not in keep:
            keep = [primary] + keep[:-1]
        cols = keep

    work = df[cols].apply(pd.to_numeric, errors="coerce")
    work[f"TARGET_{primary}_t+1"] = pd.to_numeric(df[primary], errors="coerce").shift(-1)

    work = work.dropna(how="all")
    corr = work.corr(method="spearman")

    plt.figure(figsize=(max(6, 0.6 * corr.shape[1]), max(5, 0.6 * corr.shape[0])))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, vmin=-1.0, vmax=1.0, square=True)
    plt.title("Spearman Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_missingness_timeline(df: pd.DataFrame, out_path: Path) -> None:
    cols = list(df.columns)
    if not cols:
        return

    miss = df[cols].isna().sum(axis=1)
    plt.figure(figsize=(10, 3))
    plt.plot(df.index, miss, linewidth=1)
    plt.title("Missingness Over Time (#NaNs across columns)")
    plt.xlabel("Date")
    plt.ylabel("NaN count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_nan_fraction_bar(df: pd.DataFrame, out_path: Path, max_cols: int = 50) -> None:
    cols = list(df.columns)
    if not cols:
        return
    if len(cols) > max_cols:
        cols = cols[:max_cols]

    frac = df[cols].isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(max(6, 0.35 * len(frac)), 3.5))
    plt.bar(frac.index.astype(str), frac.values)
    plt.title("NaN Fraction per Column")
    plt.ylabel("NaN fraction")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    if not RAW2_DIR.exists():
        raise FileNotFoundError(RAW2_DIR)

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted([p for p in RAW2_DIR.glob("*.parquet") if p.is_file()])
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {RAW2_DIR}")

    for p in parquet_files:
        name = p.stem
        out_dir = GRAPHS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        df = load_parquet_file(p)
        if df.empty:
            continue

        primary = pick_primary_column(df)

        save_zoom_in_line(df, primary, out_dir / "01_zoom_in_line.png")
        save_scatter_t_t1(df, primary, out_dir / "02_scatter_t_vs_t1.png")
        save_corr_heatmap(df, primary, out_dir / "03_corr_heatmap.png")
        save_missingness_timeline(df, out_dir / "04_missingness_timeline.png")
        save_nan_fraction_bar(df, out_dir / "05_nan_fraction.png")

        print(f"Generated graphs for {name} -> {out_dir}")


if __name__ == "__main__":
    main()
