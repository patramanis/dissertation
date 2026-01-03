from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


RAW3_DIR = Path("ModularMonolith") / "data" / "raw_data_3"
GRAPHS_DIR = RAW3_DIR / "Graphs"


HIST_SUFFIXES = ("_logret", "_diff", "_asinh")


def load_parquet_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="raise")
        df = df.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
    else:
        df.index = pd.to_datetime(df.index, errors="raise")
        df = df.sort_index()

    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def _chunked(cols: list[str], chunk_size: int) -> list[list[str]]:
    return [cols[i : i + chunk_size] for i in range(0, len(cols), chunk_size)]


def pick_transformed_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if any(str(c).endswith(s) for s in HIST_SUFFIXES)]
    return sorted(cols)


def save_histograms(df: pd.DataFrame, cols: list[str], out_dir: Path) -> None:
    if not cols:
        return

    work = df[cols].apply(pd.to_numeric, errors="coerce")

    pages = _chunked(cols, chunk_size=12)
    for page_idx, page_cols in enumerate(pages, start=1):
        n = len(page_cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.2 * nrows))
        axes = np.asarray(axes).reshape(-1)

        for ax, col in zip(axes, page_cols, strict=False):
            s = pd.to_numeric(work[col], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                ax.set_axis_off()
                continue
            sns.histplot(s, bins=120, kde=True, stat="density", ax=ax)
            ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
            ax.set_title(col)
            ax.set_xlabel("")
            ax.set_ylabel("")

        for ax in axes[len(page_cols) :]:
            ax.set_axis_off()

        fig.suptitle("Distributions (hist + KDE)", y=1.02)
        fig.tight_layout()
        out_path = out_dir / f"01_histograms_p{page_idx:02d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_transformed_timeseries(df: pd.DataFrame, cols: list[str], out_dir: Path) -> None:
    if not cols:
        return

    work = df[cols].apply(pd.to_numeric, errors="coerce")

    pages = _chunked(cols, chunk_size=6)
    for page_idx, page_cols in enumerate(pages, start=1):
        n = len(page_cols)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 2.4 * n), sharex=True)
        axes = np.asarray(axes).reshape(-1)

        for ax, col in zip(axes, page_cols, strict=False):
            s = pd.to_numeric(work[col], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan)
            ax.plot(s.index, s.values, linewidth=0.8, alpha=0.85)
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
            ax.set_title(col)
            ax.set_ylabel("")

        fig.suptitle("Transformed series over time", y=1.01)
        fig.tight_layout()
        out_path = out_dir / f"02_timeseries_p{page_idx:02d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_corr_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    cols = list(df.columns)
    if not cols:
        return

    work = df[cols].apply(pd.to_numeric, errors="coerce")
    work = work.dropna(how="all")
    if work.empty:
        return

    corr = work.corr(method="spearman")

    plt.figure(figsize=(max(7, 0.55 * corr.shape[1]), max(6, 0.55 * corr.shape[0])))
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
    plt.figure(figsize=(12, 3))
    plt.plot(df.index, miss, linewidth=1)
    plt.title("Missingness Over Time (#NaNs across columns)")
    plt.xlabel("Date")
    plt.ylabel("NaN count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_nan_fraction_bar(df: pd.DataFrame, out_path: Path, max_cols: int = 80) -> None:
    cols = list(df.columns)
    if not cols:
        return

    if len(cols) > max_cols:
        cols = cols[:max_cols]

    frac = df[cols].isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(max(7, 0.32 * len(frac)), 3.8))
    plt.bar(frac.index.astype(str), frac.values)
    plt.title("NaN Fraction per Column")
    plt.ylabel("NaN fraction")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    if not RAW3_DIR.exists():
        raise FileNotFoundError(RAW3_DIR)

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted([p for p in RAW3_DIR.glob("*.parquet") if p.is_file()])
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {RAW3_DIR}")

    for p in parquet_files:
        name = p.stem
        out_dir = GRAPHS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        df = load_parquet_file(p)
        if df.empty:
            continue

        transformed_cols = pick_transformed_columns(df)

        save_histograms(df, transformed_cols, out_dir)
        save_transformed_timeseries(df, transformed_cols, out_dir)
        save_corr_heatmap(df, out_dir / "03_corr_heatmap.png")
        save_missingness_timeline(df, out_dir / "04_missingness_timeline.png")
        save_nan_fraction_bar(df, out_dir / "05_nan_fraction.png")

        print(f"Generated graphs for {name} -> {out_dir}")


if __name__ == "__main__":
    main()
