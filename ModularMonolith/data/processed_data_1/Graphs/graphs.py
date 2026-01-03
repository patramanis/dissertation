from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None


PROCESSED_DIR = Path("ModularMonolith") / "data" / "processed_data_1"
GRAPHS_DIR = PROCESSED_DIR / "Graphs"


def _safe_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:180] if len(name) > 180 else name


def _maybe_style() -> None:
    if sns is not None:
        sns.set_style("whitegrid")


def _infer_horizon_tag(path: Path) -> str:
    m = re.search(r"h(5|21|63)", path.stem)
    if not m:
        raise ValueError(f"Cannot infer horizon from filename: {path.name}")
    return f"h{m.group(1)}"


def _pick_sectors(df: pd.DataFrame) -> tuple[str, str]:
    sectors = sorted([s for s in df["Sector"].dropna().unique().tolist() if isinstance(s, str)])
    if "XLK" in sectors:
        volatile = "XLK"
    else:
        volatile = sectors[0] if sectors else ""

    if "XLP" in sectors:
        defensive = "XLP"
    else:
        defensive = sectors[1] if len(sectors) > 1 else (sectors[0] if sectors else "")

    return volatile, defensive


def _is_rank_feature(name: str, s: pd.Series) -> bool:
    if name.startswith("rank_") or name.endswith("_rank"):
        return True
    x = pd.to_numeric(s, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return False
    return (x.min() >= -1e-6) and (x.max() <= 1 + 1e-6)


def _compute_daily_nan_frac(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("Date", sort=True)[col]
    return g.apply(lambda s: float(s.isna().mean()))


def _plot_timeseries_two_sectors(
    df: pd.DataFrame,
    feature: str,
    out_path: Path,
    volatile: str,
    defensive: str,
) -> None:
    panel = df.loc[df["Sector"].isin([volatile, defensive]), ["Date", "Sector", feature]].copy()
    if panel.empty:
        return

    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce")
    panel = panel.dropna(subset=["Date"])
    panel[feature] = pd.to_numeric(panel[feature], errors="coerce")

    piv = panel.pivot_table(index="Date", columns="Sector", values=feature, aggfunc="last").sort_index()

    fig, ax = plt.subplots(figsize=(14, 4))
    for col in [volatile, defensive]:
        if col in piv.columns:
            ax.plot(piv.index, piv[col], linewidth=1.0, label=col)

    is_rank = _is_rank_feature(feature, df[feature])
    if is_rank:
        ax.axhline(0.5, color="black", linewidth=0.8, alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
    elif "_z" in feature or feature.endswith("z"):
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)

    _stack_src = piv[[c for c in [volatile, defensive] if c in piv.columns]]
    try:
        y = _stack_src.stack(future_stack=True).dropna()
    except (TypeError, ValueError):
        y = _stack_src.stack(dropna=True)
    if not y.empty:
        thr = float(y.abs().quantile(0.999))
        if math.isfinite(thr) and thr > 0:
            mask = piv.abs() > thr
            for c in mask.columns:
                if c in piv.columns:
                    idx = piv.index[mask[c].fillna(False)]
                    ax.scatter(idx, piv.loc[idx, c], s=10, alpha=0.6)
            ax.set_title(f"{feature} | spikes marked if |x| > p99.9={thr:.3g}")
        else:
            ax.set_title(feature)
    else:
        ax.set_title(feature)

    ax.set_xlabel("Date")
    ax.set_ylabel(feature)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_hist(
    df: pd.DataFrame,
    feature: str,
    out_path: Path,
) -> None:
    x = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(7, 4))

    if x.empty:
        ax.set_title(f"{feature} | EMPTY")
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return

    is_rank = _is_rank_feature(feature, df[feature])
    if not is_rank:
        lo, hi = x.quantile([0.01, 0.99]).tolist()
        if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
            x = x.clip(lower=lo, upper=hi)

    if sns is not None:
        sns.histplot(x=x, bins=60, kde=False, ax=ax)
    else:
        ax.hist(x.to_numpy(), bins=60)

    ax.set_title(f"{feature} | n={len(x)}")
    ax.set_xlabel(feature)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_missingness(
    df: pd.DataFrame,
    feature: str,
    out_path: Path,
) -> None:
    s = _compute_daily_nan_frac(df, feature)
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(s.index, s.values, linewidth=1.0)
    ax.set_ylim(-0.02, 1.02)
    overall = float(pd.to_numeric(df[feature], errors="coerce").isna().mean())
    ax.set_title(f"{feature} | daily NaN fraction (across sectors) | overall={overall:.3%}")
    ax.set_xlabel("Date")
    ax.set_ylabel("NaN fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_sector_boxplot(df: pd.DataFrame, feature: str, out_path: Path) -> None:
    tmp = df[["Sector", feature]].copy()
    tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["Sector"])

    fig, ax = plt.subplots(figsize=(10, 4))

    if tmp[feature].dropna().empty:
        ax.set_title(f"{feature} | EMPTY")
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return

    is_rank = _is_rank_feature(feature, df[feature])
    if not is_rank:
        x = tmp[feature].replace([np.inf, -np.inf], np.nan).dropna()
        lo, hi = x.quantile([0.01, 0.99]).tolist()
        if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
            tmp[feature] = tmp[feature].clip(lower=lo, upper=hi)

    order = sorted(tmp["Sector"].dropna().unique().tolist())
    if sns is not None:
        sns.boxplot(data=tmp, x="Sector", y=feature, order=order, ax=ax)
    else:
        data = [tmp.loc[tmp["Sector"] == s, feature].dropna().to_numpy() for s in order]
        ax.boxplot(data, labels=order, showfliers=False)

    ax.set_title(f"{feature} distribution by sector (clipped p1–p99 if non-rank)")
    ax.set_xlabel("Sector")
    ax.set_ylabel(feature)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_corr_heatmap(df: pd.DataFrame, cols: list[str], out_path: Path) -> None:
    num = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    corr = num.corr(method="spearman", min_periods=500)

    fig, ax = plt.subplots(figsize=(22, 18))
    if sns is not None:
        sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", center=0, ax=ax)
    else:
        im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=6)
        ax.set_yticklabels(cols, fontsize=6)

    ax.set_title("Spearman correlation heatmap (pooled across all sectors/dates)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_label_corr(df: pd.DataFrame, feature_cols: list[str], label_col: str, out_path: Path) -> None:
    cols = feature_cols + [label_col]
    num = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    corr = num.corr(method="spearman", min_periods=500)[label_col].drop(label_col)

    top = corr.reindex(corr.abs().sort_values(ascending=False).head(35).index)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_title(f"Spearman corr(feature, {label_col}) | top 35 | pooled")
    ax.set_xlabel("Spearman rho")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_rank_heatmaps(df: pd.DataFrame, rank_cols: list[str], out_dir: Path) -> None:
    if not rank_cols:
        return

    sectors = sorted(df["Sector"].dropna().unique().tolist())
    for col in rank_cols:
        tmp = df[["Date", "Sector", col]].copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp = tmp.dropna(subset=["Date", "Sector"])
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        tmp["Month"] = tmp["Date"].dt.to_period("M").dt.to_timestamp("M")
        mat = tmp.pivot_table(index="Sector", columns="Month", values=col, aggfunc="mean")
        mat = mat.reindex(sectors)

        if mat.empty:
            continue

        n_months = mat.shape[1]
        width = max(12, min(28, n_months * 0.22))
        fig, ax = plt.subplots(figsize=(width, 6))

        if sns is not None:
            sns.heatmap(mat, vmin=0, vmax=1, cmap="RdYlGn", ax=ax)
        else:
            im = ax.imshow(mat.to_numpy(), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
            fig.colorbar(im, ax=ax)
            ax.set_yticks(range(len(sectors)))
            ax.set_yticklabels(sectors)
            ax.set_xticks(range(n_months))
            ax.set_xticklabels([d.strftime("%Y-%m") for d in mat.columns], rotation=90, fontsize=7)

        ax.set_title(f"Sector rank heatmap (monthly mean) | {col}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sector")
        fig.tight_layout()
        fig.savefig(out_dir / f"SUMMARY__rank_heatmap__{_safe_name(col)}.png", dpi=140)
        plt.close(fig)


def _plot_feature_nan_summary(df: pd.DataFrame, feature_cols: list[str], out_path: Path) -> None:
    nan_rates = df[feature_cols].isna().mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(nan_rates.index[::-1], nan_rates.values[::-1])
    ax.set_title("NaN rate per feature (pooled across all sectors/dates)")
    ax.set_xlabel("NaN fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_feature_range_summary(df: pd.DataFrame, feature_cols: list[str], out_path: Path) -> None:
    qs = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    p01 = qs.quantile(0.01)
    p99 = qs.quantile(0.99)

    order = (p99 - p01).abs().sort_values(ascending=False).index
    p01 = p01.reindex(order)
    p99 = p99.reindex(order)

    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(order))
    ax.hlines(y=y, xmin=p01.values, xmax=p99.values, color="tab:blue", alpha=0.8)
    ax.plot(p01.values, y, "o", color="tab:orange", markersize=3)
    ax.plot(p99.values, y, "o", color="tab:green", markersize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7)
    ax.set_title("Feature ranges: p01 (orange) to p99 (green)")
    ax.set_xlabel("value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_stationarity_summary(df: pd.DataFrame, candidate_features: list[str], out_path: Path) -> None:
    volatile, defensive = _pick_sectors(df)
    if not volatile or not defensive:
        return

    feats = [f for f in candidate_features if f in df.columns]
    feats = feats[:6]
    if not feats:
        return

    panel = df.loc[df["Sector"].isin([volatile, defensive]), ["Date", "Sector"] + feats].copy()
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce")
    panel = panel.dropna(subset=["Date"])

    n = len(feats)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(14, max(3, n * 2.3)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, feats, strict=False):
        piv = (
            panel.pivot_table(index="Date", columns="Sector", values=feat, aggfunc="last")
            .sort_index()
            .replace([np.inf, -np.inf], np.nan)
        )
        for sec in [volatile, defensive]:
            if sec in piv.columns:
                ax.plot(piv.index, pd.to_numeric(piv[sec], errors="coerce"), linewidth=1.0, label=sec)
        if "_z" in feat or feat.endswith("z"):
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(feat)
        ax.legend(loc="best")

    fig.suptitle(f"Time-series stationarity check | {volatile} (volatile) vs {defensive} (defensive)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def generate_for_file(path: Path) -> None:
    horizon_tag = _infer_horizon_tag(path)
    out_dir = GRAPHS_DIR / horizon_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==> Loading {path} -> {horizon_tag}")
    df = pd.read_parquet(path, engine="pyarrow")

    required = {"Date", "Sector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.tz_localize(None)

    numeric_cols = [c for c in df.columns if c not in ("Date", "Sector")]

    label_col = None
    label_candidates = [c for c in numeric_cols if c.startswith("label_")]
    preferred = f"label_excess_{horizon_tag[1:]}d"
    if preferred in df.columns:
        label_col = preferred
    elif label_candidates:
        label_col = label_candidates[0]

    feature_cols = [c for c in numeric_cols if c != label_col]

    print(f"Rows={len(df):,} features={len(feature_cols)} label={label_col}")

    _maybe_style()

    candidates = [
        "ratio_z21",
        "ratio_z63",
        "relmom5",
        "relmom10",
        "relmom21",
        "beta21",
        "corr21",
        "idio_vol21",
        "vol21",
    ]
    _plot_stationarity_summary(df, candidates, out_dir / "SUMMARY__stationarity_XLK_vs_XLP.png")

    corr_cols = feature_cols.copy()
    if label_col is not None:
        corr_cols = feature_cols + [label_col]

    _plot_corr_heatmap(df, corr_cols, out_dir / "SUMMARY__corr_heatmap_spearman.png")
    if label_col is not None:
        _plot_label_corr(df, feature_cols, label_col, out_dir / "SUMMARY__label_corr_top35.png")

    rank_cols = [c for c in feature_cols if c.startswith("rank_") or c.endswith("_rank")]
    _plot_rank_heatmaps(df, rank_cols, out_dir)

    _plot_feature_nan_summary(df, numeric_cols, out_dir / "SUMMARY__nan_rate_per_column.png")
    _plot_feature_range_summary(df, numeric_cols, out_dir / "SUMMARY__p01_p99_ranges.png")

    volatile, defensive = _pick_sectors(df)
    if not volatile or not defensive:
        raise ValueError("No sectors found in data")

    for i, feature in enumerate(numeric_cols, start=1):
        safe = _safe_name(feature)
        feature_dir = out_dir / safe
        feature_dir.mkdir(parents=True, exist_ok=True)

        _plot_timeseries_two_sectors(
            df,
            feature,
            feature_dir / f"timeseries_{volatile}_{defensive}.png",
            volatile,
            defensive,
        )
        _plot_hist(df, feature, feature_dir / "hist.png")
        _plot_missingness(df, feature, feature_dir / "missingness.png")
        _plot_sector_boxplot(df, feature, feature_dir / "sector_boxplot.png")

        if i % 20 == 0:
            print(f"  plotted {i}/{len(numeric_cols)} columns...")

    print(f"Done: {horizon_tag} -> {out_dir}")


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("features_h*.parquet"))
    if not files:
        raise FileNotFoundError(f"No features_h*.parquet found in {PROCESSED_DIR}")

    for path in files:
        generate_for_file(path)


if __name__ == "__main__":
    main()
