from __future__ import annotations
import argparse
import sys

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def equity_from_log_returns(log_returns: pd.Series, *, start: float = 1.0) -> pd.Series:
    lr = pd.to_numeric(log_returns, errors="coerce").fillna(0.0)
    eq = np.exp(lr.cumsum()) * float(start)
    return pd.Series(eq, index=log_returns.index, name="equity")


def plot_regimes(*, equity: pd.Series, feats: pd.DataFrame, out_path: Path, show: bool) -> None:
    df = pd.DataFrame({"equity": equity}).join(feats, how="left")

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]},
        constrained_layout=True,
    )

    ax_top.plot(df.index, df["equity"], color="black", linewidth=1.5)
    ax_top.set_title("SPY Equity Curve with HMM Regime Background")
    ax_top.set_ylabel("Equity (from log returns)")

    y0, y1 = ax_top.get_ylim()

    hi = df["hmm_trend"] > 0.5
    lo = df["hmm_trend"] <= 0.5

    ax_top.fill_between(df.index, y0, y1, where=hi.fillna(False), color="red", alpha=0.12)
    ax_top.fill_between(df.index, y0, y1, where=lo.fillna(False), color="green", alpha=0.10)
    ax_top.set_ylim(y0, y1)

    ax_mid.plot(df.index, df["hmm_raw"], linewidth=1.0, label="hmm_raw")
    ax_mid.plot(df.index, df["hmm_trend"], linewidth=1.6, label="hmm_trend")
    ax_mid.axhline(0.5, color="black", linewidth=1.0, linestyle="--")
    ax_mid.set_title("HMM High-Vol Probability (raw vs trend)")
    ax_mid.set_ylabel("prob_high_vol")
    ax_mid.set_ylim(-0.02, 1.02)
    ax_mid.legend(loc="upper left")

    thr_lo = 0.2
    thr_hi = 0.8
    trend = df["hmm_trend"].astype(float)
    safe = trend.where(trend < thr_lo)
    mid = trend.where((trend >= thr_lo) & (trend <= thr_hi))
    danger = trend.where(trend > thr_hi)

    ax_bot.plot(df.index, safe, linewidth=1.8, color="tab:green", label="Safe (hmm_trend < 0.2)")
    ax_bot.plot(
        df.index,
        mid,
        linewidth=1.8,
        color="goldenrod",
        label="Uncertainty (0.2 ≤ hmm_trend ≤ 0.8)",
    )
    ax_bot.plot(df.index, danger, linewidth=1.8, color="tab:red", label="Danger (hmm_trend > 0.8)")
    ax_bot.axhline(thr_lo, color="black", linewidth=1.0, linestyle=":")
    ax_bot.axhline(thr_hi, color="black", linewidth=1.0, linestyle=":")
    ax_bot.set_title("Probability Thresholding (traffic light on hmm_trend)")
    ax_bot.set_ylabel("hmm_trend")
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.legend(loc="upper left")

    ax_top.grid(True, alpha=0.25)
    ax_mid.grid(True, alpha=0.25)
    ax_bot.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")

    if show:
        plt.show()


def _load_spy_logret() -> pd.Series:
    spy_path = _repo_root() / "ModularMonolith" / "data" / "raw_data_3" / "SPY.parquet"
    df = pd.read_parquet(spy_path)

    if "SPY_logret" in df.columns:
        logret = pd.to_numeric(df["SPY_logret"], errors="coerce")
        if "Date" in df.columns:
            logret.index = pd.to_datetime(df["Date"], errors="coerce")
        logret = logret.sort_index()
    elif "SPY" in df.columns and "Date" in df.columns:
        px = pd.to_numeric(df["SPY"], errors="coerce")
        px.index = pd.to_datetime(df["Date"], errors="coerce")
        px = px.sort_index()
        logret = np.log(px).diff()
        logret.name = "SPY_logret"
    else:
        raise ValueError("Could not locate SPY_logret or compute from SPY + Date")

    logret = logret.dropna()
    return logret


def _default_regimes_dir() -> Path:
    return _repo_root() / "ModularMonolith" / "data" / "regimes"


def save_regime_features(*, logret: pd.Series, feats: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"SPY_logret": pd.to_numeric(logret, errors="coerce")}).join(feats, how="left")
    df.index.name = "Date"
    df.to_parquet(out_path)
    print(f"Saved regimes parquet to: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot SPY HMM regime features.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window.")
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=252)
    parser.add_argument(
        "--window-mode",
        type=str,
        default="expanding",
        choices=["rolling", "expanding"],
        help="rolling uses a fixed-length lookback; expanding uses all history since start (slower).",
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=1260,
        help="Minimum history length for expanding-window fits (e.g., 1260 for ~5 years).",
    )
    parser.add_argument("--ewm-span", type=int, default=10, help="EWM span for hmm_trend smoothing.")
    parser.add_argument("--start", type=str, default=None, help="Optional start date filter (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Optional end date filter (YYYY-MM-DD).")
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="Optional max number of rows (after date filtering). Useful for quicker runs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. Default: save into ModularMonolith/data/regimes.",
    )
    parser.add_argument(
        "--out-parquet",
        type=str,
        default=None,
        help="Output parquet path. Default: ModularMonolith/data/regimes/spy_regimes.parquet.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(_repo_root()))

    from ModularMonolith.data.train_window import TRAIN_DATE_END, TRAIN_DATE_START

    from ModularMonolith.src.features.regime import RobustMarketRegimeModel

    logret_full = _load_spy_logret()

    out_start = args.start if args.start is not None else str(TRAIN_DATE_START)
    out_end = args.end if args.end is not None else str(TRAIN_DATE_END)

    logret_fit = logret_full
    if args.max_n is not None:
        logret_fit = logret_fit.iloc[: int(args.max_n)]

    model = RobustMarketRegimeModel(
        window_mode=str(args.window_mode),
        min_train_size=int(args.min_train_size) if args.min_train_size is not None else None,
        window_size=int(args.window_size),
        n_iter=int(args.n_iter),
        n_restarts=int(args.n_restarts),
    )

    feats = model.predict_features(logret_fit, ewm_span=int(args.ewm_span))

    valid = feats["hmm_trend"].dropna()
    if len(valid):
        share_hi = float((valid > 0.5).mean())
        print(f"Valid points: {len(valid)}/{len(feats)} | Share high-vol (trend>0.5): {share_hi:.3f}")

    equity = equity_from_log_returns(logret_fit)

    if out_start is not None:
        s = pd.Timestamp(out_start)
        feats = feats.loc[s:]
        equity = equity.loc[s:]
        logret_out = logret_fit.loc[s:]
    else:
        logret_out = logret_fit
    if out_end is not None:
        e = pd.Timestamp(out_end)
        feats = feats.loc[:e]
        equity = equity.loc[:e]
        logret_out = logret_out.loc[:e]

    regimes_dir = _default_regimes_dir()

    default_plot_out = regimes_dir / "regime_spy.png"
    out_path = Path(args.out) if args.out is not None else default_plot_out

    default_parquet_out = regimes_dir / "spy_regimes.parquet"
    parquet_out = Path(args.out_parquet) if args.out_parquet is not None else default_parquet_out

    save_regime_features(logret=logret_out, feats=feats, out_path=parquet_out)

    show = not args.no_show
    plot_regimes(equity=equity, feats=feats, out_path=out_path, show=show)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())