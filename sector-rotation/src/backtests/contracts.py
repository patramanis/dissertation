from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd

SECTORS = [
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
]

ALL_MODES = [
    "unfiltered",
    "relative",
    "absolute",
    "dual",
    "pure_momentum",
    "regime",
    "regime_invol",
    "regime_dynamic",
    "regime_scalar",
    "regime_streak_decay",
    "layered_adaptive",
    "trend_vix",
    "vix_zscore",
]

ML_MODES = [m for m in ALL_MODES if m != "pure_momentum"]
REGIME_MODES = ["regime", "regime_invol", 
                "regime_dynamic", "regime_scalar", "regime_streak_decay",
                "layered_adaptive", "trend_vix", "vix_zscore"]

HORIZONS = [5, 21, 63]
COST_BPS = 10
RISK_FREE_ANNUAL = 0.025

@dataclass(frozen=True)
class PredictionsSchema:
    REQUIRED = [
        "Date",
        "Sector",
        "label_excess",
        "pred_excess",
        "rank",
        "passed_gate",
        "selected",
        "portfolio_weight",
    ]
    
    OPTIONAL = [
        "pred_excess_raw",
        "abs_momentum",
        "relative_signal",
        "absolute_signal",
        "vol",
        "regime_signal",
        "regime_weight_mult",
        "allocation",
        "zone",
        "risk_scalar",
        "s_trend",
        "s_vix",
        "in_bear_trend",
        "bullish_votes",
        "in_cooldown",
    ]


@dataclass(frozen=True)
class WeightsSchema:
    COLUMNS = SECTORS + ["cash"]

@dataclass(frozen=True)
class ReturnsSchema:
    COLUMNS = [
        "Date",
        "gross_ret",
        "net_ret",
        "cash_weight",
        "exposure",
        "turnover",
        "cost",
        "n_positions",
        "rf_contribution",
    ]


@dataclass(frozen=True)
class TradesSchema:
    COLUMNS = [
        "Date",
        "Sector",
        "direction",
        "weight_before",
        "weight_after",
        "delta_weight",
        "abs_delta",
    ]


@dataclass(frozen=True)
class RegimeSchema:
    COLUMNS = [
        "Date",
        "regime_signal",
        "allocation",
        "zone",
        "vix",
        "s_trend",
        "s_vix",
        "risk_scalar",
        "bullish_votes",
        "in_cooldown",
        "in_bear_trend",
    ]

def get_workspace_root() -> Path:
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def get_run_paths(
    run_id: str,
    horizon: int,
    category: str = "test",
) -> dict[str, Path]:
    ws = get_workspace_root()
    run_dir = ws / "runs" / category / run_id
    backtest_dir = run_dir / "backtests" / f"h{horizon}"
    
    return {
        "run_dir": run_dir,
        "backtest_dir": backtest_dir,
        "config": run_dir / "config.json",
        "rankings": run_dir / "rankings.csv",
        "weights": backtest_dir / "weights.csv",
        "returns": backtest_dir / "returns.csv",
        "trades": backtest_dir / "trades.csv",
        "turnover": backtest_dir / "turnover.csv",
        "regime_series": backtest_dir / "regime_series.csv",
        "vectorbt_stats": backtest_dir / "vectorbt_stats.json",
        "quantstats_html": backtest_dir / "quantstats.html",
        "alphalens_html": backtest_dir / "alphalens.html",
        "alphalens_factor": backtest_dir / "alphalens_factor.csv",
        "manifest": backtest_dir / "manifest.json",
    }


def get_data_paths() -> dict[str, Path]:
    ws = get_workspace_root()
    
    return {
        "spdr_prices": ws / "data" / "interim" / "aligned_pit" / "SPDR.csv",
        "spy_prices": ws / "data" / "interim" / "aligned_pit" / "SPY.csv",
        "vix_prices": ws / "data" / "interim" / "aligned_pit" / "Futures.csv",
        "panel_h5": ws / "data" / "processed" / "panel" / "panel_h5.csv",
        "panel_h21": ws / "data" / "processed" / "panel" / "panel_h21.csv",
        "panel_h63": ws / "data" / "processed" / "panel" / "panel_h63.csv",
    }

def validate_rankings(df, mode: str) -> list[str]:
    errors = []
    
    for col in PredictionsSchema.REQUIRED:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    if df.duplicated(subset=["Date", "Sector"]).any():
        errors.append("Duplicate (Date, Sector) pairs found")
    
    sectors_per_date = df.groupby("Date")["Sector"].nunique()
    bad_dates = sectors_per_date[sectors_per_date != 9]
    if len(bad_dates) > 0:
        errors.append(f"{len(bad_dates)} dates don't have exactly 9 sectors")
    
    weight_sums = df.groupby("Date")["portfolio_weight"].sum()
    if (weight_sums > 1.001).any():
        errors.append("Weights sum exceeds 1.0 on some dates")
    if (weight_sums < -0.001).any():
        errors.append("Negative weight sum on some dates")
    
    return errors


def validate_weights(df) -> list[str]:
    errors = []
    
    for sector in SECTORS:
        if sector not in df.columns:
            errors.append(f"Missing sector column: {sector}")
    
    if "cash" not in df.columns:
        errors.append("Missing cash column")
    
    weight_cols = [c for c in df.columns if c in SECTORS + ["cash"]]
    for col in weight_cols:
        if (df[col] < -0.001).any():
            errors.append(f"Negative weights in column: {col}")
    
    row_sums = df[weight_cols].sum(axis=1)
    if not ((row_sums > 0.999) & (row_sums < 1.001)).all():
        errors.append("Weight rows don't sum to 1.0")
    
    return errors


def validate_returns(df) -> list[str]:
    errors = []
    
    for col in ReturnsSchema.COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    if "exposure" in df.columns:
        if (df["exposure"] < -0.001).any() or (df["exposure"] > 1.001).any():
            errors.append("Exposure outside [0, 1] range")
    
    if "cost" in df.columns:
        if (df["cost"] < -0.001).any():
            errors.append("Negative transaction costs found")
    
    return errors

@dataclass
class RunManifest:
    
    run_id: str
    mode: str
    horizon: int
    category: str
    created_at: str
    config_path: str
    n_folds: int = 1
    n_rankings: int = 0
    validated: bool = False
    validation_passed: bool = False
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "horizon": self.horizon,
            "category": self.category,
            "created_at": self.created_at,
            "config_path": self.config_path,
            "n_folds": self.n_folds,
            "n_rankings": self.n_rankings,
            "validated": self.validated,
            "validation_passed": self.validation_passed,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "RunManifest":
        return cls(**d)


def get_run_dir(base_dir: Path | str, mode: str, horizon: int) -> Path:
    base_dir = Path(base_dir)
    
    candidates = [
        base_dir / mode / f"h{horizon}",
        base_dir / f"{mode}_h{horizon}",
        base_dir / mode / str(horizon),
        base_dir / mode,
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    return base_dir / mode / f"h{horizon}"


def load_rankings(run_dir: Path | str) -> "pd.DataFrame":
    import pandas as pd
    
    run_dir = Path(run_dir)
    
    candidates = [
        run_dir / "rankings.csv",
        run_dir / "fold_0" / "rankings.csv",
        run_dir / "oof_rankings.csv",
    ]
    
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"])
            return df
    
    raise FileNotFoundError(f"rankings.csv not found in {run_dir}")


def load_weights(run_dir: Path | str) -> "pd.DataFrame":
    import pandas as pd
    
    run_dir = Path(run_dir)
    
    path = run_dir / "weights.csv"
    if not path.exists():
        path = run_dir / "backtests" / "weights.csv"
    
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    
    raise FileNotFoundError(f"weights.csv not found in {run_dir}")


def load_returns(run_dir: Path | str) -> "pd.DataFrame":
    import pandas as pd
    
    run_dir = Path(run_dir)
    
    path = run_dir / "returns.csv"
    if not path.exists():
        path = run_dir / "backtests" / "returns.csv"
    
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    
    raise FileNotFoundError(f"returns.csv not found in {run_dir}")

def save_manifest(manifest: RunManifest, output_dir: Path | str) -> None:
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2),
        encoding="utf-8",
    )