from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

def run_alphalens(
    factor: pd.Series | pd.DataFrame,
    prices: pd.DataFrame,
    periods: tuple[int, ...] = (5, 21),
    quantiles: int = 5,
    output_dir: Path | None = None,
    max_loss: float = 0.35,
) -> dict[str, Any]:
    try:
        import alphalens  # type: ignore[import-untyped]
        from alphalens.utils import get_clean_factor_and_forward_returns  # type: ignore[import-untyped]
        from alphalens.tears import create_full_tear_sheet  # type: ignore[import-untyped]
        from alphalens.performance import (  # type: ignore[import-untyped]
            factor_information_coefficient,
            mean_information_coefficient,
            factor_returns,
            mean_return_by_quantile,
        )
    except ImportError:
        log.error("Alphalens not installed")
        return {"error": "alphalens not installed"}
    
    if isinstance(factor, pd.DataFrame):
        if "Date" in factor.columns and "Sector" in factor.columns:
            factor_col = [c for c in factor.columns 
                         if c not in ["Date", "Sector"]][0]
            factor = factor.drop_duplicates(subset=["Date", "Sector"], keep="last")
            factor = factor.set_index(["Date", "Sector"])[factor_col]
    
    if not isinstance(factor.index, pd.MultiIndex):
        log.error("Factor must have MultiIndex (Date, Sector)")
        return {"error": "invalid_factor_index"}
    
    unique_dates = pd.to_datetime(factor.index.get_level_values(0).unique())
    factor.index = factor.index.set_levels(unique_dates, level=0)
    
    factor.index = factor.index.rename(['date', 'asset'])
    
    if factor.index.duplicated().any():
        log.warning("Removing %d duplicate index entries from factor", factor.index.duplicated().sum())
        factor = factor[~factor.index.duplicated(keep="last")]
    
    prices.index = pd.to_datetime(prices.index)
    
    try:
        factor_data = get_clean_factor_and_forward_returns(
            factor,
            prices,
            periods=periods,
            quantiles=quantiles,
            max_loss=max_loss,
        )
    except Exception as e:
        log.error("Failed to create factor data: %s", e)
        return {"error": str(e)}
    
    results = {}
    
    ic = None
 
    try:
        ic = factor_information_coefficient(factor_data)
        mean_ic = mean_information_coefficient(factor_data)
        
        results["ic_by_period"] = {
            str(p): {
                "mean": float(ic[p].mean()),
                "std": float(ic[p].std()),
                "ir": float(ic[p].mean() / ic[p].std()) if ic[p].std() > 0 else 0,
                "n": int(len(ic[p])),
            }
            for p in periods
        }
        results["ic_mean"] = float(mean_ic["IC Mean"].mean())
        results["ic_std"] = float(mean_ic["IC Mean"].std())
        results["ic_ir"] = float(results["ic_mean"] / results["ic_std"]) if results["ic_std"] > 0 else 0
        
        log.info("IC analysis: mean=%.4f, std=%.4f, IR=%.2f",
                 results["ic_mean"], results["ic_std"], results["ic_ir"])
        
    except Exception as e:
        log.warning("IC calculation failed: %s", e)
        results["ic_error"] = str(e)
    
    try:
        qret = mean_return_by_quantile(factor_data)
        
        results["quantile_returns"] = {}
        for period in periods:
            results["quantile_returns"][str(period)] = {
                str(q): float(qret[0].loc[q, f"{period}D"])
                for q in qret[0].index
            }
            
    except Exception as e:
        log.warning("Quantile returns calculation failed: %s", e)
    
    try:
        fret = factor_returns(factor_data)
        
        results["factor_returns"] = {
            str(p): {
                "mean": float(fret[p].mean()),
                "std": float(fret[p].std()),
                "sharpe": float(fret[p].mean() / fret[p].std() * np.sqrt(252/p)) if fret[p].std() > 0 else 0,
            }
            for p in periods
        }
        
    except Exception as e:
        log.warning("Factor returns calculation failed: %s", e)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if ic is not None:
                ic_path = output_dir / "alphalens_ic.csv"
                ic.to_csv(ic_path)
                results["ic_path"] = str(ic_path)
        except:
            pass
        
        try:
            factor_path = output_dir / "alphalens_factor.csv"
            factor_data.to_csv(factor_path)
            results["factor_path"] = str(factor_path)
        except:
            pass
        
        results_path = output_dir / "alphalens_results.json"
        results_path.write_text(
            json.dumps(results, indent=2, default=str),
            encoding="utf-8",
        )
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            html_path = output_dir / "alphalens.html"
            
            fig = plt.figure(figsize=(16, 20))
            create_full_tear_sheet(factor_data, by_group=False)
            
            pdf_path = output_dir / "alphalens_tearsheet.pdf"
            plt.savefig(str(pdf_path), format="pdf", bbox_inches="tight")
            plt.close()
            
            results["tearsheet_path"] = str(pdf_path)
            log.info("Alphalens tearsheet saved to %s", pdf_path)
            
        except Exception as e:
            log.warning("Failed to generate Alphalens tearsheet: %s", e)
    
    return results


def prepare_factor_for_alphalens(
    rankings: pd.DataFrame,
    factor_col: str = "pred_excess",
) -> pd.Series:
    df = rankings[["Date", "Sector", factor_col]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["Date", "Sector"], keep="last")
    factor = df.set_index(["Date", "Sector"])[factor_col]
    factor = factor.sort_index()
    
    return factor


def prepare_prices_for_alphalens(
    prices: pd.DataFrame,
    sectors: list[str] | None = None,
) -> pd.DataFrame:
    from .contracts import SECTORS
    
    if sectors is None:
        sectors = SECTORS
    
    if "Date" in prices.columns:
        prices = prices.set_index("Date")
    
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = None
    available = [s for s in sectors if s in prices.columns]
    return prices[available]

def compute_rolling_ic(
    factor: pd.Series,
    forward_returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    common = factor.index.intersection(forward_returns.index)
    f = factor.loc[common]
    r = forward_returns.loc[common]
    dates = f.index.get_level_values(0).unique()
    
    ics = []
    ic_dates = []
    
    for i, date in enumerate(dates):
        if i < window:
            continue
        
        window_dates = dates[i-window:i]
        mask = f.index.get_level_values(0).isin(window_dates)
        f_window = f[mask]
        r_window = r[mask]
        
        ic = f_window.corr(r_window, method="spearman")
        ics.append(ic)
        ic_dates.append(date)
    
    return pd.Series(ics, index=pd.DatetimeIndex(ic_dates))


def compute_ic_by_regime(
    factor: pd.Series,
    forward_returns: pd.Series,
    regime: pd.Series,
) -> pd.DataFrame:
    results = []
    regimes = regime.unique()
    
    for reg in regimes:
        reg_dates = regime[regime == reg].index
        
        mask = factor.index.get_level_values(0).isin(reg_dates)
        f_reg = factor[mask]
        r_reg = forward_returns[mask]
        
        if len(f_reg) < 10:
            continue
        
        dates = f_reg.index.get_level_values(0).unique()
        ics = []
        
        for date in dates:
            f_d = f_reg.xs(date, level=0)
            r_d = r_reg.xs(date, level=0)
            if len(f_d) >= 3:
                ic = f_d.corr(r_d, method="spearman")
                if not np.isnan(ic):
                    ics.append(ic)
        
        if ics:
            results.append({
                "regime": reg,
                "n_dates": len(ics),
                "ic_mean": np.mean(ics),
                "ic_std": np.std(ics),
                "ic_ir": np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0,
            })
    
    return pd.DataFrame(results)