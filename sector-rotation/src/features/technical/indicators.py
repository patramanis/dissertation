from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from data.pit.policies import SECTORS
from features.common.returns import assert_prices_sane_after_first_valid, safe_log_returns

@dataclass(frozen=True)
class TechHorizonSpec:

    horizon: int
    window: int
    eps: float = 1e-12
    annualize_vol: bool = False

def _safe_log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return safe_log_returns(prices, name="prices")

def _rolling_zscore(x: pd.DataFrame, *, window: int, eps: float) -> pd.DataFrame:
    mean = x.rolling(window=window, min_periods=window).mean()
    var0 = _rolling_var0(x, window=window)
    std = pd.DataFrame(np.sqrt(var0)).where(var0 > eps)
    return (x - mean) / std

def _rolling_var0(x: pd.DataFrame | pd.Series, *, window: int) -> pd.DataFrame | pd.Series:
    m1 = x.rolling(window=window, min_periods=window).mean()
    m2 = (x * x).rolling(window=window, min_periods=window).mean()
    return (m2 - (m1 * m1)).clip(lower=0.0)

def _rolling_cov0(x: pd.Series, y: pd.Series, *, window: int) -> pd.Series:
    ex = x.rolling(window=window, min_periods=window).mean()
    ey = y.rolling(window=window, min_periods=window).mean()
    exy = (x * y).rolling(window=window, min_periods=window).mean()
    return exy - (ex * ey)

def _rolling_beta(
    sector_rets: pd.DataFrame,
    spy_rets: pd.Series[float],
    *,
    window: int,
    eps: float,
) -> pd.DataFrame:
    var_m = _rolling_var0(spy_rets, window=window)
    assert isinstance(var_m, pd.Series)
    var_m = var_m.where(var_m > eps)

    betas: dict[str, pd.Series] = {}
    for c in sector_rets.columns:
        cov0 = _rolling_cov0(sector_rets[c], spy_rets, window=window)
        betas[c] = cov0 / var_m

    out = pd.DataFrame(betas, index=sector_rets.index)
    out = out.reindex(columns=list(SECTORS))
    return out

def _rolling_corr(
    sector_rets: pd.DataFrame,
    spy_rets: pd.Series[float],
    *,
    window: int,
    eps: float,
) -> pd.DataFrame:
    var_m = _rolling_var0(spy_rets, window=window)
    assert isinstance(var_m, pd.Series)
    std_m = pd.Series(np.sqrt(var_m)).where(var_m > eps)

    corrs: dict[str, pd.Series[float]] = {}
    for c in sector_rets.columns:
        var_s = _rolling_var0(sector_rets[c], window=window)
        assert isinstance(var_s, pd.Series)
        std_s = pd.Series(np.sqrt(var_s)).where(var_s > eps)
        cov0 = _rolling_cov0(sector_rets[c], spy_rets, window=window)
        corrs[c] = cov0 / (std_s * std_m)

    out = pd.DataFrame(corrs, index=sector_rets.index)
    out = out.reindex(columns=list(SECTORS))
    return out

def _rolling_vol(rets: pd.DataFrame | pd.Series[float], *, window: int, annualize: bool) -> pd.DataFrame | pd.Series[float]:
    v = rets.rolling(window=window, min_periods=window).std(ddof=0)
    if annualize:
        v = v * float(np.sqrt(252.0))
    return v

def _rolling_idio_vol(
    sector_rets: pd.DataFrame | pd.Series[float],
    spy_rets: pd.Series[float],
    betas: pd.DataFrame,
    *,
    window: int,
    annualize: bool,
) -> pd.DataFrame:
    if isinstance(sector_rets, pd.Series):
        sector_rets = sector_rets.to_frame()
    resid: dict[str, pd.Series[float]] = {}
    for c in sector_rets.columns:
        resid[c] = sector_rets[c] - (betas[c] * spy_rets)
    resid_df = pd.DataFrame(resid, index=sector_rets.index).reindex(columns=list(SECTORS))
    vol_result = _rolling_vol(resid_df, window=window, annualize=annualize)
    return vol_result if isinstance(vol_result, pd.DataFrame) else vol_result.to_frame()

def _rolling_downside_deviation(rets: pd.DataFrame | pd.Series[float], *, window: int, annualize: bool) -> pd.DataFrame | pd.Series[float]:
    cond = rets < 0.0
    neg = rets.where(cond, 0.0)  # type: ignore[arg-type]
    semi = (neg.pow(2.0)).rolling(window=window, min_periods=window).mean().pow(0.5)
    if annualize:
        semi = semi * float(np.sqrt(252.0))
    return semi

def _rolling_max_drawdown(prices: pd.DataFrame, *, window: int) -> pd.DataFrame:
    roll_max = prices.rolling(window=window, min_periods=window).max()
    dd = (prices / roll_max) - 1.0
    mdd = dd.rolling(window=window, min_periods=window).min()
    return mdd

def build_wide_features_for_horizon(
    spec: TechHorizonSpec,
    spdr_prices: pd.DataFrame,
    spy_prices: pd.Series[float],
    *,
    feature_set: str = "default",
) -> dict[str, pd.DataFrame]:
    if feature_set != "default":
        raise ValueError(f"Unknown feature_set: {feature_set}")

    missing = [c for c in SECTORS if c not in spdr_prices.columns]
    if missing:
        raise ValueError(f"SPDR missing sectors: {missing}")
    spdr = spdr_prices.reindex(columns=list(SECTORS)).astype("float64")

    spy = spy_prices.astype("float64")
    if spy.name != "SPY":
        spy = spy.rename("SPY")

    if not spdr.index.equals(spy.index):
        raise AssertionError("Calendar mismatch: SPDR vs SPY")

    assert_prices_sane_after_first_valid(spdr, name="spdr_prices")
    assert_prices_sane_after_first_valid(spy, name="spy_prices")

    w = int(spec.window)
    eps = float(spec.eps)

    spdr_lr_raw = _safe_log_returns(spdr)
    spy_lr_raw = _safe_log_returns(spy)
    spdr_lr = spdr_lr_raw if isinstance(spdr_lr_raw, pd.DataFrame) else spdr_lr_raw.to_frame()
    spy_lr = spy_lr_raw if isinstance(spy_lr_raw, pd.Series) else spy_lr_raw.iloc[:, 0]
    
    relmom = spdr_lr.rolling(window=w, min_periods=w).sum().sub(spy_lr.rolling(window=w, min_periods=w).sum(), axis=0)
    ratio = pd.DataFrame(np.log(spdr.div(spy, axis=0)))
    ratio_z = _rolling_zscore(ratio, window=w, eps=eps)

    beta = _rolling_beta(spdr_lr, spy_lr, window=w, eps=eps)
    corr = _rolling_corr(spdr_lr, spy_lr, window=w, eps=eps)

    vol_result = _rolling_vol(spdr_lr, window=w, annualize=spec.annualize_vol)
    vol = vol_result if isinstance(vol_result, pd.DataFrame) else vol_result.to_frame()
    idio_vol = _rolling_idio_vol(spdr_lr, spy_lr, beta, window=w, annualize=spec.annualize_vol)

    semi_result = _rolling_downside_deviation(spdr_lr, window=w, annualize=spec.annualize_vol)
    semi = semi_result if isinstance(semi_result, pd.DataFrame) else semi_result.to_frame()
    mdd = _rolling_max_drawdown(spdr, window=w)

    out: dict[str, pd.DataFrame] = {
        "relmom": relmom,
        "ratio_z": ratio_z,
        "beta": beta,
        "corr": corr,
        "vol": vol,
        "idio_vol": idio_vol,
        "mdd": mdd,
        "semi": semi,
    }

    for k in list(out.keys()):
        out[k] = out[k].reindex(columns=list(SECTORS))

    return out