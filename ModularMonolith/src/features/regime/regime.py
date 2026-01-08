from __future__ import annotations

import contextlib
import io
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

try:
    from hmmlearn.hmm import GaussianHMM
except Exception as e:
    raise ImportError("hmmlearn error") from e


class RobustMarketRegimeModel:
    def __init__(
        self,
        *,
        n_components: int = 2,
        window_size: int = 252,
        n_iter: int = 100,
        n_restarts: int = 5,
        tol: float = 0.01,
        window_mode: str = "expanding",
        min_train_size: int | None = None,
        feature_set: str = "return_abs",
        min_var_scaled: float = 0.05,
        max_var_scaled: float = 50.0,
        min_effective_mass: float = 0.01,
        min_var_ratio: float = 1.5,
        max_any_var_scaled: float = 10_000.0,
        require_converged: bool = True,
        debug_every: int = 0,
    ) -> None:
        if n_components != 2:
            raise ValueError("This implementation expects n_components=2")
        if window_size <= 5:
            raise ValueError("window_size must be > 5")
        if n_iter <= 0:
            raise ValueError("n_iter must be positive")
        if n_restarts <= 0:
            raise ValueError("n_restarts must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if window_mode not in {"rolling", "expanding"}:
            raise ValueError("window_mode must be 'rolling' or 'expanding'")
        if min_train_size is not None and min_train_size <= 5:
            raise ValueError("min_train_size must be > 5")
        if feature_set not in {"return", "abs_return", "return_abs", "return_rolling_std"}:
            raise ValueError("feature_set must be one of: 'return', 'abs_return', 'return_abs', 'return_rolling_std'")
        if min_var_scaled <= 0 or max_var_scaled <= 0 or min_var_scaled >= max_var_scaled:
            raise ValueError("min_var_scaled/max_var_scaled must define a positive range")
        if not (0.0 < min_effective_mass < 0.5):
            raise ValueError("min_effective_mass must be in (0, 0.5)")
        if min_var_ratio <= 1.0:
            raise ValueError("min_var_ratio must be > 1.0")
        if max_any_var_scaled <= 0:
            raise ValueError("max_any_var_scaled must be positive")
        if debug_every < 0:
            raise ValueError("debug_every must be >= 0")

        self.n_components = n_components
        self.window_size = window_size
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.tol = tol
        self.window_mode = window_mode
        self.min_train_size = int(min_train_size) if min_train_size is not None else None
        self.feature_set = feature_set
        self.min_var_scaled = float(min_var_scaled)
        self.max_var_scaled = float(max_var_scaled)
        self.min_effective_mass = float(min_effective_mass)
        self.min_var_ratio = float(min_var_ratio)
        self.max_any_var_scaled = float(max_any_var_scaled)
        self.require_converged = bool(require_converged)
        self.debug_every = int(debug_every)

    @staticmethod
    def _coerce_returns(returns: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(returns, pd.DataFrame):
            if returns.empty:
                raise ValueError("returns is empty")
            if returns.shape[1] != 1:
                raise ValueError("returns DataFrame must have exactly 1 column")
            s = returns.iloc[:, 0]
        elif isinstance(returns, pd.Series):
            s = returns
        else:
            raise TypeError("returns must be a pandas Series or single-column DataFrame")

        s = pd.to_numeric(s, errors="coerce")
        idx = pd.to_datetime(s.index, errors="raise")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        s.index = idx
        s = s.sort_index()
        return s

    def _build_features(self, window_r: pd.Series) -> np.ndarray:
        r = pd.to_numeric(window_r, errors="coerce")
        if self.feature_set == "return":
            X = r.to_numpy(dtype=float).reshape(-1, 1)
        elif self.feature_set == "abs_return":
            X = r.abs().to_numpy(dtype=float).reshape(-1, 1)
        elif self.feature_set == "return_abs":
            X = np.column_stack([r.to_numpy(dtype=float), r.abs().to_numpy(dtype=float)])
        elif self.feature_set == "return_rolling_std":
            roll = r.rolling(21, min_periods=2).std(ddof=0)
            X = np.column_stack([r.to_numpy(dtype=float), roll.to_numpy(dtype=float)])
        else:
            raise RuntimeError(f"Unexpected feature_set: {self.feature_set}")

        return X * 100.0

    def _best_hmm_fit(
        self,
        X_window: np.ndarray,
        *,
        n_components: int,
        n_iter: int,
        tol: float,
        n_restarts: int,
        seed_base: int,
    ) -> GaussianHMM | None:
        if X_window.ndim != 2 or X_window.shape[1] < 1:
            raise ValueError(
                f"Expected X_window as shape (n_samples, n_features>=1); got {X_window.shape}"
            )

        best_model: GaussianHMM | None = None
        best_ll = -np.inf

        stickiness = np.array([[10.0, 1.0], [1.0, 10.0]], dtype=float)

        for i in range(n_restarts):
            model = GaussianHMM(
                n_components=n_components,
                covariance_type="diag",
                min_covar=1e-4,
                params="stmc",
                init_params="stmc",
                transmat_prior=stickiness,
                n_iter=n_iter,
                tol=tol,
                random_state=seed_base + i,
            )

            x0 = X_window[:, 0]
            q_lo = float(np.nanquantile(x0, 0.25))
            q_hi = float(np.nanquantile(x0, 0.75))
            means = np.zeros((n_components, X_window.shape[1]), dtype=float)
            means[0, 0] = q_lo
            means[1, 0] = q_hi
            model.means_ = means

            covars = np.full((n_components, X_window.shape[1]), 1.0, dtype=float)
            covars[0, 0] = 0.8
            covars[1, 0] = 8.0
            model.covars_ = covars

            model.init_params = "st"

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", message=r"Model is not converging.*")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    model.fit(X_window)

            converged = bool(getattr(getattr(model, "monitor_", None), "converged", False))
            if self.require_converged and not converged:
                continue

            cov = np.asarray(model.covars_, dtype=float)
            if not np.isfinite(cov).all():
                continue
            if float(np.nanmax(cov)) > self.max_any_var_scaled:
                continue

            var0 = cov[:, 0] if cov.ndim == 2 else cov[:, 0, 0]
            if (var0 < self.min_var_scaled).any() or (var0 > self.max_var_scaled).any():
                continue

            v_low = float(np.min(var0))
            v_high = float(np.max(var0))
            if not np.isfinite(v_low) or not np.isfinite(v_high) or v_low <= 0.0:
                continue
            if (v_high / v_low) < self.min_var_ratio:
                continue

            try:
                _, post = model.score_samples(X_window)
            except Exception:
                continue
            mass = post.mean(axis=0)
            if (mass < self.min_effective_mass).any():
                continue

            ll: float
            try:
                ll = float(model.monitor_.history[-1])
            except Exception:
                try:
                    ll = float(model.score(X_window))
                except Exception:
                    continue

            if np.isfinite(ll) and ll > best_ll:
                best_ll = ll
                best_model = model

        return best_model

    @staticmethod
    def _returns_variance_by_state(model: GaussianHMM) -> np.ndarray:
        cov = np.asarray(model.covars_)
        if cov.ndim == 2:
            if cov.shape[1] < 1:
                raise RuntimeError(f"Unexpected covars feature dimension: {cov.shape}")
            returns_var = cov[:, 0]
        elif cov.ndim == 3:
            if cov.shape[1] < 1 or cov.shape[2] < 1:
                raise RuntimeError(f"Unexpected covars feature dimension: {cov.shape}")
            returns_var = cov[:, 0, 0]
        else:
            raise RuntimeError(f"Unexpected covars shape for covariance_type='diag': {cov.shape}")

        return np.asarray(returns_var, dtype=float)

    @staticmethod
    def _sorted_state_order_by_returns_variance(model: GaussianHMM) -> np.ndarray:

        returns_var = RobustMarketRegimeModel._returns_variance_by_state(model)
        order = np.argsort(returns_var)
        return order

    @staticmethod
    def _low_high_state_from_returns_variance(model: GaussianHMM) -> tuple[int, int]:
        order = RobustMarketRegimeModel._sorted_state_order_by_returns_variance(model)
        low_state = int(order[0])
        high_state = int(order[-1])

        if low_state == high_state:
            raise RuntimeError("Failed to identify distinct low/high states")

        return low_state, high_state

    @staticmethod
    def _returns_variance_for_state(model: GaussianHMM, state: int) -> float:
        cov = np.asarray(model.covars_)
        if cov.ndim == 2:
            return float(cov[state, 0])
        if cov.ndim == 3:
            return float(cov[state, 0, 0])
        raise RuntimeError(f"Unexpected covars_ shape: {cov.shape}")

    def predict_rolling(self, returns: pd.Series | pd.DataFrame) -> pd.DataFrame:

        r = self._coerce_returns(returns)
        out = pd.DataFrame(index=r.index, columns=["prob_low_vol", "prob_high_vol"], dtype=float)
        dates = r.index

        effective_min_train = self.window_size
        if self.min_train_size is not None:
            effective_min_train = max(effective_min_train, self.min_train_size)

        if self.window_mode == "rolling":
            t_start = self.window_size
        else:
            t_start = effective_min_train

        start_pos = 0
        first_valid = r.first_valid_index()
        if first_valid is not None:
            start_pos = int(r.index.get_indexer([first_valid])[0])

        n_attempted = 0
        n_skip_nan_window = 0
        n_skip_too_short = 0
        n_skip_nonfinite_features = 0
        n_fit_failed = 0
        n_score_failed = 0
        n_written = 0

        for t_pos in range(t_start, len(r)):
            if self.window_mode == "rolling":
                window_r = r.iloc[t_pos - self.window_size : t_pos]
            else:
                window_r = r.iloc[start_pos:t_pos]
            if self.window_mode == "rolling":
                if window_r.isna().any():
                    n_skip_nan_window += 1
                    continue
            else:
                window_r = window_r.dropna()
                if len(window_r) < effective_min_train:
                    n_skip_too_short += 1
                    continue

            n_attempted += 1

            X_window = self._build_features(window_r)
            if not np.isfinite(X_window).all():
                n_skip_nonfinite_features += 1
                continue

            best_model = self._best_hmm_fit(
                X_window,
                n_components=self.n_components,
                n_iter=self.n_iter,
                tol=self.tol,
                n_restarts=self.n_restarts,
                seed_base=t_pos,
            )
            if best_model is None:
                n_fit_failed += 1
                continue

            order = self._sorted_state_order_by_returns_variance(best_model)
            low_state = int(order[0])
            high_state = int(order[1])

            if self.debug_every and (t_pos % self.debug_every == 0):
                iters = int(getattr(getattr(best_model, "monitor_", None), "iter", -1))
                mean0 = float(best_model.means_[low_state][0])
                var0 = self._returns_variance_for_state(best_model, low_state)
                mean1 = float(best_model.means_[high_state][0])
                var1 = self._returns_variance_for_state(best_model, high_state)
                print(
                    f"Date: {dates[t_pos - 1]} | iters={iters} | "
                    f"LowVol(mean={mean0:.4f}, var_scaled={var0:.4f}) | "
                    f"HighVol(mean={mean1:.4f}, var_scaled={var1:.4f})"
                )

            try:
                _, post = best_model.score_samples(X_window)
                last_post = post[-1]
            except Exception:
                n_score_failed += 1
                continue

            last_post_sorted = last_post[order]

            s = float(np.nansum(last_post_sorted))
            if not np.isfinite(s) or s <= 0.0:
                continue
            last_post_sorted = last_post_sorted / s

            out.loc[dates[t_pos], "prob_low_vol"] = float(last_post_sorted[0])
            out.loc[dates[t_pos], "prob_high_vol"] = float(last_post_sorted[1])
            n_written += 1

            if self.debug_every and (t_pos % self.debug_every == 0):
                cov = float(out["prob_high_vol"].notna().mean())
                print(
                    f"[HMM status] t_pos={t_pos} attempted={n_attempted} written={n_written} "
                    f"coverage={cov:.3f} skip_nan_window={n_skip_nan_window} "
                    f"skip_too_short={n_skip_too_short} skip_nonfinite={n_skip_nonfinite_features} "
                    f"fit_failed={n_fit_failed} score_failed={n_score_failed}"
                )

        if self.debug_every:
            cov = float(out["prob_high_vol"].notna().mean())
            print(
                f"[HMM final] attempted={n_attempted} written={n_written} coverage={cov:.3f} "
                f"skip_nan_window={n_skip_nan_window} skip_too_short={n_skip_too_short} "
                f"skip_nonfinite={n_skip_nonfinite_features} fit_failed={n_fit_failed} score_failed={n_score_failed}"
            )

        return out

    def predict_features(
        self,
        returns: pd.Series | pd.DataFrame,
        *,
        ewm_span: int = 10,
    ) -> pd.DataFrame:
        """
        Generate regime features from rolling HMM predictions.
        
        SIGNAL POLARITY DOCUMENTATION:
        - hmm_raw (prob_high_vol): Probability of being in HIGH volatility state.
          Higher values → turbulent market regime.
          Interpretation: When hmm_raw is high, defensive sectors (XLP, XLU, XLV)
          may outperform cyclicals (XLK, XLE, XLI).
          
        - hmm_trend: EWM smoothed version of hmm_raw for stability.
        
        - hmm_delta: Rate of change in regime probability.
          Positive → transitioning toward higher volatility.
          Negative → transitioning toward lower volatility.
          
        - hmm_low_vol: Probability of being in LOW volatility state (1 - hmm_raw).
          Higher values → calm market regime.
          This is the inverse of hmm_raw for clearer interpretation when needed.
        
        Args:
            returns: Daily log returns series
            ewm_span: EWM span for trend smoothing (default: 10)
            
        Returns:
            DataFrame with hmm_raw, hmm_trend, hmm_delta, hmm_low_vol columns
        """
        if ewm_span <= 0:
            raise ValueError("ewm_span must be positive")

        probs = self.predict_rolling(returns)
        hmm_raw = probs["prob_high_vol"].rename("hmm_raw")
        hmm_low_vol = (1.0 - hmm_raw).rename("hmm_low_vol")
        hmm_trend = hmm_raw.ewm(span=int(ewm_span), adjust=False).mean().rename("hmm_trend")
        hmm_delta = hmm_trend.diff().fillna(0.0).rename("hmm_delta")

        return pd.concat([hmm_raw, hmm_low_vol, hmm_trend, hmm_delta], axis=1)
