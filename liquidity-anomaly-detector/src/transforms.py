"""
transforms.py
=============

Signal-processing utilities for converting non-stationary financial time
series (prices, spreads, order-book depth) into stationary feature inputs
suitable for sequence models such as LSTMs.

The centerpiece is a fixed-width-window fractional differencing
implementation (Lopez de Prado, 2018) that removes just enough of a
series' unit-root behaviour to achieve stationarity while retaining as
much of its long-memory / autocorrelation structure as possible -- unlike
integer-order differencing (e.g. simple returns), which is
stationarity-guaranteed but memory-destroying.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)


class FinancialTimeTransform:
    """
    Collection of stationarity-inducing transforms for financial sequences.

    All methods operate on ``pandas.Series`` indexed by a monotonically
    increasing timestamp (or an equivalent ordered index) and return
    ``pandas.Series`` aligned to a (possibly truncated) subset of that
    index, so callers should always re-align outputs via an inner join
    before assembling a feature matrix.

    Parameters
    ----------
    numerical_epsilon:
        Small constant used to guard against division-by-zero in
        ratio-based features (e.g. relative spread, mid-price normalizers).
    """

    def __init__(self, numerical_epsilon: float = 1e-12) -> None:
        if numerical_epsilon <= 0:
            raise ValueError("numerical_epsilon must be strictly positive.")
        self.numerical_epsilon = numerical_epsilon

    # ------------------------------------------------------------------
    # Basic stationarity-inducing transforms
    # ------------------------------------------------------------------
    def log_returns(self, prices: pd.Series) -> pd.Series:
        """
        Compute log returns: r_t = ln(p_t / p_{t-1}).

        Raises
        ------
        ValueError
            If the series is empty or contains any non-positive price
            (log undefined).
        """
        if prices.empty:
            raise ValueError("Cannot compute log returns on an empty series.")
        if (prices <= 0).any():
            raise ValueError(
                "Log returns require strictly positive prices; found "
                f"{int((prices <= 0).sum())} non-positive observation(s)."
            )
        returns = np.log(prices / prices.shift(1))
        return returns.dropna().rename(f"{prices.name or 'price'}_log_return")

    def rolling_volatility(
        self,
        returns: pd.Series,
        window: int = 20,
        annualize: bool = False,
        periods_per_year: float = 252.0 * 390.0,
    ) -> pd.Series:
        """
        Rolling realized volatility: the trailing standard deviation of
        returns over a fixed window.

        Parameters
        ----------
        window:
            Trailing window size, in observations.
        annualize:
            If True, scale by sqrt(periods_per_year) to express volatility
            on an annualized basis (default assumes ~390 one-minute
            bars/session).
        """
        if window < 2:
            raise ValueError("window must be >= 2 to compute a standard deviation.")
        vol = returns.rolling(window=window, min_periods=window).std(ddof=1)
        if annualize:
            vol = vol * np.sqrt(periods_per_year)
        return vol.dropna().rename(f"rolling_vol_{window}")

    def rolling_spread(
        self,
        bid: pd.Series,
        ask: pd.Series,
        window: int = 20,
        relative: bool = True,
    ) -> pd.Series:
        """
        Rolling mean bid-ask spread.

        Parameters
        ----------
        relative:
            If True, normalize by the mid-price ((ask + bid) / 2) at each
            tick before averaging, yielding a scale-free spread comparable
            across price regimes. If False, returns the raw absolute
            spread.
        """
        bid, ask = bid.align(ask, join="inner")
        if (bid > ask).any():
            n_violations = int((bid > ask).sum())
            raise ValueError(
                f"Found {n_violations} crossed-book observation(s) where bid > ask."
            )
        spread = ask - bid
        if relative:
            mid = (ask + bid) / 2.0
            spread = spread / mid.clip(lower=self.numerical_epsilon)
        label = f"rolling_spread_{window}{'_rel' if relative else ''}"
        return spread.rolling(window=window, min_periods=window).mean().dropna().rename(label)

    # ------------------------------------------------------------------
    # Fractional differencing (Lopez de Prado, fixed-width window / "FFD")
    # ------------------------------------------------------------------
    @staticmethod
    def _ffd_weights(d: float, threshold: float, max_size: int = 20_000) -> np.ndarray:
        """
        Compute fixed-width-window fractional differencing weights.

        The binomial-series weights w_k = (-1)^k * C(d, k) are generated
        via the recursion w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k,
        truncated as soon as |w_k| falls below `threshold` -- this defines
        the effective (fixed) window width.

        Returns
        -------
        np.ndarray
            Weights ordered from the oldest lag to the most recent
            (lag 0), i.e. ready for a left-to-right dot product against a
            chronologically ordered window of observations.
        """
        if not (0.0 <= d <= 1.0):
            logger.warning("Fractional order d=%.4f is outside the typical [0, 1] range.", d)
        weights = [1.0]
        k = 1
        while k < max_size:
            w_k = -weights[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            weights.append(w_k)
            k += 1
        return np.array(weights[::-1], dtype=np.float64)

    def fractional_difference(
        self, series: pd.Series, d: float, threshold: float = 1e-5
    ) -> pd.Series:
        """
        Apply fixed-width-window fractional differencing to `series`.

        Unlike integer differencing (d=1, e.g. simple returns), a
        fractional `d` in (0, 1) can remove a unit root / non-stationary
        trend while retaining much more of the series' long-memory
        autocorrelation structure -- valuable for volatility and
        volume-profile features where the "memory" of the process is
        itself informative.

        Parameters
        ----------
        series:
            Input series, assumed free of interior NaNs (forward-fill or
            drop missing values first).
        d:
            Fractional differencing order, typically searched over [0, 1].
        threshold:
            Weight-loss cutoff controlling the effective window width;
            smaller values retain more history at the cost of a longer
            burn-in period.

        Returns
        -------
        pd.Series
            Differenced series, truncated at the front by (window - 1)
            observations (insufficient history to apply the full weight
            vector).
        """
        if series.isna().any():
            raise ValueError(
                "fractional_difference requires a series with no interior NaNs; "
                "forward-fill or drop missing values first."
            )
        weights = self._ffd_weights(d=d, threshold=threshold)
        width = len(weights)
        values = series.to_numpy(dtype=np.float64)
        if len(values) < width:
            raise ValueError(
                f"Series length ({len(values)}) is shorter than the fractional "
                f"differencing window width ({width}) implied by d={d}, threshold={threshold}. "
                "Use a larger threshold or a longer series."
            )
        windows = sliding_window_view(values, window_shape=width)
        diffed = windows @ weights
        index = series.index[width - 1 :]
        return pd.Series(diffed, index=index, name=f"{series.name or 'series'}_fracdiff_d{d:.3f}")

    # ------------------------------------------------------------------
    # Stationarity diagnostics (manual ADF -- no statsmodels dependency)
    # ------------------------------------------------------------------
    @staticmethod
    def augmented_dickey_fuller(series: pd.Series, lags: Optional[int] = None) -> Dict[str, float]:
        """
        Augmented Dickey-Fuller unit-root test, implemented from scratch
        via OLS (no external stats dependency beyond numpy).

        Tests H0: series has a unit root (non-stationary) against
        H1: series is stationary, using a constant-only specification
        (no deterministic trend term).

        Note
        ----
        Critical values are asymptotic approximations (MacKinnon) valid
        for reasonably large samples; for small-sample or trend-inclusive
        inference, cross-check against
        `statsmodels.tsa.stattools.adfuller`.
        """
        y = series.dropna().to_numpy(dtype=np.float64)
        n = len(y)
        if n < 30:
            raise ValueError("Series too short for a meaningful ADF test (need >= 30 observations).")
        if lags is None:
            lags = int(np.ceil(12.0 * (n / 100.0) ** 0.25))  # Schwert (1989) rule of thumb

        y_diff = np.diff(y)
        y_lag = y[:-1]
        n_obs = len(y_diff) - lags
        if n_obs <= lags + 3:
            raise ValueError(
                "Series too short to support the implied lag order; provide more data or fewer lags."
            )

        regressors = [np.ones(n_obs), y_lag[lags:]]
        for i in range(1, lags + 1):
            regressors.append(y_diff[lags - i : len(y_diff) - i])
        X = np.column_stack(regressors)
        Y = y_diff[lags:]

        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        residuals = Y - X @ beta
        dof = n_obs - X.shape[1]
        if dof <= 0:
            raise ValueError("Non-positive degrees of freedom; reduce lag order or supply more data.")
        sigma2 = float(np.sum(residuals**2) / dof)
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * xtx_inv))

        gamma_hat = beta[1]
        gamma_se = se[1]
        t_stat = float(gamma_hat / gamma_se)

        critical_values = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
        return {
            "adf_statistic": t_stat,
            "critical_values": critical_values,
            "n_obs": int(n_obs),
            "lags_used": int(lags),
            "is_stationary_95pct": bool(t_stat < critical_values["5%"]),
        }

    def find_minimum_d(
        self,
        series: pd.Series,
        d_grid: Optional[np.ndarray] = None,
        threshold: float = 1e-5,
    ) -> pd.DataFrame:
        """
        Grid-search the minimum fractional order `d` that achieves
        stationarity (by the 95%-confidence ADF criterion) while
        reporting correlation with the original series as a proxy for
        retained memory -- the practical goal being the *smallest* d that
        still passes the stationarity test, per Lopez de Prado's
        minimum-effective-differencing principle.

        Returns
        -------
        pd.DataFrame
            Columns: d, adf_statistic, corr_with_original, is_stationary_95pct.
        """
        if d_grid is None:
            d_grid = np.linspace(0.0, 1.0, 21)
        records = []
        for d in d_grid:
            try:
                diffed = self.fractional_difference(series, d=float(d), threshold=threshold)
                if len(diffed) < 30:
                    continue
                adf_result = self.augmented_dickey_fuller(diffed)
                aligned_original = series.reindex(diffed.index)
                corr = float(np.corrcoef(diffed.to_numpy(), aligned_original.to_numpy())[0, 1])
                records.append(
                    {
                        "d": float(d),
                        "adf_statistic": adf_result["adf_statistic"],
                        "corr_with_original": corr,
                        "is_stationary_95pct": adf_result["is_stationary_95pct"],
                    }
                )
            except ValueError as exc:
                logger.debug("Skipping d=%.3f: %s", d, exc)
                continue
        if not records:
            raise RuntimeError("No candidate d value produced a valid fractional differencing result.")
        return pd.DataFrame.from_records(records)
