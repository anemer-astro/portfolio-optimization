#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portfolio_opt_plus_regime.py
============================

A carefully commented, slightly optimized, and safe version of 
portfolio-optimization script with the core features:

- Mean-Variance Optimization (Min-Variance & Max-Sharpe), long-only, fully invested
- Monte Carlo random portfolios + Efficient Frontier plot highlighting Min-Var & Max-Sharpe
- Risk Parity (equal risk contributions) via SLSQP
- Black–Litterman (equilibrium prior returns with optional simple views)
- Simple in-sample backtest vs a chosen benchmark column

Examples:
    python portfolio_opt_plus.py --rf 0.02 --weekly --benchmark SPY
    python portfolio_opt_plus.py --rf 0.00 --benchmark VTI --view "VTI:+0.01@0.5,GLD:+0.02@0.3"
    python portfolio_opt_plus.py --download --tickers "VTI,VEA,VWO,AGG,GLD" --start 2015-01-01 --end 2025-01-01

Enhancements over portfolio_opt_plus.py:
- Adds **Regime-Aware Risk Targeting** via --regime and related flags:
    * Detect high/low volatility regimes from a proxy series (benchmark or mean of assets)
    * Scale daily/weekly portfolio exposure down in high-vol regimes and up in low-vol regimes
    * Parameters: window length, percentiles, and scale multipliers

Typical usage:
    python portfolio_opt_plus_regime.py --download --rf 0.045 --benchmark VTI \
      --market_equal \
      --view "BTC-USD:+0.05@0.005,GLD:+0.02@0.005" \
      --tau 0.2 --delta 2.5 \
      --regime --regime-window 60 --regime-proxy VTI \
      --regime-low-pct 0.2 --regime-high-pct 0.8 \
      --regime-low-scale 1.3 --regime-mid-scale 1.0 --regime-high-scale 0.7

Outputs (by default):
- data/optimal_weights.csv
- data/efficient_frontier.csv
- data/random_portfolios.csv
- data/risk_parity_weights.csv
- data/black_litterman_weights.csv
- figures/efficient_frontier_mc.png  (unless --no-plots)
- figures/backtest_cum_returns.png (baseline)
- figures/backtest_cum_returns_regime.png (with regime scaling, if enabled)
"""

import os
import math
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:
    yf = None

from scipy.optimize import minimize

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# I/O utils
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create the default output folders if missing."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)


def load_prices(prices_csv: str = "data/prices.csv") -> pd.DataFrame:
    """
    Load prices from CSV and align to business days.
    The file must contain Adj Close-like columns for each asset.
    """
    if not os.path.exists(prices_csv):
        raise FileNotFoundError(
            f"Missing {prices_csv}. Provide --prices or run with --download."
        )
    df = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    df = df.asfreq("B").ffill().bfill()
    df = df.dropna(axis=1, how="all")
    return df


def compute_log_returns(prices: pd.DataFrame, weekly: bool = False) -> pd.DataFrame:
    """
    Compute (optionally weekly) log returns.
    Weekly returns are resampled to W-FRI and summed to match log compounding.
    """
    rets = np.log(prices / prices.shift(1)).dropna(how="all")
    if weekly:
        rets = rets.resample("W-FRI").sum().dropna(how="all")
    return rets


def annualize_stats(returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Annualize mean and covariance from per-period returns.
    """
    mu = returns.mean() * periods_per_year
    cov = returns.cov() * periods_per_year
    return mu, cov


def shrink_covariance(cov: pd.DataFrame, lam: float = 0.0, jitter: float = 0.0) -> pd.DataFrame:
    """
    Optional diagonal shrinkage and numerical jitter for better conditioning:
        C_shrunk = (1 - lam) * C + lam * diag(diag(C)) + jitter * I

    Parameters
    ----------
    cov : pd.DataFrame
        Annualized covariance matrix.
    lam : float
        Shrinkage intensity in [0, 1]. 0 = no shrinkage, 1 = fully diagonal.
    jitter : float
        Small value added to diagonal to avoid near-singularity (e.g., 1e-10).
    """
    C = cov.values.copy()
    if lam > 0.0:
        diag = np.diag(np.diag(C))
        C = (1.0 - lam) * C + lam * diag
    if jitter > 0.0:
        C = C + np.eye(C.shape[0]) * jitter
    return pd.DataFrame(C, index=cov.index, columns=cov.columns)


def portfolio_performance(w: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0
) -> Tuple[float, float, float]:
    """
    Compute portfolio expected return, volatility, and Sharpe ratio.

    Notes
    -----
    The '@' operator in Python means matrix multiplication.
    Examples:
        w @ mu.values            -> dot product (portfolio return)
        w @ cov.values @ w       -> quadratic form (portfolio variance)
    """
    w = np.array(w, dtype=float)
    r = float(w @ mu.values)
    var = float((w @ cov.values @ w).item())
    v = float(np.sqrt(max(var, 1e-16)))
    s = (r - rf) / v if v > 0 else float("nan")
    return r, v, s


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def _sum_to_one(w: np.ndarray) -> float:
    return float(np.sum(w) - 1.0)


def _long_only_bounds(n: int):
    return tuple((0.0, 1.0) for _ in range(n))


def solve_min_variance(mu: pd.Series, cov: pd.DataFrame) -> np.ndarray:
    """
    Long-only global minimum-variance portfolio.
    """
    n = len(mu)
    x0 = np.ones(n) / n
    bnds = _long_only_bounds(n)
    cons = ({"type": "eq", "fun": _sum_to_one},)
    """
    Allow shorting: use e.g. (-0.3, 1.0) per asset, or even (-1.0, 1.0).

    Cap single-name risk: change upper bound to, say, 0.25 → tuple((0.0, 0.25) for _ in range(n)).

    Allow cash (not fully invested): replace equality with an inequality like sum(w) ≤ 1:
    """
    def obj(w): return float((w @ cov.values @ w).item())
    res = minimize(obj, x0=x0, method="SLSQP", bounds=bnds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Min-Var failed: {res.message}")
    return res.x


def solve_max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> np.ndarray:
    """
    Long-only maximum Sharpe portfolio.
    """
    n = len(mu)
    x0 = np.ones(n) / n
    bnds = _long_only_bounds(n)
    cons = ({"type": "eq", "fun": _sum_to_one},)
    def neg_sharpe(w):
        r = float(w @ mu.values)
        var = float((w @ cov.values @ w).item())
        v = math.sqrt(max(var, 1e-16))
        return - (r - rf) / v
    res = minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=bnds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Max-Sharpe failed: {res.message}")
    return res.x


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int = 60, rf: float = 0.0
) -> pd.DataFrame:
    """
    Trace the long-only efficient frontier by sweeping target returns.
    """
    n = len(mu)
    bnds = _long_only_bounds(n)
    cons_base = [{"type": "eq", "fun": _sum_to_one}]
    targets = np.linspace(float(mu.min()), float(mu.max()), n_points)
    rows = []
    for t in targets:
        def obj(w): return float((w @ cov.values @ w).item())
        def target_ret(w, tt=t): return float(w @ mu.values - tt)
        cons = cons_base + [{"type": "eq", "fun": target_ret}]
        x0 = np.ones(n) / n
        res = minimize(obj, x0=x0, method="SLSQP", bounds=bnds, constraints=cons)
        if not res.success:
            continue
        w = res.x
        r, v, s = portfolio_performance(w, mu, cov, rf=rf)
        rows.append({
            "target": t,
            "return": r,
            "volatility": v,
            "sharpe": s,
            **{f"w_{a}": w[i] for i, a in enumerate(mu.index)}
        })
    return pd.DataFrame(rows)


def random_portfolios(mu: pd.Series, cov: pd.DataFrame, n_samples: int = 5000, seed: int = 7, rf: float = 0.0
) -> pd.DataFrame:
    """
    Sample random long-only portfolios to form a 'cloud' around the EF.
    """
    rng = np.random.default_rng(seed)
    n = len(mu)
    W = rng.random((n_samples, n))
    W = W / W.sum(axis=1, keepdims=True)
    rets = W @ mu.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov.values, W))
    sharpes = (rets - rf) / np.maximum(vols, 1e-12)
    df = pd.DataFrame({"return": rets, "volatility": vols, "sharpe": sharpes})
    return df


# ---------------------------------------------------------------------------
# Risk Parity (equal risk contributions) via SLSQP
# ---------------------------------------------------------------------------

def _risk_contribution(w: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Risk contributions RC_i = w_i * (C w)_i / (w' C w).
    """
    w = w.reshape(-1, 1)
    port_var = float((w.T @ C @ w).item())  # or: float(np.dot(w.T, C @ w))
    mrc = (C @ w).flatten()                   # marginal risk contributions
    rc = mrc * w.flatten() / max(port_var, 1e-16)
    return rc


def solve_risk_parity(cov: pd.DataFrame) -> np.ndarray:
    """
    Solve for weights whose risk contributions are (approximately) equal by
    minimizing the squared deviation from equal contribution under simplex constraints.
    """
    n = cov.shape[0]
    x0 = np.ones(n) / n
    bnds = _long_only_bounds(n)
    cons = ({"type": "eq", "fun": _sum_to_one},)
    C = cov.values
    def obj(w):
        rc = _risk_contribution(w, C)
        target = np.mean(rc)
        return float(np.sum((rc - target) ** 2))
    res = minimize(obj, x0=x0, method="SLSQP", bounds=bnds, constraints=cons)
    if not res.success:
        # Fallback to naive equal-weights if optimizer struggles
        return x0
    return res.x


# ---------------------------------------------------------------------------
# Black–Litterman
# ---------------------------------------------------------------------------

def reverse_optimization(mu: pd.Series, cov: pd.DataFrame, market_weights: np.ndarray, delta: float = 2.5) -> pd.Series:
    """
    Equilibrium implied *excess* returns Pi = delta * C * w_mkt
    delta ~ risk aversion; 2.5 is a common prior for annual data.
    """
    Pi = delta * (cov.values @ market_weights)
    return pd.Series(Pi, index=mu.index)


def parse_views(views_arg: Optional[str], assets: List[str], alias_map: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Parse multi-view string of the form:
        "VTI:+0.01@0.5,GLD:+0.02@0.3"
    -> {"VTI": (0.01, 0.5), "GLD": (0.02, 0.3)}
    """
    if not views_arg:
        return None
    out: Dict[str, Tuple[float, float]] = {}
    for chunk in views_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            asset, rhs = chunk.split(":")
            q_str, omega_str = rhs.split("@")
            asset_in = asset.strip()
            if alias_map:
                asset_in = alias_map.get(asset_in, asset_in)
            if asset_in not in assets:
                raise ValueError(f"View asset '{asset}' not in columns {assets}")
            out[asset_in] = (float(q_str), float(omega_str))
        except Exception as e:
            raise ValueError(f"Could not parse view '{chunk}'; expected 'Asset:+0.01@0.5'") from e
    return out


def black_litterman(mu: pd.Series, cov: pd.DataFrame, market_weights: np.ndarray, tau: float = 0.05,
                    delta: float = 2.5, views: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.Series:
    """
    Basic BL posterior mean returns.

    - mu: historical (ignored in vanilla BL; used only if you encode "views" relative to mu)
    - cov: annualized covariance
    - market_weights: prior capitalization weights (sum=1)
    - tau: scale of uncertainty in prior
    - delta: risk aversion for implied returns
    - views: dict { "AssetName": (view_excess_return, omega_diag) }
      where omega_diag is the variance (confidence) of the view (smaller -> stronger).

    Returns
    -------
    pd.Series of posterior mean *excess* returns (same index as mu).
    """
    assets = list(mu.index)
    n = len(assets)
    C = cov.values
    w_mkt = market_weights

    # Prior (equilibrium) returns
    Pi = reverse_optimization(mu, cov, w_mkt, delta=delta).values
    Sigma_prior = tau * C

    if not views:
        # No views -> posterior equals prior
        return pd.Series(Pi, index=assets)

    # Build pick matrix P and views Q, Omega
    P = np.zeros((len(views), n))
    Q = np.zeros((len(views),))
    Omega = np.zeros((len(views), len(views)))

    for i, (asset, (q, omega_diag)) in enumerate(views.items()):
        j = assets.index(asset)
        P[i, j] = 1.0
        Q[i] = q
        Omega[i, i] = omega_diag

    # BL posterior mean: Pi* = Pi + Sigma_prior P^T (P Sigma_prior P^T + Omega)^{-1} (Q - P Pi)
    M = P @ Sigma_prior @ P.T + Omega
    adj = Sigma_prior @ P.T @ np.linalg.inv(M) @ (Q - P @ Pi)
    Pi_post = Pi + adj
    return pd.Series(Pi_post, index=assets)


# ---------------------------------------------------------------------------
# Backtesting (with optional regime risk targeting)
# ---------------------------------------------------------------------------

def backtest_fixed_weights(prices: pd.DataFrame, weights: np.ndarray, benchmark: Optional[str] = None,
                           scaler: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    In-sample naive backtest: cumulative growth with fixed weights on entire price history.
    If scaler is provided (e.g., regime-based exposure), multiply daily/weekly returns by scaler.
    Returns cumulative return series for portfolio and benchmark.
    """
    rets = np.log(prices / prices.shift(1)).fillna(0.0)
    port_rets = (rets * weights).sum(axis=1)
    if scaler is not None:
        scaler = scaler.reindex(port_rets.index).ffill().fillna(1.0)
        port_rets = port_rets * scaler
    port_cum = np.exp(port_rets.cumsum())
    out = pd.DataFrame({"Portfolio": port_cum})
    if benchmark and (benchmark in prices.columns):
        bench_rets = np.log(prices[benchmark] / prices[benchmark].shift(1)).fillna(0.0)
        out[benchmark] = np.exp(bench_rets.cumsum())
    return out.dropna()


def compute_regime_scaler(returns: pd.DataFrame, window: int = 60, proxy: Optional[str] = None,
                          low_pct: float = 0.2, high_pct: float = 0.8,
                          low_scale: float = 1.3, mid_scale: float = 1.0, high_scale: float = 0.7
) -> pd.Series:
    """
    Build a time-varying scaler in {low_scale, mid_scale, high_scale} based on rolling volatility of a proxy series.
    - proxy: column name to use as proxy; if None or missing, use cross-sectional mean of returns each period.
    - window: rolling window (in periods) for std; annualization factor is inferred from returns frequency by caller.
    """
    if proxy and proxy in returns.columns:
        series = returns[proxy]
    else:
        series = returns.mean(axis=1)  # average across assets as a simple market proxy
    # Rolling std (not annualized here; we just use relative ranking via percentiles)
    roll_std = series.rolling(window).std()
    # Determine thresholds
    lo_th = roll_std.quantile(low_pct)
    hi_th = roll_std.quantile(high_pct)
    scaler = pd.Series(mid_scale, index=roll_std.index)
    scaler[roll_std <= lo_th] = low_scale
    scaler[roll_std >= hi_th] = high_scale
    # Forward-fill early NaNs due to rolling window
    scaler = scaler.ffill().bfill()
    return scaler


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = {
    "VTI": "US Total Equity",
    "VEA": "Developed ex-US Equity",
    "VWO": "Emerging Markets Equity",
    "AGG": "US Aggregate Bonds",
    "TLT": "US Treasuries (Long)",
    "TIP": "US TIPS",
    "VNQ": "US REITs",
    "DBC": "Broad Commodities",
    "BTC-USD": "Bitcoin (USD)",
    "BIL": "Short Treasuries (cash proxy)",
    "LQD": "Corporates IG",
    "HYG": "High yield",
    "GLD": "Gold",
}

def download_prices(tickers: Optional[List[str]] = None, start: str = "2015-01-01", end: str = "2025-01-01",
                    use_friendly_names: bool = True) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed; pip install yfinance to use --download.")
    tickers = list(tickers or DEFAULT_TICKERS.keys())
    df = yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"]
    # Ensure DataFrame structure
    if isinstance(df, pd.Series):
        df = df.to_frame()
    # Standardize column order & names
    df = df[tickers]
    if use_friendly_names:
        df.columns = [DEFAULT_TICKERS.get(t, t) for t in tickers]
    # Align to business days (avoid crypto weekends causing index misalignment)
    df = df.asfreq("B").ffill().bfill()
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Portfolio Optimization with Regime-Aware Risk Targeting")
    parser.add_argument("--prices", type=str, default="data/prices.csv")
    parser.add_argument("--rf", type=float, default=0.0)
    parser.add_argument("--weekly", action="store_true")
    parser.add_argument("--ef_points", type=int, default=60)
    parser.add_argument("--mc_samples", type=int, default=5000)
    parser.add_argument("--benchmark", type=str, default=None)

    # Covariance conditioning
    parser.add_argument("--cov_shrink", type=float, default=0.0)
    parser.add_argument("--cov_jitter", type=float, default=0.0)

    # BL controls
    parser.add_argument("--tau", type=float, default=0.05, help="BL prior uncertainty")
    parser.add_argument("--delta", type=float, default=2.5, help="Risk aversion for implied returns")
    parser.add_argument("--market_equal", action="store_true", help="Use equal weights for BL prior anchor")
    parser.add_argument("--market-weights", type=str, default=None, help="CSV with one row of market weights")
    parser.add_argument("--view", type=str, default=None, help='Views like "VTI:+0.01@0.5,GLD:+0.02@0.3"')

    # Regime targeting
    parser.add_argument("--regime", action="store_true", help="Enable regime-aware risk targeting")
    parser.add_argument("--regime-window", type=int, default=60, help="Rolling window (periods) for proxy vol")
    parser.add_argument("--regime-proxy", type=str, default=None, help="Column name used as proxy; default market mean")
    parser.add_argument("--regime-low-pct", type=float, default=0.2, help="Lower percentile for low-vol regime")
    parser.add_argument("--regime-high-pct", type=float, default=0.8, help="Upper percentile for high-vol regime")
    parser.add_argument("--regime-low-scale", type=float, default=1.3, help="Scale in low-vol regime (>1)")
    parser.add_argument("--regime-mid-scale", type=float, default=1.0, help="Scale in mid regime (=1)")
    parser.add_argument("--regime-high-scale", type=float, default=0.7, help="Scale in high-vol regime (<1)")

    # Plots
    parser.add_argument("--no-plots", dest="no_plots", action="store_true", help="Disable figure generation")

    # Download
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--tickers", type=str, default=None, help='Comma-separated tickers for --download')
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--friendly-names", action="store_true", help="Rename columns to friendly names")

    args = parser.parse_args()
    ensure_dirs()

    # Optional download
    if args.download:
        tlist = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
        df_dl = download_prices(tickers=tlist, start=args.start, end=args.end, use_friendly_names=args.friendly_names)
        os.makedirs(os.path.dirname(args.prices) or ".", exist_ok=True)
        df_dl.to_csv(args.prices)

    # Load data & stats
    prices = load_prices(args.prices)
    returns = compute_log_returns(prices, weekly=args.weekly)
    periods = 52 if args.weekly else 252
    mu, cov = annualize_stats(returns, periods_per_year=periods)
    cov = shrink_covariance(cov, lam=args.cov_shrink, jitter=args.cov_jitter)

    # Step 3: MVO
    w_minvar = solve_min_variance(mu, cov)
    w_maxsharpe = solve_max_sharpe(mu, cov, rf=args.rf)

    # Step 4: EF + cloud
    ef = efficient_frontier(mu, cov, n_points=args.ef_points, rf=args.rf)
    cloud = random_portfolios(mu, cov, n_samples=args.mc_samples, rf=args.rf)

    # Save basics
    pd.DataFrame({
        "Asset": mu.index,
        "MinVariance_Weight": w_minvar,
        "MaxSharpe_Weight": w_maxsharpe
    }).to_csv("data/optimal_weights.csv", index=False)
    ef.to_csv("data/efficient_frontier.csv", index=False)
    cloud.to_csv("data/random_portfolios.csv", index=False)

    # Plot EF + cloud
    r_mv, v_mv, _ = portfolio_performance(w_minvar, mu, cov, rf=args.rf)
    r_ms, v_ms, _ = portfolio_performance(w_maxsharpe, mu, cov, rf=args.rf)
    if not args.no_plots:
        plt.figure(figsize=(8,6))
        plt.scatter(cloud["volatility"], cloud["return"], s=6, alpha=0.35, label="Random Portfolios")
        plt.plot(ef["volatility"], ef["return"], linewidth=2, label="Efficient Frontier")
        plt.scatter([v_mv], [r_mv], s=80, marker="o", label="Min Variance")
        plt.scatter([v_ms], [r_ms], s=80, marker="^", label="Max Sharpe")
        plt.xlabel("Annualized Volatility")
        plt.ylabel("Annualized Return")
        plt.title("Efficient Frontier with Monte Carlo Cloud")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/efficient_frontier_mc.png", dpi=150)
        plt.close()

    # Step 5a: Risk Parity
    w_rp = solve_risk_parity(cov)
    pd.DataFrame({"Asset": mu.index, "RiskParity_Weight": w_rp}).to_csv("data/risk_parity_weights.csv", index=False)

    # Step 5b: Black–Litterman
    assets = list(mu.index)

    # Alias map so --view accepts either tickers or friendly names
    alias = {}
    for tkr, nice in DEFAULT_TICKERS.items():
        alias[tkr] = nice
        alias[nice] = nice
    # Also map raw column names to themselves so they always match
    for col in assets:
        alias[col] = col

    # Market weights
    if args.market_weights:
        mw = pd.read_csv(args.market_weights)
        mw = mw[assets].iloc[0].values.astype(float)
        w_mkt = mw / mw.sum()
    elif args.market_equal:
        w_mkt = np.ones(len(mu)) / len(mu)
    else:
        w_mkt = w_minvar / w_minvar.sum()

    views = parse_views(args.view, assets, alias_map=alias) if args.view else None
    mu_bl = black_litterman(mu, cov, w_mkt, tau=args.tau, delta=args.delta, views=views)
    Pi = reverse_optimization(mu, cov, w_mkt, delta=args.delta)

    w_bl = solve_max_sharpe(mu_bl, cov, rf=args.rf)
    pd.DataFrame({"Asset": mu.index, "BlackLitterman_Weight": w_bl}).to_csv("data/black_litterman_weights.csv", index=False)

    # Step 5c: Backtests (baseline and optional regime-scaled)
    back = backtest_fixed_weights(prices, w_maxsharpe, benchmark=args.benchmark)
    if not args.no_plots and not back.empty:
        plt.figure(figsize=(8,6))
        for col in back.columns:
            plt.plot(back.index, back[col], label=col)
        plt.xlabel("Date"); plt.ylabel("Cumulative Growth (normalized)")
        plt.title("In-Sample Backtest: Max-Sharpe vs Benchmark")
        plt.legend(); plt.tight_layout()
        plt.savefig("figures/backtest_cum_returns.png", dpi=150)
        plt.close()

    # Regime-aware risk targeting
    if args.regime:
        scaler = compute_regime_scaler(
            returns=compute_log_returns(prices, weekly=args.weekly),  # use same freq
            window=args.regime_window,
            proxy=args.regime_proxy,
            low_pct=args.regime_low_pct,
            high_pct=args.regime_high_pct,
            low_scale=args.regime_low_scale,
            mid_scale=args.regime_mid_scale,
            high_scale=args.regime_high_scale
        )
        back_reg = backtest_fixed_weights(prices, w_maxsharpe, benchmark=args.benchmark, scaler=scaler)
        if not args.no_plots and not back_reg.empty:
            plt.figure(figsize=(8,6))
            # Plot both portfolio lines
            plt.plot(back.index, back["Portfolio"], label="Portfolio (baseline)")
            plt.plot(back_reg.index, back_reg["Portfolio"], label="Portfolio (regime-targeted)")
            # Optional: benchmark
            if args.benchmark and args.benchmark in back_reg.columns:
                plt.plot(back_reg.index, back_reg[args.benchmark], label=args.benchmark)
            plt.xlabel("Date"); plt.ylabel("Cumulative Growth (normalized)")
            plt.title("Backtest: Baseline vs Regime-Targeted")
            plt.legend(); plt.tight_layout()
            plt.savefig("figures/backtest_cum_returns_regime.png", dpi=150)
            plt.close()

    # Console summary
    perf_labels = ["Min-Variance", "Max-Sharpe", "Risk-Parity", "BL-MaxSharpe"]
    weights_mat = np.vstack([w_minvar, w_maxsharpe, w_rp, w_bl])
    perf_rows = []
    for lbl, w in zip(perf_labels, weights_mat):
        r, v, s = portfolio_performance(w, mu, cov, rf=args.rf)
        perf_rows.append([lbl, r, v, s])
    perf_df = pd.DataFrame(perf_rows, columns=["Portfolio", "Annual Return", "Annual Volatility", "Sharpe"])

    print("\n=== Portfolio Metrics (annualized) ===")
    print(perf_df.to_string(index=False))

    print("\nSaved:")
    print("- data/optimal_weights.csv")
    print("- data/efficient_frontier.csv")
    print("- data/random_portfolios.csv")
    print("- data/risk_parity_weights.csv")
    print("- data/black_litterman_weights.csv")
    if not args.no_plots:
        print("- figures/efficient_frontier_mc.png")
        print("- figures/backtest_cum_returns.png")
        if args.regime:
            print("- figures/backtest_cum_returns_regime.png")


if __name__ == "__main__":
    main()
