# Portfolio Optimization & Risk Modeling (Python)
End-to-end MVO (Min-Var / Max-Sharpe), Efficient Frontier with Monte Carlo, Risk Parity, Black–Litterman, and optional regime-aware risk targeting. Runs on Yahoo Finance data.

## Quick start
```bash
pip install -r requirements.txt
python portfolio_opt_plus_regime.py --download --rf 0.045 --benchmark VTI   --market_equal --tau 0.2   --regime --regime-window 60 --regime-proxy VTI   --regime-low-pct 0.2 --regime-high-pct 0.8   --regime-low-scale 1.3 --regime-mid-scale 1.0 --regime-high-scale 0.7 --view "BTC-USD:+0.08@0.001,BIL:+0.02@0.001"

## 🧠 How It Works

**Data ➜ Returns ➜ Optimization ➜ Risk Models ➜ Views ➜ Backtest**

1. **Data & Cleaning** — Downloads Adjusted Close prices (Yahoo Finance), aligns to business days, forward-fills small gaps.
2. **Returns** — Computes log returns (daily or weekly).  
3. **Optimization** — Builds **Mean–Variance** portfolios under constraints (long-only, fully invested):  
   - Min-Variance (minimizes \( w^\top \Sigma w \))  
   - Max-Sharpe (maximizes \( \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}} \))  
4. **Efficient Frontier** — Plots the long-only frontier; overlays **Monte Carlo** random portfolios for context.  
5. **Risk Parity** — Solves for **equal risk contributions** (each asset contributes equally to total variance).  
6. **Black–Litterman** — Blends equilibrium returns with **investor views** (e.g., `"BTC-USD:+0.08@0.001,BIL:+0.02@0.001"`), where `@` is the view variance (smaller = higher confidence).  
7. **Backtest** — In-sample fixed-weight backtest vs a benchmark (e.g., VTI) + optional **regime-aware scaling** of risk based on a volatility proxy.

## 📐 Key Formulas

**Portfolio Variance**

$$
\sigma_p^2 = w^\top \Sigma w
$$

**Sharpe Ratio**

$$
\text{Sharpe}(w) = \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}
$$

**Risk Contribution**

$$
RC_i = \frac{w_i \cdot (\Sigma w)_i}{w^\top \Sigma w}
$$


### Reproduce the Figures

```bash
pip install -r requirements.txt
python scripts/portfolio_opt_plus_regime.py --download --rf 0.045 --benchmark VTI \
  --market_equal --tau 0.2 \
  --regime --regime-window 60 --regime-proxy VTI \
  --regime-low-pct 0.2 --regime-high-pct 0.8 \
  --regime-low-scale 1.3 --regime-mid-scale 1.0 --regime-high-scale 0.7 \
  --view "BTC-USD:+0.08@0.001,BIL:+0.02@0.001"
```
### Example output
![Efficient Frontier](figures/efficient_frontier_mc.png)
![Backtest](figures/backtest_cum_returns_regime.png)
![Backtest](figures/backtest_cum_returns_regime.png)
