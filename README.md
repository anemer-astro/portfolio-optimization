# Portfolio Optimization & Risk Modeling (Python)
End-to-end MVO (Min-Var / Max-Sharpe), Efficient Frontier with Monte Carlo, Risk Parity, Black–Litterman, and optional regime-aware risk targeting. Runs on Yahoo Finance data.

## Quick start
```bash
pip install -r requirements.txt
python scripts/portfolio_opt_plus_regime.py --download --preset global-core --rf 0.02 --benchmark VTI

