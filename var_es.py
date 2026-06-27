"""
Value-at-Risk & Expected Shortfall
====================================
Three VaR methodologies: Historical Simulation, Parametric (GARCH),
Monte Carlo. Backtests with Kupiec proportion-of-failures test.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ── 1. Simulate returns ───────────────────────────────────────────────────────

def generate_portfolio_returns(n=1000, seed=42):
    """
    Simulate daily portfolio returns with GARCH(1,1) volatility clustering.
    In production: replace with actual portfolio return series.
    """
    np.random.seed(seed)
    omega, alpha, beta_ = 0.00001, 0.08, 0.90   # GARCH(1,1) params
    eps = np.random.standard_t(df=5, size=n) / np.sqrt(5/3)  # fat tails

    sigma2 = np.zeros(n); sigma2[0] = omega / (1 - alpha - beta_)
    for t in range(1, n):
        sigma2[t] = omega + alpha * (eps[t-1]*np.sqrt(sigma2[t-1]))**2 + beta_*sigma2[t-1]
    returns = eps * np.sqrt(sigma2)

    dt = pd.date_range("2021-01-04", periods=n, freq="B")
    return pd.Series(returns, index=dt, name="portfolio_return")

# ── 2. Historical Simulation VaR ─────────────────────────────────────────────

def var_historical(returns, confidence=0.99, window=250):
    var = returns.rolling(window).quantile(1 - confidence)
    es  = returns.rolling(window).apply(
        lambda x: x[x <= np.quantile(x, 1 - confidence)].mean(), raw=True)
    return var.rename("VaR_HS"), es.rename("ES_HS")

# ── 3. GARCH(1,1) Parametric VaR ─────────────────────────────────────────────

def fit_garch(returns):
    """Fit GARCH(1,1) by maximum likelihood (normal innovations)."""
    r = returns.values
    n = len(r)

    def neg_loglik(params):
        om, al, be = params
        if om <= 0 or al <= 0 or be <= 0 or al + be >= 1:
            return 1e10
        h = np.zeros(n); h[0] = np.var(r)
        ll = 0.0
        for t in range(1, n):
            h[t] = om + al * r[t-1]**2 + be * h[t-1]
            if h[t] <= 0: return 1e10
            ll += -0.5 * (np.log(2*np.pi) + np.log(h[t]) + r[t]**2 / h[t])
        return -ll

    res = minimize(neg_loglik, [1e-6, 0.08, 0.88],
                   method="L-BFGS-B",
                   bounds=[(1e-8,0.01),(0.001,0.3),(0.5,0.999)])
    om, al, be = res.x
    h = np.zeros(n); h[0] = np.var(r)
    for t in range(1, n):
        h[t] = om + al * r[t-1]**2 + be * h[t-1]
    return h, om, al, be

def var_garch(returns, confidence=0.99):
    h, om, al, be = fit_garch(returns)
    z = stats.norm.ppf(1 - confidence)
    var = pd.Series(-z * np.sqrt(h), index=returns.index, name="VaR_GARCH")
    es  = pd.Series(
        np.sqrt(h) * stats.norm.pdf(z) / (1 - confidence),
        index=returns.index, name="ES_GARCH")
    print(f"GARCH(1,1) params — ω:{om:.2e}  α:{al:.4f}  β:{be:.4f}  "
          f"persistence:{al+be:.4f}")
    return var, es

# ── 4. Monte Carlo VaR ───────────────────────────────────────────────────────

def var_montecarlo(returns, confidence=0.99, n_sim=10000, horizon=1, seed=0):
    np.random.seed(seed)
    mu, sigma = returns.mean(), returns.std()
    sims = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), n_sim)
    var_mc = -np.percentile(sims, (1 - confidence) * 100)
    es_mc  = -sims[sims <= -var_mc].mean()
    return var_mc, es_mc, sims

# ── 5. Kupiec Backtest ────────────────────────────────────────────────────────

def kupiec_test(returns, var_series, confidence=0.99):
    """Proportion of Failures (POF) test."""
    exc = (returns < -var_series.abs()).dropna()
    T, n_exc = len(exc), exc.sum()
    if T == 0: return None, None
    p = 1 - confidence
    p_hat = n_exc / T
    if p_hat == 0: p_hat = 1e-9
    lr = -2 * (n_exc*np.log(p/p_hat) + (T-n_exc)*np.log((1-p)/(1-p_hat)))
    pval = 1 - stats.chi2.cdf(lr, df=1)
    result = "PASS ✓" if pval > 0.05 else "FAIL ✗"
    return pval, result, n_exc, T

# ── 6. Plotting ───────────────────────────────────────────────────────────────

def plot_all(returns, var_hs, es_hs, var_garch_s, es_garch_s, sims, var_mc):
    fig, axes = plt.subplots(3, 1, figsize=(13, 13))
    fig.suptitle("Value-at-Risk & Expected Shortfall\n"
                 "Historical Simulation · GARCH(1,1) · Monte Carlo",
                 fontsize=13, fontweight="bold")

    # Panel 1: Returns + VaR bands
    ax = axes[0]
    ax.fill_between(returns.index, returns, 0,
                    where=returns < 0, color="#D62728", alpha=0.4, label="Losses")
    ax.fill_between(returns.index, returns, 0,
                    where=returns > 0, color="#2CA02C", alpha=0.3, label="Gains")
    ax.plot(var_hs.index,      -var_hs.abs(),      color="#1A4F8A", lw=1.2,
            ls="--", label="99% VaR (HS)")
    ax.plot(var_garch_s.index, -var_garch_s.abs(), color="#FF7F0E", lw=1.2,
            ls="-",  label="99% VaR (GARCH)")
    exc_hs = returns[returns < -var_hs.abs()]
    ax.scatter(exc_hs.index, exc_hs, color="red", s=15, zorder=5,
               label=f"Breaches ({len(exc_hs)})")
    ax.set_title("Portfolio Returns with 99% VaR Bands")
    ax.set_ylabel("Daily Return"); ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Panel 2: VaR comparison
    ax = axes[1]
    valid = var_hs.dropna().index
    ax.plot(valid, var_hs.abs()[valid],      color="#1A4F8A", lw=1.2, ls="--", label="VaR (HS)")
    ax.plot(valid, var_garch_s.abs()[valid], color="#FF7F0E", lw=1.2,          label="VaR (GARCH)")
    ax.plot(valid, es_hs.abs()[valid],       color="#1A4F8A", lw=1.0, ls=":",  label="ES (HS)")
    ax.plot(valid, es_garch_s.abs()[valid],  color="#FF7F0E", lw=1.0, ls=":",  label="ES (GARCH)")
    ax.set_title("VaR & ES Comparison (99% confidence)")
    ax.set_ylabel("Risk Estimate"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Panel 3: Monte Carlo distribution
    ax = axes[2]
    ax.hist(sims, bins=80, color="#9467BD", alpha=0.6, edgecolor="none",
            density=True, label="MC simulated returns")
    ax.axvline(-var_mc, color="#D62728", lw=2, ls="--",
               label=f"99% VaR = {var_mc*100:.2f}%")
    x = np.linspace(sims.min(), sims.max(), 300)
    ax.plot(x, stats.norm.pdf(x, sims.mean(), sims.std()),
            color="black", lw=1.2, label="Normal fit")
    ax.set_title("Monte Carlo Return Distribution (10,000 simulations, 1-day horizon)")
    ax.set_xlabel("Return"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("var_es_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved: var_es_analysis.png")
    plt.close()

# ── 7. Main ───────────────────────────────────────────────────────────────────

def main():
    print("Generating GARCH(1,1) portfolio returns...")
    returns = generate_portfolio_returns(n=1000)

    print("Computing Historical Simulation VaR/ES...")
    var_hs, es_hs = var_historical(returns)

    print("Fitting GARCH and computing parametric VaR/ES...")
    var_g, es_g = var_garch(returns)

    print("Monte Carlo VaR (10,000 sims, 1-day)...")
    var_mc, es_mc, sims = var_montecarlo(returns)
    print(f"MC 99% VaR: {var_mc*100:.4f}%  |  ES: {es_mc*100:.4f}%")

    print("\n── Kupiec Backtest (99% VaR) ──────────────────────────")
    for label, var_s in [("Historical Simulation", var_hs),
                          ("GARCH(1,1)",            var_g)]:
        pval, result, nexc, T = kupiec_test(returns, var_s)
        print(f"  {label:<25} p={pval:.4f}  {result}  "
              f"(breaches: {nexc}/{T}, expected: {int(T*0.01)})")

    print("\nGenerating plots...")
    plot_all(returns, var_hs, es_hs, var_g, es_g, sims, var_mc)

if __name__ == "__main__":
    main()
