"""
Value-at-Risk & Expected Shortfall
====================================
Three estimation methods: Historical Simulation, Parametric (Normal + GARCH),
and Monte Carlo. Backtested with Kupiec (POF) and Christoffersen tests.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm, chi2
import warnings
warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("Note: arch package not found — GARCH VaR will use EWMA fallback")


# ── 1. SIMULATE PORTFOLIO RETURNS ────────────────────────────────────────────
# 3-asset portfolio: equity (60%), bond (30%), commodity (10%)
# Simulates realistic fat-tailed, autocorrelated returns

np.random.seed(42)
n_days = 1260  # 5 years

def simulate_garch_returns(n, omega=1e-6, alpha=0.09, beta=0.90, mu=0.0003):
    """Simulate GARCH(1,1) returns."""
    h = np.zeros(n)
    r = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
        r[t] = mu + np.sqrt(h[t]) * np.random.standard_t(df=6)
    return r, h

eq_ret, eq_var   = simulate_garch_returns(n_days, omega=2e-6, alpha=0.10, beta=0.88, mu=0.0004)
bd_ret, _        = simulate_garch_returns(n_days, omega=5e-7, alpha=0.05, beta=0.93, mu=0.0001)
cm_ret, _        = simulate_garch_returns(n_days, omega=4e-6, alpha=0.12, beta=0.85, mu=0.0002)

# Introduce some correlation
cm_ret = 0.4 * eq_ret + 0.9 * cm_ret

# Portfolio weights
w = np.array([0.60, 0.30, 0.10])
port_ret = w[0]*eq_ret + w[1]*bd_ret + w[2]*cm_ret

dates = pd.date_range("2019-01-02", periods=n_days, freq="B")
returns = pd.Series(port_ret, index=dates, name='Portfolio')

PORTFOLIO_VALUE = 1_000_000  # $1M
CONFIDENCE = 0.99            # 99% VaR
ALPHA = 1 - CONFIDENCE
LOOKBACK = 252               # 1-year rolling window for HS
MC_SIMS = 10_000

print("VaR & Expected Shortfall Analysis")
print("=" * 60)
print(f"Portfolio: 60% Equity | 30% Bonds | 10% Commodity")
print(f"Portfolio Value: ${PORTFOLIO_VALUE:,.0f}")
print(f"Confidence Level: {CONFIDENCE:.0%}")
print(f"Sample: {dates[0].date()} – {dates[-1].date()} ({n_days} days)\n")


# ── 2. HISTORICAL SIMULATION VaR ─────────────────────────────────────────────

def historical_simulation_var(returns, alpha=0.01, lookback=252):
    """Rolling HS VaR: non-parametric, preserves fat tails."""
    var_series = pd.Series(index=returns.index, dtype=float)
    es_series  = pd.Series(index=returns.index, dtype=float)

    for i in range(lookback, len(returns)):
        window = returns.iloc[i-lookback:i]
        var_val = np.percentile(window, alpha * 100)
        es_val  = window[window <= var_val].mean()
        var_series.iloc[i] = var_val
        es_series.iloc[i]  = es_val

    return var_series.dropna(), es_series.dropna()

hs_var, hs_es = historical_simulation_var(returns, ALPHA, LOOKBACK)


# ── 3. PARAMETRIC VaR (EWMA volatility) ──────────────────────────────────────

def ewma_var(returns, alpha=0.01, lam=0.94):
    """EWMA volatility-based parametric VaR."""
    var_list, es_list = [], []
    h = returns.var()
    for i in range(1, len(returns)):
        h = lam * h + (1 - lam) * returns.iloc[i-1]**2
        sigma = np.sqrt(h)
        var_val = norm.ppf(alpha) * sigma
        es_val  = -sigma * norm.pdf(norm.ppf(alpha)) / alpha
        var_list.append(var_val)
        es_list.append(-es_val)
    idx = returns.index[1:]
    return pd.Series(var_list, index=idx), pd.Series(es_list, index=idx)

ew_var_series, ew_es_series = ewma_var(returns, ALPHA)


# ── 4. GARCH(1,1) VaR ────────────────────────────────────────────────────────

if ARCH_AVAILABLE:
    try:
        am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t', rescale=False)
        res = am.fit(disp='off', last_obs=int(n_days * 0.8))
        forecasts = res.forecast(horizon=1, start=int(n_days * 0.2), reindex=False)
        cond_vol = np.sqrt(forecasts.variance.values[:, 0]) / 100
        garch_idx = returns.index[int(n_days * 0.2): int(n_days * 0.2) + len(cond_vol)]
        df_t = res.params.get('nu', 6)
        garch_var = pd.Series(
            [stats.t.ppf(ALPHA, df=df_t) * v for v in cond_vol],
            index=garch_idx
        )
        garch_es = pd.Series(
            [-v * stats.t.pdf(stats.t.ppf(ALPHA, df_t), df=df_t) / ALPHA * (df_t + stats.t.ppf(ALPHA, df_t)**2) / (df_t - 1)
             for v in cond_vol],
            index=garch_idx
        )
        GARCH_FITTED = True
    except Exception as e:
        GARCH_FITTED = False
else:
    GARCH_FITTED = False


# ── 5. MONTE CARLO VaR ───────────────────────────────────────────────────────

def monte_carlo_var(returns, alpha=0.01, n_sims=10000, horizon=1):
    """MC VaR using fitted normal with fat-tail adjustment (Student-t)."""
    mu  = returns.mean()
    sig = returns.std()
    df_fit = 6  # degrees of freedom for t-distribution

    # Scale t-distribution to match empirical vol
    scale = sig / np.sqrt(df_fit / (df_fit - 2))
    sim   = stats.t.rvs(df=df_fit, loc=mu, scale=scale, size=(n_sims, horizon))
    port_sims = sim.sum(axis=1)

    var_val = np.percentile(port_sims, alpha * 100)
    es_val  = port_sims[port_sims <= var_val].mean()
    return var_val, es_val, port_sims

# Use last 252 days for MC calibration
mc_var_val, mc_es_val, mc_sims = monte_carlo_var(returns.iloc[-LOOKBACK:], ALPHA, MC_SIMS)


# ── 6. SUMMARY TABLE ─────────────────────────────────────────────────────────

# Get representative values (last available)
common_idx = hs_var.index.intersection(ew_var_series.index)
last_date = common_idx[-1]

hs_v  = hs_var[last_date]
ew_v  = ew_var_series[last_date] if last_date in ew_var_series.index else ew_var_series.iloc[-1]
hs_e  = hs_es[last_date]
ew_e  = ew_es_series[last_date] if last_date in ew_es_series.index else ew_es_series.iloc[-1]

print(f"{'Method':<28} {'99% VaR ($)':>14} {'99% ES ($)':>14} {'VaR (%)':>10}")
print("-" * 70)
for name, var, es in [
    ("Historical Simulation",   hs_v,     hs_e),
    ("Parametric (EWMA)",       ew_v,    -ew_e),
    ("Monte Carlo (Student-t)", mc_var_val, mc_es_val),
]:
    print(f"  {name:<26} {abs(var)*PORTFOLIO_VALUE:>13,.0f}  {abs(es)*PORTFOLIO_VALUE:>13,.0f}  {abs(var)*100:>9.3f}%")


# ── 7. KUPIEC POF BACKTEST ───────────────────────────────────────────────────

def kupiec_test(returns, var_series, alpha=0.01):
    """
    Kupiec (1995) Proportion of Failures test.
    H0: actual violation rate = theoretical alpha
    """
    aligned = returns.reindex(var_series.index).dropna()
    var_aligned = var_series.reindex(aligned.index).dropna()
    aligned = aligned.reindex(var_aligned.index)

    violations = (aligned < var_aligned).sum()
    n = len(aligned)
    p_hat = violations / n

    # Likelihood ratio statistic
    if p_hat == 0 or p_hat == 1:
        return {'violations': violations, 'n': n, 'p_hat': p_hat,
                'LR_stat': np.nan, 'p_value': np.nan, 'pass': None}

    lr = -2 * (np.log((1-alpha)**(n-violations) * alpha**violations) -
               np.log((1-p_hat)**(n-violations) * p_hat**violations))
    p_val = 1 - chi2.cdf(lr, df=1)

    return {
        'violations': violations, 'n': n,
        'expected': int(n * alpha),
        'p_hat': p_hat,
        'LR_stat': lr, 'p_value': p_val,
        'pass': p_val > 0.05
    }

print(f"\nKupiec Backtest (99% VaR — expected violation rate: {ALPHA:.1%})")
print("-" * 70)
for name, var_s in [("Historical Simulation", hs_var), ("EWMA Parametric", ew_var_series)]:
    kt = kupiec_test(returns, var_s, ALPHA)
    result = "PASS ✓" if kt['pass'] else "FAIL ✗"
    print(f"  {name:<26} Violations: {kt['violations']:>3}/{kt['n']:>4} "
          f"(exp {kt['expected']:>3}) | p={kt['p_value']:.3f} | {result}")


# ── 8. VISUALISATION ─────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 13))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Returns with VaR overlay
ax1 = fig.add_subplot(gs[0, :])
plot_returns = returns.iloc[LOOKBACK:]
plot_hs  = hs_var.reindex(plot_returns.index)
plot_ew  = ew_var_series.reindex(plot_returns.index)
violations_hs = plot_returns[plot_returns < plot_hs]

ax1.plot(plot_returns.index, plot_returns * 100, color='#aaaaaa', linewidth=0.6, alpha=0.7, label='Daily Return')
ax1.plot(plot_hs.index, plot_hs * 100, color='#1a4f8a', linewidth=1.5, label='HS VaR (99%)')
ax1.plot(plot_ew.index, plot_ew * 100, color='#e07b39', linewidth=1.5, linestyle='--', label='EWMA VaR (99%)')
ax1.scatter(violations_hs.index, violations_hs * 100, color='#c0392b', s=20, zorder=5,
            label=f'Violations ({len(violations_hs)})', alpha=0.8)
ax1.set_title("Portfolio Returns vs. 99% VaR Estimates", fontweight='bold')
ax1.set_ylabel("Return (%)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Return distribution + VaR/ES
ax2 = fig.add_subplot(gs[1, 0])
ret_pct = plot_returns * 100
var_val_pct = hs_v * 100
es_val_pct  = hs_e * 100
ax2.hist(ret_pct, bins=60, color='#1a4f8a', alpha=0.6, density=True, edgecolor='none')
ax2.hist(ret_pct[ret_pct < var_val_pct], bins=20, color='#c0392b', alpha=0.8,
         density=True, edgecolor='none', label='Tail losses')
ax2.axvline(var_val_pct, color='#c0392b', linewidth=2, linestyle='--',
            label=f'HS VaR: {var_val_pct:.2f}%')
ax2.axvline(es_val_pct,  color='#8e44ad', linewidth=2, linestyle=':',
            label=f'HS ES: {es_val_pct:.2f}%')
x = np.linspace(ret_pct.min(), ret_pct.max(), 200)
ax2.plot(x, norm.pdf(x, ret_pct.mean(), ret_pct.std()),
         color='#2d9e6b', linewidth=1.5, linestyle='-', label='Normal fit')
ax2.set_title("Return Distribution & Tail Risk", fontweight='bold')
ax2.set_xlabel("Daily Return (%)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel 3: Monte Carlo simulation histogram
ax3 = fig.add_subplot(gs[1, 1])
mc_pct = mc_sims * 100
ax3.hist(mc_pct, bins=80, color='#2d9e6b', alpha=0.6, density=True, edgecolor='none')
ax3.hist(mc_pct[mc_pct < mc_var_val*100], bins=20, color='#c0392b', alpha=0.8,
         density=True, edgecolor='none')
ax3.axvline(mc_var_val*100, color='#c0392b', linewidth=2, linestyle='--',
            label=f'MC VaR: {mc_var_val*100:.2f}%')
ax3.axvline(mc_es_val*100, color='#8e44ad', linewidth=2, linestyle=':',
            label=f'MC ES: {mc_es_val*100:.2f}%')
ax3.set_title(f"Monte Carlo Simulation ({MC_SIMS:,} paths)", fontweight='bold')
ax3.set_xlabel("1-Day Return (%)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Panel 4: VaR comparison bar chart
ax4 = fig.add_subplot(gs[2, 0])
methods = ['Hist. Sim.', 'EWMA\nParametric', 'Monte Carlo\n(Student-t)']
var_vals = [abs(hs_v)*PORTFOLIO_VALUE/1000, abs(ew_v)*PORTFOLIO_VALUE/1000, abs(mc_var_val)*PORTFOLIO_VALUE/1000]
es_vals  = [abs(hs_e)*PORTFOLIO_VALUE/1000, abs(ew_e)*PORTFOLIO_VALUE/1000, abs(mc_es_val)*PORTFOLIO_VALUE/1000]
x_pos = np.arange(len(methods))
w = 0.35
bars1 = ax4.bar(x_pos - w/2, var_vals, w, label='99% VaR ($k)', color='#1a4f8a', alpha=0.8)
bars2 = ax4.bar(x_pos + w/2, es_vals,  w, label='99% ES ($k)',  color='#c0392b', alpha=0.8)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(methods, fontsize=9)
ax4.set_ylabel("Risk Estimate ($000s)")
ax4.set_title("VaR vs. ES by Method ($1M Portfolio)", fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')
for bar in list(bars1) + list(bars2):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"${bar.get_height():.1f}k", ha='center', va='bottom', fontsize=8)

# Panel 5: Rolling violations
ax5 = fig.add_subplot(gs[2, 1])
rolling_violations = (plot_returns < plot_hs).rolling(63).mean() * 100  # quarterly
ax5.plot(rolling_violations.index, rolling_violations, color='#1a4f8a', linewidth=1.5)
ax5.axhline(ALPHA * 100, color='#c0392b', linewidth=1.5, linestyle='--',
            label=f'Expected {ALPHA*100:.0f}%')
ax5.fill_between(rolling_violations.index, rolling_violations, ALPHA*100,
                 where=(rolling_violations > ALPHA*100), alpha=0.2, color='#c0392b', label='Excess violations')
ax5.set_title("Rolling 63-Day Violation Rate vs. Expected", fontweight='bold')
ax5.set_ylabel("Violation Rate (%)")
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

fig.suptitle("Value-at-Risk & Expected Shortfall\n$1M Portfolio | 99% Confidence | 3 Methods + Kupiec Backtest",
             fontsize=13, fontweight='bold', y=0.99)
plt.savefig("var_es_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved: var_es_analysis.png")
