"""
metrics.py — Performance Metrics and Fee-Drag Calculations

Computes the wealth reconstruction, wealth gap, annual fee drag,
and all summary statistics for the audit report.

Public API
----------
compute_growth(returns, initial_value=1.0)          -> pd.Series
compute_wealth_gap(fund_growth, replica_growth)     -> pd.Series
compute_metrics(fund_net_returns, replica_returns,
                weights, r_squared, ter_annual,
                tickers)                            -> dict
"""

import warnings
import numpy as np
import pandas as pd


def compute_growth(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """
    Compute the cumulative growth of an initial investment from a return series.

    growth_t = initial_value * prod_{s=1}^{t} (1 + r_s)

    The first value in the output is initial_value * (1 + r_1), reflecting
    invested capital after the first period.

    Parameters
    ----------
    returns       : pd.Series of daily returns (decimal, e.g. 0.01 = 1%).
    initial_value : Starting portfolio value in EUR (default 1.0).

    Returns
    -------
    pd.Series of cumulative portfolio values, same index as returns.
    """
    return initial_value * (1 + returns).cumprod()


def compute_wealth_gap(
    fund_growth: pd.Series,
    replica_growth: pd.Series,
) -> pd.Series:
    """
    Compute the daily wealth gap: replica_t - fund_t.

    A positive gap means the passive replica has accumulated more value
    than the fund — the difference is money "eaten away" by fees.
    A negative gap means the fund has outperformed the replica.

    Parameters
    ----------
    fund_growth    : Cumulative growth Series for the fund (net returns).
    replica_growth : Cumulative growth Series for the synthetic replica.

    Returns
    -------
    pd.Series of wealth gap values, same index as inputs.
    """
    return replica_growth - fund_growth


def compute_metrics(
    fund_net_returns: pd.Series,
    replica_returns: pd.Series,
    weights: np.ndarray,
    r_squared: float,
    ter_annual: float,
    tickers: list[str],
    benchmark_returns: pd.Series | None = None,
    benchmark_label: str = "SWDA",
    risk_free_annual: float = 0.0,
) -> dict:
    """
    Compute all summary statistics for the audit report.

    Parameters
    ----------
    fund_net_returns  : pd.Series of fund daily net returns on aligned dates.
    replica_returns   : pd.Series of synthetic replica daily returns.
    weights           : np.ndarray of NNLS factor weights.
    r_squared         : float, R² from NNLS (on net returns).
    ter_annual        : float, stated annual TER as a decimal (e.g. 0.0145).
    tickers           : list of ticker symbols in the same order as weights.
    benchmark_returns : Optional pd.Series of a single-ETF benchmark (e.g. SWDA)
                        on the same aligned dates. When supplied, risk-translation
                        metrics (beta, volatility, M² alpha) are computed.
    benchmark_label   : Human-readable name for the benchmark (default "SWDA").
    risk_free_annual  : Annual risk-free rate as decimal (default 0.0).
                        Use C3M annualised return or ECB deposit rate if available.

    Returns
    -------
    dict with keys:
        fund_final_value     : float — portfolio value at end (starting from 1 EUR)
        replica_final_value  : float — replica value at end (starting from 1 EUR)
        wealth_gap_eur       : float — replica_final - fund_final (per 1 EUR)
        wealth_gap_pct       : float — wealth gap as % of initial investment
        annual_fee_drag_pct  : float — annualized geometric mean fee drag (%)
        annual_fee_drag_bps  : float — annual drag in basis points
        cumulative_fee_drag_pct : float — total drag over entire period (%)
        r_squared            : float — R² (explanatory power)
        is_closet_indexer    : bool  — True if R² > 0.90
        weights_dict         : dict  — {ticker: weight}
        stated_ter_pct       : float — ter_annual * 100
        n_trading_days       : int
        years                : float — approximate number of years
        --- risk translation (only when benchmark_returns is supplied) ---
        benchmark_label      : str
        fund_vol_annual_pct  : float — fund annualised volatility (%)
        bench_vol_annual_pct : float — benchmark annualised volatility (%)
        beta_vs_bench        : float — fund beta relative to benchmark
        swda_equiv_pct       : float — risk-equivalent benchmark allocation (%)
        fund_sharpe          : float — fund annualised Sharpe ratio
        bench_sharpe         : float — benchmark annualised Sharpe ratio
        m2_alpha_pct         : float — M² alpha in % per year
    """
    n = len(fund_net_returns)

    # Auto-detect observation frequency for correct annualisation.
    gaps = fund_net_returns.index.to_series().diff().dt.days.dropna()
    median_gap = float(gaps.median())
    if median_gap <= 3:
        periods_per_year = 252
        freq_label = "daily"
    elif median_gap <= 10:
        periods_per_year = 52
        freq_label = "weekly"
    else:
        periods_per_year = 12
        freq_label = "monthly"

    years = n / float(periods_per_year)

    # --- Wealth reconstruction ---
    fund_growth    = compute_growth(fund_net_returns)
    replica_growth = compute_growth(replica_returns)

    fund_final    = float(fund_growth.iloc[-1])
    replica_final = float(replica_growth.iloc[-1])

    wealth_gap_eur = replica_final - fund_final
    wealth_gap_pct = wealth_gap_eur / 1.0 * 100.0   # per 1 EUR initial

    # --- Annual fee drag (geometric, empirical) ---
    # Ratio of (1 + replica_t) / (1 + fund_net_t) for each day
    # Then annualize via (product)^(252/T) - 1
    ratio = (1.0 + replica_returns.values) / (1.0 + fund_net_returns.values)

    # Guard against division edge cases (fund return = -1)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)

    product = np.prod(ratio)
    if product > 0:
        annual_fee_drag = float(product ** (periods_per_year / n) - 1.0)
    else:
        warnings.warn("[metrics] Geometric mean of return ratios is non-positive — fee drag undefined.")
        annual_fee_drag = float("nan")

    annual_fee_drag_pct = annual_fee_drag * 100.0
    annual_fee_drag_bps = annual_fee_drag * 10_000.0

    # Cumulative drag over entire period
    if replica_final > 0 and fund_final > 0:
        cumulative_fee_drag_pct = (replica_final / fund_final - 1.0) * 100.0
    else:
        cumulative_fee_drag_pct = float("nan")

    # --- Closet indexer flag ---
    is_closet_indexer = (not np.isnan(r_squared)) and (r_squared > 0.90)

    # --- Short period warning (frequency-aware) ---
    if years < 1.0:
        warnings.warn(
            f"[metrics] Only {n} {freq_label} observations (~{years:.1f} years) in the analysis. "
            "Annual fee drag extrapolation is unreliable for periods shorter than 1 year."
        )

    metrics = {
        "fund_final_value":        fund_final,
        "replica_final_value":     replica_final,
        "wealth_gap_eur":          wealth_gap_eur,
        "wealth_gap_pct":          wealth_gap_pct,
        "annual_fee_drag_pct":     annual_fee_drag_pct,
        "annual_fee_drag_bps":     annual_fee_drag_bps,
        "cumulative_fee_drag_pct": cumulative_fee_drag_pct,
        "r_squared":               r_squared,
        "is_closet_indexer":       is_closet_indexer,
        "weights_dict":            dict(zip(tickers, weights.tolist())),
        "stated_ter_pct":          ter_annual * 100.0,
        "n_trading_days":          n,
        "years":                   years,
        "freq_label":              freq_label,
        "periods_per_year":        periods_per_year,
        "benchmark_label":         benchmark_label,
    }

    # --- Risk translation vs benchmark (optional) ---
    if benchmark_returns is not None:
        f = fund_net_returns.values.astype(float)
        b = benchmark_returns.reindex(fund_net_returns.index).values.astype(float)

        # Drop any NaN rows (benchmark may have missing dates)
        mask   = np.isfinite(f) & np.isfinite(b)
        f, b   = f[mask], b[mask]
        n_risk = len(f)

        if n_risk < 20:
            warnings.warn("[metrics] Too few overlapping observations for risk metrics.")
        else:
            rf_per_period = risk_free_annual / periods_per_year

            fund_vol_ann  = float(np.std(f, ddof=1) * np.sqrt(periods_per_year))
            bench_vol_ann = float(np.std(b, ddof=1) * np.sqrt(periods_per_year))

            # Beta = cov(fund, bench) / var(bench)
            cov_matrix = np.cov(f, b, ddof=1)
            beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])

            # Risk-equivalent benchmark allocation
            swda_equiv_pct = (fund_vol_ann / bench_vol_ann) * 100.0 if bench_vol_ann > 0 else float("nan")

            # Sharpe ratios
            f_excess = f - rf_per_period
            b_excess = b - rf_per_period
            fund_sharpe  = float((f_excess.mean() / np.std(f_excess, ddof=1)) * np.sqrt(periods_per_year))
            bench_sharpe = float((b_excess.mean() / np.std(b_excess, ddof=1)) * np.sqrt(periods_per_year))

            # M² alpha = (SR_fund − SR_bench) × σ_bench  [annualised, in decimal]
            m2_alpha = (fund_sharpe - bench_sharpe) * bench_vol_ann

            metrics.update({
                "fund_vol_annual_pct":  fund_vol_ann  * 100.0,
                "bench_vol_annual_pct": bench_vol_ann * 100.0,
                "beta_vs_bench":        beta,
                "swda_equiv_pct":       swda_equiv_pct,
                "fund_sharpe":          fund_sharpe,
                "bench_sharpe":         bench_sharpe,
                "m2_alpha_pct":         m2_alpha * 100.0,
            })

    return metrics


def print_metrics(metrics: dict) -> None:
    """
    Pretty-print the metrics dictionary to stdout.
    Convenience function for notebook Cell 5.
    """
    sep = "─" * 52
    print(sep)
    print(f"{'FUND AUDIT REPORT':^52}")
    print(sep)

    r2 = metrics["r_squared"]
    r2_str = f"{r2:.4f}" if not (isinstance(r2, float) and r2 != r2) else "N/A"
    ci_str = "YES  ⚠  (R² > 0.90)" if metrics["is_closet_indexer"] else "NO"

    rows = [
        ("R² (explanatory power)",          r2_str),
        ("Closet Indexer?",                 ci_str),
        ("Period",                          f"{metrics['n_trading_days']} {metrics['freq_label']} obs. ({metrics['years']:.1f} years)"),
        ("",                                ""),
        ("Fund final value (1 EUR in)",     f"{metrics['fund_final_value']:.4f} EUR"),
        ("Replica final value (1 EUR in)",  f"{metrics['replica_final_value']:.4f} EUR"),
        ("",                                ""),
        ("Wealth Gap (EUR, per 1 EUR)",     f"{metrics['wealth_gap_eur']:+.4f} EUR"),
        ("Wealth Gap (%)",                  f"{metrics['wealth_gap_pct']:+.2f}%"),
        ("",                                ""),
        ("Annual Fee Drag (empirical)",     f"{metrics['annual_fee_drag_pct']:+.2f}% ({metrics['annual_fee_drag_bps']:+.1f} bps)"),
        ("Cumulative Fee Drag",             f"{metrics['cumulative_fee_drag_pct']:+.2f}%"),
        ("Stated TER",                      f"{metrics['stated_ter_pct']:.2f}%"),
        ("",                                ""),
    ]

    for label, value in rows:
        if label == "":
            print("")
        else:
            print(f"  {label:42s}: {value}")

    print(sep)
    print("  Factor Weights:")
    for ticker, w in metrics["weights_dict"].items():
        bar = "█" * int(w * 30)
        print(f"    {ticker:12s}: {w*100:5.1f}%  {bar}")
    print(sep)

    # --- Risk translation section (only if benchmark metrics were computed) ---
    if "fund_vol_annual_pct" not in metrics:
        return

    bl = metrics["benchmark_label"]
    print()
    print(sep)
    print(f"  RISK PROFILE vs {bl} ALTERNATIVE".center(52))
    print(sep)

    fund_vol   = metrics["fund_vol_annual_pct"]
    bench_vol  = metrics["bench_vol_annual_pct"]
    beta       = metrics["beta_vs_bench"]
    equiv      = metrics["swda_equiv_pct"]
    f_sharpe   = metrics["fund_sharpe"]
    b_sharpe   = metrics["bench_sharpe"]
    m2         = metrics["m2_alpha_pct"]

    # Leverage / underweight label
    if equiv > 105:
        equiv_note = f"(≈ {equiv:.0f}% {bl} — needs leverage to replicate)"
    elif equiv < 95:
        equiv_note = f"(≈ {equiv:.0f}% {bl} + {100-equiv:.0f}% cash)"
    else:
        equiv_note = f"(≈ {equiv:.0f}% {bl} — similar risk)"

    m2_verdict = "outperformed" if m2 >= 0 else "underperformed"
    m2_color   = "+" if m2 >= 0 else ""

    risk_rows = [
        ("Annual volatility — Fund",       f"{fund_vol:.1f}%"),
        (f"Annual volatility — {bl}",      f"{bench_vol:.1f}%"),
        ("Beta vs benchmark",              f"{beta:.2f}×"),
        ("Risk-equivalent allocation",     equiv_note),
        ("",                               ""),
        ("Sharpe ratio — Fund",            f"{f_sharpe:.3f}"),
        (f"Sharpe ratio — {bl}",           f"{b_sharpe:.3f}"),
        ("M² alpha (risk-adjusted)",       f"{m2_color}{m2:.2f}%/yr"),
        ("Verdict",
         f"Fund {m2_verdict} {bl} by {abs(m2):.2f}%/yr on equal-risk basis"),
    ]

    for label, value in risk_rows:
        if label == "":
            print("")
        else:
            print(f"  {label:42s}: {value}")
    print(sep)
