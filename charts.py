"""
charts.py — Matplotlib Visualization

Produces the two-panel "Fund Audit Report" figure:
  - Left panel:  Growth of 1 EUR — Fund (net) vs Synthetic Replica,
                 with fill_between showing the Wealth Gap
  - Right panel: Horizontal bar chart of NNLS factor weight exposures

Public API
----------
plot_audit_report(fund_growth, replica_growth, weights, tickers,
                  ticker_labels, metrics,
                  figsize=(14, 8), save_path=None) -> matplotlib.figure.Figure
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

# Bond keys imported from engine — the single source of truth.
# Charts use this to colour-code equity (blue) vs bond (teal) bars.
from engine import BOND_KEYS as _BOND_KEYS

# Equity factor colors (blue family) — enough shades for up to 12 equity factors
_EQUITY_COLORS = [
    "#0D47A1", "#1565C0", "#1976D2", "#1E88E5",
    "#2196F3", "#42A5F5", "#64B5F6", "#90CAF9",
    "#BBDEFB", "#1A237E", "#283593", "#303F9F",
]
# Bond factor colors (teal/green family) — enough for up to 4 bond factors
_BOND_COLORS = ["#00695C", "#00897B", "#26A69A", "#4DB6AC"]


def _assign_bar_colors(tickers: list[str]) -> list[str]:
    """Assign colors by asset class: blue shades for equity, teal for bonds.
    Anything not in BOND_KEYS (from engine.py) is treated as equity."""
    colors = []
    eq_idx, bd_idx = 0, 0
    for t in tickers:
        if t in _BOND_KEYS:
            colors.append(_BOND_COLORS[bd_idx % len(_BOND_COLORS)])
            bd_idx += 1
        else:
            colors.append(_EQUITY_COLORS[eq_idx % len(_EQUITY_COLORS)])
            eq_idx += 1
    return colors


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_audit_report(
    fund_growth: pd.Series,
    replica_growth: pd.Series,
    weights: np.ndarray,
    tickers: list[str],
    ticker_labels: dict[str, str],
    metrics: dict,
    figsize: tuple = (14, 8),
    save_path: str = None,
) -> matplotlib.figure.Figure:
    """
    Generate the two-panel audit report figure.

    Parameters
    ----------
    fund_growth    : Cumulative growth Series for the fund (net returns).
    replica_growth : Cumulative growth Series for the synthetic replica.
    weights        : np.ndarray of NNLS factor weights (same order as tickers).
    tickers        : List of ticker symbols.
    ticker_labels  : Dict mapping ticker → human-readable label.
    metrics        : Dict from compute_metrics().
    figsize        : Figure dimensions in inches (width, height).
    save_path      : If provided, save figure as PNG to this path (150 dpi).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = GridSpec(
        1, 2,
        width_ratios=[2.2, 1],
        wspace=0.30,
        left=0.07, right=0.97,
        top=0.88, bottom=0.10,
    )
    ax_growth = fig.add_subplot(gs[0])
    ax_bar    = fig.add_subplot(gs[1])

    # -----------------------------------------------------------------------
    # Panel 1: Growth of 1 EUR
    # -----------------------------------------------------------------------
    dates = fund_growth.index

    # Lines
    ax_growth.plot(
        dates, replica_growth.values,
        color="#1565C0", linewidth=1.8, label="Synthetic Replica (passive ETFs)",
        zorder=3,
    )
    ax_growth.plot(
        dates, fund_growth.values,
        color="#E65100", linewidth=1.8, label=f"Fund (net, TER={metrics['stated_ter_pct']:.2f}%)",
        zorder=3,
    )

    # Wealth gap fill
    replica_arr = replica_growth.values
    fund_arr    = fund_growth.values

    # Red fill: replica > fund (fee drag — investor loses to fees)
    ax_growth.fill_between(
        dates,
        fund_arr, replica_arr,
        where=(replica_arr > fund_arr),
        alpha=0.15,
        color="#C62828",
        label="Wealth Gap (cost of fees)",
        zorder=2,
    )
    # Green fill: fund > replica (fund outperforms — relatively rare)
    ax_growth.fill_between(
        dates,
        fund_arr, replica_arr,
        where=(replica_arr <= fund_arr),
        alpha=0.15,
        color="#2E7D32",
        label="Fund Outperforms Replica",
        zorder=2,
    )

    # Annotation: wealth gap at final date
    gap_eur = metrics["wealth_gap_eur"]
    gap_pct = metrics["wealth_gap_pct"]
    fund_end    = float(fund_growth.iloc[-1])
    replica_end = float(replica_growth.iloc[-1])
    midpoint_y  = (fund_end + replica_end) / 2

    annotation_text = (
        f"Wealth Gap\n"
        f"{gap_eur:+.4f} EUR\n"
        f"({gap_pct:+.1f}%)"
    )
    ax_growth.annotate(
        annotation_text,
        xy=(dates[-1], midpoint_y),
        xytext=(-90, 0),
        textcoords="offset points",
        fontsize=8.5,
        color="#C62828" if gap_eur > 0 else "#2E7D32",
        va="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9),
        zorder=5,
    )

    # Info text box (top-left)
    r2 = metrics["r_squared"]
    r2_str = f"{r2:.4f}" if not (isinstance(r2, float) and r2 != r2) else "N/A"
    ci_str = "CLOSET INDEXER" if metrics["is_closet_indexer"] else "Active"
    drag_str = f"{metrics['annual_fee_drag_pct']:+.2f}%/yr ({metrics['annual_fee_drag_bps']:+.0f} bps)"

    info_text = (
        f"R² = {r2_str}  [{ci_str}]\n"
        f"Annual drag: {drag_str}\n"
        f"Period: {metrics['n_trading_days']} days ({metrics['years']:.1f} yr)"
    )
    ax_growth.text(
        0.02, 0.97, info_text,
        transform=ax_growth.transAxes,
        fontsize=8.5,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="lightgray", alpha=0.95),
        zorder=5,
    )

    # X-axis date formatting based on period length
    n_years = metrics["years"]
    if n_years > 5:
        ax_growth.xaxis.set_major_locator(mdates.YearLocator())
        ax_growth.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    elif n_years > 1:
        ax_growth.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax_growth.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        ax_growth.xaxis.set_major_locator(mdates.MonthLocator())
        ax_growth.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.setp(ax_growth.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax_growth.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_growth.set_ylabel("Portfolio Value (EUR, starting at 1.00)", fontsize=10)
    ax_growth.set_xlabel("")
    ax_growth.set_title("Growth of 1 EUR: Fund vs Synthetic Replica", fontsize=12, fontweight="bold", pad=10)
    ax_growth.grid(True, which="major", linestyle="--", alpha=0.4, zorder=1)
    ax_growth.spines[["top", "right"]].set_visible(False)

    # Only show "Fund Outperforms" in legend if it actually occurs
    handles, labels = ax_growth.get_legend_handles_labels()
    if not any(replica_arr <= fund_arr):
        handles = [h for h, l in zip(handles, labels) if "Outperforms" not in l]
        labels  = [l for l in labels if "Outperforms" not in l]
    ax_growth.legend(handles, labels, fontsize=8.5, loc="upper left",
                     bbox_to_anchor=(0.02, 0.80), framealpha=0.9)

    # -----------------------------------------------------------------------
    # Panel 2: Factor Weight Bar Chart
    # -----------------------------------------------------------------------
    # Sort by weight descending
    order = np.argsort(weights)[::-1]
    sorted_weights  = weights[order]
    sorted_tickers  = [tickers[i] for i in order]
    sorted_labels   = [ticker_labels.get(t, t) for t in sorted_tickers]
    sorted_colors   = _assign_bar_colors(sorted_tickers)

    y_pos = np.arange(len(sorted_tickers))

    bars = ax_bar.barh(
        y_pos,
        sorted_weights * 100,
        color=sorted_colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.65,
        zorder=3,
    )

    # Cross-hatch zero-weight bars
    for bar, w in zip(bars, sorted_weights):
        if w < 1e-4:
            bar.set_hatch("///")
            bar.set_alpha(0.4)

    # Value labels on bars
    for bar, w in zip(bars, sorted_weights):
        x_val = w * 100
        label = f"{x_val:.1f}%"
        x_offset = max(x_val + 0.5, 0.5)
        ax_bar.text(
            x_offset,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha="left",
            fontsize=8, color="#333333",
        )

    # Ticker labels on Y-axis
    tick_labels = [f"{l}\n({t})" for l, t in zip(sorted_labels, sorted_tickers)]
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(tick_labels, fontsize=7.5)
    ax_bar.invert_yaxis()  # Largest weight at top

    max_w = max(weights) * 100
    ax_bar.set_xlim(0, max_w * 1.30)
    ax_bar.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_bar.set_xlabel("Weight (%)", fontsize=9)

    r2_str_bar = f"{r2:.1%}" if not (isinstance(r2, float) and r2 != r2) else "N/A"
    ax_bar.set_title(f"Factor Exposures\n(R² = {r2_str_bar})", fontsize=11, fontweight="bold", pad=10)
    ax_bar.grid(True, axis="x", linestyle="--", alpha=0.4, zorder=1)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # -----------------------------------------------------------------------
    # Figure-level title
    # -----------------------------------------------------------------------
    fig.suptitle(
        "Historical Fund Auditor — Style Analysis & Wealth Gap Report",
        fontsize=13, fontweight="bold", y=0.96,
    )

    # -----------------------------------------------------------------------
    # Save if requested
    # -----------------------------------------------------------------------
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"[charts] Figure saved to '{save_path}'")

    return fig
