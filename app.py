"""
Portfolio Analysis Tool — Streamlit App

Run locally:
    cd portfolio_tool
    streamlit run app.py

Deploy:
    Push this folder to a GitHub repo, then connect it to share.streamlit.io.
    Set the main file to app.py. No data/ folder needed — users upload their CSVs.
"""

import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before other matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import seaborn as sns
import streamlit as st

# Ensure local modules (loader.py, engine.py, metrics.py) are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import load_benchmark_csv
from metrics import compute_growth


# =============================================================================
# BLOCK 2 — HELPER FUNCTIONS
# =============================================================================

def extract_ticker(filename: str) -> str:
    """
    Extract short ticker from an ETF CSV filename.
    'SWDA ETF Stock Price History.csv' -> 'SWDA'
    'BRK.B.csv'                        -> 'BRK.B'
    'IS3S(mscifactor).csv'             -> 'IS3S(mscifactor)'
    """
    suffix = " ETF Stock Price History.csv"
    if filename.endswith(suffix):
        return filename[: -len(suffix)].strip()
    return os.path.splitext(filename)[0].strip()


@st.cache_data(show_spinner="Loading price data…")
def load_all_uploads(file_contents: dict) -> dict:
    """
    Load uploaded CSVs into price Series.

    Parameters
    ----------
    file_contents : dict  {filename (str) -> raw bytes}
        Bytes are hashable so st.cache_data can key on them correctly.
        Invalidates automatically when any file's content changes.

    Returns
    -------
    dict  {ticker (str) -> pd.Series of prices}
    """
    result = {}
    errors = []
    for filename, content in file_contents.items():
        ticker = extract_ticker(filename)
        suffix = os.path.splitext(filename)[1] or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                series = load_benchmark_csv(tmp_path, label=ticker)
            result[ticker] = series
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    if errors:
        raise RuntimeError("\n".join(errors))
    return result


def build_prices_df(series_dict: dict) -> pd.DataFrame:
    """Concatenate price Series into a single DataFrame, sorted by date."""
    df = pd.concat(series_dict, axis=1)
    df.columns = list(series_dict.keys())
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()


def compute_max_drawdown(wealth_s: pd.Series) -> float:
    """Maximum peak-to-trough drawdown. Returns a negative float."""
    peak = wealth_s.cummax()
    return float(((wealth_s - peak) / peak).min())


def simulate_gbm(
    mu_daily: float,
    sigma_daily: float,
    n_years: int,
    n_sims: int,
    initial_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Geometric Brownian Motion with Ito drift correction.
    Returns array of shape (n_steps + 1, n_sims). Row 0 = initial_value.
    """
    n_steps  = n_years * 252
    drift    = mu_daily - 0.5 * sigma_daily ** 2
    Z        = rng.standard_normal((n_steps, n_sims))
    log_rets = drift + sigma_daily * Z
    paths    = initial_value * np.exp(np.cumsum(log_rets, axis=0))
    return np.vstack([np.full((1, n_sims), initial_value), paths])


def simulate_block_bootstrap(
    hist_returns: np.ndarray,
    block_length: int,
    n_years: int,
    n_sims: int,
    initial_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorized Block Bootstrap. Preserves autocorrelation and fat tails.
    Returns array of shape (n_steps + 1, n_sims). Row 0 = initial_value.
    """
    n_steps   = n_years * 252
    T         = len(hist_returns)
    eff_block = min(block_length, T // 5)
    max_start = T - eff_block
    if max_start < 1:
        raise ValueError(
            f"History too short ({T} days) for block_length={eff_block}."
        )
    n_blocks     = int(np.ceil(n_steps / eff_block))
    start_idx    = rng.integers(0, max_start, size=(n_sims, n_blocks))
    offsets      = np.arange(eff_block)
    indices      = start_idx[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
    indices      = np.clip(indices, 0, T - 1)
    sampled      = hist_returns[indices.reshape(n_sims, -1)][:, :n_steps]
    wealth_paths = initial_value * np.cumprod(1.0 + sampled, axis=1)
    return np.vstack([np.full((1, n_sims), initial_value), wealth_paths.T])


def compute_percentile_bands(paths: np.ndarray, percentiles=(5, 25, 50, 75, 95)) -> dict:
    """Return {percentile: array of shape (n_steps+1,)} for each requested percentile."""
    return {p: np.percentile(paths, p, axis=1) for p in percentiles}


@st.cache_data(show_spinner="Running Monte Carlo simulations… (this may take a moment)")
def run_monte_carlo_cached(
    port_returns_tuple: tuple,
    initial_value: float,
    n_sims: int,
    block_length: int,
    horizons: tuple = (10, 20, 30),
    seed: int = 42,
) -> dict:
    """
    Run GBM and Block Bootstrap for all horizons.

    port_returns_tuple is tuple(port_returns.values.tolist()) — tuples are
    hashable so st.cache_data can key on them. Cache invalidates when weights,
    portfolio value, n_sims, or block_length change.
    """
    port_returns = np.array(port_returns_tuple)
    rng = np.random.default_rng(seed)

    log_ret      = np.log1p(port_returns)
    mu_daily     = float(log_ret.mean())
    sigma_daily  = float(log_ret.std(ddof=1))
    mu_annual    = float(np.exp(mu_daily * 252) - 1)
    sigma_annual = float(sigma_daily * np.sqrt(252))

    gbm_paths  = {}
    boot_paths = {}

    for horizon in horizons:
        gbm_paths[horizon] = simulate_gbm(
            mu_daily=mu_daily, sigma_daily=sigma_daily,
            n_years=horizon, n_sims=n_sims,
            initial_value=initial_value, rng=rng,
        )
        boot_paths[horizon] = simulate_block_bootstrap(
            hist_returns=port_returns, block_length=block_length,
            n_years=horizon, n_sims=n_sims,
            initial_value=initial_value, rng=rng,
        )

    return {
        "gbm":       gbm_paths,
        "bootstrap": boot_paths,
        "calibration": {
            "mu_daily":    mu_daily,
            "sigma_daily": sigma_daily,
            "mu_annual":   mu_annual,
            "sigma_annual": sigma_annual,
        },
    }


# =============================================================================
# BLOCK 3 — CHART HELPERS (all return Figure; caller does st.pyplot + plt.close)
# =============================================================================

def make_correlation_heatmap(common_returns: pd.DataFrame) -> plt.Figure:
    """Seaborn lower-triangle Pearson correlation heatmap."""
    corr     = common_returns.corr(method="pearson")
    n        = len(corr)
    side     = max(5, n * 1.1)
    mask     = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(side, side * 0.85))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn_r", vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 12},
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    date_start = common_returns.index[0].date()
    date_end   = common_returns.index[-1].date()
    ax.set_title(
        f"ETF Pairwise Correlation (Pearson daily log-returns)\n"
        f"{date_start}  to  {date_end}   |   {len(common_returns)} shared trading days",
        fontsize=12, pad=14, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def make_backtest_chart(
    port_wealth_plot: pd.Series,
    etf_wealth_dict: dict,
    w_dict: dict,
    portfolio_eur: float,
    port_start,
    total_years: float,
) -> plt.Figure:
    """Cumulative wealth: portfolio line + dashed individual ETF lines."""
    tickers    = list(etf_wealth_dict.keys())
    colors_etf = plt.cm.tab10(np.linspace(0, 1, max(len(tickers), 1)))

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, t in enumerate(tickers):
        s = etf_wealth_dict[t]
        ax.plot(
            s.index, s.values, alpha=0.45, lw=1.3, ls="--", color=colors_etf[i],
            label=f"{t} only ({w_dict[t]:.0%} weight)",
        )

    ax.plot(
        port_wealth_plot.index, port_wealth_plot.values,
        color="#1565C0", lw=2.8, label="Portfolio (buy & hold)", zorder=5,
    )
    ax.axhline(y=portfolio_eur, color="gray", ls=":", lw=1.0, alpha=0.7,
               label="Initial investment")

    final_val = port_wealth_plot.iloc[-1]
    ax.annotate(
        f"EUR {final_val:,.0f}",
        xy=(port_wealth_plot.index[-1], final_val),
        xytext=(-90, 12), textcoords="offset points",
        fontsize=10, color="#1565C0", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.2),
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"EUR {v:,.0f}"))
    ax.set_title(
        f"Buy & Hold Portfolio — EUR {portfolio_eur:,.0f} invested on {port_start}\n"
        "Individual ETF lines: hypothetical 100% allocation to each instrument",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Portfolio Value (EUR)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _draw_fan_on_ax(ax, bands, hist_wealth, horizon_years, method_label, color, initial_value):
    """Draw a single Monte Carlo fan chart panel onto an existing Axes."""
    n_steps = len(bands[50]) - 1
    x_sim   = np.linspace(0, horizon_years, n_steps + 1)

    hist_arr   = hist_wealth.values
    hist_years = len(hist_arr) / 252.0
    x_hist     = np.linspace(-hist_years, 0, len(hist_arr))

    ax.plot(x_hist, hist_arr, color="gray", lw=1.6, alpha=0.75,
            label="Historical portfolio", zorder=4)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    ax.fill_between(x_sim, bands[5],  bands[95], alpha=0.12, color=color)
    ax.fill_between(x_sim, bands[25], bands[75], alpha=0.26, color=color)
    ax.plot(x_sim, bands[50], color=color, lw=2.2,
            label="Median (50th pct)", zorder=5)
    ax.scatter([0], [initial_value], color=color, s=35, zorder=6)

    med_t = bands[50][-1]
    p5_t  = bands[5][-1]
    p95_t = bands[95][-1]
    ax.text(
        0.98, 0.04,
        f"Median: EUR {med_t:,.0f}\n5th–95th: EUR {p5_t:,.0f}–{p95_t:,.0f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"EUR {v:,.0f}"))
    ax.set_xlabel("Years from today", fontsize=9)
    ax.set_title(f"{method_label}  —  {horizon_years}-Year Projection",
                 fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def make_fan_chart(
    gbm_paths: dict,
    boot_paths: dict,
    port_wealth_full: pd.Series,
    initial_val: float,
    n_sims: int,
    port_returns_index,
    horizons: tuple = (10, 20, 30),
) -> plt.Figure:
    """3-row × 2-column Monte Carlo fan chart grid (GBM left, Bootstrap right)."""
    COLORS  = {"GBM": "#1565C0", "Bootstrap": "#2E7D32"}
    METHODS = [("GBM", gbm_paths), ("Bootstrap", boot_paths)]
    PCTS    = [5, 25, 50, 75, 95]

    fig, axes = plt.subplots(len(horizons), 2, figsize=(16, 5 * len(horizons)))
    fig.suptitle(
        f"Monte Carlo Portfolio Projections\n"
        f"Starting value: EUR {initial_val:,.0f}   |   "
        f"Historical: {port_returns_index[0].date()} to {port_returns_index[-1].date()} "
        f"({len(port_returns_index)} trading days)\n"
        f"N = {n_sims:,} simulations per method  |  "
        "Bands: 5th / 25th / 50th / 75th / 95th percentile",
        fontsize=11, fontweight="bold", y=0.99,
    )

    legend_added = False
    for row_i, horizon in enumerate(horizons):
        for col_i, (method_name, paths_dict) in enumerate(METHODS):
            ax    = axes[row_i, col_i]
            bands = compute_percentile_bands(paths_dict[horizon], PCTS)
            _draw_fan_on_ax(
                ax=ax, bands=bands, hist_wealth=port_wealth_full,
                horizon_years=horizon, method_label=method_name,
                color=COLORS[method_name], initial_value=initial_val,
            )
            if not legend_added:
                ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
                legend_added = True

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# BLOCK 4 — SIDEBAR
# =============================================================================

def render_sidebar(uploaded_files) -> dict:
    """Render sidebar controls. Returns a config dict for the main area."""

    with st.sidebar:
        st.title("Portfolio Configuration")

        st.subheader("Portfolio Settings")
        portfolio_eur = st.number_input(
            "Initial investment (EUR)",
            min_value=1_000.0, max_value=100_000_000.0,
            value=320_000.0, step=1_000.0, format="%.0f",
        )
        rf_pct = st.number_input(
            "Risk-free rate (% per year)",
            min_value=0.0, max_value=20.0,
            value=2.5, step=0.1, format="%.2f",
        )
        risk_free_annual = rf_pct / 100.0

        st.markdown("---")
        st.subheader("Instrument Weights")
        st.caption("Fractions that sum to 1 (e.g. 0.15). Auto-normalised if they don't.")

        # Determine tickers from uploaded filenames (no full CSV parse needed here)
        uploaded_tickers = [extract_ticker(f.name) for f in uploaded_files] if uploaded_files else []

        # Initialise session_state keys only on first appearance (prevents flicker)
        default_w = round(1.0 / max(len(uploaded_tickers), 1), 4)
        for t in uploaded_tickers:
            if f"weight_{t}" not in st.session_state:
                st.session_state[f"weight_{t}"] = default_w

        raw_weights = {}
        if uploaded_tickers:
            for t in uploaded_tickers:
                raw_weights[t] = st.number_input(
                    label=t,
                    min_value=0.0, max_value=1.0,
                    value=st.session_state[f"weight_{t}"],
                    step=0.01, format="%.2f",
                    key=f"weight_{t}",
                )
            total = sum(raw_weights.values())
            if total > 1e-9:
                w_dict = {t: v / total for t, v in raw_weights.items()}
            else:
                w_dict = {t: 1.0 / len(uploaded_tickers) for t in uploaded_tickers}
                st.warning("All weights are zero — using equal weights.")

            if abs(total - 1.0) > 0.005:
                st.caption(f"Sum = {total:.3f} → auto-normalised to 1.0")

            st.markdown("**Effective weights:**")
            for t, w in w_dict.items():
                st.caption(f"  {t}: {w:.1%}")
        else:
            w_dict = {}
            st.info("Upload CSV files to configure weights.")

        st.markdown("---")
        st.subheader("Monte Carlo Settings")
        n_sims = st.select_slider(
            "Simulations",
            options=[500, 1_000, 2_000, 5_000, 10_000],
            value=5_000,
        )
        block_length = st.slider(
            "Block length (trading days)",
            min_value=5, max_value=63, value=20, step=1,
            help="Consecutive days per block in Block Bootstrap. "
                 "Longer = more autocorrelation preserved.",
        )

    return {
        "portfolio_eur":    portfolio_eur,
        "risk_free_annual": risk_free_annual,
        "w_dict":           w_dict,
        "n_sims":           n_sims,
        "block_length":     block_length,
    }


# =============================================================================
# BLOCK 5 — MAIN
# =============================================================================

def main():
    st.set_page_config(
        page_title="Portfolio Analysis Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Portfolio Analysis Tool")
    st.markdown(
        "Upload your ETF price CSVs below, configure the portfolio in the sidebar, "
        "then explore each analysis tab."
    )

    # --- File uploader ---
    uploaded_files = st.file_uploader(
        "Upload ETF price CSV files",
        type="csv",
        accept_multiple_files=True,
        help=(
            "Supported formats: Investing.com ('SWDA ETF Stock Price History.csv') "
            "or plain ticker ('SWDA.csv'). Drop as many files as you like."
        ),
    )

    cfg = render_sidebar(uploaded_files)
    portfolio_eur    = cfg["portfolio_eur"]
    risk_free_annual = cfg["risk_free_annual"]
    w_dict           = cfg["w_dict"]
    n_sims           = cfg["n_sims"]
    block_length     = cfg["block_length"]

    if not uploaded_files:
        st.info("Upload one or more ETF CSV files to get started.")
        st.stop()

    # --- Load all CSVs (cached by file contents) ---
    file_contents = {f.name: f.getvalue() for f in uploaded_files}
    try:
        series_dict = load_all_uploads(file_contents)
    except Exception as exc:
        st.error(f"Failed to load CSV files:\n\n{exc}")
        st.stop()

    prices_df   = build_prices_df(series_dict)
    all_tickers = list(prices_df.columns)

    # --- Validate and apply weights ---
    if not w_dict:
        w_dict = {t: 1.0 / len(all_tickers) for t in all_tickers}

    missing = [t for t in w_dict if t not in prices_df.columns]
    if missing:
        st.error(
            f"Tickers not found in uploaded files: **{missing}**\n\n"
            f"Available tickers: {all_tickers}"
        )
        st.stop()

    tickers = list(w_dict.keys())
    w_arr   = np.array([w_dict[t] for t in tickers])

    # --- Shared computations ---
    # Log-returns for ALL uploaded ETFs (used by Correlation tab)
    log_returns_all    = np.log(prices_df / prices_df.shift(1)).iloc[1:]
    common_returns_all = log_returns_all.dropna(how="any")

    # Portfolio returns on selected tickers (used by Backtest, Diversification, Monte Carlo)
    prices_selected = prices_df[tickers].ffill().dropna()
    port_start      = prices_selected.index[0].date()
    etf_returns     = prices_selected.pct_change().iloc[1:]
    port_returns    = (etf_returns[tickers] * w_arr).sum(axis=1)
    port_wealth     = compute_growth(port_returns, initial_value=portfolio_eur)
    total_years     = (port_returns.index[-1] - port_returns.index[0]).days / 365.25

    # =========================================================================
    # TABS
    # =========================================================================
    tab_corr, tab_bt, tab_div, tab_mc = st.tabs([
        "📊 Correlation",
        "📈 Backtest",
        "🔀 Diversification",
        "🎲 Monte Carlo",
    ])

    # =========================================================================
    # TAB 1 — CORRELATION
    # =========================================================================
    with tab_corr:
        st.header("Pairwise ETF Correlation")

        n_shared = len(common_returns_all)
        if n_shared < 60:
            st.error(
                f"Only **{n_shared}** shared trading days across all uploaded ETFs "
                "(need at least 60). Check that your CSV date ranges overlap."
            )
        else:
            date_start = common_returns_all.index[0].date()
            date_end   = common_returns_all.index[-1].date()
            st.caption(
                f"Correlation window: **{date_start}** to **{date_end}** "
                f"({n_shared} shared trading days across all {len(all_tickers)} ETFs)"
            )

            fig = make_correlation_heatmap(common_returns_all)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(
                "**Green** = low/negative correlation (diversification benefit).  \n"
                "**Red** = high positive correlation (instruments move together)."
            )

            with st.expander("Show correlation matrix as table"):
                corr_table = common_returns_all.corr(method="pearson")
                st.dataframe(
                    corr_table.style
                    .format("{:.3f}")
                    .background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1),
                    use_container_width=True,
                )

    # =========================================================================
    # TAB 2 — BACKTEST
    # =========================================================================
    with tab_bt:
        st.header("Buy & Hold Backtest")

        final_val  = port_wealth.iloc[-1]
        alloc_str  = "  |  ".join(f"{t} {w:.0%}" for t, w in zip(tickers, w_arr))
        st.markdown(
            f"**EUR {portfolio_eur:,.0f}  →  EUR {final_val:,.0f}** "
            f"over **{total_years:.1f} years**  \n{alloc_str}"
        )

        # Annual statistics
        rows = []
        for yr, grp in port_returns.groupby(port_returns.index.year):
            n_yr = len(grp)
            if n_yr < 20:
                rows.append({
                    "Year": yr, "Return (%)": None, "Vol (%)": None,
                    "Sharpe": None, "Max DD (%)": None,
                    "Days": n_yr, "Note": "partial year",
                })
                continue
            ann_ret  = float((1 + grp).prod() - 1)
            ann_vol  = float(grp.std(ddof=1) * np.sqrt(252))
            sharpe   = (
                round((ann_ret - risk_free_annual) / ann_vol, 3)
                if ann_vol > 1e-9 else None
            )
            yr_wealth = compute_growth(grp, initial_value=1.0)
            max_dd   = compute_max_drawdown(yr_wealth)
            rows.append({
                "Year": yr,
                "Return (%)": round(ann_ret * 100, 2),
                "Vol (%)":    round(ann_vol * 100, 2),
                "Sharpe":     sharpe,
                "Max DD (%)": round(max_dd * 100, 2),
                "Days": n_yr, "Note": "",
            })

        overall_ann_ret = float(port_wealth.iloc[-1] / portfolio_eur) ** (1.0 / total_years) - 1
        overall_ann_vol = float(port_returns.std(ddof=1) * np.sqrt(252))
        overall_sharpe  = (
            round((overall_ann_ret - risk_free_annual) / overall_ann_vol, 3)
            if overall_ann_vol > 1e-9 else None
        )
        rows.append({
            "Year": "OVERALL",
            "Return (%)": round(overall_ann_ret * 100, 2),
            "Vol (%)":    round(overall_ann_vol * 100, 2),
            "Sharpe":     overall_sharpe,
            "Max DD (%)": round(compute_max_drawdown(port_wealth) * 100, 2),
            "Days": len(port_returns),
            "Note": f"{total_years:.1f} yrs",
        })

        stats_df = pd.DataFrame(rows)
        stats_df["Year"] = stats_df["Year"].astype(str)
        stats_df = stats_df.set_index("Year")
        st.dataframe(
            stats_df[["Return (%)", "Vol (%)", "Sharpe", "Max DD (%)", "Note"]],
            use_container_width=True,
        )

        # Wealth chart
        start_date       = prices_selected.index[0]
        port_wealth_plot = pd.concat([
            pd.Series([portfolio_eur], index=[start_date]),
            port_wealth,
        ])
        etf_wealth_dict = {}
        for t in tickers:
            ew = compute_growth(etf_returns[t], initial_value=portfolio_eur)
            etf_wealth_dict[t] = pd.concat([
                pd.Series([portfolio_eur], index=[start_date]), ew
            ])

        fig = make_backtest_chart(
            port_wealth_plot=port_wealth_plot,
            etf_wealth_dict=etf_wealth_dict,
            w_dict=w_dict,
            portfolio_eur=portfolio_eur,
            port_start=port_start,
            total_years=total_years,
        )
        st.pyplot(fig)
        plt.close(fig)

    # =========================================================================
    # TAB 3 — DIVERSIFICATION
    # =========================================================================
    with tab_div:
        st.header("Diversification Quality")

        sel_returns = etf_returns[tickers]
        n_inst      = len(tickers)

        # Average pairwise correlation
        if n_inst > 1:
            corr_sel   = sel_returns.corr(method="pearson")
            idx_upper  = np.triu_indices(n_inst, k=1)
            avg_corr   = float(corr_sel.values[idx_upper].mean())
            corr_label = "LOW" if avg_corr < 0.30 else ("MODERATE" if avg_corr < 0.60 else "HIGH")
            corr_comment = {
                "LOW":      "Low average correlation — strong diversification benefit.",
                "MODERATE": "Moderate correlation — partial diversification benefit.",
                "HIGH":     "High correlation — limited diversification benefit.",
            }[corr_label]
        else:
            avg_corr, corr_label, corr_comment = 1.0, "N/A", "Single instrument."

        # Diversification Ratio
        annual_vols      = sel_returns.std(ddof=1) * np.sqrt(252)
        weighted_avg_vol = float(np.dot(w_arr, annual_vols.values))
        cov_mat          = sel_returns.cov() * 252
        port_var         = float(w_arr @ cov_mat.values @ w_arr)
        port_vol         = np.sqrt(max(port_var, 1e-12))
        DR               = weighted_avg_vol / port_vol
        var_red_pct      = (1.0 - 1.0 / DR) * 100.0
        dr_label         = "LOW" if DR < 1.10 else ("MODERATE" if DR < 1.50 else "STRONG")

        # PCA effective factors
        if n_inst > 1:
            eigvals  = np.sort(np.linalg.eigvalsh(cov_mat.values))[::-1]
            eigvals  = eigvals[eigvals > 1e-10]
            props    = eigvals / eigvals.sum()
            eff_n    = float(np.exp(-np.sum(props * np.log(props + 1e-15))))
            pct_dom  = float(props[0] * 100)
            pca_label = "LOW" if eff_n / n_inst < 0.50 else ("MODERATE" if eff_n / n_inst < 0.75 else "HIGH")
            pca_comment = {
                "LOW":      f"One dominant factor drives {pct_dom:.0f}% of variance.",
                "MODERATE": f"{eff_n:.1f} of {n_inst} independent factors active.",
                "HIGH":     f"{eff_n:.1f} of {n_inst} factors contribute meaningfully.",
            }[pca_label]
        else:
            eff_n, pca_label, pca_comment = 1.0, "N/A", "Single instrument."

        # Summary metric cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Pairwise Correlation", f"{avg_corr:.3f}", delta=corr_label, delta_color="off")
        col2.metric("Diversification Ratio", f"{DR:.3f}", delta=dr_label, delta_color="off")
        col3.metric("PCA Effective Factors", f"{eff_n:.1f} / {n_inst}", delta=pca_label, delta_color="off")

        st.markdown(
            f"- **Correlation ({corr_label}):** {corr_comment}\n"
            f"- **Diversification Ratio = {DR:.2f}** → portfolio volatility is "
            f"**{var_red_pct:.0f}% lower** than the weighted average of individual ETF volatilities.\n"
            f"- **PCA:** {pca_comment}"
        )

        # Per-ETF volatility bar chart
        st.subheader("Individual ETF Annualized Volatility")
        BOND_TICKERS = {"AGGH", "C3M"}
        bar_colors = ["#00695C" if t in BOND_TICKERS else "#1565C0" for t in tickers]
        vol_vals   = annual_vols[tickers].values * 100

        fig, ax = plt.subplots(figsize=(max(6, n_inst * 1.3), 4))
        bars = ax.bar(tickers, vol_vals, color=bar_colors, edgecolor="white", width=0.6)
        ax.axhline(y=port_vol * 100, color="#C62828", ls="--", lw=1.5,
                   label=f"Portfolio vol: {port_vol*100:.2f}%")
        for bar, v in zip(bars, vol_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Annualized Volatility (%)")
        ax.set_title("Individual ETF Volatility vs Portfolio Volatility")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # =========================================================================
    # TAB 4 — MONTE CARLO
    # =========================================================================
    with tab_mc:
        st.header("Monte Carlo Projections")

        n_history = len(port_returns)
        if n_history < 252:
            st.warning(
                f"Portfolio history is only **{n_history}** trading days (< 1 year). "
                "GBM calibration is unreliable for short histories. Results are illustrative only."
            )

        # GBM calibration info
        log_ret      = np.log1p(port_returns.values)
        mu_daily     = float(log_ret.mean())
        sigma_daily  = float(log_ret.std(ddof=1))
        mu_annual    = float(np.exp(mu_daily * 252) - 1)
        sigma_annual = float(sigma_daily * np.sqrt(252))
        sharpe_gbm   = (
            (mu_annual - risk_free_annual) / sigma_annual
            if sigma_annual > 1e-9 else float("nan")
        )

        with st.expander("GBM Calibration Parameters"):
            cal_df = pd.DataFrame({
                "Parameter": [
                    "Historical period", "Trading days",
                    "Daily log-return mean", "Daily log-return vol",
                    "Implied annual return", "Implied annual vol", "Implied Sharpe",
                ],
                "Value": [
                    f"{port_returns.index[0].date()} to {port_returns.index[-1].date()}",
                    str(n_history),
                    f"{mu_daily*100:.4f}%",
                    f"{sigma_daily*100:.4f}%",
                    f"{mu_annual*100:.2f}%",
                    f"{sigma_annual*100:.2f}%",
                    f"{sharpe_gbm:.3f}",
                ],
            })
            st.dataframe(cal_df, hide_index=True, use_container_width=True)

        # Run simulations (cached)
        initial_val          = float(port_wealth.iloc[-1])
        port_returns_tuple   = tuple(port_returns.values.tolist())
        mc_results           = run_monte_carlo_cached(
            port_returns_tuple=port_returns_tuple,
            initial_value=initial_val,
            n_sims=n_sims,
            block_length=block_length,
        )

        gbm_paths  = mc_results["gbm"]
        boot_paths = mc_results["bootstrap"]

        # Terminal value summary table
        st.subheader("Terminal Value Summary")
        summary_rows = []
        for horizon in (10, 20, 30):
            for method, paths in [("GBM", gbm_paths), ("Block Bootstrap", boot_paths)]:
                terminal = paths[horizon][-1]
                summary_rows.append({
                    "Horizon": f"{horizon} yr",
                    "Method":  method,
                    "5th pct (EUR)":  f"{np.percentile(terminal, 5):,.0f}",
                    "Median (EUR)":   f"{np.median(terminal):,.0f}",
                    "95th pct (EUR)": f"{np.percentile(terminal, 95):,.0f}",
                })
        st.dataframe(
            pd.DataFrame(summary_rows),
            hide_index=True,
            use_container_width=True,
        )

        # Fan charts
        port_wealth_full = pd.concat([
            pd.Series([portfolio_eur], index=[prices_selected.index[0]]),
            port_wealth,
        ])
        fig = make_fan_chart(
            gbm_paths=gbm_paths,
            boot_paths=boot_paths,
            port_wealth_full=port_wealth_full,
            initial_val=initial_val,
            n_sims=n_sims,
            port_returns_index=port_returns.index,
        )
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
**Interpretation:**
- The **gray line** shows actual historical portfolio wealth (before t = 0).
- The **dark band** (25th–75th percentile) is the central 50% of simulated outcomes.
- The **light band** (5th–95th percentile) is the 90% probability interval.
- **GBM** assumes log-normal returns. **Block Bootstrap** resamples actual return sequences, preserving fat tails and autocorrelation.
- The Block Bootstrap 5th-percentile is a useful stress scenario: it has a 5% chance of occurring.
        """)


if __name__ == "__main__":
    main()
