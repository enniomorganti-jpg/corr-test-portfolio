"""
engine.py — Factor Data Loading, Date Alignment, and Style Analysis

Loads ETF benchmark prices from local CSVs (no network calls), aligns them
with the fund's observation dates using a "Common Window" approach, and runs
sum-to-one constrained NNLS (Non-Negative Least Squares) style analysis.

Public API
----------
load_etf_local(etf_folder)           -> pd.DataFrame
align_returns(fund_df, etf_prices)   -> (fund_aligned, etf_returns_aligned)
run_nnls(etf_returns, fund_gross_returns, fund_net_returns, lambda_constraint)
                                     -> (weights, r_squared)
build_replica(etf_returns, weights)  -> pd.Series
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls

import loader  # local module — load_benchmark_csv


# ---------------------------------------------------------------------------
# Factor universe — dynamically discovered from the ETF folder at runtime.
#
# BOND_KEYS: the only thing you need to maintain manually.
#   Add a key here if you add a bond ETF to the folder.
#   Everything else is treated as equity for chart color purposes.
#
# FACTOR_TICKERS: populated by load_etf_local() — do not set manually.
#   After load_etf_local() runs, it maps {key: key} for every discovered ETF.
# ---------------------------------------------------------------------------

BOND_KEYS: set[str] = {"AGGH", "C3M"}

# Populated at runtime by load_etf_local(); starts empty.
FACTOR_TICKERS: dict[str, str] = {}


def _extract_ticker(filename: str) -> str:
    """
    Extract the short ticker key from an Investing.com ETF CSV filename.

    Expected pattern: "{TICKER} ETF Stock Price History.csv"
    Example: "SWDA ETF Stock Price History.csv" → "SWDA"

    Falls back to the filename stem (without extension) if the pattern
    is not matched, so arbitrary CSV names also work.
    """
    suffix = " ETF Stock Price History.csv"
    if filename.endswith(suffix):
        return filename[: -len(suffix)].strip()
    # Fallback: use stem (e.g., "SWDA.csv" → "SWDA")
    return os.path.splitext(filename)[0].strip()


def discover_etf_files(etf_folder: str) -> dict[str, str]:
    """
    Scan a folder for ETF CSV files and return a mapping of key → filename.

    Any file ending in .csv is treated as a factor. The short key is
    extracted from the filename via _extract_ticker().

    Parameters
    ----------
    etf_folder : Path to the folder containing ETF CSVs.

    Returns
    -------
    dict mapping short key (e.g. "SWDA") → filename (e.g. "SWDA ETF ...csv").
    Sorted alphabetically by key for deterministic ordering.
    """
    if not os.path.isdir(etf_folder):
        raise FileNotFoundError(
            f"ETF folder not found: '{etf_folder}'. "
            f"Check that ETF_FOLDER points to the correct directory."
        )

    csv_files = sorted(
        f for f in os.listdir(etf_folder)
        if f.lower().endswith(".csv") and not f.startswith(".")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{etf_folder}'. "
            f"Please place your ETF CSVs in that folder."
        )

    return {_extract_ticker(f): f for f in csv_files}


# ---------------------------------------------------------------------------
# Local ETF price loader (replaces download_etf_prices)
# ---------------------------------------------------------------------------

def load_etf_local(etf_folder: str) -> pd.DataFrame:
    """
    Auto-discover and load all ETF price CSVs from a local folder.

    Scans the folder for any .csv file, extracts the ticker from the
    filename, and loads each one. No hardcoded list — just drop a new
    CSV in the folder and it will be picked up automatically.

    Side effect: updates the module-level FACTOR_TICKERS dict in-place
    so that downstream code (charts, notebook) sees the discovered factors.

    Parameters
    ----------
    etf_folder : Path to the folder containing the ETF CSVs.

    Returns
    -------
    DataFrame indexed by date (pd.Timestamp, ascending), one column per
    discovered ETF key. Contains raw prices.
    """
    discovered = discover_etf_files(etf_folder)

    print(f"[engine] Discovered {len(discovered)} ETF CSV(s) in folder:")
    print(f"[engine] Folder: {etf_folder}\n")

    price_series = {}
    for key, filename in discovered.items():
        filepath = os.path.join(etf_folder, filename)
        series = loader.load_benchmark_csv(filepath, label=key)
        price_series[key] = series

    # Update the module-level FACTOR_TICKERS dict in-place so that
    # `from engine import FACTOR_TICKERS` in the notebook sees the live values.
    FACTOR_TICKERS.clear()
    FACTOR_TICKERS.update({key: key for key in discovered})

    # Concatenate into a single price DataFrame (outer join — alignment handles dates)
    prices_df = pd.concat(price_series, axis=1)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df.index.name = "date"
    prices_df = prices_df.sort_index()

    print(f"\n[engine] Loaded {len(prices_df.columns)} factors: {list(prices_df.columns)}")
    print(f"[engine] Price data spans: {prices_df.index[0].date()} → {prices_df.index[-1].date()}")

    return prices_df


# ---------------------------------------------------------------------------
# Date alignment — Common Window approach
# ---------------------------------------------------------------------------

def align_returns(
    fund_df: pd.DataFrame,
    etf_prices: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align fund returns and ETF returns using the fund's own trading calendar
    as the master timeline (inner join approach).

    WHY THIS MATTERS — the phantom-zero bug:
      The naive approach (outer-join all ETF dates → ffill → pct_change) injects
      artificial 0% returns into the regression matrix X on every day a European
      exchange is closed while the US fund is open. For a US tech fund, those days
      often have large moves. NNLS correctly sees 0% for EQQQ on those days and
      assigns it 0 weight. This method eliminates that noise entirely.

    Algorithm
    ---------
    0.  (Optional) Clip fund_df to [start_date, end_date] before alignment.
    1.  Determine Common Window:
          common_start = max(fund start date, earliest ETF price date)
          common_end   = min(fund end date,   latest ETF price date)
    2.  Raise ValueError with clear message if no overlap.
    3.  Take the fund's own trading dates inside the Common Window as the
        MASTER CALENDAR — the only dates that matter for the regression.
    4.  Reindex ETF PRICES to those fund dates using forward-fill.
        (If an ETF didn't trade on a fund date, use the last known price.)
    5.  Compute ETF pct_change on this fund-calendar-aligned price DataFrame.
        Returns are now measured on exactly the same intervals as the fund NAV.
        No phantom 0% returns from exchange holidays.
    6.  Drop the first row (NaN from pct_change) and any rows with NaN
        (ETF inception dates before the ETF existed).
    7.  Require at least min_periods aligned rows (frequency-aware).

    Parameters
    ----------
    fund_df    : Output of load_nav_csv (date-indexed, columns: nav, r_net, r_gross).
    etf_prices : Raw price DataFrame from load_etf_local (date-indexed, one col per ETF).
    start_date : Optional ISO date string (e.g. "2021-01-01") to restrict the
                 analysis window. Rows before this date are dropped from the fund.
    end_date   : Optional ISO date string (e.g. "2022-12-31") to restrict the
                 analysis window. Rows after this date are dropped from the fund.

    Returns
    -------
    fund_aligned         : fund_df sliced to aligned dates
    etf_returns_aligned  : DataFrame of ETF daily returns on the same aligned dates
    """
    etf_prices.index = pd.to_datetime(etf_prices.index)
    fund_df.index    = pd.to_datetime(fund_df.index)

    # Step 0: Apply optional user-supplied date window to the fund series.
    if start_date is not None:
        fund_df = fund_df.loc[pd.Timestamp(start_date):]
        print(f"[engine] Analysis start clipped to : {pd.Timestamp(start_date).date()}")
    if end_date is not None:
        fund_df = fund_df.loc[:pd.Timestamp(end_date)]
        print(f"[engine] Analysis end   clipped to : {pd.Timestamp(end_date).date()}")
    if fund_df.empty:
        raise ValueError(
            f"[engine] No fund data remains after applying start_date={start_date!r} "
            f"/ end_date={end_date!r}. Check your date range."
        )

    # Step 1: Common Window
    fund_start = fund_df.index[0]
    fund_end   = fund_df.index[-1]
    etf_start  = etf_prices.index[0]
    etf_end    = etf_prices.index[-1]

    common_start = max(fund_start, etf_start)
    common_end   = min(fund_end,   etf_end)

    print(f"[engine] Fund range      : {fund_start.date()} → {fund_end.date()}")
    print(f"[engine] ETF price range : {etf_start.date()} → {etf_end.date()}")
    print(f"[engine] Common Window   : {common_start.date()} → {common_end.date()}")

    # Step 2: Guard — no overlap
    if common_start >= common_end:
        raise ValueError(
            f"\n[engine] No overlapping dates found between benchmarks and fund.\n"
            f"  Fund range:      {fund_start.date()} → {fund_end.date()}\n"
            f"  ETF price range: {etf_start.date()} → {etf_end.date()}\n"
            f"  Ensure your fund CSV date range overlaps with the benchmark CSVs."
        )

    # Step 3: Fund trading dates inside the Common Window = master calendar
    fund_window  = fund_df.loc[common_start:common_end]
    master_dates = fund_window.index  # these are the only dates we care about

    if len(master_dates) == 0:
        raise ValueError(
            "[engine] No overlapping dates found between benchmarks and fund."
        )

    # Step 4: Reindex ETF PRICES to fund dates — fill gaps with last known price.
    #   This is the key fix: we compute returns only on fund-valuation days.
    #   An ETF closed on a US trading day gets its previous closing price carried
    #   forward, so its "return" for that fund-date interval is measured correctly
    #   across however many calendar days elapsed — not injected as 0%.
    etf_on_fund_dates = etf_prices.reindex(master_dates, method="ffill")

    # Step 5: ETF returns on the fund calendar
    etf_returns = etf_on_fund_dates.pct_change()

    # Step 6: Drop first row (NaN from pct_change) and align both sides
    etf_returns  = etf_returns.iloc[1:]
    fund_aligned = fund_window.iloc[1:]

    # Inner join: keep only rows with no NaN in any column
    combined = pd.concat([fund_aligned[["r_net", "r_gross"]], etf_returns], axis=1)
    n_before = len(combined)
    combined = combined.dropna()
    n_dropped = n_before - len(combined)

    if n_dropped > 0:
        pct = n_dropped / n_before * 100
        msg = (
            f"[engine] Dropped {n_dropped} rows ({pct:.1f}%) with NaN — "
            f"likely dates before some ETFs began trading."
        )
        if pct > 10:
            warnings.warn(msg + " High NaN rate — check benchmark CSV date coverage.")
        else:
            print(msg)

    # Step 7: Minimum data check — frequency-aware.
    n_aligned = len(combined)
    if n_aligned == 0:
        raise ValueError(
            "[engine] No overlapping dates found between benchmarks and fund.\n"
            "  Check that fund CSV dates and ETF CSV dates are in compatible formats."
        )

    gaps = combined.index.to_series().diff().dt.days.dropna()
    median_gap = float(gaps.median())
    if median_gap <= 3:
        freq_label, min_periods = "daily", 252
    elif median_gap <= 10:
        freq_label, min_periods = "weekly", 104
    else:
        freq_label, min_periods = "monthly", 36

    if n_aligned < min_periods:
        raise ValueError(
            f"[engine] Only {n_aligned} aligned {freq_label} observations after inner join — "
            f"need at least {min_periods} for reliable style analysis.\n"
            f"  Extend the fund CSV history or verify benchmark date coverage.\n"
            f"  Common Window was: {common_start.date()} → {common_end.date()}"
        )

    fund_aligned        = fund_aligned.loc[combined.index]
    etf_returns_aligned = etf_returns.loc[combined.index]

    print(
        f"[engine] Aligned {n_aligned} {freq_label} observations | "
        f"{combined.index[0].date()} → {combined.index[-1].date()}"
    )

    return fund_aligned, etf_returns_aligned


# ---------------------------------------------------------------------------
# NNLS Style Analysis
# ---------------------------------------------------------------------------

def run_nnls(
    etf_returns: pd.DataFrame,
    fund_gross_returns: pd.Series,
    fund_net_returns: pd.Series,
    lambda_constraint: float = 100.0,
) -> tuple[np.ndarray, float]:
    """
    Run sum-to-one constrained Non-Negative Least Squares (NNLS).

    The regression is performed on GROSS returns (to find the true factor
    exposures before fee drag). R² is computed on NET returns (conservative;
    reflects the investor's actual experience).

    Math
    ----
    Objective:  min ||Xw - y_gross||²   s.t. w_i >= 0, sum(w) = 1

    Augmented system (converts equality constraint to soft penalty):
        X_aug = vstack([X,  λ * 1ᵀ])   shape: (T+1) × n_factors
        y_aug = hstack([y,  λ * 1  ])   shape: (T+1,)
    Then: w_raw, _ = scipy.nnls(X_aug, y_aug)
          w = w_raw / w_raw.sum()        # exact normalization

    Parameters
    ----------
    etf_returns        : T × n_factors DataFrame of ETF daily returns.
    fund_gross_returns : T-length Series of fund gross daily returns.
    fund_net_returns   : T-length Series of fund net daily returns (for R²).
    lambda_constraint  : Penalty strength for sum=1 constraint (default 100).

    Returns
    -------
    weights   : np.ndarray of shape (n_factors,), non-negative, sum to 1.0
    r_squared : float, R² of net returns explained by the weighted replica.
    """
    X = etf_returns.values.astype(float)           # shape: (T, n_factors)
    y = fund_gross_returns.values.astype(float)    # shape: (T,)
    n_factors = X.shape[1]
    tickers   = list(etf_returns.columns)

    # -----------------------------------------------------------------------
    # Diagnostic: Pearson correlation of each ETF vs fund gross returns.
    # Printed BEFORE NNLS so the user can see raw signal strength.
    # If EQQQ shows near-zero here, the issue is data quality (dates / FX).
    # If EQQQ shows high correlation but still gets 0% weight, a leveraged
    # ETF in the basket is crowding it out — remove that ETF from the folder.
    # -----------------------------------------------------------------------
    etf_vol_annual = np.std(X, axis=0) * np.sqrt(252)
    print("[engine] ETF diagnostics (before NNLS):")
    print(f"  {'Ticker':<8}  {'Corr vs fund':>12}  {'Ann.Vol':>8}  {'Note'}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*30}")
    for i, t in enumerate(tickers):
        corr = float(np.corrcoef(X[:, i], y)[0, 1])
        vol  = float(etf_vol_annual[i])
        note = ""
        if vol > 0.30:
            note = "⚠ HIGH VOL — may be leveraged!"
        bar  = "█" * int(abs(corr) * 20)
        sign = "+" if corr >= 0 else "-"
        print(f"  {t:<8}  {sign}{abs(corr):.4f}        {vol*100:6.1f}%  {note}  {bar}")
    print()

    # Inter-ETF collinearity check.
    # When two ETFs in the basket are highly correlated with each other,
    # NNLS will assign weight to one and zero to the other, even if both
    # have high correlation with the fund.  This is correct NNLS behaviour.
    # The fix is to remove the redundant factor from the ETF folder.
    corr_matrix = np.corrcoef(X.T)   # shape: (n_factors, n_factors)
    high_corr_pairs = []
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            c = float(corr_matrix[i, j])
            if abs(c) > 0.85:
                high_corr_pairs.append((tickers[i], tickers[j], c))

    if high_corr_pairs:
        high_corr_pairs.sort(key=lambda x: -abs(x[2]))
        print(
            "[engine] ⚠  Highly correlated factor pairs (|r| > 0.85).\n"
            "         NNLS will zero out one of each pair even when both\n"
            "         correlate strongly with the fund.\n"
            "         ACTION: remove the less-appropriate ETF from the ETF folder.\n"
        )
        for t1, t2, c in high_corr_pairs:
            print(f"  {t1} <-> {t2} : {c:+.4f}")
        print()
    else:
        print("[engine] No highly correlated factor pairs found (all |r| <= 0.85).\n")

    # Augmented system: add one row to enforce sum(w) = 1
    constraint_row = lambda_constraint * np.ones((1, n_factors))
    X_aug = np.vstack([X, constraint_row])
    y_aug = np.hstack([y, lambda_constraint * 1.0])

    w_raw, _ = nnls(X_aug, y_aug)

    # Guard against degenerate all-zero solution
    if w_raw.sum() < 1e-10:
        warnings.warn(
            "[engine] NNLS returned all-zero weights. "
            "The fund returns may have no correlation with the ETF universe. "
            "Defaulting to equal weights."
        )
        w_raw = np.ones(n_factors)

    # Exact normalization to enforce sum = 1
    weights = w_raw / w_raw.sum()

    assert abs(weights.sum() - 1.0) < 1e-6, (
        f"Weight normalization failed: sum = {weights.sum():.8f}"
    )

    # R² on net returns (conservative — reflects investor's actual experience)
    y_net = fund_net_returns.values.astype(float)
    y_hat = X @ weights  # replica returns on aligned dates

    ss_res = np.sum((y_net - y_hat) ** 2)
    ss_tot = np.sum((y_net - y_net.mean()) ** 2)

    if ss_tot < 1e-12:
        warnings.warn("[engine] Fund net returns are essentially flat — R² is undefined.")
        r_squared = float("nan")
    else:
        r_squared = float(1.0 - ss_res / ss_tot)

    # Print results
    if not np.isnan(r_squared):
        ci_label = " [CLOSET INDEXER]" if r_squared > 0.90 else ""
        print(f"[engine] R² = {r_squared:.4f}{ci_label}")

    print("[engine] Factor weights:")
    for t, w in zip(tickers, weights):
        bar   = "█" * int(w * 40)
        label = FACTOR_TICKERS.get(t, t)
        print(f"  {label:40s} ({t}): {w*100:5.1f}%  {bar}")

    return weights, r_squared


# ---------------------------------------------------------------------------
# Replica construction
# ---------------------------------------------------------------------------

def build_replica(
    etf_returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """
    Construct the synthetic replica's daily return series.

    replica_return_t = sum_i( w_i * ETF_return_i_t )

    Parameters
    ----------
    etf_returns : T × n_factors DataFrame of ETF daily returns.
    weights     : np.ndarray of factor weights (non-negative, sum to 1).

    Returns
    -------
    pd.Series of weighted daily returns, same index as etf_returns.
    """
    replica = pd.Series(
        etf_returns.values @ weights,
        index=etf_returns.index,
        name="replica_return",
    )
    return replica
