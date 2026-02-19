"""
loader.py — Smart NAV CSV Parser

Ingests a fund's historical NAV CSV (from Investing.com, Borsa Italiana, etc.)
and computes daily net and gross returns. Also provides a lightweight function
for loading benchmark ETF CSVs (price series only, no gross-up).

Public API
----------
load_nav_csv(filepath, ter_annual, date_col=None, price_col=None) -> pd.DataFrame
    Returns a DataFrame indexed by date with columns [nav, r_net, r_gross].

load_benchmark_csv(filepath, label=None) -> pd.Series
    Returns a price Series indexed by date (ascending). No return computation.
    Used by engine.py to load local ETF benchmark CSVs.
"""

import os
import warnings
import pandas as pd
import numpy as np

# Column name keywords for auto-detection
_DATE_KEYWORDS  = {"date", "data", "giorno", "datum", "fecha"}
_PRICE_KEYWORDS = {"price", "nav", "prezzo", "close", "last", "valore", "prix"}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Auto-detect date and price columns from a raw DataFrame.
    Returns (date_col_name, price_col_name).
    """
    lower_map = {col: col.lower().strip() for col in df.columns}

    date_candidates  = [col for col, lc in lower_map.items() if any(k in lc for k in _DATE_KEYWORDS)]
    price_candidates = [col for col, lc in lower_map.items() if any(k in lc for k in _PRICE_KEYWORDS)]

    if not date_candidates:
        raise ValueError(
            f"Cannot auto-detect a date column. "
            f"Column names found: {list(df.columns)}. "
            f"Pass date_col='<your column name>' explicitly."
        )
    if not price_candidates:
        raise ValueError(
            f"Cannot auto-detect a price/NAV column. "
            f"Column names found: {list(df.columns)}. "
            f"Pass price_col='<your column name>' explicitly."
        )

    if len(date_candidates) > 1:
        warnings.warn(
            f"Multiple date column candidates: {date_candidates}. "
            f"Using '{date_candidates[0]}'. Pass date_col= to override."
        )
    if len(price_candidates) > 1:
        warnings.warn(
            f"Multiple price column candidates: {price_candidates}. "
            f"Using '{price_candidates[0]}'. Pass price_col= to override."
        )

    return date_candidates[0], price_candidates[0]


def _parse_date_series(series: pd.Series) -> pd.Series:
    """
    Robust date parser supporting all formats encountered in practice.

    Strategy (tried in order):
      1. "%b %d, %Y"    — "Feb 13, 2026"  (Investing.com string months)
      2. "%Y-%m-%d"     — "2026-02-13"    (ISO 8601)
      3. dayfirst=True  — "17/02/2026"    (European DD/MM/YYYY)
                          Will produce many NaT for US-format dates where
                          the day field > 12 (e.g. "02/17/2026" → month 17
                          is invalid), so the threshold check rejects it.
      4. dayfirst=False — "02/17/2026"    (US MM/DD/YYYY, Investing.com default)
      5. "%d-%m-%Y"     — "17-02-2026"
      6. "%d.%m.%Y"     — "17.02.2026"
      7. Auto-inference fallback

    Accepts a strategy if fewer than 5% of values become NaT.
    All results are guaranteed timezone-naive.

    WHY dayfirst=True BEFORE dayfirst=False:
      European CSV "17/02/2026" → dayfirst=True succeeds (day=17, month=02).
      US CSV "02/17/2026" → dayfirst=True tries month=17, which is invalid
      → ~60% NaT → threshold exceeded → rejected.
      US CSV then succeeds on dayfirst=False. This order is unambiguous.
    """
    n = len(series)
    threshold = max(1, int(n * 0.05))

    strategies = [
        # Explicit format strings
        {"format": "%b %d, %Y"},                       # "Feb 13, 2026"
        {"format": "%Y-%m-%d"},                        # "2026-02-13" ISO
        # Flexible numeric: European first, then US
        {"dayfirst": True},                            # "17/02/2026" or "17-02-2026"
        {"dayfirst": False},                           # "02/17/2026" US
        # Less-common separators
        {"format": "%d-%m-%Y"},                        # "17-02-2026"
        {"format": "%d.%m.%Y"},                        # "17.02.2026"
        {"format": "%m/%d/%Y"},                        # explicit US fallback
    ]

    for kwargs in strategies:
        try:
            # Suppress pandas UserWarning about format conflicts — these fire
            # during intentional try-and-reject probing (e.g. dayfirst=True on
            # US-format dates) and are expected noise, not real problems.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(series, errors="coerce", **kwargs)
        except Exception:
            continue

        nat_count = int(parsed.isna().sum())
        if nat_count <= threshold:
            if nat_count > 0:
                warnings.warn(
                    f"[loader] Date parsing ({kwargs}): "
                    f"{nat_count} value(s) could not be parsed and will be dropped."
                )
            # Ensure timezone-naive
            if parsed.dt.tz is not None:
                parsed = parsed.dt.tz_localize(None)
            return parsed

    # Final fallback
    warnings.warn(
        "[loader] No date format matched. Falling back to pandas auto-inference. "
        "Verify parsed dates in the output."
    )
    parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _clean_price_series(series: pd.Series) -> pd.Series:
    """
    Strip currency symbols, thousands separators, and whitespace; cast to float64.
    """
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"[^\d.\-]", "", regex=True)
    cleaned = cleaned.replace("", np.nan)

    result = pd.to_numeric(cleaned, errors="coerce")
    n_failed = int(result.isna().sum())
    if n_failed > 0:
        warnings.warn(
            f"[loader] {n_failed} price value(s) could not be converted to float → set to NaN."
        )
    return result


def _build_clean_series(dates: pd.Series, prices: pd.Series, label: str) -> pd.Series:
    """
    Combine parsed dates and prices into a clean, sorted, deduplicated Series.
    Shared by both load functions.
    """
    s = pd.Series(prices.values, index=dates, name=label).dropna()

    if len(s) < 2:
        raise ValueError(
            f"[loader] {label}: fewer than 2 valid rows after cleaning. "
            "Check date and price column detection."
        )

    # Strict ascending chronological sort
    s = s.sort_index(ascending=True)

    # Deduplicate (keep last — matches Investing.com export behavior)
    n_before = len(s)
    s = s[~s.index.duplicated(keep="last")]
    if len(s) < n_before:
        warnings.warn(
            f"[loader] {label}: {n_before - len(s)} duplicate date(s) removed."
        )

    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_nav_csv(
    filepath: str,
    ter_annual: float,
    date_col: str = None,
    price_col: str = None,
) -> pd.DataFrame:
    """
    Parse a fund NAV CSV and compute daily net and gross returns.

    Parameters
    ----------
    filepath   : Path to the CSV file.
    ter_annual : Annual TER as a decimal (e.g. 0.0145 = 1.45%).
    date_col   : Override auto-detected date column name.
    price_col  : Override auto-detected price column name.

    Returns
    -------
    DataFrame indexed by date (pd.Timestamp, timezone-naive, ascending):
        nav     : float — cleaned NAV price
        r_net   : float — (NAV_t / NAV_{t-1}) - 1
        r_gross : float — r_net + TER/252
    """
    try:
        raw = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Fund CSV not found: '{filepath}'")
    except Exception as e:
        raise ValueError(f"Could not read fund CSV '{filepath}': {e}")

    if len(raw) < 2:
        raise ValueError(
            f"Fund CSV contains only {len(raw)} row(s) — need at least 2."
        )

    if date_col is None or price_col is None:
        detected_date, detected_price = _detect_columns(raw)
        date_col  = date_col  or detected_date
        price_col = price_col or detected_price

    if date_col not in raw.columns:
        raise ValueError(f"Date column '{date_col}' not found. Available: {list(raw.columns)}")
    if price_col not in raw.columns:
        raise ValueError(f"Price column '{price_col}' not found. Available: {list(raw.columns)}")

    dates  = _parse_date_series(raw[date_col])
    prices = _clean_price_series(raw[price_col])

    nav_series = _build_clean_series(dates, prices, label=os.path.basename(filepath))

    df = nav_series.rename("nav").to_frame()

    df["r_net"]   = df["nav"].pct_change()
    df = df.iloc[1:]  # drop first row (NaN returns, no prior day)

    # Auto-detect observation frequency to apply the right TER gross-up.
    gaps = df.index.to_series().diff().dt.days.dropna()
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

    period_fee       = ter_annual / periods_per_year
    df["r_gross"]    = df["r_net"] + period_fee

    print(
        f"Loaded Fund:  {df.index[0].date()} to {df.index[-1].date()} "
        f"({len(df)} {freq_label} observations) | TER {freq_label} gross-up: {period_fee*100:.4f}%"
    )
    return df


def load_benchmark_csv(
    filepath: str,
    label: str = None,
) -> pd.Series:
    """
    Load a benchmark ETF price CSV and return a clean price Series.

    Handles all Investing.com / Borsa Italiana CSV formats:
      - "Date","Price","Open","High","Low","Vol.","Change %"
      - Date formats: "02/17/2026", "17/02/2026", "Feb 13, 2026"

    Parameters
    ----------
    filepath : Path to the ETF CSV file.
    label    : Name for the returned Series (e.g. "SWDA").

    Returns
    -------
    pd.Series of float prices, indexed by pd.Timestamp (timezone-naive, ascending).
    """
    if label is None:
        label = os.path.splitext(os.path.basename(filepath))[0]

    try:
        raw = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Benchmark CSV not found: '{filepath}'")
    except Exception as e:
        raise ValueError(f"Could not read benchmark CSV '{filepath}': {e}")

    if len(raw) < 2:
        raise ValueError(f"Benchmark CSV '{label}' contains only {len(raw)} row(s).")

    date_col, price_col = _detect_columns(raw)
    dates  = _parse_date_series(raw[date_col])
    prices = _clean_price_series(raw[price_col])

    s = _build_clean_series(dates, prices, label=label)

    print(
        f"Loaded {label:6s}: {s.index[0].date()} to {s.index[-1].date()} "
        f"({len(s)} rows)"
    )
    return s
