"""
preprocess.py - Data cleaning and feature engineering
Digital Shield Ticket Anomaly Detection System

Cleans, normalises, and engineers features from the raw tickets_1 DataFrame.
"""

import pandas as pd
import numpy as np
from src.utils import setup_logger

logger = setup_logger()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline on the raw tickets DataFrame.

    Steps:
        1. Strip whitespace from column names and string columns
        2. Handle null values safely
        3. Remove duplicate rows
        4. Convert age to numeric
        5. Parse txn_date  → datetime
        6. Parse jrny_date → datetime
        7. Parse txn_time  → time (then combine with txn_date)
        8. Create booking_timestamp (txn_date + txn_time)
        9. Create journey_gap_days  (jrny_date − txn_date)

    Args:
        df (pd.DataFrame): Raw dataframe loaded from tickets_1.

    Returns:
        pd.DataFrame: Cleaned and feature-enriched dataframe.
    """
    logger.info("🔧 Starting data preprocessing ...")
    initial_rows = len(df)

    # ── 1. Strip whitespace from column names ────────────────────────────
    df.columns = df.columns.str.strip()
    logger.debug("  Column names stripped.")

    # ── 2. Strip whitespace from all string columns ───────────────────────
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        # Replace literal "nan" strings created by astype(str) with NaN
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    logger.debug("  String columns stripped.")

    # ── 3. Remove duplicate rows ─────────────────────────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates()
    removed = before_dedup - len(df)
    if removed > 0:
        logger.info(f"  🗑️  Removed {removed:,} duplicate rows.")

    # ── 4. Convert age to numeric ─────────────────────────────────────────
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].fillna(df["age"].median() if not df["age"].dropna().empty else 0)
    df["age"] = df["age"].clip(lower=0, upper=120)  # Sanity clip
    logger.debug("  'age' column converted to numeric.")

    # ── 5. Convert txn_date to datetime ───────────────────────────────────
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce", dayfirst=True)
    logger.debug("  'txn_date' converted to datetime.")

    # ── 6. Convert jrny_date to datetime ─────────────────────────────────
    df["jrny_date"] = pd.to_datetime(df["jrny_date"], errors="coerce", dayfirst=True)
    logger.debug("  'jrny_date' converted to datetime.")

    # ── 7. Parse txn_time properly ────────────────────────────────────────
    df["txn_time"] = _parse_time_column(df["txn_time"])
    logger.debug("  'txn_time' parsed successfully.")

    # ── 8. Create booking_timestamp ───────────────────────────────────────
    df["booking_timestamp"] = _combine_date_time(df["txn_date"], df["txn_time"])
    logger.debug("  'booking_timestamp' created.")

    # ── 9. Create journey_gap_days ────────────────────────────────────────
    df["journey_gap_days"] = (df["jrny_date"] - df["txn_date"]).dt.days
    df["journey_gap_days"] = df["journey_gap_days"].fillna(-1).astype(int)
    logger.debug("  'journey_gap_days' calculated.")

    # ── Final null report ─────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        logger.warning(f"  ⚠️  Null values remaining after preprocessing:\n{cols_with_nulls.to_string()}")

    final_rows = len(df)
    logger.info(
        f"✅ Preprocessing complete. Rows: {initial_rows:,} → {final_rows:,} "
        f"(removed {initial_rows - final_rows:,})."
    )

    return df


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _parse_time_column(time_series: pd.Series) -> pd.Series:
    """
    Parse a mixed-format time series into pandas time objects.
    Handles formats like '14:30:00', '14:30', '2:30 PM', etc.

    Returns a Series of datetime.time objects (NaT where parsing fails).
    """
    parsed = pd.to_datetime(time_series, errors="coerce", format="%H:%M:%S")

    # Try secondary format for values that failed
    mask_failed = parsed.isna()
    if mask_failed.any():
        secondary = pd.to_datetime(
            time_series[mask_failed], errors="coerce", format="%H:%M"
        )
        parsed[mask_failed] = secondary

    # Try 12-hour format
    mask_failed2 = parsed.isna()
    if mask_failed2.any():
        try:
            tertiary = pd.to_datetime(
                time_series[mask_failed2], errors="coerce", format="%I:%M %p"
            )
            parsed[mask_failed2] = tertiary
        except Exception:
            pass

    return parsed


def _combine_date_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    """
    Combine parsed date and time Series into a single datetime Series.

    Args:
        date_series: Series of datetime64 dates.
        time_series: Series of datetime64 times (from _parse_time_column).

    Returns:
        Combined timestamp Series.
    """
    combined = []
    for date, time in zip(date_series, time_series):
        if pd.isna(date):
            combined.append(pd.NaT)
        elif pd.isna(time):
            combined.append(date)  # Use just the date if time is missing
        else:
            try:
                combined.append(
                    date.replace(
                        hour=time.hour,
                        minute=time.minute,
                        second=time.second
                    )
                )
            except Exception:
                combined.append(date)

    return pd.Series(combined, index=date_series.index)
