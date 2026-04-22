"""
anomalies.py - Rule-based anomaly detection engine
Digital Shield Ticket Anomaly Detection System

Implements 8 distinct rule-based anomaly detectors.
Each detector returns a set of (user_id, pnrno) tuples that are flagged.
"""

import pandas as pd
import numpy as np
from src.utils import setup_logger

logger = setup_logger()

# ─────────────────────────────────────────────
# THRESHOLDS (tweak without changing logic)
# ─────────────────────────────────────────────
BULK_BOOKING_THRESHOLD = 5          # Rule 2: bookings per user per day
SAME_IP_THRESHOLD = 5               # Rule 3: bookings per IP per day
LAST_MINUTE_DAYS = 1                # Rule 4: journey_gap_days < this
RAPID_TXN_MINUTES = 10              # Rule 5: window in minutes
REPEATED_ROUTE_THRESHOLD = 3        # Rule 6: same route count
LARGE_GROUP_THRESHOLD = 6           # Rule 7: passengers per PNR
SAME_BANK_DAILY_THRESHOLD = 20      # Rule 8: transactions per bank per day


def run_all_anomaly_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all 8 rule-based anomaly detectors on the preprocessed DataFrame.

    Each rule appends a boolean flag column to the DataFrame.
    Returns the augmented DataFrame with all rule columns attached.

    Args:
        df (pd.DataFrame): Preprocessed tickets data.

    Returns:
        pd.DataFrame: DataFrame with anomaly flag columns added.
    """
    logger.info("🔍 Running 8 rule-based anomaly detectors ...")

    df = df.copy()

    df = rule_adult_with_many_minors(df)
    df = rule_bulk_booking_by_user(df)
    df = rule_same_ip_multiple_bookings(df)
    df = rule_last_minute_booking(df)
    df = rule_rapid_consecutive_transactions(df)
    df = rule_repeated_same_route(df)
    df = rule_large_group_size(df)
    df = rule_same_bank_high_frequency(df)

    total_flagged = (
        df[RULE_FLAG_COLUMNS].any(axis=1).sum()
    )
    logger.info(f"✅ Rule engine complete. Flagged rows: {total_flagged:,} (may overlap).")
    return df


# ─────────────────────────────────────────────
# RULE 1: Adult + Many Minors
# ─────────────────────────────────────────────

def rule_adult_with_many_minors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag PNR groups where exactly 1 adult is travelling with 3+ minors.

    Logic: Group by pnrno. Count adults (age >= 18) and minors (age < 18).
    If adults == 1 and minors >= 3, flag ALL rows of that PNR.
    """
    logger.debug("  Rule 1: Adult + Many Minors ...")

    df["_is_minor"] = df["age"].apply(lambda a: 1 if pd.notna(a) and a < 18 else 0)
    df["_is_adult"] = df["age"].apply(lambda a: 1 if pd.notna(a) and a >= 18 else 0)

    pnr_group = df.groupby("pnrno").agg(
        adults=("_is_adult", "sum"),
        minors=("_is_minor", "sum")
    ).reset_index()

    suspicious_pnr = pnr_group[
        (pnr_group["adults"] == 1) & (pnr_group["minors"] >= 3)
    ]["pnrno"].tolist()

    df["flag_adult_minors"] = df["pnrno"].isin(suspicious_pnr)

    count = df["flag_adult_minors"].sum()
    logger.info(f"    Rule 1 (Adult+Minors):          {count:>8,} rows flagged")

    # Clean helper columns
    df.drop(columns=["_is_minor", "_is_adult"], inplace=True)
    return df


# ─────────────────────────────────────────────
# RULE 2: Bulk Booking by Same User
# ─────────────────────────────────────────────

def rule_bulk_booking_by_user(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag users who make more than BULK_BOOKING_THRESHOLD bookings on the same day.
    """
    logger.debug("  Rule 2: Bulk Booking by User ...")

    df["_txn_date_only"] = df["txn_date"].dt.date

    daily_counts = (
        df.groupby(["user_id", "_txn_date_only"])
        .size()
        .reset_index(name="daily_count")
    )

    bulk_users = daily_counts[
        daily_counts["daily_count"] > BULK_BOOKING_THRESHOLD
    ][["user_id", "_txn_date_only"]]

    bulk_set = set(zip(bulk_users["user_id"], bulk_users["_txn_date_only"]))

    df["flag_bulk_booking"] = list(
        zip(df["user_id"], df["_txn_date_only"])
    ) if len(df) > 0 else []
    df["flag_bulk_booking"] = df.apply(
        lambda r: (r["user_id"], r["_txn_date_only"]) in bulk_set, axis=1
    )

    count = df["flag_bulk_booking"].sum()
    logger.info(f"    Rule 2 (Bulk Booking):          {count:>8,} rows flagged")

    df.drop(columns=["_txn_date_only"], inplace=True)
    return df


# ─────────────────────────────────────────────
# RULE 3: Same IP Multiple Bookings
# ─────────────────────────────────────────────

def rule_same_ip_multiple_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag IPs used for more than SAME_IP_THRESHOLD bookings on the same day.
    """
    logger.debug("  Rule 3: Same IP Multiple Bookings ...")

    df["_txn_date_only"] = df["txn_date"].dt.date

    ip_daily = (
        df.groupby(["ip_addrs", "_txn_date_only"])
        .size()
        .reset_index(name="ip_count")
    )

    hot_ips = ip_daily[
        ip_daily["ip_count"] > SAME_IP_THRESHOLD
    ][["ip_addrs", "_txn_date_only"]]

    hot_set = set(zip(hot_ips["ip_addrs"], hot_ips["_txn_date_only"]))

    df["flag_same_ip"] = df.apply(
        lambda r: (r["ip_addrs"], r["_txn_date_only"]) in hot_set, axis=1
    )

    count = df["flag_same_ip"].sum()
    logger.info(f"    Rule 3 (Same IP):               {count:>8,} rows flagged")

    df.drop(columns=["_txn_date_only"], inplace=True)
    return df


# ─────────────────────────────────────────────
# RULE 4: Last-Minute Booking
# ─────────────────────────────────────────────

def rule_last_minute_booking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag bookings where journey_gap_days < 1 (booking day of travel or in the past).
    """
    logger.debug("  Rule 4: Last-Minute Booking ...")

    df["flag_last_minute"] = df["journey_gap_days"] < LAST_MINUTE_DAYS

    count = df["flag_last_minute"].sum()
    logger.info(f"    Rule 4 (Last Minute):           {count:>8,} rows flagged")
    return df


# ─────────────────────────────────────────────
# RULE 5: Rapid Consecutive Transactions
# ─────────────────────────────────────────────

def rule_rapid_consecutive_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag users who make more than 1 booking within a RAPID_TXN_MINUTES window.

    Uses booking_timestamp sorted per user. If the gap between consecutive
    bookings is less than the threshold, both are flagged.
    """
    logger.debug("  Rule 5: Rapid Consecutive Transactions ...")

    if "booking_timestamp" not in df.columns or df["booking_timestamp"].isna().all():
        df["flag_rapid_txn"] = False
        logger.warning("    Rule 5 skipped: booking_timestamp unavailable.")
        return df

    df_sorted = df[["user_id", "booking_timestamp"]].copy()
    df_sorted = df_sorted.sort_values(["user_id", "booking_timestamp"])

    df_sorted["_prev_ts"] = df_sorted.groupby("user_id")["booking_timestamp"].shift(1)
    df_sorted["_gap_minutes"] = (
        (df_sorted["booking_timestamp"] - df_sorted["_prev_ts"])
        .dt.total_seconds()
        .div(60)
    )

    # Mark current row as rapid if gap to previous < threshold
    df_sorted["_is_rapid"] = df_sorted["_gap_minutes"] < RAPID_TXN_MINUTES

    # Also mark the previous row rapid (since it was booking just before)
    df_sorted["_prev_rapid"] = df_sorted.groupby("user_id")["_is_rapid"].shift(-1).fillna(False)
    df_sorted["flag_rapid_txn"] = df_sorted["_is_rapid"] | df_sorted["_prev_rapid"]

    df["flag_rapid_txn"] = df_sorted["flag_rapid_txn"].values

    count = df["flag_rapid_txn"].sum()
    logger.info(f"    Rule 5 (Rapid Txn):             {count:>8,} rows flagged")
    return df


# ─────────────────────────────────────────────
# RULE 6: Repeated Same Route
# ─────────────────────────────────────────────

def rule_repeated_same_route(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag users who repeatedly book the exact same from_stn → to_stn route
    more than REPEATED_ROUTE_THRESHOLD times.
    """
    logger.debug("  Rule 6: Repeated Same Route ...")

    route_counts = (
        df.groupby(["user_id", "from_stn", "to_stn"])
        .size()
        .reset_index(name="route_count")
    )

    repeated = route_counts[
        route_counts["route_count"] > REPEATED_ROUTE_THRESHOLD
    ][["user_id", "from_stn", "to_stn"]]

    repeated_set = set(
        zip(repeated["user_id"], repeated["from_stn"], repeated["to_stn"])
    )

    df["flag_repeated_route"] = df.apply(
        lambda r: (r["user_id"], r["from_stn"], r["to_stn"]) in repeated_set, axis=1
    )

    count = df["flag_repeated_route"].sum()
    logger.info(f"    Rule 6 (Repeated Route):        {count:>8,} rows flagged")
    return df


# ─────────────────────────────────────────────
# RULE 7: Large Group Size
# ─────────────────────────────────────────────

def rule_large_group_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag PNRs that have more than LARGE_GROUP_THRESHOLD passenger rows.
    """
    logger.debug("  Rule 7: Large Group Size ...")

    pnr_sizes = df.groupby("pnrno").size().reset_index(name="group_size")
    large_pnrs = pnr_sizes[pnr_sizes["group_size"] > LARGE_GROUP_THRESHOLD]["pnrno"].tolist()

    df["flag_large_group"] = df["pnrno"].isin(large_pnrs)

    count = df["flag_large_group"].sum()
    logger.info(f"    Rule 7 (Large Group):           {count:>8,} rows flagged")
    return df


# ─────────────────────────────────────────────
# RULE 8: Same Bank High Frequency
# ─────────────────────────────────────────────

def rule_same_bank_high_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag records where a single bank_name is used for an unusually high
    number of transactions on the same day (> SAME_BANK_DAILY_THRESHOLD).
    """
    logger.debug("  Rule 8: Same Bank High Frequency ...")

    df["_txn_date_only"] = df["txn_date"].dt.date

    bank_daily = (
        df.groupby(["bank_name", "_txn_date_only"])
        .size()
        .reset_index(name="bank_count")
    )

    hot_banks = bank_daily[
        bank_daily["bank_count"] > SAME_BANK_DAILY_THRESHOLD
    ][["bank_name", "_txn_date_only"]]

    hot_set = set(zip(hot_banks["bank_name"], hot_banks["_txn_date_only"]))

    df["flag_same_bank_freq"] = df.apply(
        lambda r: (r["bank_name"], r["_txn_date_only"]) in hot_set, axis=1
    )

    count = df["flag_same_bank_freq"].sum()
    logger.info(f"    Rule 8 (Same Bank Freq):        {count:>8,} rows flagged")

    df.drop(columns=["_txn_date_only"], inplace=True)
    return df


# ─────────────────────────────────────────────
# EXPORTED CONSTANTS
# ─────────────────────────────────────────────

# List of all rule flag column names (used by scoring.py)
RULE_FLAG_COLUMNS = [
    "flag_adult_minors",
    "flag_bulk_booking",
    "flag_same_ip",
    "flag_last_minute",
    "flag_rapid_txn",
    "flag_repeated_route",
    "flag_large_group",
    "flag_same_bank_freq",
]
