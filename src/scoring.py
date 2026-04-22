"""
scoring.py - Risk scoring engine
Digital Shield Ticket Anomaly Detection System

Assigns numeric risk scores and risk levels based on triggered anomaly rules.
Also collects the human-readable list of triggered reasons per row.
"""

import pandas as pd
from src.utils import setup_logger
from src.anomalies import RULE_FLAG_COLUMNS

logger = setup_logger()

# ─────────────────────────────────────────────
# RULE → SCORE WEIGHTS
# ─────────────────────────────────────────────

RULE_SCORES = {
    "flag_adult_minors":    35,
    "flag_bulk_booking":    20,
    "flag_same_ip":         20,
    "flag_last_minute":     15,
    "flag_rapid_txn":       15,
    "flag_repeated_route":  10,
    "flag_large_group":     20,
    "flag_same_bank_freq":  10,
}

# Human-readable names for each rule flag
RULE_LABELS = {
    "flag_adult_minors":    "Adult with 3+ Minors",
    "flag_bulk_booking":    "Bulk Booking (>5/day)",
    "flag_same_ip":         "Same IP Multiple Bookings",
    "flag_last_minute":     "Last-Minute Booking (<1 day)",
    "flag_rapid_txn":       "Rapid Consecutive Transactions",
    "flag_repeated_route":  "Repeated Same Route",
    "flag_large_group":     "Large Group (>6 passengers)",
    "flag_same_bank_freq":  "High Bank Transaction Frequency",
}

# Risk level thresholds
RISK_THRESHOLDS = {
    "HIGH":   60,
    "MEDIUM": 30,
    "LOW":    0,
}


def calculate_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate risk_score, risk_level, and triggered_rules for every row.

    For each row:
      - Sum the weights of all triggered rule flags → risk_score
      - Map score to LOW / MEDIUM / HIGH → risk_level
      - Collect list of triggered rule labels  → triggered_rules

    Args:
        df (pd.DataFrame): DataFrame with all anomaly flag columns present.

    Returns:
        pd.DataFrame: DataFrame with risk_score, risk_level, triggered_rules added.
    """
    logger.info("📊 Calculating risk scores ...")

    df = df.copy()

    # Ensure all flag columns exist (fill False if a rule was skipped)
    for flag_col in RULE_FLAG_COLUMNS:
        if flag_col not in df.columns:
            df[flag_col] = False
            logger.warning(f"  Missing flag column '{flag_col}' — defaulted to False.")

    # ── Risk Score ─────────────────────────────────────────────────────
    df["risk_score"] = 0
    for flag_col, weight in RULE_SCORES.items():
        df["risk_score"] += df[flag_col].astype(int) * weight

    logger.debug("  risk_score column computed.")

    # ── Risk Level ─────────────────────────────────────────────────────
    df["risk_level"] = df["risk_score"].apply(_score_to_level)
    logger.debug("  risk_level column computed.")

    # ── Triggered Rules ────────────────────────────────────────────────
    df["triggered_rules"] = df.apply(_build_triggered_list, axis=1)
    logger.debug("  triggered_rules column computed.")

    # Summary log
    level_counts = df["risk_level"].value_counts()
    logger.info(f"  Risk distribution: {level_counts.to_dict()}")

    logger.info("✅ Risk scoring complete.")
    return df


def _score_to_level(score: int) -> str:
    """Map a numeric risk score to a human-readable level string."""
    if score >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif score >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    else:
        return "LOW"


def _build_triggered_list(row: pd.Series) -> str:
    """
    Build a pipe-separated string of rule labels that fired for this row.

    Returns:
        str: e.g. "Bulk Booking (>5/day) | Same IP Multiple Bookings"
    """
    triggered = [
        RULE_LABELS[flag]
        for flag in RULE_FLAG_COLUMNS
        if flag in row and bool(row[flag])
    ]
    return " | ".join(triggered) if triggered else "None"


def get_suspicious_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame to only include rows with at least one triggered rule.

    Args:
        df (pd.DataFrame): Scored dataframe.

    Returns:
        pd.DataFrame: Subset with risk_score > 0.
    """
    suspicious = df[df["risk_score"] > 0].copy()
    suspicious = suspicious.sort_values("risk_score", ascending=False)
    logger.info(f"🚨 Suspicious rows isolated: {len(suspicious):,} out of {len(df):,} total.")
    return suspicious
