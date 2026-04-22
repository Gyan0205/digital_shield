"""
model.py - ML anomaly detection using Isolation Forest
Digital Shield Ticket Anomaly Detection System

Builds engineered features from the preprocessed DataFrame and applies
scikit-learn's IsolationForest to detect statistically unusual booking records.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.utils import setup_logger

logger = setup_logger()

# ─────────────────────────────────────────────
# ISOLATION FOREST CONFIGURATION
# ─────────────────────────────────────────────
CONTAMINATION = 0.05        # Assumed proportion of anomalies (5%)
N_ESTIMATORS = 100          # Number of trees
RANDOM_STATE = 42           # For reproducibility
N_JOBS = -1                 # Use all CPU cores


def run_ml_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Isolation Forest ML anomaly detection on engineered features.

    Steps:
        1. Engineer six numeric features from the DataFrame
        2. Scale features with StandardScaler
        3. Fit Isolation Forest
        4. Append ml_anomaly_flag column (True = anomaly detected)

    Args:
        df (pd.DataFrame): Preprocessed and rule-scored DataFrame.

    Returns:
        pd.DataFrame: Original DataFrame with 'ml_anomaly_flag' column added.
    """
    logger.info("🤖 Running Isolation Forest ML anomaly detection ...")

    df = df.copy()

    # ── Feature Engineering ────────────────────────────────────────────
    features_df = _engineer_features(df)

    if features_df is None or features_df.empty:
        logger.warning("⚠️  Could not engineer ML features. Skipping ML detection.")
        df["ml_anomaly_flag"] = False
        return df

    feature_columns = features_df.columns.tolist()
    logger.debug(f"  Feature columns: {feature_columns}")

    # Fill remaining NaNs with 0 before scaling
    features_df = features_df.fillna(0)

    # ── Scale Features ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    # ── Fit Isolation Forest ───────────────────────────────────────────
    model = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    logger.info("  Training Isolation Forest ...")
    predictions = model.fit_predict(X_scaled)

    # IsolationForest returns: -1 = anomaly, 1 = normal
    df["ml_anomaly_flag"] = predictions == -1

    ml_anomaly_count = df["ml_anomaly_flag"].sum()
    logger.info(
        f"✅ ML detection complete. "
        f"Flagged {ml_anomaly_count:,} / {len(df):,} rows as anomalous "
        f"({100 * ml_anomaly_count / max(len(df), 1):.2f}%)."
    )

    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Engineer six numeric features for the Isolation Forest model.

    Features:
        1. bookings_per_user    — how many bookings each user makes
        2. same_ip_count        — how many bookings share the same IP
        3. group_size           — how many passengers share same PNR
        4. minors_count         — how many minors share same PNR
        5. journey_gap_days     — days between booking and journey
        6. rapid_txn_count      — how many rapid bookings the user made

    Returns:
        pd.DataFrame of engineered features, or None on error.
    """
    try:
        feat = pd.DataFrame(index=df.index)

        # 1. Bookings per user
        bpu = df.groupby("user_id")["pnrno"].transform("count")
        feat["bookings_per_user"] = bpu

        # 2. Same IP count (global, not daily)
        ip_counts = df.groupby("ip_addrs")["pnrno"].transform("count")
        feat["same_ip_count"] = ip_counts

        # 3. Group size (passengers per PNR)
        group_size = df.groupby("pnrno")["user_id"].transform("count")
        feat["group_size"] = group_size

        # 4. Minors count per PNR
        df["_is_minor_ml"] = (df["age"] < 18).astype(int)
        minors_per_pnr = df.groupby("pnrno")["_is_minor_ml"].transform("sum")
        feat["minors_count"] = minors_per_pnr
        df.drop(columns=["_is_minor_ml"], inplace=True)

        # 5. Journey gap days
        feat["journey_gap_days"] = df["journey_gap_days"].clip(lower=0)

        # 6. Rapid transaction count per user
        if "booking_timestamp" in df.columns and not df["booking_timestamp"].isna().all():
            feat["rapid_txn_count"] = _compute_rapid_count(df)
        else:
            feat["rapid_txn_count"] = 0

        return feat

    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        return None


def _compute_rapid_count(df: pd.DataFrame) -> pd.Series:
    """
    For each user, count how many transactions occurred within a 10-minute window.

    Uses a rolling approach: sort by user + timestamp, compute time diffs,
    and count rows where the gap to the previous booking < 10 minutes.

    Returns:
        pd.Series: rapid_txn_count per row (indexed to match df.index).
    """
    WINDOW_MINUTES = 10

    df_ts = df[["user_id", "booking_timestamp"]].copy()
    df_ts = df_ts.sort_values(["user_id", "booking_timestamp"])

    # Gap to previous booking by same user, in minutes
    df_ts["_prev_ts"] = df_ts.groupby("user_id")["booking_timestamp"].shift(1)
    df_ts["_gap_min"] = (
        (df_ts["booking_timestamp"] - df_ts["_prev_ts"])
        .dt.total_seconds()
        .div(60)
        .fillna(np.inf)
    )
    df_ts["_rapid"] = (df_ts["_gap_min"] < WINDOW_MINUTES).astype(int)

    # Sum of rapid bookings per user
    user_rapid = df_ts.groupby("user_id")["_rapid"].transform("sum")
    df_ts["rapid_txn_count"] = user_rapid

    # Return aligned to original index
    return df_ts["rapid_txn_count"].reindex(df.index).fillna(0)
