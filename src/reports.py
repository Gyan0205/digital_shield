"""
reports.py - Report generation and export
Digital Shield Ticket Anomaly Detection System

Generates CSV, JSON, and summary exports from the final scored DataFrame.
Also prints a formatted console summary.
"""

import json
import pandas as pd
from pathlib import Path
from src.utils import setup_logger

logger = setup_logger()

# Output directory
OUTPUT_DIR = Path("outputs")

# Columns that form the final output dataset
FINAL_COLUMNS = [
    "user_id",
    "pnrno",
    "from_stn",
    "to_stn",
    "txn_date",
    "jrny_date",
    "risk_score",
    "risk_level",
    "triggered_rules",
    "ml_anomaly_flag",
]


def generate_all_reports(df: pd.DataFrame) -> None:
    """
    Run the full report generation pipeline.

    Exports:
        outputs/suspicious_bookings.csv
        outputs/suspicious_bookings.json
        outputs/dashboard_summary.csv

    Also prints a summary to the console.

    Args:
        df (pd.DataFrame): Fully scored and annotated DataFrame.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("📝 Generating reports ...")

    # Build the final output dataset
    final_df = _build_final_dataset(df)

    # Filter suspicious rows (risk_score > 0)
    suspicious_df = final_df[final_df["risk_score"] > 0].copy()
    suspicious_df = suspicious_df.sort_values("risk_score", ascending=False)

    # Export reports
    _export_csv(suspicious_df)
    _export_json(suspicious_df)
    _export_dashboard_summary(suspicious_df)

    # Print summary to console
    print_console_summary(df, suspicious_df)

    logger.info("✅ All reports generated successfully.")


def _build_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and return only the final output columns from the scored DataFrame.
    Missing columns are filled with appropriate defaults.

    Args:
        df (pd.DataFrame): Fully scored DataFrame.

    Returns:
        pd.DataFrame: Cleaned final output DataFrame.
    """
    output_df = pd.DataFrame()

    for col in FINAL_COLUMNS:
        if col in df.columns:
            output_df[col] = df[col]
        else:
            # Fill missing columns with sensible defaults
            if col in ("risk_score",):
                output_df[col] = 0
            elif col in ("ml_anomaly_flag",):
                output_df[col] = False
            elif col in ("triggered_rules",):
                output_df[col] = "None"
            elif col in ("risk_level",):
                output_df[col] = "LOW"
            else:
                output_df[col] = None

    return output_df


def _export_csv(df: pd.DataFrame) -> None:
    """Export suspicious bookings to CSV."""
    path = OUTPUT_DIR / "suspicious_bookings.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"  📄 CSV exported: {path}  ({len(df):,} rows)")


def _export_json(df: pd.DataFrame) -> None:
    """Export suspicious bookings to JSON (records orientation)."""
    path = OUTPUT_DIR / "suspicious_bookings.json"

    # Convert timestamps to strings for JSON serialisation
    df_json = df.copy()
    for col in df_json.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        df_json[col] = df_json[col].astype(str)

    records = df_json.to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)

    logger.info(f"  📄 JSON exported: {path}  ({len(df):,} records)")


def _export_dashboard_summary(df: pd.DataFrame) -> None:
    """
    Export a dashboard-friendly summary CSV with aggregated stats.

    Columns: risk_level, count, pct_of_total
    """
    path = OUTPUT_DIR / "dashboard_summary.csv"

    if df.empty:
        summary = pd.DataFrame(columns=["risk_level", "count", "pct_of_total"])
        summary.to_csv(path, index=False)
        logger.info(f"  📄 Dashboard summary exported (empty): {path}")
        return

    total = len(df)
    summary = (
        df.groupby("risk_level")
        .size()
        .reset_index(name="count")
        .assign(pct_of_total=lambda x: (100 * x["count"] / total).round(2))
    )

    # Ensure all three levels are present
    for level in ["HIGH", "MEDIUM", "LOW"]:
        if level not in summary["risk_level"].values:
            summary = pd.concat(
                [summary, pd.DataFrame([{"risk_level": level, "count": 0, "pct_of_total": 0.0}])],
                ignore_index=True
            )

    # Add top-5 risky users
    top_users = (
        df.groupby("user_id")["risk_score"]
        .sum()
        .nlargest(5)
        .reset_index()
        .rename(columns={"risk_score": "total_risk_score"})
    )
    top_users_path = OUTPUT_DIR / "top_risky_users.csv"
    top_users.to_csv(top_users_path, index=False)
    logger.info(f"  📄 Top risky users: {top_users_path}")

    # Add top-5 risky routes
    top_routes = (
        df.groupby(["from_stn", "to_stn"])["risk_score"]
        .sum()
        .nlargest(5)
        .reset_index()
        .rename(columns={"risk_score": "total_risk_score"})
    )
    top_routes_path = OUTPUT_DIR / "top_risky_routes.csv"
    top_routes.to_csv(top_routes_path, index=False)
    logger.info(f"  📄 Top risky routes: {top_routes_path}")

    summary.to_csv(path, index=False)
    logger.info(f"  📄 Dashboard summary exported: {path}")


def print_console_summary(full_df: pd.DataFrame, suspicious_df: pd.DataFrame) -> None:
    """
    Print a formatted console summary with key statistics.

    Args:
        full_df (pd.DataFrame): Complete dataset (all rows).
        suspicious_df (pd.DataFrame): Subset with risk_score > 0.
    """
    total_rows = len(full_df)
    suspicious_rows = len(suspicious_df)
    high_risk = (suspicious_df["risk_level"] == "HIGH").sum() if "risk_level" in suspicious_df else 0

    # Top 5 risky users
    top_users = pd.DataFrame()
    if not suspicious_df.empty and "user_id" in suspicious_df.columns and "risk_score" in suspicious_df.columns:
        top_users = (
            suspicious_df.groupby("user_id")["risk_score"]
            .sum()
            .nlargest(5)
            .reset_index()
        )

    # Top 5 risky routes
    top_routes = pd.DataFrame()
    if (not suspicious_df.empty
            and "from_stn" in suspicious_df.columns
            and "to_stn" in suspicious_df.columns
            and "risk_score" in suspicious_df.columns):
        top_routes = (
            suspicious_df.groupby(["from_stn", "to_stn"])["risk_score"]
            .sum()
            .nlargest(5)
            .reset_index()
        )

    separator = "=" * 60

    print(f"\n{separator}")
    print("  [DIGITAL SHIELD] ANOMALY DETECTION SUMMARY")
    print(separator)
    print(f"  {'Total rows scanned:':<35} {total_rows:>10,}")
    print(f"  {'Suspicious rows found:':<35} {suspicious_rows:>10,}")
    print(f"  {'High risk count:':<35} {high_risk:>10,}")
    print(separator)

    if not top_users.empty:
        print("\n  [!!] Top 5 Risky Users:")
        for _, row in top_users.iterrows():
            print(f"     User {row['user_id']:>20}  |  Score: {int(row['risk_score'])}")

    if not top_routes.empty:
        print("\n  [ROUTE] Top 5 Risky Routes:")
        for _, row in top_routes.iterrows():
            route = f"{row['from_stn']} -> {row['to_stn']}"
            print(f"     {route:<40}  |  Score: {int(row['risk_score'])}")

    print(f"\n  [OUTPUT] Reports saved in: outputs/")
    print(separator + "\n")
