"""
app.py - Main entrypoint for Digital Shield Ticket Anomaly Detection System
==========================================================================

Run command:
    python app.py

Pipeline:
    1. Connect to PostgreSQL database
    2. Load tickets_1 data
    3. Preprocess data
    4. Run 8 rule-based anomaly detectors
    5. Run Isolation Forest ML anomaly detection
    6. Calculate risk scores
    7. Generate and export reports
    8. Print summary to console
"""

import sys
import time
from pathlib import Path

# Ensure 'src' is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logger, ensure_output_dirs, format_duration, dashboard_api_hook
from src.db import get_connection, close_connection
from src.loader import load_tickets, get_row_count
from src.preprocess import preprocess
from src.anomalies import run_all_anomaly_rules
from src.scoring import calculate_risk_scores, get_suspicious_df
from src.model import run_ml_anomaly_detection
from src.reports import generate_all_reports

logger = setup_logger()


def main():
    """
    Full anomaly detection pipeline orchestrator.
    """
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("  🛡️  DIGITAL SHIELD TICKET ANOMALY DETECTION — START")
    logger.info("=" * 60)

    ensure_output_dirs()

    conn = None
    try:
        # ──────────────────────────────────────────────────────────────
        # STEP 1: Connect to database
        # ──────────────────────────────────────────────────────────────
        logger.info("── STEP 1: Connecting to database ...")
        conn = get_connection()

        # Optional: log total row count before loading
        get_row_count(conn)

        # ──────────────────────────────────────────────────────────────
        # STEP 2: Load data
        # ──────────────────────────────────────────────────────────────
        logger.info("── STEP 2: Loading ticket data ...")
        df = load_tickets(conn)

        if df.empty:
            logger.error("❌ No data loaded from tickets_1. Exiting.")
            sys.exit(1)

        logger.info(f"   Rows loaded: {len(df):,}")

        # ──────────────────────────────────────────────────────────────
        # STEP 3: Preprocess
        # ──────────────────────────────────────────────────────────────
        logger.info("── STEP 3: Preprocessing data ...")
        df = preprocess(df)

        # ──────────────────────────────────────────────────────────────
        # STEP 4: Rule-based anomaly detection
        # ──────────────────────────────────────────────────────────────
        logger.info("── STEP 4: Running rule-based anomaly detection ...")
        df = run_all_anomaly_rules(df)

        # ──────────────────────────────────────────────────────────────
        # STEP 5: ML anomaly detection
        # ──────────────────────────────────────────────────────────────
        logger.info("── STEP 5: Running ML anomaly detection (Isolation Forest) ...")
        df = run_ml_anomaly_detection(df)

        # ──────────────────────────────────────────────────────────────
        # STEP 6: Risk scoring
        # ──────────────────────────────────────────────────────────────
        logger.info("── STEP 6: Calculating risk scores ...")
        df = calculate_risk_scores(df)

        suspicious_df = get_suspicious_df(df)

        # ──────────────────────────────────────────────────────────────
        # STEP 7: Generate reports
        # ──────────────────────────────────────────────────────────────
        logger.info("── STEP 7: Generating reports ...")
        generate_all_reports(df)

        # ──────────────────────────────────────────────────────────────
        # STEP 8: Future integration hooks
        # ──────────────────────────────────────────────────────────────
        _trigger_integration_hooks(df, suspicious_df)

    except EnvironmentError as e:
        logger.error(f"❌ Configuration error: {e}")
        logger.error("   Make sure your .env file is set up correctly (see .env.example).")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("⚠️  Pipeline interrupted by user.")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"❌ Unexpected error in pipeline: {e}")
        sys.exit(1)

    finally:
        if conn:
            close_connection(conn)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"  ✅ Pipeline completed in {format_duration(elapsed)}")
    logger.info("=" * 60)


def _trigger_integration_hooks(df, suspicious_df):
    """
    Fire all future integration hooks with current pipeline data.
    These are NOOPs in V1 — placeholders for future development.
    """
    # Dashboard API hook
    summary_payload = {
        "total_rows": len(df),
        "suspicious_rows": len(suspicious_df),
        "high_risk": int((suspicious_df["risk_level"] == "HIGH").sum()) if not suspicious_df.empty else 0,
        "medium_risk": int((suspicious_df["risk_level"] == "MEDIUM").sum()) if not suspicious_df.empty else 0,
    }
    dashboard_api_hook(summary_payload)

    # Email alert hook (only if HIGH risk rows exist)
    if not suspicious_df.empty:
        high_risk_count = int((suspicious_df["risk_level"] == "HIGH").sum())
        if high_risk_count > 0:
            from src.utils import email_alert_hook
            email_alert_hook(
                subject=f"[Digital Shield] {high_risk_count} HIGH RISK bookings detected",
                body=f"{high_risk_count} high risk booking anomalies were detected. Check outputs/ for details.",
                recipients=["security@railwayshield.gov.in"]
            )


if __name__ == "__main__":
    main()
