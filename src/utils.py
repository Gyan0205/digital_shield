"""
utils.py - Shared helpers and logger setup
Digital Shield Ticket Anomaly Detection System
"""

import logging
import os
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────
# LOGGER SETUP
# ─────────────────────────────────────────────

def setup_logger(name: str = "digital_shield", log_file: str = "logs/app.log") -> logging.Logger:
    """
    Set up and return a configured logger.
    Logs to both console and rotating log file.
    """
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ─────────────────────────────────────────────
# GENERAL HELPERS
# ─────────────────────────────────────────────

def get_timestamp() -> str:
    """Return the current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_output_dirs():
    """Create all required output directories if they don't exist."""
    directories = ["outputs", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def format_duration(seconds: float) -> str:
    """Format elapsed seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.2f}s"


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Safely divide two numbers, returning fallback on zero division."""
    try:
        return numerator / denominator if denominator != 0 else fallback
    except Exception:
        return fallback


# ─────────────────────────────────────────────
# FUTURE INTEGRATION HOOKS
# ─────────────────────────────────────────────

def cctv_integration_hook(event_data: dict):
    """
    PLACEHOLDER: Future hook for CCTV anomaly integration.
    This function will receive real-time event data from CCTV feeds
    and correlate suspicious physical behavior with ticket anomalies.

    Args:
        event_data (dict): CCTV event payload (camera_id, timestamp, alert_type, etc.)
    """
    logger = logging.getLogger("digital_shield")
    logger.debug(f"[CCTV HOOK] Event received (integration pending): {event_data}")
    # Future: call CCTV anomaly classifier here


def email_alert_hook(subject: str, body: str, recipients: list):
    """
    PLACEHOLDER: Future hook for Email alert notifications.
    Will send automated alert emails to security/admin staff
    when HIGH risk anomalies are detected.

    Args:
        subject (str): Email subject line.
        body (str): Email body content.
        recipients (list): List of email addresses.
    """
    logger = logging.getLogger("digital_shield")
    logger.debug(f"[EMAIL HOOK] Alert pending (integration pending): subject='{subject}', to={recipients}")
    # Future: integrate smtplib or SendGrid/Mailgun API here


def dashboard_api_hook(summary_payload: dict):
    """
    PLACEHOLDER: Future hook for Dashboard Frontend API.
    Will POST anomaly summary data to a REST API or WebSocket
    so a live dashboard can display real-time alerts.

    Args:
        summary_payload (dict): JSON-serialisable summary data.
    """
    logger = logging.getLogger("digital_shield")
    logger.debug(f"[DASHBOARD HOOK] Payload ready (integration pending): keys={list(summary_payload.keys())}")
    # Future: use requests.post() to push to FastAPI / Flask endpoint here
