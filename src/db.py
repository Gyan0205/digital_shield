"""
db.py - PostgreSQL database connection manager
Digital Shield Ticket Anomaly Detection System

Uses psycopg2 only (no SQLAlchemy).
Credentials loaded from .env file via python-dotenv.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from src.utils import setup_logger

logger = setup_logger()

# Load environment variables from .env file
load_dotenv()


def get_connection() -> psycopg2.extensions.connection:
    """
    Establish and return a psycopg2 connection to the PostgreSQL database.
    All credentials are loaded from environment variables.

    Returns:
        psycopg2 connection object.

    Raises:
        EnvironmentError: If required env variables are missing.
        psycopg2.OperationalError: If connection fails.
    """
    required_vars = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}. "
            "Please copy .env.example to .env and fill in your credentials."
        )

    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 5432)),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=30,
            options="-c statement_timeout=120000",  # 120 seconds query timeout
        )
        logger.info("✅ Database connected successfully.")
        return conn

    except psycopg2.OperationalError as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise


def close_connection(conn: psycopg2.extensions.connection):
    """
    Safely close a psycopg2 database connection.

    Args:
        conn: An open psycopg2 connection object.
    """
    try:
        if conn and not conn.closed:
            conn.close()
            logger.info("🔒 Database connection closed.")
    except Exception as e:
        logger.warning(f"Warning while closing connection: {e}")


def test_connection() -> bool:
    """
    Test the database connection by executing a simple query.

    Returns:
        True if connection is successful, False otherwise.
    """
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            result = cur.fetchone()
        close_connection(conn)
        if result:
            logger.info("✅ Connection test passed.")
            return True
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        return False
    return False
