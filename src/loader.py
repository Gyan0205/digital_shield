"""
loader.py - Data loading from tickets_1 PostgreSQL table
Digital Shield Ticket Anomaly Detection System

Efficiently loads data using server-side cursors for large datasets (900k+ rows).
"""

import pandas as pd
import psycopg2
from src.utils import setup_logger

logger = setup_logger()

# Columns expected in the tickets_1 table
EXPECTED_COLUMNS = [
    "user_id", "psgn_name", "train_number", "cls", "txn_date",
    "txn_time", "ip_addrs", "jrny_date", "pnrno", "from_stn",
    "to_stn", "age", "sex", "quota", "coach_no_seat_no",
    "txntype", "bank_name", "txn_no"
]

TABLE_NAME = "tickets_1"
CHUNK_SIZE = 50_000  # Rows per fetch batch for memory efficiency


def load_tickets(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    """
    Load all rows from the tickets_1 table into a pandas DataFrame.

    Uses a named server-side cursor so large result sets are streamed
    in chunks rather than loaded all at once (handles 900k+ rows safely).

    Args:
        conn: An open psycopg2 database connection.

    Returns:
        pd.DataFrame: All rows from tickets_1.

    Raises:
        Exception: On any database query failure.
    """
    logger.info(f"📥 Loading data from table: {TABLE_NAME} ...")

    query = f"SELECT * FROM {TABLE_NAME};"

    try:
        chunks = []

        # Named cursor = server-side cursor — streams data in batches
        with conn.cursor(name="ticket_loader_cursor") as cur:
            cur.execute(query)

            while True:
                rows = cur.fetchmany(CHUNK_SIZE)
                if not rows:
                    break

                col_names = [desc[0] for desc in cur.description]
                chunk_df = pd.DataFrame(rows, columns=col_names)
                chunks.append(chunk_df)
                logger.debug(f"  Fetched chunk of {len(chunk_df)} rows ...")

        if not chunks:
            logger.warning(f"⚠️  Table '{TABLE_NAME}' returned zero rows.")
            return pd.DataFrame(columns=EXPECTED_COLUMNS)

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"✅ Loaded {len(df):,} rows from '{TABLE_NAME}'.")
        return df

    except psycopg2.Error as e:
        logger.error(f"❌ Failed to load data from '{TABLE_NAME}': {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during data loading: {e}")
        raise


def load_tickets_sample(conn: psycopg2.extensions.connection, limit: int = 1000) -> pd.DataFrame:
    """
    Load a small sample of rows from tickets_1 for testing/development.

    Args:
        conn: An open psycopg2 database connection.
        limit (int): Maximum number of rows to return.

    Returns:
        pd.DataFrame: Sample rows from tickets_1.
    """
    logger.info(f"📥 Loading sample of {limit} rows from '{TABLE_NAME}' ...")

    query = f"SELECT * FROM {TABLE_NAME} LIMIT %s;"

    try:
        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]

        df = pd.DataFrame(rows, columns=col_names)
        logger.info(f"✅ Sample loaded: {len(df):,} rows.")
        return df

    except psycopg2.Error as e:
        logger.error(f"❌ Failed to load sample data: {e}")
        raise


def get_row_count(conn: psycopg2.extensions.connection) -> int:
    """
    Quickly fetch the total row count from tickets_1.

    Args:
        conn: An open psycopg2 database connection.

    Returns:
        int: Number of rows in the table.
    """
    query = f"SELECT COUNT(*) FROM {TABLE_NAME};"
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            count = cur.fetchone()[0]
        logger.info(f"📊 Total rows in '{TABLE_NAME}': {count:,}")
        return count
    except Exception as e:
        logger.error(f"❌ Failed to count rows: {e}")
        return 0
