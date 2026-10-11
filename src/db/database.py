"""SQLite helpers: connect, initialise schema, and read/write DataFrames.

Writers are idempotent: `replace_table` DELETEs existing rows then appends,
preserving the explicit schema & indexes defined in schema.sql (unlike
``DataFrame.to_sql(if_exists="replace")`` which would drop them).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.logging_utils import get_logger

log = get_logger(__name__)
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection, schema_path: Path = SCHEMA_PATH) -> None:
    with open(schema_path, "r", encoding="utf-8") as fh:
        conn.executescript(fh.read())
    conn.commit()
    log.info("Database schema initialised (%s)", schema_path.name)


def write_df(conn: sqlite3.Connection, df: pd.DataFrame, table: str,
             if_exists: str = "append") -> int:
    """Append a DataFrame to an existing table (columns must be a subset)."""
    if df is None or df.empty:
        log.warning("write_df: empty frame for table '%s' — nothing written", table)
        return 0
    df.to_sql(table, conn, if_exists=if_exists, index=False)
    conn.commit()
    log.info("Wrote %d rows to '%s'", len(df), table)
    return len(df)


def replace_table(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> int:
    """Idempotently replace a table's contents, keeping its schema & indexes."""
    conn.execute(f"DELETE FROM {table}")
    conn.commit()
    return write_df(conn, df, table, if_exists="append")


def read_df(conn: sqlite3.Connection, query: str,
            params: Iterable | None = None) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params)


def table_count(conn: sqlite3.Connection, table: str) -> int:
    cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cur.fetchone()[0])
