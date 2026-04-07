#!/usr/bin/env python3
"""
Compute daily clicks by SKU and region from the existing SQLite database.

Defaults are designed for the JD datasets in this repo, but you can override
table/column names via CLI flags if your schema differs.
"""

import argparse
import csv
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from project_paths import DATABASE_DIR, PROCESSED_DATA_DIR, ensure_dir, resolve_path

DEFAULT_DB_PATH = str(DATABASE_DIR / "click_orders_new.db")

# Column candidates
CLICK_TIME_COLS = ["request_time", "click_time", "time", "timestamp", "ts"]
REGION_COLS = ["region", "province", "city", "city_level", "area", "district"]


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [r[0] for r in cur.fetchall()]


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def pick_click_table(conn: sqlite3.Connection) -> Tuple[str, str]:
    tables = list_tables(conn)
    best = None
    best_score = -1
    best_time_col = None
    for t in tables:
        cols = table_columns(conn, t)
        if "user_ID" in cols and "sku_ID" in cols:
            time_col = next((c for c in CLICK_TIME_COLS if c in cols), None)
            score = 0
            if "click" in t.lower():
                score += 2
            if time_col:
                score += 2
            if score > best_score:
                best_score = score
                best = t
                best_time_col = time_col
    if not best or not best_time_col:
        raise RuntimeError(
            "Could not auto-detect click table or time column. "
            "Use --click-table and --click-time-col."
        )
    return best, best_time_col


def pick_user_table_and_region(conn: sqlite3.Connection) -> Tuple[str, str]:
    tables = list_tables(conn)
    best = None
    best_score = -1
    best_region_col = None
    for t in tables:
        cols = table_columns(conn, t)
        if "user_ID" not in cols:
            continue
        region_col = next((c for c in REGION_COLS if c in cols), None)
        score = 0
        if "user" in t.lower():
            score += 2
        if region_col:
            score += 2
        if score > best_score:
            best_score = score
            best = t
            best_region_col = region_col
    if not best or not best_region_col:
        raise RuntimeError(
            "Could not auto-detect user table or region column. "
            "Use --user-table and --region-col."
        )
    return best, best_region_col


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite database")
    p.add_argument(
        "--out",
        default=str(PROCESSED_DATA_DIR / "clicks_by_sku_day_region.csv"),
        help="Output CSV path",
    )
    p.add_argument("--click-table", default=None, help="Click table name")
    p.add_argument("--click-time-col", default=None, help="Datetime column in click table")
    p.add_argument("--user-table", default=None, help="User table name")
    p.add_argument("--region-col", default=None, help="Region column in user table")
    p.add_argument(
        "--include-unknown-user",
        action="store_true",
        help="Count user_ID='-' as a distinct user in unique user counts",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.db = str(resolve_path(args.db))
    args.out = str(resolve_path(args.out))
    ensure_dir(Path(args.out).parent)

    if not os.path.exists(args.db):
        raise SystemExit(f"DB not found: {args.db}")

    # Read-only connection to avoid lock contention
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=30)

    if args.click_table and args.click_time_col:
        click_table = args.click_table
        click_time_col = args.click_time_col
    else:
        click_table, click_time_col = pick_click_table(conn)

    if args.user_table and args.region_col:
        user_table = args.user_table
        region_col = args.region_col
    else:
        user_table, region_col = pick_user_table_and_region(conn)

    # Unique user expression
    if args.include_unknown_user:
        uniq_expr = "COUNT(DISTINCT c.user_ID)"
    else:
        uniq_expr = "COUNT(DISTINCT CASE WHEN c.user_ID IS NOT NULL AND c.user_ID != '-' THEN c.user_ID END)"

    query = f"""
        SELECT
            date(c.{click_time_col}) AS day,
            COALESCE(u.{region_col}, 'UNKNOWN') AS region,
            c.sku_ID AS sku_ID,
            {uniq_expr} AS unique_users,
            COUNT(*) AS total_clicks
        FROM {click_table} c
        LEFT JOIN {user_table} u
            ON c.user_ID = u.user_ID
        GROUP BY day, region, sku_ID
        ORDER BY day, region, sku_ID
    """

    cur = conn.execute(query)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["day", "region", "sku_ID", "unique_users", "total_clicks"])
        for row in cur:
            writer.writerow(row)

    conn.close()


if __name__ == "__main__":
    main()
