#!/usr/bin/env python3
"""
Compute per-user click behavior metrics from a SQLite database and output a CSV.

Usage:
    python user_click_behavior_metrics.py --db data/database/click_orders.db --out data/processed/user_click_behavior.csv
"""
import argparse
import csv
import sqlite3
import threading
import time
from pathlib import Path

from project_paths import DATABASE_DIR, PROCESSED_DATA_DIR, ensure_dir, resolve_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', default=str(DATABASE_DIR / 'click_orders.db'))
    p.add_argument('--out', default=str(PROCESSED_DATA_DIR / 'user_click_behavior.csv'))
    p.add_argument('--user_col', default='user_ID')
    p.add_argument('--sku_col', default='sku_ID')
    p.add_argument('--click_time_col', default='request_time', help='Timestamp column name in clicks table')
    p.add_argument('--click_channel_col', default='channel', help='Channel column name in clicks table')
    args = p.parse_args()
    args.db = str(resolve_path(args.db))
    args.out = str(resolve_path(args.out))
    ensure_dir(Path(args.out).parent)

    # set a long connection timeout so this process will wait for locks
    conn = sqlite3.connect(args.db, timeout=600)
    cur = conn.cursor()

    # Ask SQLite to wait (milliseconds) when DB is locked, to avoid failing immediately
    try:
        cur.execute("PRAGMA busy_timeout = 600000")
    except Exception:
        pass

    print('Connected to DB:', args.db)

    # quick sanity check: ensure clicks table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='clicks'")
    if cur.fetchone() is None:
        print('ERROR: table "clicks" not found in', args.db)
        conn.close()
        return

    print('Computing per-user click behavior (SQLite)...')
    print('Dropping existing `user_behavior` table if it exists...')
    cur.execute('DROP TABLE IF EXISTS user_behavior')
    print('Dropped existing table (if present).')

    user_behavior_sql = f"""
    CREATE TABLE user_behavior AS
    WITH click_base AS (
        SELECT
            {args.user_col} AS user_id,
            {args.sku_col} AS sku_id,
            {args.click_channel_col} AS channel,
            datetime({args.click_time_col}) AS ts,
            date({args.click_time_col}) AS click_date,
            CAST(strftime('%H', {args.click_time_col}) AS INT) AS hour,
            ((CAST(strftime('%w', {args.click_time_col}) AS INT) + 6) % 7) AS dow
        FROM clicks
        WHERE {args.user_col} IS NOT NULL AND {args.user_col} != '-'
    ),
    totals AS (
        SELECT
            user_id,
            COUNT(*) AS total_clicks,
            COUNT(DISTINCT click_date) AS active_days,
            MIN(ts) AS first_click_time,
            MAX(ts) AS last_click_time,
            MIN(click_date) AS first_click_date,
            MAX(click_date) AS last_click_date,
            COUNT(DISTINCT sku_id) AS unique_skus,
            COUNT(DISTINCT channel) AS unique_channels
        FROM click_base
        GROUP BY user_id
    ),
    sku_rank AS (
        SELECT user_id, sku_id, COUNT(*) AS cnt,
               ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC, sku_id) AS rn
        FROM click_base
        GROUP BY user_id, sku_id
    ),
    channel_rank AS (
        SELECT user_id, channel, COUNT(*) AS cnt,
               ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC, channel) AS rn
        FROM click_base
        GROUP BY user_id, channel
    ),
    hour_rank AS (
        SELECT user_id, hour, COUNT(*) AS cnt,
               ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC, hour) AS rn
        FROM click_base
        GROUP BY user_id, hour
    ),
    dow_rank AS (
        SELECT user_id, dow, COUNT(*) AS cnt,
               ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC, dow) AS rn
        FROM click_base
        GROUP BY user_id, dow
    )
    SELECT
        t.user_id,
        t.total_clicks,
        t.active_days,
        ROUND(CAST(t.total_clicks AS REAL) / NULLIF(t.active_days, 0), 4) AS avg_clicks_per_day,
        t.first_click_time,
        t.last_click_time,
        (julianday(t.last_click_date) - julianday(t.first_click_date) + 1) AS active_span_days,
        t.unique_skus,
        s.sku_id AS top_sku,
        s.cnt AS top_sku_clicks,
        ROUND(CAST(s.cnt AS REAL) / NULLIF(t.total_clicks, 0), 4) AS top_sku_share,
        t.unique_channels,
        c.channel AS top_channel,
        c.cnt AS top_channel_clicks,
        ROUND(CAST(c.cnt AS REAL) / NULLIF(t.total_clicks, 0), 4) AS top_channel_share,
        h.hour AS peak_hour,
        h.cnt AS peak_hour_clicks,
        ROUND(CAST(h.cnt AS REAL) / NULLIF(t.total_clicks, 0), 4) AS peak_hour_share,
        CASE d.dow
            WHEN 0 THEN 'Mon'
            WHEN 1 THEN 'Tue'
            WHEN 2 THEN 'Wed'
            WHEN 3 THEN 'Thu'
            WHEN 4 THEN 'Fri'
            WHEN 5 THEN 'Sat'
            WHEN 6 THEN 'Sun'
            ELSE NULL
        END AS peak_dow,
        d.cnt AS peak_dow_clicks,
        ROUND(CAST(d.cnt AS REAL) / NULLIF(t.total_clicks, 0), 4) AS peak_dow_share
    FROM totals t
    LEFT JOIN sku_rank s ON s.user_id = t.user_id AND s.rn = 1
    LEFT JOIN channel_rank c ON c.user_id = t.user_id AND c.rn = 1
    LEFT JOIN hour_rank h ON h.user_id = t.user_id AND h.rn = 1
    LEFT JOIN dow_rank d ON d.user_id = t.user_id AND d.rn = 1
    """
    # PRAGMA tuning to improve performance for large operations
    try:
        print('Applying PRAGMA tuning (may speed up heavy queries)...')
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.execute("PRAGMA cache_size=100000")
        print('PRAGMA tuning applied.')
    except Exception:
        print('Warning: could not set some PRAGMA values; continuing.')

    # Create helpful indexes if they don't exist (these may take time but speed GROUP/BY)
    try:
        print('Ensuring indexes on click columns (this may take a while)...')
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_clicks_user ON clicks({args.user_col})")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_clicks_sku ON clicks({args.sku_col})")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_clicks_time ON clicks({args.click_time_col})")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_clicks_channel ON clicks({args.click_channel_col})")
        conn.commit()
        print('Index creation commands issued (indexes may be building in background).')
    except Exception:
        print('Warning: index creation failed or is unsupported; continuing without indexes.')

    # Heartbeat thread to print progress while the heavy CREATE TABLE runs
    stop_event = threading.Event()

    def heartbeat():
        i = 0
        while not stop_event.is_set():
            i += 1
            print(f'Working on user_behavior CREATE... heartbeat {i} (sleeping 60s)')
            # wait up to 60s but exit early if stop_event set
            stop_event.wait(60)

    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    print('Starting heartbeat thread to report progress every 60s...')
    hb_thread.start()

    print('Executing CREATE TABLE for `user_behavior` (this may take a while)...')
    cur.execute(user_behavior_sql)
    conn.commit()
    stop_event.set()
    hb_thread.join()
    print('Finished creating `user_behavior` table.')

    print('Exporting user behavior to CSV (streaming)...')
    cur.execute('SELECT * FROM user_behavior')
    cols = [d[0] for d in cur.description]
    exported = 0
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in cur:
            writer.writerow(row)
            exported += 1
            if exported % 100000 == 0:
                print(f'Exported {exported} rows...')

    conn.close()
    print(f'User behavior metrics saved to {args.out} ({exported} rows exported)')


if __name__ == '__main__':
    main()
