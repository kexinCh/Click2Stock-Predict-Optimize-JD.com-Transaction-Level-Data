#!/usr/bin/env python3
"""
Compute metrics linking clicks -> orders for (user, sku) using data already loaded in SQLite.
Outputs a CSV with per (user_id, sku): first_click_time, first_order_time, clicks_before_order, days_between.

Usage:
    python compute_click_order_metrics.py --db data/database/click_orders.db --out data/processed/click_to_order_metrics.csv
"""
import argparse
import csv
import sqlite3
from pathlib import Path

from project_paths import DATABASE_DIR, PROCESSED_DATA_DIR, ensure_dir, resolve_path


def run_sqlite_aggregation(db_path, out_csv, user_col, sku_col, click_time_col, order_time_col):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print('Computing first orders (SQL)...')
    cur.execute('DROP TABLE IF EXISTS first_orders')
    cur.execute(
        f"CREATE TABLE first_orders AS "
        f"SELECT {user_col}, {sku_col}, MIN({order_time_col}) AS first_order_time "
        f"FROM orders GROUP BY {user_col}, {sku_col}"
    )
    conn.commit()

    print('Aggregating clicks vs orders (SQL)...')
    cur.execute('DROP TABLE IF EXISTS results')
    cur.execute(f"""
    CREATE TABLE results AS
    SELECT
        o.{user_col} as user_id,
        o.{sku_col} as sku,
        MIN(c.{click_time_col}) AS first_click_time,
        o.first_order_time,
        SUM(CASE WHEN c.{click_time_col} < o.first_order_time THEN 1 ELSE 0 END) AS clicks_before_order,
        (julianday(o.first_order_time) - julianday(MIN(c.{click_time_col}))) AS days_between
    FROM first_orders o
    JOIN clicks c ON c.{user_col} = o.{user_col} AND c.{sku_col} = o.{sku_col}
    GROUP BY o.{user_col}, o.{sku_col}
    """)
    conn.commit()

    print('Exporting results to CSV...')
    cur.execute('SELECT user_id, sku, first_click_time, first_order_time, clicks_before_order, days_between FROM results')
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)

    conn.close()
    print('Results saved to', out_csv)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', default=str(DATABASE_DIR / 'click_orders.db'))
    p.add_argument('--out', default=str(PROCESSED_DATA_DIR / 'click_to_order_metrics.csv'))
    p.add_argument('--user_col', default='user_ID')
    p.add_argument('--sku_col', default='sku_ID')
    p.add_argument('--click_time_col', default='request_time', help='Timestamp column name in clicks table')
    p.add_argument('--order_time_col', default='order_time', help='Timestamp column name in orders table')
    args = p.parse_args()

    args.db = str(resolve_path(args.db))
    args.out = str(resolve_path(args.out))
    ensure_dir(Path(args.out).parent)

    run_sqlite_aggregation(
        args.db,
        args.out,
        args.user_col,
        args.sku_col,
        args.click_time_col,
        args.order_time_col,
    )


if __name__ == '__main__':
    main()
