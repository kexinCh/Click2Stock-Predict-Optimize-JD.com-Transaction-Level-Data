#!/usr/bin/env python3
"""
Load JD CSVs into a single SQLite database.

Usage:
    python load_jd_data_to_db.py --db data/database/click_orders.db \\
        --clicks data/raw/JD_click_data.csv \\
        --orders data/raw/JD_order_data.csv \\
        --users data/raw/JD_user_data.csv
"""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from project_paths import DATABASE_DIR, RAW_DATA_DIR, ensure_dir, resolve_path


def import_csv_to_sqlite(conn, csv_path, table_name, cols, time_col=None, chunksize=200000):
    cur = conn.cursor()
    cur.execute(f'DROP TABLE IF EXISTS {table_name}')
    col_defs = ','.join([f'{c} TEXT' for c in cols])
    cur.execute(f'CREATE TABLE {table_name} ({col_defs})')
    conn.commit()
    insert_sql = f'INSERT INTO {table_name} ({",".join(cols)}) VALUES ({",".join(["?" for _ in cols])})'
    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize):
        chunk = chunk[cols]
        if time_col is not None and time_col in chunk.columns:
            try:
                chunk[time_col] = pd.to_datetime(chunk[time_col]).dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass
        records = list(chunk.itertuples(index=False, name=None))
        cur.executemany(insert_sql, records)
        conn.commit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', default=str(DATABASE_DIR / 'click_orders.db'))
    p.add_argument('--clicks', default=str(RAW_DATA_DIR / 'JD_click_data.csv'))
    p.add_argument('--orders', default=str(RAW_DATA_DIR / 'JD_order_data.csv'))
    p.add_argument('--users', required=False)
    p.add_argument('--chunksize', type=int, default=200000)
    p.add_argument('--user_col', default='user_ID')
    p.add_argument('--sku_col', default='sku_ID')
    p.add_argument('--click_time_col', default='request_time')
    p.add_argument('--order_time_col', default='order_time')
    p.add_argument('--channel_col', default='channel')
    args = p.parse_args()

    if args.users is None:
        default_users = RAW_DATA_DIR / 'JD_user_data.csv'
        if default_users.exists():
            args.users = str(default_users)

    args.db = str(resolve_path(args.db))
    args.clicks = str(resolve_path(args.clicks))
    args.orders = str(resolve_path(args.orders))
    if args.users:
        args.users = str(resolve_path(args.users))

    ensure_dir(Path(args.db).parent)

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    print('Loading clicks...')
    import_csv_to_sqlite(
        conn,
        args.clicks,
        'clicks',
        [args.user_col, args.sku_col, args.click_time_col, args.channel_col],
        time_col=args.click_time_col,
        chunksize=args.chunksize,
    )
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_clicks_user ON clicks({args.user_col})')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_clicks_user_sku ON clicks({args.user_col}, {args.sku_col})')
    conn.commit()

    print('Loading orders...')
    import_csv_to_sqlite(
        conn,
        args.orders,
        'orders',
        [args.user_col, args.sku_col, args.order_time_col],
        time_col=args.order_time_col,
        chunksize=args.chunksize,
    )
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_orders_user_sku ON orders({args.user_col}, {args.sku_col})')
    conn.commit()

    if args.users:
        print('Loading users...')
        users_cols = list(pd.read_csv(args.users, nrows=0).columns)
        import_csv_to_sqlite(conn, args.users, 'users', users_cols, time_col=None, chunksize=args.chunksize)
        conn.commit()

    print('Validating DB...')
    def table_cols(name):
        cur.execute(f'PRAGMA table_info({name})')
        return [r[1] for r in cur.fetchall()]

    def table_count(name):
        cur.execute(f'SELECT COUNT(*) FROM {name}')
        return cur.fetchone()[0]

    clicks_cols = table_cols('clicks')
    orders_cols = table_cols('orders')
    users_cols = table_cols('users') if args.users else []

    required_clicks = {args.user_col, args.sku_col, args.click_time_col, args.channel_col}
    required_orders = {args.user_col, args.sku_col, args.order_time_col}

    missing_clicks = required_clicks - set(clicks_cols)
    missing_orders = required_orders - set(orders_cols)

    print('Row counts:')
    print('  clicks:', table_count('clicks'))
    print('  orders:', table_count('orders'))
    if args.users:
        print('  users:', table_count('users'))

    print('Column checks:')
    print('  clicks missing:', sorted(missing_clicks))
    print('  orders missing:', sorted(missing_orders))
    if args.users:
        print('  users columns:', users_cols)

    if missing_clicks or missing_orders:
        raise SystemExit('DB validation failed: missing required columns.')

    conn.close()
    print('Done. DB:', args.db)


if __name__ == '__main__':
    main()
