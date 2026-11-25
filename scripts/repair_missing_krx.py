"""Repair missing KRX cap/value and/or flows on existing rows, never delete prices.
Usage:
  python scripts/repair_missing_krx.py --code 005090 --dry-run
  python scripts/repair_missing_krx.py --code 005090
  python scripts/repair_missing_krx.py --all --dry-run
  python scripts/repair_missing_krx.py --all --limit 10
By default only historical dates up to 2 days before today are repaired;
zero-volume flow gaps are left alone as they may be nontrading/suspended dates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

from sqlalchemy import text
from ai.common.db import get_engine, read_sql
from ai.ingestion.pykrx_market import _krx
from pykrx import stock


def candidates(cutoff: dt.date, code: str | None):
    params = {"cutoff": cutoff}
    filter_code = ""
    if code:
        filter_code = "AND code = :code"
        params["code"] = code
    sql = f"""
        SELECT code, trade_date, volume, market_cap, trading_value,
               foreign_net_buy, inst_net_buy
          FROM price_daily
         WHERE trade_date <= :cutoff {filter_code}
           AND (market_cap IS NULL OR trading_value IS NULL
                OR (COALESCE(volume, 0) > 0
                    AND (foreign_net_buy IS NULL OR inst_net_buy IS NULL)))
         ORDER BY code, trade_date
    """
    return read_sql(sql, params)


def repair_one(code, subset, dry_run=False):
    dates = {pd.date() if hasattr(pd, 'date') else pd for pd in subset['trade_date']}
    start = min(dates).strftime('%Y%m%d')
    end = max(dates).strftime('%Y%m%d')
    need_cap = subset['market_cap'].isna().any() or subset['trading_value'].isna().any()
    need_flow = ((subset['volume'].fillna(0) > 0) &
                 (subset['foreign_net_buy'].isna() | subset['inst_net_buy'].isna())).any()
    print(f'{code}: {len(dates)} incomplete date(s) in {start}..{end}; cap={need_cap} flow={need_flow}')
    if dry_run:
        return 0
    records = {}
    if need_cap:
        cap = _krx(stock.get_market_cap, start, end, code,
                   {'시가총액': 'market_cap', '거래대금': 'trading_value'})
        for day, row in cap.iterrows():
            d = day.date() if hasattr(day, 'date') else day
            if d in dates:
                records.setdefault(d, {}).update(row.to_dict())
    if need_flow:
        flow = _krx(stock.get_market_trading_value_by_date, start, end, code,
                    {'외국인합계': 'foreign_net_buy', '기관합계': 'inst_net_buy'})
        for day, row in flow.iterrows():
            d = day.date() if hasattr(day, 'date') else day
            if d in dates:
                records.setdefault(d, {}).update(row.to_dict())
    updates = []
    for day, values in records.items():
        clean = {k: (None if value is None or __import__('pandas').isna(value) else float(value))
                 for k, value in values.items()}
        if any(v is not None for v in clean.values()):
            updates.append({'code': code, 'trade_date': day,
                            'market_cap': clean.get('market_cap'),
                            'trading_value': clean.get('trading_value'),
                            'foreign_net_buy': clean.get('foreign_net_buy'),
                            'inst_net_buy': clean.get('inst_net_buy')})
    if not updates:
        print(f'{code}: no usable KRX values; left unchanged')
        return 0
    sql = text('''
        UPDATE price_daily AS p
           SET market_cap = COALESCE(p.market_cap, :market_cap),
               trading_value = COALESCE(p.trading_value, :trading_value),
               foreign_net_buy = COALESCE(p.foreign_net_buy, :foreign_net_buy),
               inst_net_buy = COALESCE(p.inst_net_buy, :inst_net_buy)
         WHERE p.code = :code AND p.trade_date = :trade_date
           AND (p.market_cap IS NULL OR p.trading_value IS NULL
                OR p.foreign_net_buy IS NULL OR p.inst_net_buy IS NULL)
    ''')
    with get_engine().begin() as conn:
        result = conn.execute(sql, updates)
    print(f'{code}: updated {result.rowcount} row(s) (NULL-only fills)')
    return result.rowcount


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--code', help='one stock code, e.g. 005090')
    group.add_argument('--all', action='store_true', help='all codes containing gaps')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='limit codes in --all mode')
    parser.add_argument('--include-recent', action='store_true', help='include latest two calendar days')
    args = parser.parse_args()
    cutoff = dt.date.today() if args.include_recent else dt.date.today() - dt.timedelta(days=2)
    df = candidates(cutoff, args.code)
    print(f'{len(df)} candidate rows across {df.code.nunique() if not df.empty else 0} stocks; cutoff={cutoff}')
    if df.empty:
        return
    total = 0
    for i, (code, subset) in enumerate(df.groupby('code', sort=True)):
        if args.limit and i >= args.limit:
            break
        try:
            total += repair_one(str(code), subset, args.dry_run)
        except Exception as exc:
            print(f'[error] {code}: {type(exc).__name__}: {exc}')
        time.sleep(1)
    print(f'finished; updated row count reported by DB: {total}')


if __name__ == '__main__':
    main()
