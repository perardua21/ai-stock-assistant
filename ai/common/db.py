"""
DB 연결, 읽기/쓰기 함수. DATABASE_URL은 .env에서 읽음
"""
from __future__ import annotations

import json
import os
import pathlib

import pandas as pd
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

_engine = None


def db_url() -> str:
    """SQLAlchemy 2.1부터 postgresql:// 의 기본 드라이버가 psycopg(3)로 바뀌어서 psycopg2로 고정"""
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url


def get_engine():
    # sqlalchemy는 DB를 실제로 쓸 때만 import (DB 없이 도는 테스트에서 의존성을 줄이려고)
    from sqlalchemy import create_engine

    global _engine
    if _engine is None:
        _engine = create_engine(db_url(), pool_pre_ping=True)
    return _engine


def read_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    from sqlalchemy import text

    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def execute(sql: str, params: dict | list[dict] | None = None):
    from sqlalchemy import text

    with get_engine().begin() as conn:
        return conn.execute(text(sql), params or {})


def _json_default(o):
    if hasattr(o, "item"):  # numpy 숫자
        return o.item()
    return str(o)


def _clean(v):
    """numpy/pandas 값을 DB에 넣을 수 있는 파이썬 값으로 바꾼다."""
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, default=_json_default)
    if hasattr(v, "item"):  # numpy scalar
        v = v.item()
    if isinstance(v, float) and v != v:  # NaN
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def upsert_rows(table: str, rows: list[dict], conflict_cols: list[str], chunk: int = 5000) -> int:
    """rows를 table에 upsert. conflict_cols는 UNIQUE/PK 제약과 같아야 한다."""
    from sqlalchemy import text

    if not rows:
        return 0
    cols = list(rows[0].keys())

    update_cols = [c for c in cols if c not in conflict_cols]

    # price_daily의 기존 정상 데이터가 NULL로 덮어써지는 것을 방지
    protected_cols = {
        "market_cap",
        "trading_value",
        "foreign_net_buy",
        "inst_net_buy",
    }

    update_sql = ", ".join(
        f"{c} = COALESCE(EXCLUDED.{c}, {table}.{c})"
        if table == "price_daily" and c in protected_cols
        else f"{c} = EXCLUDED.{c}"
        for c in update_cols
    )

    sql = text(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)}) "
        f"ON CONFLICT ({', '.join(conflict_cols)}) "
        + (f"DO UPDATE SET {update_sql}" if update_cols else "DO NOTHING")
    )
    total = 0
    with get_engine().begin() as conn:
        for i in range(0, len(rows), chunk):
            part = [{k: _clean(v) for k, v in r.items()} for r in rows[i : i + chunk]]
            conn.execute(sql, part)
            total += len(part)
    return total


def df_to_rows(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")
