"""
DB 저장 공통 함수. DATABASE_URL은 .env에서 읽음.
"""
import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = os.environ["DATABASE_URL"]
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def upsert_rows(table: str, rows: list[dict], conflict_cols: list[str]):
    """rows를 table에 upsert. conflict_cols는 UNIQUE/PK 제약과 일치해야 함."""
    if not rows:
        return 0

    cols = list(rows[0].keys())
    col_list = ", ".join(cols)
    val_list = ", ".join(f":{c}" for c in cols)
    update_list = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in conflict_cols)
    conflict_list = ", ".join(conflict_cols)

    sql = text(
        f"INSERT INTO {table} ({col_list}) VALUES ({val_list}) "
        f"ON CONFLICT ({conflict_list}) DO UPDATE SET {update_list}"
    )

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)
