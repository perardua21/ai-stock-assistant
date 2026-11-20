"""
db/seed/kospi_codes.csv -> stock_master 테이블 적재.

repo 루트에서, docker compose up -d 로 DB 띄운 상태에서 실행:
    python scripts/load_stock_master.py
"""
import csv
import os
import pathlib

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEED_PATH = ROOT / "db" / "seed" / "kospi_codes.csv"

UPSERT_SQL = text("""
    INSERT INTO stock_master (code, name, market, listed_date, updated_at)
    VALUES (:code, :name, :market, :listed_date, now())
    ON CONFLICT (code) DO UPDATE SET
        name = EXCLUDED.name,
        market = EXCLUDED.market,
        listed_date = COALESCE(EXCLUDED.listed_date, stock_master.listed_date),
        is_active = TRUE,
        updated_at = now()
""")


def main():
    load_dotenv(ROOT / ".env")

    with open(SEED_PATH, encoding="utf-8") as f:
        rows = [
            {
                "code": r["code"],
                "name": r["name"],
                "market": r["market"],
                "listed_date": r["listed_date"] or None,
            }
            for r in csv.DictReader(f)
        ]

    url = os.environ["DATABASE_URL"]
    if url.startswith("postgresql://"):  # SQLAlchemy 2.1 기본 드라이버가 psycopg(3)라서 psycopg2로 고정
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(UPSERT_SQL, rows)

    print(f"stock_master: {len(rows)} rows upserted")


if __name__ == "__main__":
    main()
