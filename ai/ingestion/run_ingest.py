"""
수집 실행 엔트리포인트.

사용 예:
    python run_ingest.py --source kis_price --limit 10
    python run_ingest.py --source opendart --limit 10
    python run_ingest.py --source naver_news --limit 10
"""
import argparse
import csv
import datetime
import pathlib

from dotenv import load_dotenv

import kis_price
import naver_news
import opendart_filing

SEED_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "db" / "seed" / "kospi_codes.csv"


def load_codes(limit: int | None = None) -> list[dict]:
    with open(SEED_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["kis_price", "opendart", "naver_news"])
    parser.add_argument("--limit", type=int, default=None, help="테스트용 종목 수 제한")
    parser.add_argument("--days", type=int, default=30, help="가격/공시 조회 기간 (일)")
    args = parser.parse_args()

    stocks = load_codes(args.limit)
    codes = [s["code"] for s in stocks]
    end = datetime.date.today().strftime("%Y%m%d")
    start = (datetime.date.today() - datetime.timedelta(days=args.days)).strftime("%Y%m%d")

    if args.source == "kis_price":
        n = kis_price.fetch_all(codes, start, end)
    elif args.source == "opendart":
        n = opendart_filing.fetch_all_disclosures(codes, start, end)
    elif args.source == "naver_news":
        queries = [s["name"] for s in stocks]
        n = naver_news.fetch_all(queries)

    print(f"[{args.source}] saved {n} rows for {len(codes)} stocks ({start} ~ {end})")


if __name__ == "__main__":
    main()
