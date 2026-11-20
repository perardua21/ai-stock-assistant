"""
수집 실행 (repo 루트에서)

    python -m ai.ingestion.run_ingest --source pykrx --days 1500 --limit 10
    python -m ai.ingestion.run_ingest --source pykrx --days 1500
    python -m ai.ingestion.run_ingest --source pykrx --days 1500 --resume   # 끊긴 데서 이어서
    python -m ai.ingestion.run_ingest --source opendart --days 30 --limit 10
    python -m ai.ingestion.run_ingest --source kis_price --limit 10        # 한투 API (안 써도 됨)
"""
import argparse
import csv
import datetime
import pathlib

SEED_PATH = pathlib.Path(__file__).resolve().parents[2] / "db" / "seed" / "kospi_codes.csv"


def load_codes(limit: int | None = None) -> list[dict]:
    with open(SEED_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["pykrx", "opendart", "naver_news", "kis_price"])
    parser.add_argument("--limit", type=int, default=None, help="테스트용 종목 수 제한")
    parser.add_argument("--days", type=int, default=30, help="조회 기간 (일)")
    parser.add_argument("--end", default=None, help="조회 종료일 YYYYMMDD (기본: 오늘)")
    parser.add_argument("--resume", action="store_true", help="pykrx: 이미 다 받은 종목은 건너뜀 (중간에 끊겼을 때)")
    args = parser.parse_args()

    stocks = load_codes(args.limit)
    codes = [s["code"] for s in stocks]
    end_d = datetime.datetime.strptime(args.end, "%Y%m%d").date() if args.end else datetime.date.today()
    end = end_d.strftime("%Y%m%d")
    start = (end_d - datetime.timedelta(days=args.days)).strftime("%Y%m%d")

    if args.source == "pykrx":
        from ai.ingestion import pykrx_market
        n = pykrx_market.fetch_all(codes, start, end, resume=args.resume)
    elif args.source == "opendart":
        from ai.ingestion import opendart_filing
        n = opendart_filing.fetch_all_disclosures(codes, start, end)
    elif args.source == "naver_news":
        from ai.ingestion import naver_news
        n = naver_news.fetch_all([s["name"] for s in stocks])
    else:
        from ai.ingestion import kis_price
        n = kis_price.fetch_all(codes, start, end)

    print(f"[{args.source}] saved {n} rows for {len(codes)} stocks ({start} ~ {end})")


if __name__ == "__main__":
    main()
