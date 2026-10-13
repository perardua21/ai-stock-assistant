"""
KOSPI 전체 종목코드 목록을 받아와 db/seed/kospi_codes.csv로 저장.

사용법:
    pip install pykrx pandas
    python scripts/fetch_kospi_codes.py
"""
import csv
import datetime
import pathlib

from pykrx import stock

OUTPUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "db" / "seed" / "kospi_codes.csv"


def main():
    today = datetime.date.today().strftime("%Y%m%d")
    codes = stock.get_market_ticker_list(today, market="KOSPI")

    rows = []
    for code in codes:
        name = stock.get_market_ticker_name(code)
        rows.append({"code": code, "name": name, "market": "KOSPI", "listed_date": ""})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "name", "market", "listed_date"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {len(rows)} codes -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
