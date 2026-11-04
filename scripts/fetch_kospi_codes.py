"""
KOSPI 전체 종목코드 목록을 받아와 db/seed/kospi_codes.csv로 저장.

사용법:
    pip install pykrx pandas python-dotenv
    python scripts/fetch_kospi_codes.py

pykrx가 KRX 로그인을 요구해서 .env에 KRX_ID, KRX_PW가 있어야 한다.
"""
import csv
import datetime
import pathlib

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # pykrx import 전에 KRX_ID/KRX_PW를 환경변수로 올려둔다

from pykrx import stock  # noqa: E402

OUTPUT_PATH = ROOT / "db" / "seed" / "kospi_codes.csv"


def latest_ticker_list(max_back_days: int = 10):
    # 주말/공휴일이면 빈 리스트가 오니까 하루씩 거슬러 올라가며 찾는다
    day = datetime.date.today()
    for _ in range(max_back_days):
        codes = stock.get_market_ticker_list(day.strftime("%Y%m%d"), market="KOSPI")
        if codes:
            return codes, day
        day -= datetime.timedelta(days=1)
    raise RuntimeError("최근 영업일 종목 목록을 못 받아왔음 (pykrx/KRX 응답 확인 필요)")


def main():
    codes, base_day = latest_ticker_list()
    print(f"기준일 {base_day}: {len(codes)}개 종목")

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
