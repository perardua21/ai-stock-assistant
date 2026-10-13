# seed 데이터

`kospi_codes.csv` — KOSPI 전체 종목코드 마스터. `stock_master` 테이블 초기 적재용.

`scripts/fetch_kospi_codes.py`를 실행하면 pykrx로 현재 KOSPI 상장 종목 목록을 받아와
이 폴더에 `kospi_codes.csv`를 생성합니다.

```bash
python scripts/fetch_kospi_codes.py
```

컬럼: `code,name,market,listed_date`
