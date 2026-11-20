# AI Stock Assistant

KOSPI 전 종목의 단기 급락 위험(5거래일 안에 종가 -10%)을 예측하고, 급락 위험을 반영해서 포트폴리오를 추천하는 프로젝트.

구조: Chronos-2 + 급락 분류기(LightGBM) + Self-evolving 하네스

1. Chronos-2가 일봉과 수급·거래량·지수(공변량)로 5거래일 가격 분포를 예측
2. 예측 분포에서 뽑은 가격 피처 + 정형 피처로 LightGBM이 종목별 급락 확률 계산
3. 하네스의 LLM 에이전트가 급락 확률을 받고 뉴스·공시를 직접 찾아본 뒤 판단, 포트폴리오 최적화 툴 호출
4. 5거래일 뒤 자동 채점, 틀린 사례로 스킬(판단 절차) 개선안 생성

뉴스는 학습에 안 쓴다. 네이버 뉴스 API로는 과거 기사를 못 받아서 에이전트가 그날 기사만 읽는다.

## 폴더

```
db/migrations/        0001 원본, 0002 분류 결과, 0003 라벨, 0004 Chronos·하네스
ai/common/            DB 연결, DB→pandas 로더, 테스트용 가짜 데이터
ai/ingestion/         pykrx(가격·수급·지수), OpenDART, 네이버 뉴스, 한투 API
ai/labeling/          급락 라벨
ai/features/          정형 피처
ai/forecasting/       Chronos-2 입력, 실행, 가격 피처, 추가 학습
ai/classification/    학습 데이터, LightGBM 학습/비교, 평가, 운영 추론
ai/explain/           SHAP
ai/optimization/      포트폴리오 최적화
ai/harness/           에이전트, 툴, 프롬프트, 스킬, 메모리, 채점, 스킬 개선
ai/pipeline/          매일 실행, 환경 점검
scripts/              종목 목록, DB export/import/migrate
backend/ frontend/    아직 비어 있음
```

## 설치 (Windows, repo 루트)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r ai\requirements.txt
copy .env.example .env
```

`.env`에 KRX 계정, OpenDART·네이버·OpenAI 키를 넣는다. KRX 계정은 data.krx.co.kr에서 무료로 만들 수 있다.

Chronos-2 돌릴 컴퓨터는 아래도 설치. GPU가 있으면 CUDA 버전 torch를 먼저 깔고 설치한다.

```bat
pip install -r ai\requirements-chronos.txt
```

## DB

```bat
docker compose up -d
scripts\db_migrate.bat
python scripts\fetch_kospi_codes.py
python scripts\load_stock_master.py
```

`docker compose up`을 처음 하면 migration이 자동으로 들어간다. 이미 DB가 있었으면 `db_migrate.bat`로 0004를 넣어야 한다.
DBeaver 접속 정보: `localhost:5432`, DB `ai_stock_assistant`, user `ai_stock`, pw `ai_stock_pw`

## 실행 순서

명령은 전부 repo 루트에서 `python -m`으로 실행한다.

0) 환경 점검

```bat
python -m pytest -q
python -m ai.pipeline.check_setup --api
```

1) 수집. 약 4년치(1500일). Chronos 입력에 과거 1년 정도가 필요해서 예측은 2023년 6월부터 한다.
처음엔 10종목으로 확인하고 전체를 돌린다. 전체는 1~2시간 정도 걸린다.

```bat
python -m ai.ingestion.run_ingest --source pykrx --days 1500 --limit 10
python -m ai.ingestion.run_ingest --source pykrx --days 1500
python -m ai.labeling.build_crash_label
```

중간에 끊기면 `--resume`을 붙여서 다시 돌리면 받은 종목은 건너뛴다.

2) Chronos-2 예측. 5거래일 간격, 공변량 있는 버전과 없는 버전 둘 다. CPU면 오래 걸려서 `--limit 50` 정도로 먼저 해 보는 게 좋다.
이미 저장된 날짜는 건너뛰니까 끊겨도 다시 돌리면 된다.

```bat
python -m ai.forecasting.run_forecast --start 2023-06-01 --end 2026-10-31 --stride 5 --no-covariates
python -m ai.forecasting.run_forecast --start 2023-06-01 --end 2026-10-31 --stride 5
```

3) 분류기 학습, 비교 실험

```bat
python -m ai.classification.train
```

결과표는 `models/experiments.csv`, PR-AUC가 제일 좋은 LightGBM(정형+가격)이 `models/crash_lgbm.txt`로 저장된다.
운영에 쓸 때는 `--refit-all`로 test 기간까지 넣어서 다시 학습한다.

4) 매일 실행

```bat
python -m ai.pipeline.run_daily --date 2026-11-20 --no-llm
python -m ai.pipeline.run_daily --date 2026-11-20
```

결과는 DB(`classification_result`, `agent_decision`, `portfolio_result`)와 `data/reports/날짜.json`에 남는다.

5) 스킬 개선. 채점된 판단이 30건 넘게 쌓이면

```bat
python -m ai.harness.evolve --skill crash_review
python -m ai.harness.evolve --apply 1
```

첫 줄이 `ai\harness\skills\proposed\`에 개선안을 만들고, 읽어 보고 괜찮으면 두 번째 줄로 적용한다.

## 데이터 주고받기

DB는 각자 Docker 안에 있다. 코드는 깃으로 맞추고 데이터는 파일로 넘긴다.

- 보내는 사람: `scripts\db_export.bat` 실행하면 레포 폴더에 `db.dump`가 생김
- 받는 사람: `db.dump`를 레포 폴더에 넣고 `scripts\db_import.bat`
- 학습한 모델은 `models\crash_lgbm.txt`, `models\crash_lgbm.json` 두 파일을 같이 넘긴다

주고받기 전에 둘 다 `git pull` 먼저. import하면 내 DB는 통째로 덮어써진다.

## 데이터 출처

| 데이터 | 출처 | 어디에 쓰나 |
| --- | --- | --- |
| 수정주가 일봉 | pykrx (네이버) | Chronos 입력, 정형 피처, 라벨 |
| 시가총액, 거래대금, 외국인/기관 순매수, KOSPI 지수 | pykrx (KRX, 로그인 필요) | 공변량, 정형 피처 |
| 공시 | OpenDART | 에이전트 공시 검색 |
| 뉴스 | 네이버 뉴스 검색 API | 에이전트 뉴스 검색 (학습에는 안 씀) |
