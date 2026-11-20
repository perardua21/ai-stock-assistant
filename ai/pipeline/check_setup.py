"""
실제로 돌리기 전 환경 점검

    python -m ai.pipeline.check_setup                       # 패키지, .env, DB
    python -m ai.pipeline.check_setup --api                 # KRX, OpenDART, 네이버까지
    python -m ai.pipeline.check_setup --api --llm --chronos # OpenAI, Chronos-2까지
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback

import ai.common.db  # noqa: F401  .env를 먼저 읽어 둠 (pykrx는 import할 때 KRX 계정을 봄)

results = []


def check(name, fn):
    try:
        msg = fn()
        results.append(("OK", name, msg or ""))
    except Exception as e:
        results.append(("FAIL", name, f"{type(e).__name__}: {e}"))
        if os.environ.get("CHECK_DEBUG"):
            traceback.print_exc()


def skip(name, why):
    results.append(("SKIP", name, why))


def pkg(name):
    def _():
        m = importlib.import_module(name)
        return getattr(m, "__version__", "")
    return _


def env_keys():
    from ai.common.db import ROOT  # .env 로드

    need = ["DATABASE_URL", "KRX_ID", "KRX_PW", "OPENDART_API_KEY", "NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET", "OPENAI_API_KEY"]
    missing = [k for k in need if not os.environ.get(k)]
    if not (ROOT / ".env").exists():
        raise RuntimeError(".env 파일이 없음 (copy .env.example .env)")
    if missing:
        raise RuntimeError("비어 있는 값: " + ", ".join(missing))


def db():
    from ai.common.db import read_sql

    tables = set(read_sql("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")["tablename"])
    need = {"stock_master", "price_daily", "index_daily", "label_crash", "forecast_feature", "agent_decision"}
    if need - tables:
        raise RuntimeError(f"없는 테이블 {sorted(need - tables)} → scripts\\db_migrate.bat 실행")
    n = read_sql("SELECT (SELECT count(*) FROM stock_master) s, (SELECT count(*) FROM price_daily) p, "
                 "(SELECT count(*) FROM label_crash) l, (SELECT count(*) FROM forecast_feature) f").iloc[0]
    return f"stock_master {n.s}, price_daily {n.p}, label_crash {n.l}, forecast_feature {n.f}"


def seed():
    from ai.ingestion.run_ingest import load_codes

    return f"{len(load_codes())}개 종목"


def krx():
    import datetime as dt

    from ai.ingestion import pykrx_market

    end = dt.date.today()
    rows = pykrx_market.fetch_index((end - dt.timedelta(days=14)).strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    if not rows:
        raise RuntimeError("KOSPI 지수가 비어 있음. KRX 로그인 확인")
    stock_rows = pykrx_market.fetch_stock("005930", (end - dt.timedelta(days=14)).strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    filled = sum(r["market_cap"] is not None and r["foreign_net_buy"] is not None for r in stock_rows)
    if not stock_rows or filled == 0:
        raise RuntimeError("삼성전자 시가총액/수급이 비어 있음")
    return f"지수 {len(rows)}일, 삼성전자 {len(stock_rows)}일 (시가총액·수급 {filled}일)"


def dart():
    import datetime as dt

    from ai.ingestion.opendart_filing import corp_code_of, fetch_disclosures

    cc = corp_code_of("005930")
    end = dt.date.today()
    rows = fetch_disclosures("005930", (end - dt.timedelta(days=60)).strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    return f"삼성전자 corp_code {cc}, 최근 60일 공시 {len(rows)}건"


def naver():
    from ai.ingestion.naver_news import fetch_news

    rows = fetch_news("삼성전자", display=5)
    return f"{len(rows)}건, 최신: {rows[0]['title'][:30] if rows else '-'}"


def llm():
    from ai.harness.llm import OpenAILLM

    m = OpenAILLM()
    r = m.chat([{"role": "user", "content": '{"ok": true} 를 그대로 JSON으로 답해'}], json_mode=True)
    return f"{m.model}: {str(r['content'])[:40]}"


def chronos():
    import numpy as np

    from ai.forecasting.chronos_runner import ChronosForecaster
    from ai.forecasting.windows import Window

    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))
    w = Window("TEST", None, float(close[-1]), close.astype("float32"),
               {"log_volume": rng.normal(10, 1, 200).astype("float32")})
    fc = ChronosForecaster()
    q = fc.predict_quantiles([w])
    return f"device {fc.pipeline.model.device}, 출력 {q.shape}, 5일 뒤 중앙값 {q[0, -1, 6]:.1f} (마지막 종가 {close[-1]:.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true", help="KRX·OpenDART·네이버 실제 호출")
    ap.add_argument("--llm", action="store_true", help="OpenAI 호출 (요금 조금 나옴)")
    ap.add_argument("--chronos", action="store_true", help="Chronos-2 모델 받아서 예측 한 번")
    args = ap.parse_args()

    results.append(("OK", "python", sys.version.split()[0]))
    for name in ["pandas", "numpy", "sqlalchemy", "psycopg2", "pykrx", "sklearn", "scipy", "lightgbm", "openai", "dotenv"]:
        check(f"패키지 {name}", pkg(name))
    for name in ["torch", "chronos"]:
        check(f"패키지 {name} (Chronos용)", pkg(name))
    check(".env 키", env_keys)
    check("DB 연결·테이블", db)
    check("종목 목록 db/seed/kospi_codes.csv", seed)
    if args.api:
        check("KRX (pykrx)", krx)
        check("OpenDART", dart)
        check("네이버 뉴스", naver)
    else:
        skip("KRX·OpenDART·네이버", "--api 로 확인")
    check("OpenAI", llm) if args.llm else skip("OpenAI", "--llm 으로 확인")
    check("Chronos-2", chronos) if args.chronos else skip("Chronos-2", "--chronos 로 확인")

    print()
    for status, name, msg in results:
        print(f"[{status:4}] {name:<32} {msg}")
    fails = sum(s == "FAIL" for s, _, _ in results)
    print(f"\n실패 {fails}개" if fails else "\n전부 통과")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
