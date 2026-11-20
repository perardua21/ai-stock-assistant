"""
매일 장 마감 후 돌리는 전체 파이프라인

    python -m ai.pipeline.run_daily --date 2026-11-20
    python -m ai.pipeline.run_daily --date 2026-11-20 --skip-ingest --no-llm

1 수집, 2 라벨 갱신, 3 지난 판단 채점, 4 Chronos-2, 5 분류기,
6 에이전트 검토 (위험 상위 + 추천 후보), 7 포트폴리오
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from ai.common.db import ROOT

REPORT_DIR = ROOT / "data" / "reports"


def step(msg: str):
    print(f"\n=== {msg}")


def run(date: str, skip_ingest=False, ingest_days=10, top_k=10, n_candidates=60, use_llm=True,
        chronos_model_id="amazon/chronos-2"):
    from ai.classification import predict
    from ai.common.data import load_forecast_features, load_price_panel, load_stock_names
    from ai.common.db import execute, upsert_rows
    from ai.forecasting import run_forecast
    from ai.harness import scoring
    from ai.harness.tools import MarketSnapshot
    from ai.labeling.build_crash_label import build_labels_panel
    from ai.optimization.portfolio import PortfolioParams, params_dict, recommend

    d = pd.Timestamp(date)
    if not skip_ingest:
        step("1. 수집")
        from ai.ingestion import pykrx_market
        from ai.ingestion.run_ingest import load_codes
        start = (d - pd.Timedelta(days=ingest_days)).strftime("%Y%m%d")
        pykrx_market.fetch_all([s["code"] for s in load_codes()], start, d.strftime("%Y%m%d"))

    step("2. 라벨 갱신")
    recent = load_price_panel(start=(d - pd.Timedelta(days=40)).date().isoformat(), end=d.date().isoformat())
    labels = build_labels_panel(recent)
    if len(labels):
        labels["base_date"] = pd.to_datetime(labels["base_date"]).dt.date
        print(f"label_crash {upsert_rows('label_crash', labels.to_dict(orient='records'), ['code', 'base_date', 'horizon_days'])}행")

    step("3. 지난 판단 채점")
    scoring.run()

    step("4. Chronos-2 예측")
    booster, meta = predict.load_model()
    fm = meta.get("forecast_model")
    if fm:
        run_forecast.run(date=d.date().isoformat(), use_covariates="+cov" in fm,
                         tag=fm.split(":", 1)[1] if ":" in fm else None, model_id=chronos_model_id)
    else:
        print("분류기가 Chronos 피처를 쓰지 않아 건너뜀")

    step("5. 급락 분류기")
    scored = predict.run(d.date().isoformat())
    if scored.empty:
        print("오늘 데이터가 없습니다 (휴장일이거나 수집 전).")
        return None

    snap = MarketSnapshot(
        run_date=d,
        scores=scored[["code", "crash_probability", "top_features"]],
        tabular=scored,
        forecast=load_forecast_features(fm, d.date().isoformat()) if fm else None,
        names=load_stock_names(),
        base_rate=meta.get("metrics", {}).get("base_rate"),
    )

    # 포트폴리오 후보: 유동성 상위 종목 (거래대금이 너무 작은 종목 제외)
    price_hist = load_price_panel(start=(d - pd.Timedelta(days=120)).date().isoformat(), end=d.date().isoformat())
    pool = scored.nlargest(n_candidates, "log_trading_value_20")[["code", "crash_probability"]].copy()
    if snap.forecast is not None and len(snap.forecast):
        pool = pool.merge(snap.forecast[["code", "q50_ret"]], on="code", how="left")
    adjusted: dict[str, float] = {}
    params = PortfolioParams()

    def portfolio_fn(overrides: dict) -> list[dict]:
        p = PortfolioParams(**{**params_dict(params), **overrides})
        c = pool.copy()
        c["crash_probability"] = [adjusted.get(code, prob) for code, prob in zip(c["code"], c["crash_probability"])]
        out = recommend(c, price_hist, p)
        out["name"] = out["code"].map(snap.names)
        return out.round(5).to_dict(orient="records")

    decisions = []
    if use_llm:
        step("6. 하네스 에이전트 검토")
        from ai.harness.agent import HarnessAgent
        from ai.harness.llm import OpenAILLM
        from ai.harness.memory import recall, save_decision
        from ai.harness.tools import Toolbox
        from ai.ingestion.naver_news import fetch_news
        from ai.ingestion.opendart_filing import fetch_disclosures

        tb = Toolbox(
            snapshot=snap,
            news_fn=lambda q, n: fetch_news(q, display=min(n, 100)),
            disclosure_fn=fetch_disclosures,
            memory_fn=lambda code, n: recall(code, n, before=d),
            portfolio_fn=portfolio_fn,
        )
        agent = HarnessAgent(OpenAILLM(), tb)
        prelim = pd.DataFrame(portfolio_fn({}))
        targets = list(dict.fromkeys(list(scored["code"].head(top_k)) + list(prelim["code"].head(top_k))))
        for code in targets:
            dec = agent.review_stock(code)
            save_decision(dec)
            adjusted[code] = dec["agent_probability"]
            decisions.append(dec)
            print(f"  {code} {snap.names.get(code, ''):<12} model {dec['model_probability']:.3f} → agent {dec['agent_probability']:.3f} [{dec['judgment']}]")
        step("7. 포트폴리오 (에이전트 검토)")
        port = agent.review_portfolio(decisions)
    else:
        step("7. 포트폴리오 (LLM 없이)")
        port = {"summary": "", "warnings": [], "weights": portfolio_fn({})}

    weights = port.get("weights") or portfolio_fn({})
    rows = [{"run_date": d.date(), "code": w["code"], "weight": w["weight"], "expected_return": w.get("expected_return"),
             "crash_probability": w.get("crash_probability"), "params": params_dict(params)} for w in weights]
    execute("DELETE FROM portfolio_result WHERE run_date = :d", {"d": d.date()})  # 다시 돌릴 때 예전 비중 제거
    upsert_rows("portfolio_result", rows, ["run_date", "code"])
    for w in weights[:15]:
        print(f"  {w['code']} {w.get('name') or '':<12} {w['weight']:.1%}  (급락 {w['crash_probability']:.3f})")
    if port.get("summary"):
        print("\n추천 사유:", port["summary"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_date": str(d.date()),
        "model_version": meta["version"],
        "top_risk": scored.head(20)[["code", "crash_probability"]].assign(name=lambda x: x["code"].map(snap.names)).to_dict(orient="records"),
        "decisions": [{k: v for k, v in x.items() if k != "evidence"} for x in decisions],
        "portfolio": port,
    }
    path = REPORT_DIR / f"{d.date()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n리포트: {path}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="기준일 YYYY-MM-DD (장 마감 후)")
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--ingest-days", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=10, help="에이전트가 검토할 위험 상위·추천 상위 종목 수")
    ap.add_argument("--no-llm", action="store_true", help="에이전트 없이 분류기+최적화만")
    ap.add_argument("--chronos-model-id", default="amazon/chronos-2", help="추가 학습한 모델이면 그 폴더 경로")
    args = ap.parse_args()
    run(args.date, args.skip_ingest, args.ingest_days, args.top_k, use_llm=not args.no_llm,
        chronos_model_id=args.chronos_model_id)


if __name__ == "__main__":
    main()
