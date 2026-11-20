"""
Chronos-2 예측 → 가격 피처 → forecast_feature 저장

백테스트용 (5거래일 간격):
    python -m ai.forecasting.run_forecast --start 2023-01-01 --end 2025-12-31 --stride 5
    python -m ai.forecasting.run_forecast --start 2023-01-01 --end 2025-12-31 --stride 5 --no-covariates
매일 운영:
    python -m ai.forecasting.run_forecast --date 2026-11-20

model_name: 공변량 쓰면 chronos-2+cov, 안 쓰면 chronos-2
추가 학습 모델은 --model-id 경로랑 --tag 같이 줘서 이름을 구분함
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from ai.common.data import load_index, load_price_panel
from ai.common.db import upsert_rows
from ai.forecasting.price_features import QUANTILE_LEVELS, quantiles_to_features
from ai.forecasting.windows import make_windows


def model_name_of(use_covariates: bool, tag: str | None = None) -> str:
    name = "chronos-2" + ("+cov" if use_covariates else "")
    return f"{name}:{tag}" if tag else name


def forecast_rows(windows, quantiles, model_name: str, horizon: int) -> list[dict]:
    rows = []
    for w, q in zip(windows, quantiles):
        feats = quantiles_to_features(q, w.last_close)
        rows.append({
            "code": w.code,
            "base_date": pd.Timestamp(w.base_date).date(),
            "model_name": model_name,
            "horizon": horizon,
            "last_close": w.last_close,
            **feats,
            "raw_quantiles": {"levels": QUANTILE_LEVELS, "values": q.round(2).tolist()},
        })
    return rows


def existing_keys(model_name: str) -> set[tuple[str, pd.Timestamp]]:
    from ai.common.db import read_sql

    df = read_sql("SELECT code, base_date FROM forecast_feature WHERE model_name = :m", {"m": model_name})
    return set(zip(df["code"], pd.to_datetime(df["base_date"])))


def run(start=None, end=None, date=None, stride=1, use_covariates=True, model_id="amazon/chronos-2",
        tag=None, codes=None, context_length=256, horizon=5, codes_per_batch=50, skip_existing=True,
        forecaster=None):
    """종목 50개씩 끊어서 window 만들기 → 예측 → 저장. 중간에 끊겨도 다시 돌리면 이어서 한다."""
    from ai.forecasting.windows import pick_dates

    name = model_name_of(use_covariates, tag)
    # context_length 만큼 과거가 필요해서 조회 시작일을 넉넉히 당긴다
    first = pd.Timestamp(date or start or "2000-01-01") - pd.Timedelta(days=int(context_length * 1.6))
    panel = load_price_panel(start=first.date().isoformat(), end=date or end, codes=codes)
    index_df = load_index(start=first.date().isoformat(), end=date or end)
    if panel.empty:
        print("가격 데이터가 없음")
        return 0
    calendar = index_df["trade_date"] if len(index_df) else panel["trade_date"]
    base_dates = [pd.Timestamp(date)] if date else pick_dates(calendar, start, end, stride)
    done = existing_keys(name) if skip_existing else set()

    if forecaster is None:
        from ai.forecasting.chronos_runner import ChronosForecaster
        forecaster = ChronosForecaster(model_id=model_id, horizon=horizon)

    all_codes = panel["code"].unique()
    saved, t0 = 0, time.time()
    print(f"[{name}] 종목 {len(all_codes)}개 x 기준일 {len(base_dates)}개")
    for i in range(0, len(all_codes), codes_per_batch):
        part = panel[panel["code"].isin(all_codes[i : i + codes_per_batch])]
        windows = make_windows(part, index_df, base_dates=base_dates, context_length=context_length,
                               use_covariates=use_covariates)
        windows = [w for w in windows if (w.code, w.base_date) not in done]
        if not windows:
            continue
        q = forecaster.predict_quantiles(windows)
        saved += upsert_rows("forecast_feature", forecast_rows(windows, q, name, horizon), ["code", "base_date", "model_name"])
        print(f"  종목 {min(i + codes_per_batch, len(all_codes))}/{len(all_codes)}, 누적 {saved}행 ({time.time() - t0:.0f}초)")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--date", help="하루만 예측 (운영용)")
    ap.add_argument("--stride", type=int, default=1, help="기준일 간격 (거래일)")
    ap.add_argument("--no-covariates", action="store_true")
    ap.add_argument("--model-id", default="amazon/chronos-2")
    ap.add_argument("--tag", default=None, help="추가 학습 모델 등을 구분하는 이름")
    ap.add_argument("--limit", type=int, default=None, help="테스트용 종목 수")
    ap.add_argument("--redo", action="store_true", help="이미 저장된 날짜도 다시 예측")
    args = ap.parse_args()

    codes = None
    if args.limit:
        from ai.ingestion.run_ingest import load_codes
        codes = [s["code"] for s in load_codes(args.limit)]
    run(args.start, args.end, args.date, args.stride, not args.no_covariates, args.model_id, args.tag, codes,
        skip_existing=not args.redo)


if __name__ == "__main__":
    main()
