"""
저장된 분류기로 그날 전 종목 급락 확률 계산 → classification_result, feature_importance

    python -m ai.classification.predict --date 2026-11-20
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from ai.classification.dataset import prediction_frame
from ai.classification.train import MODEL_DIR
from ai.explain.shap_explainer import explain_rows


def load_model():
    import lightgbm as lgb

    meta = json.loads((MODEL_DIR / "crash_lgbm.json").read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(MODEL_DIR / "crash_lgbm.txt")), meta


def score_date(date, tabular: pd.DataFrame, forecast: pd.DataFrame | None, booster, meta: dict) -> pd.DataFrame:
    df = prediction_frame(tabular, forecast, date)
    if df.empty:
        return df
    feats = meta["features"]
    df["crash_probability"] = booster.predict(df[feats])
    df["top_features"] = explain_rows(booster, df, feats, top_k=5)
    return df.sort_values("crash_probability", ascending=False).reset_index(drop=True)


def save_results(df: pd.DataFrame, date, version: str) -> int:
    from ai.common.db import execute, read_sql, upsert_rows

    d = pd.Timestamp(date).date()
    rows = [{"code": r.code, "run_date": d, "crash_probability": round(float(r.crash_probability), 5),
             "model_version": version} for r in df.itertuples()]
    upsert_rows("classification_result", rows, ["code", "run_date", "model_version"])
    ids = read_sql("SELECT id, code FROM classification_result WHERE run_date = :d AND model_version = :v",
                   {"d": d, "v": version})
    id_of = dict(zip(ids["code"], ids["id"]))
    execute("DELETE FROM feature_importance WHERE classification_id = ANY(:ids)", {"ids": [int(i) for i in ids["id"]]})
    imp = [{"classification_id": int(id_of[r.code]), "feature_name": t["feature"],
            "shap_value": t["shap_value"], "contribution_pct": t["contribution_pct"]}
           for r in df.itertuples() for t in r.top_features]
    if imp:
        execute(
            "INSERT INTO feature_importance (classification_id, feature_name, shap_value, contribution_pct) "
            "VALUES (:classification_id, :feature_name, :shap_value, :contribution_pct)", imp)
    return len(rows)


def run(date) -> pd.DataFrame:
    from ai.common.data import load_forecast_features, load_index, load_price_panel
    from ai.features.tabular import build_tabular_features

    booster, meta = load_model()
    start = (pd.Timestamp(date) - pd.Timedelta(days=420)).date().isoformat()  # 252거래일 피처용
    tabular = build_tabular_features(load_price_panel(start=start, end=str(date)), load_index(start=start, end=str(date)))
    forecast = load_forecast_features(meta["forecast_model"], str(date)) if meta.get("forecast_model") else None
    df = score_date(date, tabular, forecast, booster, meta)
    n = save_results(df, date, meta["version"]) if len(df) else 0
    print(f"{date}: {n}개 종목 급락 확률 저장 (모델 {meta['version']})")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    run(args.date)


if __name__ == "__main__":
    main()
