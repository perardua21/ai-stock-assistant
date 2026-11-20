"""
급락 분류기 학습 + 기준 모델 비교

    python -m ai.classification.train
    python -m ai.classification.train --forecast chronos-2 --forecast chronos-2+cov

비교하는 것 (전부 같은 test 행에서 평가)
    vol_rule                20일 변동성 큰 순서 (학습 없음)
    logistic(tabular)       정형 피처 로지스틱 회귀
    lgbm(tabular)           정형 피처 LightGBM
    chronos[m]              Chronos p_crash 그대로 (학습 없음)
    lgbm(tabular+price)[m]  정형 + Chronos 가격 피처 LightGBM

결과표는 models/experiments.csv, 모델은 models/crash_lgbm.txt / .json
"""
from __future__ import annotations

import argparse
import datetime
import json

import numpy as np
import pandas as pd

from ai.classification.dataset import FEATURE_SETS, build_dataset
from ai.classification.evaluate import evaluate, results_table
from ai.classification.split import TRAIN_END_DATE, train_test_split_by_date
from ai.common.db import ROOT
from ai.labeling.build_crash_label import CRASH_WINDOW_DAYS

MODEL_DIR = ROOT / "models"
LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "metric": "average_precision",
    "verbosity": -1,
    "seed": 42,
}


def _time_valid_split(train_df: pd.DataFrame, valid_frac: float = 0.2):
    dates = np.sort(train_df["base_date"].unique())
    cut = dates[int(len(dates) * (1 - valid_frac))]
    return train_df[train_df["base_date"] < cut], train_df[train_df["base_date"] >= cut]


def fit_lgbm(train_df: pd.DataFrame, features: list[str], params: dict | None = None, rounds: int = 2000):
    """학습 구간 마지막 20%를 검증으로 써서 early stopping. 반환: lightgbm.Booster"""
    import lightgbm as lgb

    tr, va = _time_valid_split(train_df)
    p = {**LGBM_PARAMS, **(params or {})}
    dtr = lgb.Dataset(tr[features], tr["is_crash"])
    dva = lgb.Dataset(va[features], va["is_crash"], reference=dtr)
    booster = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    # 검증 구간까지 포함해 best_iteration 만큼 다시 학습
    full = lgb.Dataset(train_df[features], train_df["is_crash"])
    return lgb.train(p, full, max(booster.best_iteration, 50))


def fit_logistic(train_df: pd.DataFrame, features: list[str]):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    return model.fit(train_df[features], train_df["is_crash"])


def run_experiments(tabular: pd.DataFrame, labels: pd.DataFrame, forecasts: dict[str, pd.DataFrame],
                    train_end: str = TRAIN_END_DATE, k: int = 20, use_lgbm: bool = True):
    """반환: (결과표 DataFrame, 최종 모델 정보 dict | None)"""
    split = dict(train_end=train_end, embargo_days=CRASH_WINDOW_DAYS)
    base = build_dataset(tabular, labels)
    base_tr, base_te = train_test_split_by_date(base, **split)

    # 평가 행을 맞춘다: Chronos 피처가 있으면 그 날짜의 test 행만 쓴다
    eval_df = base_te.copy()
    for f in forecasts.values():
        eval_df = eval_df.merge(f[["code", "base_date"]], on=["code", "base_date"], how="inner")
    if eval_df.empty or base_tr.empty:
        raise ValueError(f"학습 {len(base_tr)}행, 평가 {len(eval_df)}행이라 실험을 못 함. "
                         "train_end 날짜와 Chronos 예측 기간(run_forecast --start/--end)이 맞는지 확인")
    print(f"train {len(base_tr):,}행 / 평가 {len(eval_df):,}행 (급락 비율 {eval_df['is_crash'].mean():.2%})")

    results, scores, best = {}, eval_df[["code", "base_date", "is_crash"]].copy(), None

    scores["vol_rule"] = eval_df["vol_20"]
    results["vol_rule"] = evaluate(scores, "vol_rule", k, is_probability=False)

    tab = FEATURE_SETS["tabular"]
    logit = fit_logistic(base_tr, tab)
    scores["logistic(tabular)"] = logit.predict_proba(eval_df[tab])[:, 1]
    results["logistic(tabular)"] = evaluate(scores, "logistic(tabular)", k)

    tab_model = None
    if use_lgbm:
        tab_model = fit_lgbm(base_tr, tab)
        scores["lgbm(tabular)"] = tab_model.predict(eval_df[tab])
        results["lgbm(tabular)"] = evaluate(scores, "lgbm(tabular)", k)

    for name, f in forecasts.items():
        ds = build_dataset(tabular, labels, f)
        tr, te = train_test_split_by_date(ds, **split)
        te = eval_df[["code", "base_date"]].merge(te, on=["code", "base_date"], how="left")
        key = f"chronos[{name}]"
        scores[key] = te["p_crash"].to_numpy()
        results[key] = evaluate(scores, key, k)
        if use_lgbm:
            feats = FEATURE_SETS["tabular+price"]
            m = fit_lgbm(tr, feats)
            key = f"lgbm(tabular+price)[{name}]"
            scores[key] = m.predict(te[feats])
            results[key] = evaluate(scores, key, k)
            if best is None or results[key]["pr_auc"] > best["metrics"]["pr_auc"]:
                best = {"booster": m, "features": feats, "forecast_model": name, "metrics": results[key]}

    if use_lgbm and best is None:  # Chronos 피처가 없으면 정형 피처 모델이 최종
        best = {"booster": tab_model, "features": tab, "forecast_model": None, "metrics": results["lgbm(tabular)"]}
    return results_table(results), best


def save_model(best: dict, train_end: str) -> str:
    MODEL_DIR.mkdir(exist_ok=True)
    version = f"lgbm-{datetime.date.today():%Y%m%d}"
    best["booster"].save_model(str(MODEL_DIR / "crash_lgbm.txt"))
    meta = {
        "version": version,
        "features": best["features"],
        "forecast_model": best["forecast_model"],
        "train_end": train_end,
        "metrics": best["metrics"],
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (MODEL_DIR / "crash_lgbm.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return version


def main():
    from ai.common.data import load_forecast_features, load_index, load_labels, load_price_panel
    from ai.features.tabular import build_tabular_features

    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", action="append", default=None,
                    help="비교할 forecast_feature.model_name (여러 번 지정 가능)")
    ap.add_argument("--train-end", default=TRAIN_END_DATE)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--refit-all", action="store_true",
                    help="평가가 끝난 뒤 test 구간까지 포함한 전체 데이터로 최종 모델을 다시 학습 (운영용)")
    args = ap.parse_args()

    panel, index_df = load_price_panel(), load_index()
    tabular = build_tabular_features(panel, index_df)
    labels = load_labels()
    names = args.forecast or ["chronos-2", "chronos-2+cov"]
    forecasts = {n: load_forecast_features(n) for n in names}
    forecasts = {n: f for n, f in forecasts.items() if len(f)}
    if not forecasts:
        print("forecast_feature가 비어 있어 Chronos 없이 정형 피처 실험만 합니다.")

    table, best = run_experiments(tabular, labels, forecasts, args.train_end, args.k)
    print(table.to_string())
    MODEL_DIR.mkdir(exist_ok=True)
    table.to_csv(MODEL_DIR / "experiments.csv", encoding="utf-8-sig")
    if best and best.get("booster") is not None:
        if args.refit_all:
            ds = build_dataset(tabular, labels, forecasts.get(best["forecast_model"]))
            best["booster"] = fit_lgbm(ds, best["features"])
            print(f"전체 {len(ds):,}행으로 다시 학습")
        print("최종 모델 저장:", save_model(best, args.train_end), "| forecast:", best["forecast_model"])


if __name__ == "__main__":
    main()
