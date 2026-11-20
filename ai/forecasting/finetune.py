"""
Chronos-2 추가 학습 (기본 LoRA). GPU 권장

TRAIN_END_DATE 이전 데이터만 써서 test 기간이랑 안 섞임.
결과는 models/chronos2-kospi-cov/finetuned-ckpt 에 저장됨.

    python -m ai.forecasting.finetune --steps 1000 --mode lora
"""
from __future__ import annotations

import argparse

from ai.classification.split import TRAIN_END_DATE
from ai.common.data import load_index, load_price_panel
from ai.common.db import ROOT
from ai.forecasting.windows import COVARIATES, add_covariates


def build_inputs(panel, index_df, use_covariates: bool, min_len: int = 120) -> list[dict]:
    df = add_covariates(panel, index_df) if use_covariates else panel
    inputs = []
    for _, g in df.groupby("code", sort=False):
        if len(g) < min_len:
            continue
        item = {"target": g["close"].to_numpy(dtype="float32")}
        if use_covariates:
            item["past_covariates"] = {c: g[c].to_numpy(dtype="float32") for c in COVARIATES}
        inputs.append(item)
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="amazon/chronos-2")
    ap.add_argument("--mode", choices=["lora", "full"], default="lora")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-covariates", action="store_true")
    ap.add_argument("--train-end", default=TRAIN_END_DATE)
    args = ap.parse_args()

    from ai.forecasting.chronos_runner import ChronosForecaster

    panel = load_price_panel(end=args.train_end)
    index_df = load_index(end=args.train_end)
    inputs = build_inputs(panel, index_df, use_covariates=not args.no_covariates)
    print(f"학습 시계열 {len(inputs)}개 (~{args.train_end})")

    base = ChronosForecaster(model_id=args.model_id).pipeline
    lr = args.lr or (1e-5 if args.mode == "lora" else 1e-6)
    out_dir = ROOT / "models" / ("chronos2-kospi" + ("" if args.no_covariates else "-cov"))
    base.fit(inputs, prediction_length=5, finetune_mode=args.mode, learning_rate=lr,
             num_steps=args.steps, output_dir=out_dir)
    print(f"저장: {out_dir / 'finetuned-ckpt'}")


if __name__ == "__main__":
    main()
