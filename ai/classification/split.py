"""
train/test 날짜 분할. 모든 모델이 이 함수 하나로 나눔

모델마다 따로 나누면 test 기간이 달라서 비교가 안 되고, 섞으면 미래 데이터가 학습에 들어감.

embargo_days: 라벨이 기준일 뒤 5거래일 가격으로 만들어져서, 분할일 바로 전 5거래일 행은
라벨이 test 기간 가격을 보게 됨. 그 행들을 train에서 뺌.
테스트는 ai/classification/tests/test_split.py
"""
from __future__ import annotations

import pandas as pd

# TODO: 데이터 기간 확정되면 팀 회의에서 날짜 다시 정하기
TRAIN_END_DATE = "2025-06-30"


def train_test_split_by_date(
    df: pd.DataFrame,
    date_col: str = "base_date",
    train_end: str = TRAIN_END_DATE,
    embargo_days: int = 0,
):
    """df에서 date_col 기준으로 train_end까지는 train, 이후는 test로 나눈다."""
    cutoff = pd.Timestamp(train_end)
    dates = pd.to_datetime(df[date_col])
    train_mask = dates <= cutoff
    if embargo_days > 0:
        train_mask &= dates <= cutoff - pd.offsets.BDay(embargo_days)
    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[dates > cutoff].reset_index(drop=True)
    return train_df, test_df
