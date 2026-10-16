"""
train/test 날짜 분할 — structured/text_bert/fusion이 전부 이 함수 하나만 쓴다.

브랜치마다 따로 나누면 test 기간이 달라져서 성능 비교가 안 맞고, 무작위로
섞으면 미래 데이터가 학습에 들어간다. 그래서 날짜 기준 한 곳으로 고정.
겹치는지는 ai/classification/tests/test_split.py 에서 확인한다.
"""
from __future__ import annotations

import pandas as pd

# TODO: 데이터 기간 확정되면 팀 회의에서 날짜 다시 정하기
TRAIN_END_DATE = "2025-06-30"


def train_test_split_by_date(
    df: pd.DataFrame,
    date_col: str = "base_date",
    train_end: str = TRAIN_END_DATE,
):
    """df에서 date_col 기준으로 train_end까지는 train, 이후는 test로 나눈다."""
    cutoff = pd.Timestamp(train_end)
    dates = pd.to_datetime(df[date_col])
    train_df = df[dates <= cutoff].reset_index(drop=True)
    test_df = df[dates > cutoff].reset_index(drop=True)
    return train_df, test_df
