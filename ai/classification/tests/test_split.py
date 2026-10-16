"""split.py가 train/test를 제대로 나누는지 확인하는 테스트.

python -m pytest ai/classification/tests/test_split.py  (repo 루트에서)
"""
import pandas as pd

from ai.classification.split import train_test_split_by_date


def _dummy_df():
    dates = pd.date_range("2025-01-01", periods=400, freq="B")
    return pd.DataFrame({
        "base_date": dates,
        "code": "005930",
        "is_crash": False,
    })


def test_no_overlap():
    df = _dummy_df()
    train_df, test_df = train_test_split_by_date(df, train_end="2025-06-30")
    train_dates = set(train_df["base_date"])
    test_dates = set(test_df["base_date"])
    assert train_dates.isdisjoint(test_dates)


def test_test_is_strictly_after_train():
    df = _dummy_df()
    train_df, test_df = train_test_split_by_date(df, train_end="2025-06-30")
    assert train_df["base_date"].max() < test_df["base_date"].min()


def test_all_rows_covered():
    df = _dummy_df()
    train_df, test_df = train_test_split_by_date(df, train_end="2025-06-30")
    assert len(train_df) + len(test_df) == len(df)
