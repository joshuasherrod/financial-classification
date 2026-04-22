from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.featurization import (
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TEXT_LIKE_CATEGORICAL_FEATURES,
    build_feature_transformer,
)


def load_processed_splits(
    processed_dir: Path = Path("data/processed"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(processed_dir / "train.csv")
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    return train, val, test


def get_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_columns = [*TEXT_LIKE_CATEGORICAL_FEATURES, *NUMERIC_FEATURES]
    x = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()
    return x, y


def get_data_for_model(
    processed_dir: Path = Path("data/processed"),
):
    train, val, test = load_processed_splits(processed_dir)

    x_train, y_train = get_xy(train)
    x_val, y_val = get_xy(val)
    x_test, y_test = get_xy(test)

    feature_transformer = build_feature_transformer()
    return x_train, y_train, x_val, y_val, x_test, y_test, feature_transformer
