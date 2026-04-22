from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEXT_LIKE_CATEGORICAL_FEATURES = [
    "description_clean",
    "date_month",
    "date_day_of_week",
]
NUMERIC_FEATURES = ["amount"]
TARGET_COLUMN = "category"


def build_feature_transformer() -> ColumnTransformer:
    """Shared featurization for all models: one-hot categorical + scaled numeric."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical_one_hot",
                OneHotEncoder(handle_unknown="ignore"),
                TEXT_LIKE_CATEGORICAL_FEATURES,
            ),
            ("numeric_scaled", StandardScaler(), NUMERIC_FEATURES),
        ],
        sparse_threshold=0.3,
    )
