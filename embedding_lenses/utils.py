import pandas as pd


def encode_labels(labels: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(labels):
        return labels
    return labels.astype("category").cat.codes
