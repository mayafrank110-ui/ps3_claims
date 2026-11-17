import hashlib
import json
import pandas as pd
import numpy as np


def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """

    df = df.copy()

    if not 0 < training_frac < 1:
        raise ValueError("training_frac must be between 0 and 1")

    # Convert IDs into integers
    if pd.api.types.is_integer_dtype(df[id_column]):
        ids = df[id_column]
    else:
        ids = df[id_column].apply(
            lambda x: int(
                hashlib.sha256(str(x).encode("utf-8")).hexdigest(), 16
            )
        )

    # Assign observations to training or test samples
    df["sample"] = "test"
    df.loc[(ids % 100) < training_frac * 100, "sample"] = "train"

    return df
