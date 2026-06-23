# data_loader.py
# Loads and splits the PlaceMux dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    """Load the PlaceMux student dataset."""
    data = {
        "student_id":      [1,  2,  3,  4,  5,  6,  7,  8,  9,  10],
        "python_score":    [85, 40, 70, 55, 90, 30, 75, 60, 95, 45],
        "sql_score":       [70, 60, 80, 40, 85, 50, 65, 55, 90, 35],
        "communication":   [80, 55, 65, 70, 88, 45, 72, 60, 91, 50],
        "problem_solving": [75, 50, 70, 60, 92, 35, 68, 58, 94, 42],
        "placed":          [1,  0,  1,  0,  1,  0,  1,  0,  1,  0]
    }
    df = pd.DataFrame(data)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns ✅")
    return df


def split_data(df, config):
    """Split data into train, validation and test sets."""
    seed      = config["data"]["seed"]
    test_size = config["data"]["test_size"]
    val_size  = config["data"]["val_size"]
    target    = config["features"]["target"]
    drop_cols = config["features"]["drop_columns"]

    # Drop unnecessary columns
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=[target])
    y = df[target]

    # 80% train+val, 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=seed, stratify=y
    )

    # 75% train, 25% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size, random_state=seed
    )

    print(f"Train : {len(X_train)} rows")
    print(f"Val   : {len(X_val)} rows")
    print(f"Test  : {len(X_test)} rows")
    print("Data split complete ✅")

    return X_train, X_val, X_test, y_train, y_val, y_test