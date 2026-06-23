# features.py
# Feature engineering and preprocessing pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_feature_pipeline(config):
    """
    Build a scikit-learn Pipeline for feature preprocessing.
    Any new preprocessing steps can be added here.
    """
    numeric_features = config["features"]["numeric_features"]

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    print(f"Features selected : {numeric_features}")
    print(f"Pipeline steps    : {[step[0] for step in pipeline.steps]}")
    print("Feature pipeline built ✅")

    return pipeline, numeric_features


def apply_pipeline(pipeline, X_train, X_val, X_test):
    """
    Fit pipeline on train data only.
    Transform val and test separately to avoid leakage.
    """
    X_train_scaled = pipeline.fit_transform(X_train)
    X_val_scaled   = pipeline.transform(X_val)
    X_test_scaled  = pipeline.transform(X_test)

    print("Train scaled ✅")
    print("Val   scaled ✅")
    print("Test  scaled ✅")
    print("Pipeline applied — no leakage ✅")

    return X_train_scaled, X_val_scaled, X_test_scaled