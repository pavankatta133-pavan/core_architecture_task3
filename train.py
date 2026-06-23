import yaml
import mlflow
import random
import os
import numpy as np

from data_loader import load_data, split_data
from features    import build_feature_pipeline, apply_pipeline
from ml_model    import get_model, train_model, predict
from evaluate    import evaluate_model, log_metrics_to_mlflow


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print("Config loaded ✅")
    return config


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seeds fixed (seed={seed}) ✅")


def run_training():
    print("="*45)
    print("PLACEMUX — TRAINING HARNESS")
    print("="*45)

    config = load_config()
    fix_seeds(config["data"]["seed"])

    df = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, config)

    pipeline, features = build_feature_pipeline(config)

    X_train_s, X_val_s, X_test_s = apply_pipeline(
        pipeline, X_train, X_val, X_test
    )

    model = get_model(config)
    model = train_model(model, X_train_s, y_train)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        mlflow.log_param("model",    config["model"]["name"])
        mlflow.log_param("seed",     config["data"]["seed"])

        val_metrics = evaluate_model(model, X_val_s, y_val, "val")
        log_metrics_to_mlflow(val_metrics, "val")

        test_metrics = evaluate_model(model, X_test_s, y_test, "test")
        log_metrics_to_mlflow(test_metrics, "test")

    print("\n" + "="*45)
    print("TRAINING COMPLETE ✅")
    print("="*45)

    return model, val_metrics, test_metrics


if __name__ == "__main__":
    run_training()