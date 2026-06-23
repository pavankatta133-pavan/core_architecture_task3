# evaluate.py
# Single evaluation function — used for both val and test
# No duplicated eval logic anywhere in the project!

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
import mlflow


def evaluate_model(model, X, y, split_name="val"):
    """
    Evaluate model on any split (val or test).
    Same function used everywhere — no duplication!
    """
    preds = model.predict(X)

    metrics = {
        "accuracy"  : accuracy_score(y, preds),
        "f1_score"  : f1_score(y, preds, zero_division=0),
        "precision" : precision_score(y, preds, zero_division=0),
        "recall"    : recall_score(y, preds, zero_division=0)
    }

    print(f"\n{'='*45}")
    print(f"EVALUATION RESULTS — {split_name.upper()} SET")
    print(f"{'='*45}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y, preds, zero_division=0))

    return metrics


def log_metrics_to_mlflow(metrics, split_name="val"):
    """Log all metrics to MLflow experiment tracker."""
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"{split_name}_{metric_name}", value)

    print(f"Metrics logged to MLflow ({split_name}) ✅")