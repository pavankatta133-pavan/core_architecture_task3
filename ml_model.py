from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_model(config):
    model_name = config["model"]["name"]
    seed = config["data"]["seed"]

    if model_name == "DummyClassifier":
        model = DummyClassifier(
            strategy=config["model"]["strategy"],
            random_state=seed
        )
    elif model_name == "LogisticRegression":
        model = LogisticRegression(
            random_state=seed,
            max_iter=1000
        )
    elif model_name == "RandomForest":
        model = RandomForestClassifier(
            random_state=seed,
            n_estimators=100
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Model selected : {model_name}")
    print(f"Model ready ✅")
    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    print("Model trained ✅")
    return model


def predict(model, X):
    return model.predict(X)