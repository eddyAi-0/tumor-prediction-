import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def cross_validate(model, X_train, y_train, cv: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
    return {"mean": scores.mean(), "std": scores.std(), "scores": scores}


def train_all(X_train, y_train, X_test, y_test) -> tuple[dict, dict]:
    results, trained = {}, {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        trained[name] = model
        results[name] = evaluate(model, X_test, y_test)
        cv = cross_validate(model, X_train, y_train)
        results[name]["cv_roc_auc_mean"] = cv["mean"]
        results[name]["cv_roc_auc_std"] = cv["std"]
    return results, trained


def best_model(results: dict) -> str:
    return max(results, key=lambda k: results[k]["roc_auc"])
