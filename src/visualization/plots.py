import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_class_distribution(y: pd.Series, save_path: str = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts()
    ax.bar(["Malignant (0)", "Benign (1)"], [counts[0], counts[1]], color=["#e74c3c", "#2ecc71"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate([counts[0], counts[1]]):
        ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_feature_distributions(X: pd.DataFrame, y: pd.Series, n_features: int = 10, save_path: str = None):
    top_features = X.columns[:n_features]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    for i, feat in enumerate(top_features):
        axes[i].hist(X[feat][y == 0], bins=30, alpha=0.6, label="Malignant", color="#e74c3c")
        axes[i].hist(X[feat][y == 1], bins=30, alpha=0.6, label="Benign", color="#2ecc71")
        axes[i].set_title(feat, fontsize=8)
        axes[i].legend(fontsize=7)
    plt.suptitle("Feature Distributions by Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_correlation_heatmap(X: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(14, 11))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", center=0,
                linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, model_name: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Malignant", "Benign"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curves(models: dict, X_test, y_test, save_path: str = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_model_comparison(results: dict, save_path: str = None):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        vals = [results[name][m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=name, alpha=0.85)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
