import joblib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessing import load_dataset, preprocess
from src.models.trainer import best_model, train_all
from src.visualization.plots import plot_confusion_matrix, plot_model_comparison, plot_roc_curves

OUTPUTS_MODELS = Path("outputs/models")
OUTPUTS_FIGURES = Path("outputs/figures")
OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)


def print_results(results: dict):
    header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'CV AUC':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, m in results.items():
        print(
            f"{name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f} "
            f"{m['cv_roc_auc_mean']:>7.4f}±{m['cv_roc_auc_std']:.4f}"
        )
    print("=" * len(header))


def main():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"  Samples: {len(X)} | Features: {X.shape[1]} | Classes: {y.value_counts().to_dict()}")

    print("\nPreprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess(X, y)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    print("\nTraining models...")
    results, trained_models = train_all(X_train, y_train, X_test, y_test)
    print_results(results)

    winner = best_model(results)
    print(f"\nBest model: {winner}  (ROC-AUC={results[winner]['roc_auc']:.4f})")

    print("\nSaving artifacts...")
    joblib.dump(trained_models[winner], OUTPUTS_MODELS / "best_model.pkl")
    joblib.dump(scaler, OUTPUTS_MODELS / "scaler.pkl")
    pd.DataFrame(results).T.to_csv(OUTPUTS_MODELS / "results.csv")
    print("  Saved to outputs/models/")

    print("\nGenerating plots...")
    plot_confusion_matrix(
        trained_models[winner], X_test, y_test, winner,
        save_path=str(OUTPUTS_FIGURES / "confusion_matrix.png")
    )
    plot_roc_curves(
        trained_models, X_test, y_test,
        save_path=str(OUTPUTS_FIGURES / "roc_curves.png")
    )
    plot_model_comparison(
        results,
        save_path=str(OUTPUTS_FIGURES / "model_comparison.png")
    )
    print("  Saved to outputs/figures/")
    print("\nDone.")


if __name__ == "__main__":
    main()
