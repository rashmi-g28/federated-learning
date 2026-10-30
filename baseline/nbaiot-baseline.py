# ============================================================
# N-BaIoT MULTI-DEVICE MODEL COMPARISON
#
# Purpose:
#   Evaluate all five models (Decision Tree, Random Forest, KNN,
#   SVM, XGBoost) across every benign-device CSV inside the
#   "normal" folder, using the same Mirai attack traffic from
#   N-BaIoT for every comparison.
#
# This keeps the evaluation balanced and fair, avoiding both
# overfitting (training on low-volume data) and underfitting
# (comparing models on inconsistent sample sizes across devices).
# ============================================================

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# SETTINGS
# ============================================================

BASE_PATH = Path(__file__).resolve().parent
NORMAL_DIR = BASE_PATH / "normal"
MIRAI_DIR = BASE_PATH / "N-BaIoT" / "mirai_attacks"

MAX_SAMPLES_PER_CLASS = 5000
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# DATA PREPARATION
# ============================================================

def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.dropna(axis=1, how="all")
    return X


def build_balanced_dataset(benign_file: str):
    benign = pd.read_csv(benign_file)
    benign = benign.drop_duplicates().reset_index(drop=True)
    benign["label"] = 0

    mirai_frames = []
    for attack_file in sorted(MIRAI_DIR.glob("*.csv")):
        df = pd.read_csv(attack_file)
        df = df.drop_duplicates().reset_index(drop=True)
        df["label"] = 1
        mirai_frames.append(df)

    if not mirai_frames:
        raise FileNotFoundError(f"No Mirai attack CSV files found in {MIRAI_DIR}")

    mirai = pd.concat(mirai_frames, ignore_index=True)

    benign_count = min(len(benign), MAX_SAMPLES_PER_CLASS)
    mirai_count = min(len(mirai), MAX_SAMPLES_PER_CLASS)

    benign_sample = benign.sample(n=benign_count, random_state=RANDOM_STATE)
    mirai_sample = mirai.sample(n=mirai_count, random_state=RANDOM_STATE + 1)

    final_count = min(len(benign_sample), len(mirai_sample))
    benign_sample = benign_sample.sample(n=final_count, random_state=RANDOM_STATE + 2)
    mirai_sample = mirai_sample.sample(n=final_count, random_state=RANDOM_STATE + 3)

    data = pd.concat([benign_sample, mirai_sample], ignore_index=True)
    data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X = data.drop(columns=["label"])
    y = data["label"]
    X = clean_features(X)

    print(f"\nDevice: {os.path.basename(benign_file)}")
    print(f"Balanced dataset size: {len(data)}")
    print(data["label"].value_counts().rename(index={0: "Benign", 1: "Mirai Attack"}))
    print(f"Feature count after cleaning: {X.shape[1]}")

    return X, y


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    matthews = matthews_corrcoef(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    roc_auc = roc_auc_score(y_test, y_score)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    print(f"\n{'=' * 60}\n{model_name.upper()}\n{'=' * 60}")
    print(f"Accuracy         : {accuracy * 100:.4f}%")
    print(f"Precision        : {precision * 100:.4f}%")
    print(f"Recall           : {recall * 100:.4f}%")
    print(f"F1 Score         : {f1 * 100:.4f}%")
    print(f"Balanced Accuracy: {balanced_accuracy * 100:.4f}%")
    print(f"ROC-AUC          : {roc_auc:.4f}")
    print(f"Specificity      : {specificity * 100:.4f}%")
    print(f"MCC              : {matthews:.4f}")
    print(f"FP / FN          : {fp} / {fn}")
    print(f"Training Time    : {training_time:.2f}s")
    print(f"Prediction Time  : {prediction_time:.2f}s")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Mirai Attack"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Mirai Attack"], yticklabels=["Benign", "Mirai Attack"], ax=ax)
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=300)
    plt.close(fig)

    return {
        "Device": "",
        "Model": model_name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100,
        "Balanced Accuracy": balanced_accuracy * 100,
        "ROC-AUC": roc_auc,
        "Specificity": specificity * 100,
        "MCC": matthews,
        "False Positives": fp,
        "False Negatives": fn,
        "Training Time (s)": training_time,
        "Prediction Time (s)": prediction_time,
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    if not NORMAL_DIR.exists():
        raise FileNotFoundError(f"Normal device directory not found: {NORMAL_DIR}")

    device_files = sorted(NORMAL_DIR.glob("*.csv"))
    if not device_files:
        raise FileNotFoundError(f"No CSV files found in {NORMAL_DIR}")

    all_results = []

    for device_file in device_files:
        device_name = device_file.stem
        X, y = build_balanced_dataset(str(device_file))

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_specs = [
            ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE), X_train, X_test),
            ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), X_train, X_test),
            ("KNN", KNeighborsClassifier(n_neighbors=5, n_jobs=-1), X_train_scaled, X_test_scaled),
            ("SVM", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE), X_train_scaled, X_test_scaled),
            ("XGBoost", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss"), X_train, X_test),
        ]

        for model_name, model, Xtr, Xts in model_specs:
            result = evaluate_model(model_name, model, Xtr, Xts, y_train, y_test)
            result["Device"] = device_name
            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    print("\n\n" + "=" * 110)
    print("FINAL COMPARISON OF ALL DEVICES ACROSS MODELS")
    print("=" * 110)
    print(results_df.sort_values(["Device", "Model"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    avg_by_model = results_df.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
    best_model = avg_by_model.idxmax()
    best_model_acc = avg_by_model.max()

    print("\nAverage Accuracy by Model:")
    print(avg_by_model.round(4).to_string())
    print(f"\nBest overall model: {best_model} with {best_model_acc:.4f}% average accuracy")

    device_rank = results_df.groupby("Device")["Accuracy"].mean().sort_values(ascending=False)
    print("\nAverage Accuracy by Device:")
    print(device_rank.round(4).to_string())

    output_file = BASE_PATH / "device_model_comparison.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nFull comparison saved to: {output_file}")


if __name__ == "__main__":
    main()
