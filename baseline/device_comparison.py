import os
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

BASE = Path(__file__).resolve().parent
NORMAL_DIR = BASE / 'normal'
ATTACK_DIR = BASE / 'N-BaIoT' / 'mirai_attacks'
RANDOM_STATE = 42
MAX_SAMPLES_PER_CLASS = 5000


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    return df.dropna(axis=1, how='all')


rows = []
for device_file in sorted(NORMAL_DIR.glob('*.csv')):
    benign = pd.read_csv(device_file).drop_duplicates().reset_index(drop=True)
    benign['label'] = 0

    attack_frames = []
    for attack_file in sorted(ATTACK_DIR.glob('*.csv')):
        temp = pd.read_csv(attack_file).drop_duplicates().reset_index(drop=True)
        temp['label'] = 1
        attack_frames.append(temp)

    mirai = pd.concat(attack_frames, ignore_index=True)

    benign_sample = benign.sample(n=min(len(benign), MAX_SAMPLES_PER_CLASS), random_state=RANDOM_STATE)
    mirai_sample = mirai.sample(n=min(len(mirai), MAX_SAMPLES_PER_CLASS), random_state=RANDOM_STATE + 1)
    n = min(len(benign_sample), len(mirai_sample))
    benign_sample = benign_sample.sample(n=n, random_state=RANDOM_STATE + 2)
    mirai_sample = mirai_sample.sample(n=n, random_state=RANDOM_STATE + 3)

    data = pd.concat([benign_sample, mirai_sample], ignore_index=True)
    data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X = clean(data.drop(columns=['label']))
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = [
        ('Decision Tree', DecisionTreeClassifier(random_state=RANDOM_STATE), X_train, X_test),
        ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), X_train, X_test),
        ('KNN', KNeighborsClassifier(n_neighbors=5, n_jobs=-1), X_train_s, X_test_s),
        ('SVM', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE), X_train_s, X_test_s),
        ('XGBoost', XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'), X_train, X_test),
    ]

    for model_name, model, Xtr, Xval in models:
        model.fit(Xtr, y_train)
        pred = model.predict(Xval)
        rows.append({
            'Device': device_file.stem,
            'Model': model_name,
            'Accuracy': round(accuracy_score(y_test, pred) * 100, 4),
            'Precision': round(precision_score(y_test, pred, zero_division=0) * 100, 4),
            'Recall': round(recall_score(y_test, pred, zero_division=0) * 100, 4),
            'F1': round(f1_score(y_test, pred, zero_division=0) * 100, 4),
            'Balanced Accuracy': round(balanced_accuracy_score(y_test, pred) * 100, 4),
        })

summary = pd.DataFrame(rows)
summary.to_csv(BASE / 'device_model_summary.csv', index=False)
print(summary.to_string(index=False))
print('\nAverage accuracy by model:')
print(summary.groupby('Model')['Accuracy'].mean().sort_values(ascending=False).round(4).to_string())
print('\nAverage accuracy by device:')
print(summary.groupby('Device')['Accuracy'].mean().sort_values(ascending=False).round(4).to_string())
