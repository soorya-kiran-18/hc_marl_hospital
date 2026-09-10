import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings("ignore")

# Features known at/before the triage decision. wait_time_minutes,
# intervention_iv_fluids, race and insurance are excluded from the model:
# the first two happen after triage (would leak the label), race and
# insurance are held out and used only for the fairness check below.
FEATURES = [
    'heart_rate', 'sys_bp', 'dias_bp', 'resp_rate', 'temp', 'spo2', 'pain_score',
    'age', 'arrival_hour', 'shock_index', 'chronic_conditions',
    'ems_arrival_flag', 'injury_flag',
]

EQUITY_GROUPS = ['Private', 'Medicaid', 'Medicare', 'SelfPay', 'Other/Unknown']


def load_splits():
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    return train_df, val_df, test_df


def build_models():
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    lr = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    dt = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
    nb = GaussianNB()
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    ensemble = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('gb', gb)], voting='soft')
    return {
        'Logistic Regression': lr,
        'Decision Tree': dt,
        'Naive Bayes': nb,
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'MLP': mlp,
        'Soft-Voting Ensemble': ensemble,
    }


def evaluate():
    print("Loading processed splits...")
    try:
        train_df, val_df, test_df = load_splits()
    except FileNotFoundError:
        print("Run python3 -m src.data_processing first.")
        return

    features = [f for f in FEATURES if f in train_df.columns]
    target = 'target_triage_acuity' if 'target_triage_acuity' in train_df.columns else 'triage_level'

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(train_df[features]))
    X_val = scaler.transform(imputer.transform(val_df[features]))
    X_test = scaler.transform(imputer.transform(test_df[features]))
    y_train, y_val, y_test = train_df[target], val_df[target], test_df[target]
    equity_test = test_df['equity_group'] if 'equity_group' in test_df.columns else pd.Series(['Other/Unknown'] * len(test_df))

    sample_weight = compute_sample_weight('balanced', y_train)

    models = build_models()
    val_scores, fitted = {}, {}

    print("Training baseline classifiers...")
    for name, model in models.items():
        if name == 'Gradient Boosting':
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        fitted[name] = model
        val_pred = model.predict(X_val)
        val_scores[name] = f1_score(y_val, val_pred, average='macro')

    best_name = max(val_scores, key=val_scores.get)
    print(f"\nBest model on validation (macro-F1): {best_name} ({val_scores[best_name]:.3f})")

    os.makedirs("docs", exist_ok=True)
    rows = []
    for name, model in fitted.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        except Exception:
            roc_auc = np.nan
        rows.append({'model': name, 'accuracy': acc, 'macro_f1': macro_f1,
                     'weighted_kappa': kappa, 'roc_auc': roc_auc})

    results_df = pd.DataFrame(rows).sort_values('macro_f1', ascending=False)
    results_df.to_csv("docs/triage_model_metrics.csv", index=False)

    print("\n================ TEST SET METRICS ================")
    print(results_df.to_string(index=False))

    # Confusion matrix for the best model
    best_model = fitted[best_name]
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=[1, 2, 3, 4, 5])
    disp = ConfusionMatrixDisplay(cm, display_labels=[f"ESI {i}" for i in range(1, 6)])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name} (test set)")
    plt.tight_layout()
    plt.savefig("docs/confusion_matrix_best_model.png", dpi=150)
    plt.close(fig)

    # Fairness check: under-triage rate = fraction of true ESI 1-2 patients
    # predicted as ESI 3-5, broken down by insurance equity group.
    print("\n--- Fairness Check: Under-Triage Rate by Insurance Group (best model) ---")
    critical_mask = (y_test <= 2)
    fairness_rows = []
    for group in EQUITY_GROUPS:
        g_mask = (equity_test.values == group) & critical_mask.values
        n = g_mask.sum()
        if n > 0:
            under_triaged = (y_pred_best[g_mask] > 2).sum()
            rate = under_triaged / n * 100
            fairness_rows.append({'equity_group': group, 'n_critical': int(n),
                                   'under_triaged': int(under_triaged), 'under_triage_rate_pct': rate})
            print(f"  {group:15s}: {rate:5.1f}% ({under_triaged}/{n})")
        else:
            print(f"  {group:15s}: no critical (ESI 1-2) patients in test set")

    pd.DataFrame(fairness_rows).to_csv("docs/fairness_under_triage_rates.csv", index=False)
    print("\nSaved: docs/triage_model_metrics.csv, docs/confusion_matrix_best_model.png, docs/fairness_under_triage_rates.csv")

    return results_df, best_name


if __name__ == "__main__":
    evaluate()
