import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

def evaluate():
    print("Loading processed splits...")
    try:
        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")
    except FileNotFoundError:
        print("Run python3 -m src.data_processing first.")
        return

    features = ['heart_rate', 'sys_bp', 'resp_rate', 'temp', 'arrival_hour', 'shock_index', 'chronic_conditions']
    features = [f for f in features if f in train_df.columns]

    target = 'target_triage_acuity' if 'target_triage_acuity' in train_df.columns else 'triage_level'

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    equity_test = test_df['equity_group']

    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft')

    print("Training baseline classifiers...")
    rf.fit(X_train, y_train)
    ensemble.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_pred_ens = ensemble.predict(X_test)

    print("\n================ EVALUATION METRICS ================")
    print(f"Random Forest Accuracy:          {accuracy_score(y_test, y_pred_rf):.3f}")
    print(f"Random Forest Macro-F1:          {f1_score(y_test, y_pred_rf, average='macro'):.3f}")
    print(f"Soft-Voting Ensemble Accuracy:   {accuracy_score(y_test, y_pred_ens):.3f}")
    print(f"Soft-Voting Ensemble Macro-F1:   {f1_score(y_test, y_pred_ens, average='macro'):.3f}")

    print("\n--- Critical Under-Triage Rates (ESI 1-2 marked as 3-5) ---")
    critical_mask = (y_test <= 2)
    for group in sorted(equity_test.unique()):
        g_mask = (equity_test == group) & critical_mask
        if g_mask.sum() > 0:
            under_triaged = (y_pred_rf[g_mask] > 2).sum()
            rate = (under_triaged / g_mask.sum()) * 100
            print(f"  {group:15s}: {rate:5.1f}% ({under_triaged}/{g_mask.sum()})")

if __name__ == "__main__":
    evaluate()