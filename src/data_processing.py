import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Raw NHAMCS insurance strings observed in data/nhamcs_data_2018_22.csv.
# Anything not listed here (Unknown/Blank, Other, Workers_Comp, Charity, ...)
# falls into 'Other/Unknown'.
INSURANCE_MAP = {
    'Private': 'Private',
    'Medicare': 'Medicare',
    'Medicaid/Public': 'Medicaid',
    'Self_Pay': 'SelfPay',
}


def clean_and_process(filepath="data/nhamcs_data_2018_22.csv"):
    print(f"Loading dataset from {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    df = pd.read_csv(filepath)
    print(f"Initial raw rows: {len(df)}")

    # 1. Clean impossible physiological vitals to NaN. Median-imputation is
    # left to the modeling stage (fit on the train split only) so no test/val
    # statistics leak into training.
    print("Cleaning clinical vitals...")
    if 'heart_rate' in df.columns:
        df['heart_rate'] = pd.to_numeric(df['heart_rate'], errors='coerce')
        df.loc[(df['heart_rate'] < 20) | (df['heart_rate'] > 300), 'heart_rate'] = np.nan
    if 'sys_bp' in df.columns:
        df['sys_bp'] = pd.to_numeric(df['sys_bp'], errors='coerce')
        df.loc[(df['sys_bp'] < 40) | (df['sys_bp'] > 260), 'sys_bp'] = np.nan
    if 'dias_bp' in df.columns:
        df['dias_bp'] = pd.to_numeric(df['dias_bp'], errors='coerce')
        df.loc[(df['dias_bp'] < 20) | (df['dias_bp'] > 200), 'dias_bp'] = np.nan
    if 'resp_rate' in df.columns:
        df['resp_rate'] = pd.to_numeric(df['resp_rate'], errors='coerce')
        df.loc[(df['resp_rate'] < 4) | (df['resp_rate'] > 80), 'resp_rate'] = np.nan
    if 'temp' in df.columns:
        df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
        df.loc[(df['temp'] < 85) | (df['temp'] > 110), 'temp'] = np.nan
    if 'spo2' in df.columns:
        df['spo2'] = pd.to_numeric(df['spo2'], errors='coerce')
        df.loc[(df['spo2'] < 50) | (df['spo2'] > 100), 'spo2'] = np.nan

    # 2. Map raw insurance strings into the 5 equity groups used for the
    # fairness evaluation and the RL equity penalty.
    print("Mapping insurance equity groups...")
    if 'insurance' in df.columns:
        df['equity_group'] = df['insurance'].map(INSURANCE_MAP).fillna('Other/Unknown')
    else:
        df['equity_group'] = 'Other/Unknown'

    # 3. Feature engineering (all values known at/before the triage decision,
    # so none of this introduces post-triage leakage).
    if 'arrival_time' in df.columns:
        df['arrival_hour'] = (pd.to_numeric(df['arrival_time'], errors='coerce') // 100).clip(0, 23).fillna(12)

    if 'heart_rate' in df.columns and 'sys_bp' in df.columns:
        df['shock_index'] = df['heart_rate'] / (df['sys_bp'] + 1e-5)

    hist_cols = [c for c in df.columns if c.startswith('hist_')]
    if hist_cols:
        df['chronic_conditions'] = df[hist_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    else:
        df['chronic_conditions'] = 0

    if 'ems_arrival' in df.columns:
        df['ems_arrival_flag'] = (df['ems_arrival'] == 'Yes').astype(int)

    if 'is_injury_poison' in df.columns:
        df['injury_flag'] = (~df['is_injury_poison'].isin(['No injury'])).astype(int)

    # 4. Target variable resolution (ESI 1-5, 1 = most critical)
    target_col = 'target_triage_acuity' if 'target_triage_acuity' in df.columns else 'triage_level'
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    df = df[df[target_col].isin([1, 2, 3, 4, 5])]

    # 5. Stratified 70/15/15 split by triage level
    print("Executing 70/15/15 stratified split...")
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df[target_col], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df[target_col], random_state=42)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("Data processing complete.")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("\nEquity group counts (full dataset):")
    print(df['equity_group'].value_counts())


if __name__ == "__main__":
    clean_and_process()
