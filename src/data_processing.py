import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def clean_and_process(filepath="data/nhamcs_data_2018_22.csv"):
    print(f"Loading dataset from {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    df = pd.read_csv(filepath)
    print(f"Initial raw rows: {len(df)}")

    # 1. Clean Impossible Physiological Vitals
    print("Cleaning clinical vitals...")
    vital_cols = ['heart_rate', 'sys_bp', 'dias_bp', 'resp_rate', 'temp', 'spo2']
    for col in vital_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Filter biologically impossible limits
            if col == 'heart_rate':
                df.loc[(df[col] < 20) | (df[col] > 300), col] = np.nan
            elif col == 'sys_bp':
                df.loc[(df[col] < 40) | (df[col] > 260), col] = np.nan
            elif col == 'temp':
                df.loc[(df[col] < 85) | (df[col] > 110), col] = np.nan
            df[col] = df[col].fillna(df[col].median())

    # 2. Map Equity Groups (Insurance)
    print("Mapping insurance equity groups...")
    if 'insurance' in df.columns:
        insurance_map = {
            'Private': 'Private',
            'Medicaid': 'Medicaid',
            'Medicare': 'Medicare',
            'Self-Pay': 'SelfPay',
            'SelfPay': 'SelfPay'
        }
        df['equity_group'] = df['insurance'].map(insurance_map).fillna('Other/Unknown')
    else:
        df['equity_group'] = 'Other/Unknown'

    # 3. Feature Engineering
    if 'arrival_time' in df.columns:
        df['arrival_hour'] = (pd.to_numeric(df['arrival_time'], errors='coerce') // 100).fillna(12)
    
    if 'heart_rate' in df.columns and 'sys_bp' in df.columns:
        df['shock_index'] = df['heart_rate'] / (df['sys_bp'] + 1e-5)

    hist_cols = [c for c in df.columns if c.startswith('hist_')]
    if hist_cols:
        df['chronic_conditions'] = df[hist_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    else:
        df['chronic_conditions'] = 0

    # 4. Target Variable Resolution (target_triage_acuity)
    target_col = 'target_triage_acuity' if 'target_triage_acuity' in df.columns else 'triage_level'
    df = df.dropna(subset=[target_col])
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df[df[target_col].isin([1, 2, 3, 4, 5])]

    # 5. Stratified 70/15/15 Split
    print("Executing 70/15/15 stratified split...")
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df[target_col], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df[target_col], random_state=42)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print(f"Data processing complete.")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    clean_and_process()