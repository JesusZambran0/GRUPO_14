#!/usr/bin/env python
"""Entrena el XGBoost de rendimiento pagado y CPM.

Uso:
    python scripts/train_xgboost_paid_ads.py

Requiere los CSV en data/processed/paid_ads_xgboost/ o data/raw/paid_ads_xgboost/.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [ROOT/'data/processed/paid_ads_xgboost', ROOT/'data/raw/paid_ads_xgboost']
MODELS = ROOT/'models'
OUTPUTS = ROOT/'outputs'
MODELS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

def find_file(name: str) -> Path:
    for d in DATA_DIRS:
        p = d/name
        if p.exists(): return p
    raise FileNotFoundError(f'No encontré {name} en {DATA_DIRS}')

def main() -> None:
    train = pd.read_csv(find_file('paid_ads_xgboost_train_ready_no_leakage.csv'))
    test = pd.read_csv(find_file('paid_ads_xgboost_test_ready_no_leakage.csv'))
    y_cols = ['paid_performance_score', 'cpm']
    X_cols = [c for c in train.columns if c not in y_cols]
    cat_cols = [c for c in X_cols if train[c].dtype == 'object']
    num_cols = [c for c in X_cols if c not in cat_cols]
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), cat_cols),
        ('num', StandardScaler(), num_cols),
    ])
    xgb = XGBRegressor(
        objective='reg:squarederror', n_estimators=350, max_depth=4,
        learning_rate=0.045, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.5, random_state=42, n_jobs=2, tree_method='hist'
    )
    pipe = Pipeline([('preprocess', pre), ('model', MultiOutputRegressor(xgb, n_jobs=1))])
    pipe.fit(train[X_cols], train[y_cols])
    pred = pipe.predict(test[X_cols])
    metrics = {}
    for i, col in enumerate(y_cols):
        yt, yp = test[col].values, pred[:, i]
        metrics[col] = {
            'mae': float(mean_absolute_error(yt, yp)),
            'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
            'r2': float(r2_score(yt, yp)),
            'mape_pct': float(np.mean(np.abs((yt - yp) / np.maximum(np.abs(yt), 1e-6))) * 100),
            'test_mean': float(np.mean(yt)),
        }
    meta = {
        'model_name': 'xgboost_paid_ads_multioutput',
        'targets': y_cols,
        'feature_columns': X_cols,
        'categorical_columns': cat_cols,
        'numeric_columns': num_cols,
        'train_rows': int(len(train)),
        'test_rows': int(len(test)),
        'source_files': ['facebook_kag_conversion.csv', 'onyx_marketing_campaign.csv'],
        'metrics': metrics,
        'threshold_gate_probability': 0.51,
        'training_note': 'XGBoost estima score pagado y CPM para priorización; no prueba causalidad ni ROI garantizado.'
    }
    pipe.target_labels_ = y_cols
    joblib.dump(pipe, MODELS/'xgboost_paid_ads.joblib')
    (MODELS/'xgboost_paid_ads_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame([{'target': k, **v} for k, v in metrics.items()]).to_csv(OUTPUTS/'xgboost_paid_ads_metrics.csv', index=False)
    sample = test.copy()
    for i, col in enumerate(y_cols): sample['pred_' + col] = pred[:, i]
    sample.head(50).to_csv(OUTPUTS/'xgboost_paid_ads_predictions_sample.csv', index=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
