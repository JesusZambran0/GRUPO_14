from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, log_loss)
from sklearn.model_selection import train_test_split

# Import project constants
from src.train import TEXT_COL, NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL, build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'evaluacion_modelo_ia'
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'data' / 'processed' / 'youtube_training_dataset.csv'
MODEL_PATH = ROOT / 'models' / 'best_model.joblib'

FEATURES = [TEXT_COL] + NUMERIC_COLS + CATEGORICAL_COLS


def safe_metric(fn, y_true, y_pred, **kwargs):
    try:
        return float(fn(y_true, y_pred, **kwargs))
    except Exception:
        return float('nan')


def get_proba(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return model.predict(X).astype(float)


def metrics_row(name: str, model, X_test, y_test, commercial=False) -> Dict:
    pred = model.predict(X_test)
    proba = get_proba(model, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
    return {
        'modelo': name,
        'tipo': 'comercial/API' if commercial else 'experimental',
        'accuracy': safe_metric(accuracy_score, y_test, pred),
        'precision': safe_metric(precision_score, y_test, pred, zero_division=0),
        'recall': safe_metric(recall_score, y_test, pred, zero_division=0),
        'f1': safe_metric(f1_score, y_test, pred, zero_division=0),
        'roc_auc': safe_metric(roc_auc_score, y_test, proba) if len(set(y_test)) == 2 else float('nan'),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def group_metrics(df: pd.DataFrame, y_true: pd.Series, pred: np.ndarray, proba: np.ndarray, group_col: str, min_n: int = 100) -> pd.DataFrame:
    work = pd.DataFrame({
        'y': y_true.values.astype(int),
        'pred': pred.astype(int),
        'proba': proba.astype(float),
        'group': df[group_col].fillna('unknown').astype(str).values if group_col in df.columns else 'unknown'
    })
    rows: List[Dict] = []
    for group, g in work.groupby('group'):
        if len(g) < min_n:
            continue
        y = g['y'].values; p = g['pred'].values; pr = g['proba'].values
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
        rows.append({
            'variable_grupo': group_col,
            'grupo': str(group),
            'n': int(len(g)),
            'prevalencia_real': float(y.mean()),
            'tasa_seleccion_predicha': float(p.mean()),
            'accuracy': safe_metric(accuracy_score, y, p),
            'precision': safe_metric(precision_score, y, p, zero_division=0),
            'recall_tpr': safe_metric(recall_score, y, p, zero_division=0),
            'f1': safe_metric(f1_score, y, p, zero_division=0),
            'fpr': float(fp / (fp + tn)) if (fp + tn) else float('nan'),
            'fnr': float(fn / (fn + tp)) if (fn + tp) else float('nan'),
            'prob_media': float(pr.mean()),
        })
    result = pd.DataFrame(rows)
    return result.sort_values(['variable_grupo','n'], ascending=[True,False])


def fairness_summary(tbl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, g in tbl.groupby('variable_grupo'):
        if g.empty:
            continue
        sel = g['tasa_seleccion_predicha'].replace([np.inf, -np.inf], np.nan).dropna()
        tpr = g['recall_tpr'].replace([np.inf, -np.inf], np.nan).dropna()
        fpr = g['fpr'].replace([np.inf, -np.inf], np.nan).dropna()
        f1 = g['f1'].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            'variable_grupo': var,
            'grupos_incluidos': int(g.shape[0]),
            'n_total_en_grupos': int(g['n'].sum()),
            'demographic_parity_difference': float(sel.max() - sel.min()) if len(sel) else np.nan,
            'disparate_impact_ratio_min_max': float(sel.min() / sel.max()) if len(sel) and sel.max() > 0 else np.nan,
            'equal_opportunity_difference_tpr': float(tpr.max() - tpr.min()) if len(tpr) else np.nan,
            'fpr_difference': float(fpr.max() - fpr.min()) if len(fpr) else np.nan,
            'f1_gap': float(f1.max() - f1.min()) if len(f1) else np.nan,
        })
    return pd.DataFrame(rows)


def make_duration_bucket(s: pd.Series) -> pd.Series:
    d = pd.to_numeric(s, errors='coerce').fillna(0)
    return pd.cut(d, bins=[-0.01, 0, 15, 60, 180, np.inf], labels=['desconocida','muy_corto_1_15s','corto_16_60s','medio_61_180s','largo_180s_plus']).astype(str)


def main():
    df = pd.read_csv(DATA, low_memory=False)
    # Drop edge rows without target/features if any.
    df = df.dropna(subset=[TARGET_COL]).copy()
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0 if c in NUMERIC_COLS else 'unknown'
    X = df[FEATURES].copy()
    y = df[TARGET_COL].astype(int)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)

    rows = []
    # Baselines
    for strategy in ['most_frequent', 'stratified']:
        dummy = DummyClassifier(strategy=strategy, random_state=42)
        dummy.fit(X_train, y_train)
        rows.append(metrics_row(f'DummyClassifier_{strategy}', dummy, X_test, y_test))

    # Rule baseline using text_power_score threshold fitted on training quantile for positive rate.
    class RuleTextPower:
        def __init__(self): self.threshold = 0.0
        def fit(self, X, y):
            pos_rate = y.mean()
            self.threshold = float(pd.to_numeric(X['text_total_word_count'], errors='coerce').fillna(0).quantile(1-pos_rate))
            return self
        def predict(self, X):
            # Simple proxy: larger text/count commercial cues than fitted threshold.
            score = pd.to_numeric(X['text_total_word_count'], errors='coerce').fillna(0)
            return (score >= self.threshold).astype(int).values
        def predict_proba(self, X):
            score = pd.to_numeric(X['text_total_word_count'], errors='coerce').fillna(0)
            z = (score - score.min())/(score.max()-score.min()+1e-9)
            return np.vstack([1-z, z]).T
    rule = RuleTextPower().fit(X_train, y_train)
    rows.append(metrics_row('Regla_simple_longitud_texto', rule, X_test, y_test))

    rows.append(metrics_row('Modelo_propuesto_LogisticRegression_TFIDF', model, X_test, y_test))
    bench = pd.DataFrame(rows)
    bench.to_csv(OUT / 'benchmark_baselines_modelo.csv', index=False)

    # Fairness
    pred = model.predict(X_test)
    proba = get_proba(model, X_test)
    test_df = df.loc[idx_test].copy()
    test_df['duration_bucket'] = make_duration_bucket(test_df['duration_seconds'])
    # source_dataset is too sparse; source_file is more explicit but not used as protected group. Include as audit of dataset origin.
    group_cols = ['country', 'category_id', 'duration_bucket', 'source_file']
    fair_tables = []
    for col in group_cols:
        min_n = 50 if col in {'duration_bucket','source_file'} else 100
        fair_tables.append(group_metrics(test_df, y_test.reset_index(drop=True), pred, proba, col, min_n=min_n))
    fair = pd.concat(fair_tables, ignore_index=True)
    fair.to_csv(OUT / 'fairness_por_grupos.csv', index=False)
    fsum = fairness_summary(fair)
    fsum.to_csv(OUT / 'fairness_resumen.csv', index=False)

    # Learning curves: fixed validation subset and stratified training subsamples for tractable reproducibility.
    # The final model metrics above are computed on the full 20% test split; curves are diagnostic.
    sizes_abs = [1000, 3000, 7000, 15000, 30000]
    lc_rows = []
    rng = np.random.default_rng(42)

    # Validation subset capped for speed while preserving class balance.
    val_parts = []
    for cls in sorted(y_test.unique()):
        ids = y_test[y_test == cls].index.to_numpy()
        target_n = min(len(ids), 3500 if cls == 0 else 1500)
        val_parts.extend(rng.choice(ids, size=target_n, replace=False).tolist())
    X_val_curve = X_test.loc[val_parts]
    y_val_curve = y_test.loc[val_parts]

    for n_total in sizes_abs:
        parts = []
        for cls in sorted(y_train.unique()):
            ids = y_train[y_train == cls].index.to_numpy()
            cls_n = max(20, int(round(n_total * float((y_train == cls).mean()))))
            cls_n = min(cls_n, len(ids))
            parts.extend(rng.choice(ids, size=cls_n, replace=False).tolist())
        X_sub = X_train.loc[parts]
        y_sub = y_train.loc[parts]
        pipe = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('model', LogisticRegression(max_iter=600, class_weight='balanced', solver='liblinear'))
        ])
        pipe.fit(X_sub, y_sub)
        for split, X_eval, y_eval in [('train', X_sub, y_sub), ('validation', X_val_curve, y_val_curve)]:
            p = pipe.predict(X_eval)
            pr = get_proba(pipe, X_eval)
            lc_rows.append({
                'n_train': int(len(X_sub)),
                'split': split,
                'accuracy': accuracy_score(y_eval, p),
                'f1': f1_score(y_eval, p, zero_division=0),
                'roc_auc': roc_auc_score(y_eval, pr),
                'log_loss': log_loss(y_eval, np.clip(pr, 1e-6, 1-1e-6)),
            })
    lc = pd.DataFrame(lc_rows)
    lc.to_csv(OUT / 'learning_curves_logistic_regression.csv', index=False)

    # Plots
    for metric, ylabel, fname in [
        ('f1', 'F1-score', 'curva_aprendizaje_f1.png'),
        ('log_loss', 'Log loss', 'curva_aprendizaje_logloss.png'),
        ('roc_auc', 'ROC-AUC', 'curva_aprendizaje_auc.png'),
    ]:
        plt.figure(figsize=(7.5, 5))
        for split in ['train', 'validation']:
            sub = lc[lc['split'] == split]
            plt.plot(sub['n_train'], sub[metric], marker='o', label=split)
        plt.xlabel('Tamaño del conjunto de entrenamiento')
        plt.ylabel(ylabel)
        plt.title(f'Curva de aprendizaje - {ylabel}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / fname, dpi=180)
        plt.close()

    # Summary JSON
    summary = {
        'n_rows': int(len(df)),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'positive_rate': float(y.mean()),
        'benchmark': bench.to_dict(orient='records'),
        'fairness_summary': fsum.to_dict(orient='records'),
        'learning_curve_last': lc[lc['n_train'] == lc['n_train'].max()].to_dict(orient='records'),
    }
    (OUT / 'evaluacion_resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
