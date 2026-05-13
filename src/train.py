"""Entrenamiento y comparación de modelos para YouTube Boost AI.

Diseñado para terminar siempre, incluso en máquinas modestas:
- HistGradientBoosting es **opcional** (env var ``ENABLE_HGB``).
- Cada modelo tiene un *timeout* configurable (env var ``MODEL_TIMEOUT_S``).
- Si un modelo agota el timeout o falla, el entrenamiento continúa con los demás.
- El criterio de selección prioriza F1 → ROC-AUC → recall (no accuracy).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from .config import (
    MODEL_METADATA_PATH,
    MODEL_PATH,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from .features import create_boost_candidate_target, normalize_training_dataframe

TEXT_COL = "text_total"
# Features de ENTRENAMIENTO. Excluyen deliberadamente las variables que
# componen ``organic_performance_score`` (la etiqueta) para evitar leakage:
# views_per_day, engagement_rate, like_rate, comment_rate, text_power_score,
# duration_fit_score, y los conteos absolutos views/likes/comments y sus log.
#
# El modelo predice potencial a partir de señales independientes del score:
# texto (TF-IDF sobre title+description+tags), longitudes, flags textuales,
# categoría y duración nominal. Esto refleja la situación real de un creador
# evaluando un video ANTES de publicarlo o sin fiarse de métricas tempranas.
NUMERIC_COLS = [
    "title_len", "description_len", "text_total_word_count",
    "cta_flag", "urgency_flag", "trust_flag", "promo_flag", "benefit_flag", "price_flag",
    "duration_seconds",
]
CATEGORICAL_COLS = ["category_id"]
TARGET_COL = "boost_candidate"

# Para que ``predict.py`` use las MISMAS columnas que train, las exporta:
TRAINING_FEATURE_COLUMNS = [TEXT_COL] + NUMERIC_COLS + CATEGORICAL_COLS

# Configuración runtime via env vars.
ENABLE_HGB = os.getenv("ENABLE_HGB", "0") == "1"  # opt-in: por defecto OFF
MODEL_TIMEOUT_S = int(os.getenv("MODEL_TIMEOUT_S", "180"))  # 3 min por modelo

# Indicador del origen del dataset (sintético vs real). Se persiste en metadata.
DATASET_KIND_ENV = os.getenv("DATASET_KIND", "auto")  # auto | synthetic | real


def load_raw_csvs(raw_dir: Path = RAW_DATA_DIR) -> Tuple[pd.DataFrame, List[str]]:
    """Carga todos los CSV de ``data/raw`` y devuelve (dataframe, lista_de_archivos)."""
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron CSV en {raw_dir}. Agrega al menos un dataset (público o demo) y reintenta."
        )
    frames: List[pd.DataFrame] = []
    used: List[str] = []
    for file in files:
        try:
            df = pd.read_csv(file, low_memory=False)
        except Exception as exc:
            print(f"[train] No se pudo leer {file.name}: {exc}")
            continue
        df["source_file"] = file.name
        frames.append(df)
        used.append(file.name)
    if not frames:
        raise FileNotFoundError(f"Ninguno de los CSV en {raw_dir} pudo leerse correctamente.")
    return pd.concat(frames, ignore_index=True, sort=False), used


def detect_dataset_kind(used_files: List[str]) -> str:
    """Heurística para reportar si el dataset es real o sintético.

    - ``synthetic``: todos los archivos contienen "demo".
    - ``real_kaggle``: hay al menos un archivo cuyo nombre coincide con
      patrones de Kaggle conocidos (``trending``, ``kaggle``, ``daily_``).
    - ``real_or_mixed``: archivos sin patrón claro pero presentes.
    - ``unknown``: sin archivos.
    """
    if DATASET_KIND_ENV in {"synthetic", "real", "real_kaggle", "real_or_mixed"}:
        return DATASET_KIND_ENV
    if not used_files:
        return "unknown"
    if all("demo" in f.lower() for f in used_files):
        return "synthetic"
    real_markers = ("trending", "kaggle", "daily_", "datasnaek", "youtube_")
    if any(any(marker in f.lower() for marker in real_markers) for f in used_files):
        return "real_kaggle"
    return "real_or_mixed"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    text_pipe = TfidfVectorizer(max_features=700, ngram_range=(1, 2), min_df=2)
    return ColumnTransformer(
        [
            ("text", text_pipe, TEXT_COL),
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def get_models() -> Dict[str, object]:
    """Modelos comparativos sin dependencias externas pesadas.

    HistGradientBoosting queda detrás del flag ``ENABLE_HGB`` por dos motivos:
    1) en datasets pequeños con TF-IDF muy ancho su tiempo es desproporcionado;
    2) en CPU compartida (HF Spaces gratis) puede no terminar en tiempo razonable.

    Para activarlo: ``ENABLE_HGB=1 python -m src.train``.
    """
    models: Dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=1200, class_weight="balanced", solver="liblinear"
        ),
        "linear_svc": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=4000, random_state=42),
            method="sigmoid",
            cv=3,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
            max_depth=12,
        ),
    }
    if ENABLE_HGB:
        # Import lazy: solo si está habilitado.
        from sklearn.ensemble import HistGradientBoostingClassifier

        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_depth=6,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.15,
            random_state=42,
        )
    return models


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    preds = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        rng = scores.max() - scores.min()
        probs = (scores - scores.min()) / (rng + 1e-9) if rng > 0 else np.zeros_like(scores, dtype=float)
    else:
        probs = preds
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if len(set(y_test)) == 2 else 0.0,
    }
    cm = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return metrics


# ---------------------------------------------------------------------------
# Entrenamiento individual con timeout (proceso aislado)
# ---------------------------------------------------------------------------

def _train_one_worker(name: str, payload_path: str, out_path: str) -> None:
    """Worker ejecutado en un proceso aparte. Lee payload pickle y guarda salida pickle."""
    try:
        payload = joblib.load(payload_path)
        estimator = payload["estimator"]
        X_train = payload["X_train"]
        y_train = payload["y_train"]
        X_test = payload["X_test"]
        y_test = payload["y_test"]
        preprocessor = build_preprocessor()
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)
        metrics["model"] = name
        joblib.dump({"ok": True, "pipe": pipe, "metrics": metrics}, out_path)
    except Exception as exc:
        joblib.dump({"ok": False, "error": f"{exc}\n{traceback.format_exc()}"}, out_path)


def _train_with_timeout(
    name: str, estimator: object,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    timeout_s: int,
    workdir: Path,
) -> Optional[Dict[str, Any]]:
    """Entrena un modelo en un proceso hijo con timeout. Devuelve dict o None."""
    payload_path = workdir / f"{name}_payload.joblib"
    out_path = workdir / f"{name}_out.joblib"
    joblib.dump(
        {"estimator": estimator, "X_train": X_train, "y_train": y_train,
         "X_test": X_test, "y_test": y_test},
        payload_path,
    )

    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_train_one_worker, args=(name, str(payload_path), str(out_path)))
    t0 = time.time()
    proc.start()
    proc.join(timeout=timeout_s)
    elapsed = time.time() - t0

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        print(f"[train] {name:<25} TIMEOUT tras {elapsed:.1f}s; modelo descartado.")
        return None

    if not out_path.exists():
        print(f"[train] {name:<25} sin salida (exit_code={proc.exitcode}, {elapsed:.1f}s); descartado.")
        return None

    try:
        result = joblib.load(out_path)
    except Exception as exc:
        print(f"[train] {name:<25} no se pudo leer salida: {exc}")
        return None
    if not result.get("ok"):
        print(f"[train] {name:<25} FALLO: {result.get('error', 'sin detalle')}")
        return None
    metrics = result["metrics"]
    print(f"[train] {name:<25} f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f} recall={metrics['recall']:.4f} ({elapsed:.1f}s)")
    return result


def _validate_dataset(dataset: pd.DataFrame) -> None:
    if len(dataset) < 30:
        raise ValueError(f"Dataset muy pequeño para entrenar ({len(dataset)} filas). Mínimo recomendado: 30.")
    if TARGET_COL not in dataset.columns:
        raise ValueError(f"Falta la columna objetivo {TARGET_COL!r} tras el preprocesamiento.")
    classes = dataset[TARGET_COL].unique()
    if len(classes) < 2:
        raise ValueError(
            f"La variable objetivo {TARGET_COL!r} tiene una sola clase ({classes}). "
            "Ajusta el percentil_threshold o agrega más datos."
        )


def _ensure_required_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    work = dataset.copy()
    for col in NUMERIC_COLS:
        if col not in work.columns:
            work[col] = 0
    for col in CATEGORICAL_COLS:
        if col not in work.columns:
            work[col] = "unknown"
    if TEXT_COL not in work.columns:
        work[TEXT_COL] = ""
    return work


def _generate_eda(dataset: pd.DataFrame, target_threshold: float, percentile: float, dataset_kind: str, used_files: List[str]) -> None:
    """Genera ``outputs/eda_summary.md`` y ``outputs/eda_summary.json`` reales."""
    OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

    columns = list(dataset.columns)
    duplicates = int(dataset.duplicated().sum())
    missing = {c: int(dataset[c].isna().sum()) for c in columns if dataset[c].isna().any()}
    n_rows = int(len(dataset))
    n_cols = int(len(columns))

    def _series_stats(s: pd.Series) -> Dict[str, float]:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return {}
        return {
            "count": float(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.5)),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
        }

    interesting_cols = [
        "views", "likes", "comments",
        "engagement_rate", "like_rate", "comment_rate", "views_per_day",
        "duration_seconds", "title_len", "description_len", "text_total_word_count",
    ]
    stats = {col: _series_stats(dataset[col]) for col in interesting_cols if col in dataset.columns}

    cta_present = float(dataset["cta_flag"].mean()) if "cta_flag" in dataset.columns else 0.0
    promo_present = float(dataset["promo_flag"].mean()) if "promo_flag" in dataset.columns else 0.0
    target_balance = dataset[TARGET_COL].value_counts(normalize=True).to_dict() if TARGET_COL in dataset.columns else {}
    category_dist: Dict[str, float] = {}
    if "category_id" in dataset.columns:
        vc = dataset["category_id"].astype(str).value_counts(normalize=True).head(15)
        category_dist = {str(k): float(v) for k, v in vc.items()}

    json_summary: Dict[str, Any] = {
        "dataset_kind": dataset_kind,
        "source_files": used_files,
        "rows": n_rows,
        "cols": n_cols,
        "columns": columns,
        "duplicates": duplicates,
        "missing": missing,
        "stats": stats,
        "cta_flag_mean": round(cta_present, 4),
        "promo_flag_mean": round(promo_present, 4),
        "category_top_share": category_dist,
        "target": TARGET_COL,
        "target_threshold": float(target_threshold),
        "target_percentile": float(percentile),
        "target_balance": {str(k): float(v) for k, v in target_balance.items()},
    }
    (OUTPUTS_DIR / "eda_summary.json").write_text(
        json.dumps(json_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md: List[str] = []
    md.append("# EDA — YouTube Boost AI\n")
    md.append(f"## Origen del dataset\n- Tipo detectado: **{dataset_kind}**\n")
    if used_files:
        md.append("- Archivos usados:\n")
        for f in used_files:
            md.append(f"  - `{f}`\n")
    if dataset_kind == "synthetic":
        md.append(
            "\n> ⚠️ Este EDA proviene del **dataset sintético** incluido para validación funcional.\n"
            "> Para la versión final hay que entrenar con datasets públicos reales (ver "
            "`scripts/download_kaggle_datasets.py` y `docs/DATASETS.md`).\n"
        )
    md.append(f"\n## Dimensiones\n- Filas: **{n_rows}**\n- Columnas: **{n_cols}**\n")
    md.append("\n## Columnas\n")
    md.append(", ".join(f"`{c}`" for c in columns) + "\n")
    md.append(f"\n## Calidad\n- Duplicados: **{duplicates}**\n")
    if missing:
        md.append("- Faltantes detectados:\n")
        for c, m in missing.items():
            md.append(f"  - `{c}`: {m}\n")
    else:
        md.append("- Sin faltantes críticos.\n")
    md.append(f"\n## Señales textuales\n- Tasa de presencia de CTA: {cta_present:.2%}\n- Tasa de presencia de promo: {promo_present:.2%}\n")
    if category_dist:
        md.append("\n## Top categorías (share)\n")
        for k, v in category_dist.items():
            md.append(f"- `{k}`: {v:.2%}\n")
    md.append("\n## Estadísticas principales\n")
    for col, st in stats.items():
        if not st:
            continue
        md.append(f"### {col}\n")
        md.append(
            f"- count: {st['count']:.0f} | mean: {st['mean']:.4f} | std: {st['std']:.4f}\n"
            f"- min: {st['min']:.4f} | p25: {st['p25']:.4f} | p50: {st['p50']:.4f} | p75: {st['p75']:.4f} | max: {st['max']:.4f}\n"
        )
    md.append(f"\n## Variable objetivo `{TARGET_COL}`\n")
    md.append(f"- Percentil para etiquetar como candidato: **{percentile:.2f}**\n")
    md.append(f"- Umbral del score orgánico: **{target_threshold:.4f}**\n")
    if target_balance:
        for k, v in target_balance.items():
            md.append(f"- Clase `{k}`: {v:.2%}\n")
    md.append(
        "\n## Lectura metodológica\n"
        "Las variables de YouTube presentan asimetría fuerte; por eso el pipeline aplica `log1p` "
        "a views, likes y comments y crea tasas relativas (engagement_rate, like_rate, comment_rate) "
        "y `views_per_day`. La etiqueta `boost_candidate` representa el **potencial orgánico relativo** "
        "calculado como un score ponderado, **no un ROI causal real**.\n"
    )
    (OUTPUTS_DIR / "eda_summary.md").write_text("".join(md), encoding="utf-8")


def deduplicate_by_video_id(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplica por ``video_id`` quedándose con la fila de máximo views.

    Los datasets de "trending" suelen registrar el MISMO video varias veces
    (una entrada por día que apareció en trending). Para entrenamiento queremos
    una sola fila por video, idealmente la del pico de views.
    """
    if "video_id" not in df.columns:
        return df
    before = len(df)
    work = df.sort_values("views", ascending=False).drop_duplicates(subset=["video_id"], keep="first").reset_index(drop=True)
    print(f"[train] Dedup por video_id: {before} → {len(work)} filas (-{before - len(work)})")
    return work


def maybe_subsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Sub-muestrea de forma estratificada por boost_candidate si el dataset es enorme."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    print(f"[train] Sub-muestreo estratificado: {len(df)} → {max_rows} filas")
    if TARGET_COL in df.columns and df[TARGET_COL].nunique() == 2:
        # Sub-muestreo manteniendo proporción de clases.
        ratio = max_rows / len(df)
        sampled = (
            df.groupby(TARGET_COL, group_keys=False)
            .apply(lambda g: g.sample(n=max(int(round(len(g) * ratio)), 1), random_state=42))
            .reset_index(drop=True)
        )
        return sampled
    return df.sample(n=max_rows, random_state=42).reset_index(drop=True)


def train_all(percentile: float = 0.70) -> Tuple[Optional[Pipeline], pd.DataFrame, Dict[str, Any]]:
    OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
    PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)

    raw, used_files = load_raw_csvs()
    print(f"[train] Filas crudas leídas: {len(raw)} (de {len(used_files)} archivo/s)")
    dataset_kind = detect_dataset_kind(used_files)
    print(f"[train] Origen del dataset: {dataset_kind}")

    print("[train] Normalizando columnas y construyendo features...")
    normalized = normalize_training_dataframe(raw)
    normalized = deduplicate_by_video_id(normalized)

    # Para datasets muy grandes, opcional reducir vía MAX_TRAIN_ROWS env var.
    max_rows = int(os.getenv("MAX_TRAIN_ROWS", "0"))
    dataset = create_boost_candidate_target(normalized, percentile=percentile)
    dataset = maybe_subsample(dataset, max_rows=max_rows)
    dataset = _ensure_required_columns(dataset)
    _validate_dataset(dataset)

    dataset.to_csv(PROCESSED_DATA_DIR / "youtube_training_dataset.csv", index=False)
    print(f"[train] Dataset procesado guardado en data/processed/youtube_training_dataset.csv ({len(dataset)} filas)")
    target_threshold = float(dataset["target_threshold"].iloc[0]) if "target_threshold" in dataset.columns else 0.0
    _generate_eda(dataset, target_threshold=target_threshold, percentile=percentile,
                  dataset_kind=dataset_kind, used_files=used_files)

    X = dataset[[TEXT_COL] + NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = dataset[TARGET_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    workdir = OUTPUTS_DIR / "_train_workspace"
    workdir.mkdir(exist_ok=True, parents=True)

    rows: List[Dict[str, Any]] = []
    trained: Dict[str, Pipeline] = {}
    for name, estimator in get_models().items():
        result = _train_with_timeout(
            name, estimator, X_train, y_train, X_test, y_test,
            timeout_s=MODEL_TIMEOUT_S, workdir=workdir,
        )
        if result is None:
            continue
        rows.append(result["metrics"])
        trained[name] = result["pipe"]

    # Limpia workspace temporal.
    try:
        for f in workdir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
        workdir.rmdir()
    except Exception:
        pass

    if not rows:
        # Dejamos un metadata mínimo pero el comando termina sin levantar excepción
        # para que `python -m src.train` no rompa pipelines de CI/CD.
        print("[train] ⚠ Ningún modelo logró entrenar a tiempo. Revisa el dataset y dependencias.")
        empty_meta = {
            "best_model": None,
            "selection_criterion": "f1_then_roc_auc_then_recall",
            "target": TARGET_COL,
            "rows": int(len(dataset)),
            "dataset_kind": dataset_kind,
            "source_files": used_files,
            "metrics": [],
            "warning": "Ningún modelo terminó en el tiempo MODEL_TIMEOUT_S.",
        }
        MODEL_METADATA_PATH.write_text(json.dumps(empty_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return None, pd.DataFrame(), empty_meta

    comparison = pd.DataFrame(rows).sort_values(
        ["f1", "roc_auc", "recall"], ascending=False
    ).reset_index(drop=True)
    comparison.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    best_name = str(comparison.iloc[0]["model"])
    best_model = trained[best_name]
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(best_model, MODEL_PATH)

    metadata: Dict[str, Any] = {
        "best_model": best_name,
        "selection_criterion": "f1_then_roc_auc_then_recall",
        "target": TARGET_COL,
        "percentile_threshold": percentile,
        "rows": int(len(dataset)),
        "positive_rate": float(y.mean()),
        "feature_columns": [TEXT_COL] + NUMERIC_COLS + CATEGORICAL_COLS,
        "metrics": comparison.to_dict(orient="records"),
        "dataset_kind": dataset_kind,
        "source_files": used_files,
        "trained_with_synthetic": dataset_kind == "synthetic",
        "model_timeout_s": MODEL_TIMEOUT_S,
        "hgb_enabled": ENABLE_HGB,
        "methodological_note": (
            "boost_candidate es una etiqueta derivada del potencial orgánico; "
            "no representa ROI causal real."
        ),
        "probability_clipping": "Las probabilidades de inferencia se recortan a [0.02, 0.98] para evitar absolutos.",
    }
    MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return best_model, comparison, metadata


if __name__ == "__main__":
    model, comparison, metadata = train_all(percentile=0.70)
    print("\n== Comparación final ==")
    if not comparison.empty:
        print(comparison.to_string(index=False))
        print("\n== Mejor modelo ==", metadata.get("best_model"))
    else:
        print("Sin modelos entrenados.")
    print("== Origen del dataset ==", metadata.get("dataset_kind"))
