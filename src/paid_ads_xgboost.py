"""Segundo modelo: predictor robusto de pauta pagada.

Contrato metodológico del módulo:
1. El modelo principal de la app estima la probabilidad de candidatura publicitaria.
2. Este módulo ejecuta el modelo XGBoost si existe y lo usa como clasificador/señal de eficiencia.
3. El CPM se calcula con un calibrador dinámico: nicho, engagement, retención, views/hora,
   duración, presupuesto y probabilidad del modelo principal.
4. Si el XGBoost no existe, falla o fue entrenado solo como clasificador, el sistema NO se cae:
   conserva la predicción con reglas calibradas y deja trazabilidad en model_status/warning.

Nota importante:
- Si tu artefacto XGBoost fue entrenado para clasificar si conviene pautar, no debe venderse como
  regresor exacto de CPM. Aquí se usa como señal de score/aptitud; el CPM queda calibrado dinámicamente.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable as IterableABC
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Tuple

# Evita que XGBoost/NumPy bloqueen CPU con demasiados hilos en despliegues pequeños.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import pandas as pd

from .config import MODELS_DIR
from .features import safe_float

XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_paid_ads.joblib"
XGBOOST_METADATA_PATH = MODELS_DIR / "xgboost_paid_ads_metadata.json"
GATE_THRESHOLD = 0.51
XGBOOST_PREDICTION_TIMEOUT_SECONDS = float(os.getenv("XGBOOST_PREDICTION_TIMEOUT_SECONDS", "4"))
_MODEL_RUNTIME_DISABLED_REASON: str | None = None

NICHE_KEYWORDS = {
    "educación": ["curso", "tutorial", "aprende", "clase", "guía", "guia", "tips", "enseño", "capacitación", "webinar"],
    "retail/ecommerce": ["compra", "tienda", "producto", "envío", "envio", "descuento", "oferta", "precio", "promo", "shop"],
    "servicios": ["agenda", "reserva", "consulta", "servicio", "cotiza", "whatsapp", "asesoría", "asesoria", "cita"],
    "entretenimiento": ["meme", "humor", "comedia", "música", "musica", "cover", "show", "podcast", "viral", "trend"],
    "tecnología": ["app", "software", "ia", "inteligencia artificial", "automatiza", "digital", "tecnología", "tecnologia", "saas"],
    "salud/bienestar": ["salud", "bienestar", "fitness", "nutrición", "nutricion", "entrenamiento", "terapia"],
    "gastronomía": ["restaurante", "comida", "menú", "menu", "receta", "chef", "sabor", "delivery"],
    "inmobiliario": ["casa", "departamento", "inmueble", "terreno", "alquiler", "venta", "propiedad"],
}

SUPPORTED_AD_CATEGORIES = [
    "educación",
    "retail/ecommerce",
    "servicios",
    "entretenimiento",
    "tecnología",
    "salud/bienestar",
    "gastronomía",
    "inmobiliario",
    "general/branding",
]

BASE_CPM_BY_NICHE = {
    "entretenimiento": 3.80,
    "gastronomía": 4.60,
    "educación": 5.40,
    "retail/ecommerce": 6.20,
    "salud/bienestar": 7.00,
    "servicios": 7.60,
    "tecnología": 8.80,
    "inmobiliario": 9.80,
    "general/branding": 5.60,
}


def _load_metadata() -> Dict[str, Any]:
    if XGBOOST_METADATA_PATH.exists():
        try:
            return json.loads(XGBOOST_METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@lru_cache(maxsize=1)
def _load_model():
    if not XGBOOST_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(XGBOOST_MODEL_PATH)
    except Exception:
        return None


def normalize_ad_category(value: Any) -> str | None:
    """Normaliza la categoría elegida en UI.

    - auto/vacío/otros: permite inferencia por texto.
    - categorías soportadas: fuerzan nicho para CPM y recomendación.
    """
    raw = str(value or "").strip().lower()
    if not raw or raw in {"auto", "automático", "automatico", "otro", "otros", "other"}:
        return None
    aliases = {
        "educacion": "educación",
        "ecommerce": "retail/ecommerce",
        "retail": "retail/ecommerce",
        "tecnologia": "tecnología",
        "salud": "salud/bienestar",
        "bienestar": "salud/bienestar",
        "gastronomia": "gastronomía",
        "branding": "general/branding",
        "general": "general/branding",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in BASE_CPM_BY_NICHE else None


def infer_ad_niche(
    title: str = "",
    description: str = "",
    transcript_text: str = "",
    ocr_text: str = "",
    selected_category: Any = None,
) -> str:
    forced = normalize_ad_category(selected_category)
    if forced:
        return forced
    text = " ".join([title or "", description or "", transcript_text or "", ocr_text or ""]).lower()
    scores = {niche: sum(1 for kw in kws if kw in text) for niche, kws in NICHE_KEYWORDS.items()}
    best, value = max(scores.items(), key=lambda kv: kv[1])
    return best if value > 0 else "general/branding"


def _season_campaign(month: int) -> str:
    if month in {3, 4, 5}:
        return "Spring"
    if month in {6, 7, 8}:
        return "Summer"
    if month in {9, 10, 11}:
        return "Fall"
    return "Winter"


def _ad_type_from_niche(niche: str, features: Dict[str, Any]) -> str:
    if niche in {"retail/ecommerce", "servicios", "gastronomía", "inmobiliario"}:
        if safe_float(features.get("promo_flag", 0)) or safe_float(features.get("price_flag", 0)):
            return "Discount"
        return "Collection"
    if niche == "educación":
        return "Story"
    if niche == "entretenimiento":
        return "Video"
    return "Brand"


def _channel_from_niche(niche: str) -> str:
    if niche in {"retail/ecommerce", "entretenimiento", "gastronomía"}:
        return "Instagram"
    if niche in {"servicios", "educación", "inmobiliario"}:
        return "Facebook"
    return "Facebook"


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _stable_jitter(text: str, span: float = 0.16) -> float:
    """Variación determinista pequeña para evitar empates sin aleatoriedad."""
    digest = hashlib.sha256((text or "general").encode("utf-8", errors="ignore")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return 1.0 + (raw - 0.5) * 2 * span


def _positive_probability_from_proba(model: Any, proba: Any) -> float | None:
    """Extrae probabilidad positiva para clasificadores binarios/multiclase."""
    try:
        if isinstance(proba, list):
            # Algunos multioutput devuelven lista de arrays. Tomamos la primera salida.
            proba = proba[0]
        row = proba[0]
        if not isinstance(row, IterableABC) or isinstance(row, (str, bytes)):
            return _clip(float(row), 0.0, 1.0)
        values = list(row)
        if not values:
            return None
        classes = getattr(model, "classes_", None)
        if classes is not None:
            try:
                class_values = list(classes)
                positive_markers = {1, "1", True, "true", "True", "yes", "apto", "eligible", "impulsar"}
                for idx, cls in enumerate(class_values):
                    if cls in positive_markers or str(cls).lower() in positive_markers:
                        return _clip(float(values[idx]), 0.0, 1.0)
            except Exception:
                pass
        return _clip(float(values[-1]), 0.0, 1.0)
    except Exception:
        return None


def _flatten_prediction(pred: Any) -> list[float]:
    """Convierte predicciones sklearn/xgboost a lista simple sin depender de numpy."""
    try:
        if hasattr(pred, "tolist"):
            pred = pred.tolist()
        if isinstance(pred, list) and pred and isinstance(pred[0], list):
            pred = pred[0]
        elif isinstance(pred, tuple) and pred and isinstance(pred[0], (list, tuple)):
            pred = pred[0]
        if isinstance(pred, (list, tuple)):
            return [float(x) for x in pred]
        return [float(pred)]
    except Exception:
        return []


def _expected_input_columns(meta: Dict[str, Any], row: Dict[str, Any]) -> list[str]:
    cols = meta.get("feature_columns") or meta.get("input_columns") or meta.get("columns")
    if isinstance(cols, list) and cols:
        return [str(c) for c in cols]
    return [k for k in row.keys() if k not in {"ad_niche", "selected_ad_category"}]


def build_paid_ad_feature_row(
    features: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
    ops: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    ops = ops or {}
    title = metadata.get("title", "") or features.get("title", "")
    description = metadata.get("description", "") or features.get("description", "")
    selected_category = (
        metadata.get("xgboost_category")
        or metadata.get("ad_category")
        or features.get("xgboost_category")
        or features.get("ad_category")
    )
    niche = infer_ad_niche(
        title,
        description,
        features.get("transcript_text", ""),
        features.get("ocr_text", ""),
        selected_category=selected_category,
    )
    now = datetime.utcnow()
    month = now.month
    weekday = now.isoweekday()
    views = max(safe_float(metadata.get("views", features.get("views", 0))), 0.0)
    likes = max(safe_float(metadata.get("likes", features.get("likes", 0))), 0.0)
    comments = max(safe_float(metadata.get("comments", features.get("comments", 0))), 0.0)
    duration = max(safe_float(metadata.get("duration_seconds", features.get("duration_seconds", 0))), 0.0)
    engagement = safe_float(features.get("engagement_rate", 0.0))
    if engagement <= 0 and views > 0:
        engagement = (likes + comments) / views
    retention = safe_float(ops.get("retention_rate", features.get("retention_rate", 0.0)))
    hours = max(safe_float(ops.get("hours_since_publication", 24)), 1.0)
    views_per_hour = views / hours if views else 0.0

    return {
        # Columnas originales compatibles con el modelo anterior.
        "source_dataset": "onyx_marketing_campaign",
        "campaign": _season_campaign(month),
        "channel": _channel_from_niche(niche),
        "device": "Mobile",
        "ad_type": _ad_type_from_niche(niche, features),
        "age": "unknown",
        "gender": "unknown",
        "city": "unknown",
        "campaign_numeric_id": 0.0,
        "ad_numeric_id": 0.0,
        "interest": 0.0,
        "month": float(month),
        "weekday": float(weekday),
        "ctr_prior": _clip(engagement, 0.001, 0.20),
        "cvr_prior": _clip(retention * 0.12, 0.001, 0.30),
        "ad_niche": niche,
        "selected_ad_category": selected_category or "auto",
        # Columnas extra defensivas: si el metadata del modelo las pide, ya existen.
        "views": views,
        "likes": likes,
        "comments": comments,
        "engagement_rate": engagement,
        "retention_rate": _clip(retention, 0.0, 1.0),
        "hours_since_publication": hours,
        "views_per_hour": views_per_hour,
        "duration_seconds": duration,
        "followers_count": safe_float(ops.get("followers_count", features.get("followers_count", 0))),
        "avg_channel_reach": safe_float(ops.get("avg_channel_reach", features.get("avg_channel_reach", 0))),
    }


def _rules_estimate(
    features: Dict[str, Any],
    metadata: Dict[str, Any],
    ops: Dict[str, Any],
    probability: float,
    budget: float,
) -> Dict[str, Any]:
    """Calibrador dinámico de CPM/CTR/VTR.

    Este bloque es intencionalmente transparente: evita CPM fijo y no depende de que el XGBoost
    sea regresor. Es el respaldo metodológico cuando el artefacto XGBoost clasifica pero no predice CPM.
    """
    title = metadata.get("title", "") or features.get("title", "")
    description = metadata.get("description", "") or features.get("description", "")
    selected_category = (
        metadata.get("xgboost_category")
        or metadata.get("ad_category")
        or features.get("xgboost_category")
        or features.get("ad_category")
    )
    niche = infer_ad_niche(
        title,
        description,
        features.get("transcript_text", ""),
        features.get("ocr_text", ""),
        selected_category=selected_category,
    )

    views = max(safe_float(metadata.get("views", features.get("views", 0))), 0.0)
    likes = max(safe_float(metadata.get("likes", features.get("likes", 0))), 0.0)
    comments = max(safe_float(metadata.get("comments", features.get("comments", 0))), 0.0)
    engagement_rate = safe_float(features.get("engagement_rate", 0.0))
    if engagement_rate <= 0 and views > 0:
        engagement_rate = (likes + comments) / views
    engagement_rate = _clip(engagement_rate, 0.0, 0.25)

    retention = _clip(safe_float(ops.get("retention_rate", features.get("retention_rate", 0.0))), 0.0, 1.0)
    hours = max(safe_float(ops.get("hours_since_publication", 24)), 1.0)
    views_per_hour = views / hours if views else 0.0
    duration = max(safe_float(metadata.get("duration_seconds", features.get("duration_seconds", 30))), 1.0)

    base_cpm = BASE_CPM_BY_NICHE.get(niche, BASE_CPM_BY_NICHE["general/branding"])

    quality = (
        _clip(probability, 0, 1) * 0.42
        + _clip(engagement_rate / 0.075, 0, 1) * 0.23
        + _clip(retention / 0.62, 0, 1) * 0.22
        + _clip(views_per_hour / 700.0, 0, 1) * 0.09
        + _clip((75 - min(duration, 75)) / 75, 0, 1) * 0.04
    )

    # CPM dinámico: calidad alta tiende a bajar CPM efectivo; baja calidad/ritmo lo sube.
    cpm = base_cpm * (1.30 - 0.42 * _clip(quality, 0, 1))

    if probability >= 0.75:
        cpm *= 0.93
    elif probability < 0.40:
        cpm *= 1.10

    if engagement_rate >= 0.08:
        cpm *= 0.92
    elif engagement_rate < 0.015:
        cpm *= 1.13

    if retention >= 0.60:
        cpm *= 0.93
    elif retention and retention < 0.25:
        cpm *= 1.12

    if views_per_hour >= 1000:
        cpm *= 0.94
    elif views_per_hour < 50:
        cpm *= 1.06

    if budget >= 100:
        cpm *= 1.06
    elif 0 < budget <= 20:
        cpm *= 0.97

    if views < 100:
        cpm *= 1.08

    fingerprint = "|".join([
        str(title),
        str(description),
        str(niche),
        f"v{round(views, -2) if views >= 100 else views}",
        f"e{engagement_rate:.4f}",
        f"r{retention:.4f}",
        f"h{round(hours, 1)}",
        f"p{probability:.4f}",
    ])
    cpm *= _stable_jitter(fingerprint, span=0.115)
    cpm = _clip(cpm, 2.20, 18.00)

    vtr = _clip(0.10 + 0.34 * quality, 0.08, 0.48)
    ctr = _clip(0.004 + engagement_rate * 0.40 + quality * 0.020, 0.003, 0.085)

    impressions = (max(budget, 0.0) / cpm) * 1000 if budget > 0 else 0.0
    paid_views = impressions * vtr
    clicks = impressions * ctr
    score = _clip(32 + probability * 44 + quality * 30 - (cpm / 18.0) * 7, 0, 100)

    return {
        "ad_niche": niche,
        "selected_ad_category": selected_category or "auto",
        "rules_quality_score": round(quality * 100, 2),
        "rules_cpm": round(cpm, 4),
        "rules_paid_performance_score": round(score, 2),
        "estimated_view_through_rate": round(vtr, 4),
        "estimated_ctr": round(ctr, 4),
        "estimated_budget_impressions": round(impressions, 0),
        "estimated_paid_views": round(paid_views, 0),
        "estimated_clicks": round(clicks, 0),
    }


def _predict_model_direct(row: Dict[str, Any]) -> Tuple[float | None, float | None, str]:
    """Predice con el artefacto XGBoost sin asumir que es regresor multi-output.

    Casos soportados:
    - Clasificador sklearn/xgboost con predict_proba: devuelve score 0-100 y cpm=None.
    - Clasificador con predict: devuelve 0/100 si la etiqueta es binaria.
    - Regresor/multioutput: intenta leer paid_performance_score y cpm.
    - Cualquier fallo: devuelve None sin romper la app.
    """
    model = _load_model()
    meta = _load_metadata()
    if model is None:
        return None, None, "modelo_no_disponible"

    input_cols = _expected_input_columns(meta, row)
    X = pd.DataFrame([{c: row.get(c, 0) for c in input_cols}])

    # 1) Preferimos predict_proba si el modelo es clasificador.
    try:
        if hasattr(model, "predict_proba"):
            prob = _positive_probability_from_proba(model, model.predict_proba(X))
            if prob is not None:
                return round(prob * 100, 4), None, "modelo_xgboost_clasificador"
    except Exception:
        pass

    # 2) Luego intentamos predict genérico.
    try:
        pred_raw = model.predict(X)
        pred_values = _flatten_prediction(pred_raw[0] if hasattr(pred_raw, "__len__") else pred_raw)
        if not pred_values:
            return None, None, "modelo_sin_salida"

        targets = getattr(model, "target_labels_", None) or meta.get("targets") or meta.get("target_columns")
        if isinstance(targets, list) and len(targets) >= 2 and len(pred_values) >= 2:
            pred_map = {str(t): float(v) for t, v in zip(targets, pred_values)}
            score = pred_map.get("paid_performance_score") or pred_map.get("score") or pred_map.get("performance_score")
            cpm = pred_map.get("cpm") or pred_map.get("predicted_cpm") or pred_map.get("target_cpm")
            return (
                _clip(float(score), 0.0, 100.0) if score is not None else None,
                _clip(float(cpm), 1.0, 25.0) if cpm is not None else None,
                "modelo_xgboost_regresor",
            )

        # Si es un único valor, lo tratamos como etiqueta/score de clasificador.
        value = float(pred_values[0])
        if 0.0 <= value <= 1.0:
            return round(value * 100, 4), None, "modelo_xgboost_clasificador_label"
        if value in {0.0, 1.0}:
            return round(value * 100, 4), None, "modelo_xgboost_clasificador_label"
        return _clip(value, 0.0, 100.0), None, "modelo_xgboost_score"
    except Exception as exc:
        return None, None, f"error_modelo:{type(exc).__name__}"


def _predict_model(row: Dict[str, Any]) -> Tuple[float | None, float | None, str]:
    """Ejecuta el artefacto XGBoost con timeout y bloqueo anti-cuelgue.

    Si el artefacto responde dentro del tiempo permitido, se usa su predicción real.
    Si se queda colgado, se desactiva solo para la sesión actual y el módulo sigue
    con predictor calibrado dinámico. Esto evita que Gradio quede cargando.
    """
    global _MODEL_RUNTIME_DISABLED_REASON

    if _MODEL_RUNTIME_DISABLED_REASON:
        return None, None, _MODEL_RUNTIME_DISABLED_REASON

    timeout_s = max(float(XGBOOST_PREDICTION_TIMEOUT_SECONDS or 4), 1.0)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xgboost_paid_ads")
    future = executor.submit(_predict_model_direct, row)
    try:
        result = future.result(timeout=timeout_s)
        executor.shutdown(wait=False, cancel_futures=True)
        return result
    except TimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        _MODEL_RUNTIME_DISABLED_REASON = f"timeout_modelo:{timeout_s:.0f}s_desactivado_en_sesion"
        return None, None, _MODEL_RUNTIME_DISABLED_REASON
    except Exception as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        _MODEL_RUNTIME_DISABLED_REASON = f"error_modelo:{type(exc).__name__}_desactivado_en_sesion"
        return None, None, _MODEL_RUNTIME_DISABLED_REASON

def _paid_recommendation(score: float, cpm: float, quality: float, passed_gate: bool) -> str:
    if not passed_gate:
        return "No se recomienda pautar todavía: el video no supera el gate principal. El CPM se muestra como referencia para planificación, no como autorización de inversión."
    if score >= 70 and cpm <= 8:
        return "Impulso viable: buen score estimado y CPM controlado. Probar con presupuesto pequeño y escalar si CTR/CPM reales confirman."
    if score >= 58:
        return "Impulso moderado: lanzar prueba A/B con presupuesto limitado y optimizar hook/segmentación según primeros resultados."
    if quality >= 55:
        return "Potencial creativo aceptable, pero eficiencia incierta. Mejorar segmentación o pieza antes de invertir más presupuesto."
    return "Riesgo de eficiencia: conviene ajustar creativo, mensaje o audiencia antes de pautar."


def _build_response(
    *,
    eligible: bool,
    probability: float,
    row: Dict[str, Any],
    rules: Dict[str, Any],
    meta: Dict[str, Any],
    model_score: float | None,
    model_cpm: float | None,
    model_status: str,
    calibrated_score: float,
    calibrated_cpm: float,
    budget: float,
    method: str,
    warning: str = "",
) -> Dict[str, Any]:
    budget_v = max(safe_float(budget, 0.0), 0.0)
    impressions = (budget_v / calibrated_cpm) * 1000 if budget_v > 0 and calibrated_cpm > 0 else 0.0
    vtr = safe_float(rules.get("estimated_view_through_rate", 0.18), 0.18)
    ctr = safe_float(rules.get("estimated_ctr", 0.012), 0.012)
    paid_views = impressions * vtr
    clicks = impressions * ctr
    model_loaded = _load_model() is not None

    return {
        "eligible_for_paid_xgboost": bool(eligible),
        "model_available": model_loaded,
        "model_status": model_status,
        "model_role": "clasificador_xgboost + calibrador_dinamico_cpm",
        "gate_threshold": GATE_THRESHOLD,
        "gate_passed": bool(probability >= GATE_THRESHOLD),
        "logistic_probability": round(probability, 4),
        "ad_niche": rules.get("ad_niche") or row.get("ad_niche") or "general/branding",
        "selected_ad_category": rules.get("selected_ad_category") or row.get("selected_ad_category", "auto"),
        "paid_ad_feature_row": row,
        "raw_xgboost_paid_performance_score": round(model_score, 4) if model_score is not None else None,
        "raw_xgboost_cpm": round(model_cpm, 4) if model_cpm is not None else None,
        "rules_cpm": rules.get("rules_cpm"),
        "rules_quality_score": rules.get("rules_quality_score"),
        "rules_paid_performance_score": rules.get("rules_paid_performance_score"),
        "predicted_paid_performance_score": round(calibrated_score, 2),
        "predicted_cpm": round(calibrated_cpm, 4),
        "estimated_cost_per_1000_impressions": round(calibrated_cpm, 4),
        "estimated_impressions_per_dollar": round(1000.0 / calibrated_cpm, 2) if calibrated_cpm > 0 else 0.0,
        "estimated_paid_views_per_dollar": round((1000.0 / calibrated_cpm) * vtr, 2) if calibrated_cpm > 0 else 0.0,
        "estimated_budget_impressions": round(impressions, 0),
        "estimated_paid_views": round(paid_views, 0),
        "estimated_clicks": round(clicks, 0),
        "estimated_view_through_rate": round(vtr, 4),
        "estimated_ctr": round(ctr, 4),
        "recommendation": _paid_recommendation(calibrated_score, calibrated_cpm, safe_float(rules.get("rules_quality_score", 0)), bool(eligible)),
        "reason": "Gate superado; se evalúa pauta pagada." if eligible else "Gate no superado; se entrega predicción de referencia pero no se recomienda pautar.",
        "calibration_method": method,
        "warning": warning,
        "model_metadata": meta,
        "methodological_warning": "Estimación no causal. Validar con campañas reales antes de decidir presupuestos altos.",
    }


def predict_paid_ad_boost(
    *,
    features: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
    operational_metrics: Dict[str, Any] | None = None,
    model_probability: float,
    budget: float = 0.0,
) -> Dict[str, Any]:
    """Predice pauta pagada sin fallar aunque el XGBoost esté ausente o sea clasificador.

    La salida siempre incluye predicted_cpm, impresiones, views pagadas y score.
    La bandera eligible_for_paid_xgboost indica si conviene activar pauta, no si el cálculo existe.
    """
    try:
        probability = _clip(safe_float(model_probability, 0.0), 0.0, 1.0)
        metadata = metadata or {}
        operational_metrics = operational_metrics or {}
        row = build_paid_ad_feature_row(features, metadata, operational_metrics)
        meta = _load_metadata()
        rules = _rules_estimate(features, metadata, operational_metrics, probability, safe_float(budget, 0.0))
        model_score, model_cpm, model_status = _predict_model(row)

        passed_gate = probability >= GATE_THRESHOLD
        rules_cpm = safe_float(rules.get("rules_cpm", BASE_CPM_BY_NICHE.get(row.get("ad_niche"), 5.6)), 5.6)
        rules_score = safe_float(rules.get("rules_paid_performance_score", 0), 0)

        if model_score is not None and model_cpm is not None:
            # Caso poco común: artefacto multioutput/regresor. Se mezcla con reglas para evitar rigidez.
            raw_score = _clip(float(model_score), 0.0, 100.0)
            raw_cpm = _clip(float(model_cpm), 1.0, 25.0)
            calibrated_score = _clip(raw_score * 0.55 + rules_score * 0.45, 0.0, 100.0)
            calibrated_cpm = _clip(raw_cpm * 0.35 + rules_cpm * 0.65, 2.20, 18.00)
            method = "XGBoost regresor/multioutput disponible + calibrador dinámico por señales reales del video."
            warning = "" if abs(raw_cpm - calibrated_cpm) <= 3 else "El CPM bruto del modelo fue calibrado para evitar una salida rígida o poco sensible."
        elif model_score is not None:
            # Caso esperado si el XGBoost entrenado es clasificador.
            raw_score = _clip(float(model_score), 0.0, 100.0)
            calibrated_score = _clip(raw_score * 0.60 + rules_score * 0.40, 0.0, 100.0)
            # El clasificador no predice CPM: se mantiene calibrador dinámico.
            calibrated_cpm = rules_cpm
            method = "XGBoost clasificador disponible; CPM calculado con calibrador dinámico por nicho, engagement, retención y velocidad."
            warning = "El XGBoost disponible clasifica aptitud/score; el CPM no sale como regresión directa sino como estimación calibrada."
        else:
            calibrated_score = rules_score
            calibrated_cpm = rules_cpm
            method = "Predictor calibrado por señales reales porque el artefacto XGBoost no estuvo disponible o no respondió a tiempo."
            warning = "El artefacto XGBoost no entregó predicción utilizable; se protegió la app para evitar bloqueo."

        return _build_response(
            eligible=passed_gate,
            probability=probability,
            row=row,
            rules=rules,
            meta=meta,
            model_score=model_score,
            model_cpm=model_cpm,
            model_status=model_status,
            calibrated_score=calibrated_score,
            calibrated_cpm=calibrated_cpm,
            budget=budget,
            method=method,
            warning=warning,
        )
    except Exception as exc:
        # Última red de seguridad: la app nunca debe caerse por este módulo.
        fallback_cpm = BASE_CPM_BY_NICHE["general/branding"]
        budget_v = max(safe_float(budget, 0.0), 0.0)
        impressions = (budget_v / fallback_cpm) * 1000 if budget_v > 0 else 0.0
        return {
            "eligible_for_paid_xgboost": False,
            "model_available": _load_model() is not None,
            "model_status": f"proteccion_app:{type(exc).__name__}",
            "model_role": "proteccion_app_sin_bloqueo",
            "gate_threshold": GATE_THRESHOLD,
            "gate_passed": False,
            "logistic_probability": round(_clip(safe_float(model_probability, 0.0), 0.0, 1.0), 4),
            "ad_niche": "general/branding",
            "selected_ad_category": "auto",
            "predicted_paid_performance_score": 0.0,
            "predicted_cpm": fallback_cpm,
            "estimated_cost_per_1000_impressions": fallback_cpm,
            "estimated_impressions_per_dollar": round(1000.0 / fallback_cpm, 2),
            "estimated_paid_views_per_dollar": round((1000.0 / fallback_cpm) * 0.18, 2),
            "estimated_budget_impressions": round(impressions, 0),
            "estimated_paid_views": round(impressions * 0.18, 0),
            "estimated_clicks": round(impressions * 0.012, 0),
            "estimated_view_through_rate": 0.18,
            "estimated_ctr": 0.012,
            "recommendation": "No se recomienda pautar hasta revisar el error del predictor de pauta.",
            "reason": "Protección de aplicación ejecutada para evitar bloqueo del dashboard.",
            "calibration_method": "Protección mínima con CPM base general/branding; revisar logs del predictor.",
            "warning": f"Predictor de pauta no pudo completarse: {type(exc).__name__}.",
            "model_metadata": _load_metadata(),
            "methodological_warning": "Estimación no causal. Revisar logs antes de usar para decisión real.",
        }


def render_paid_ad_boost_markdown(xgb: Dict[str, Any]) -> str:
    if not xgb:
        return ""

    if not xgb.get("eligible_for_paid_xgboost"):
        return f"""
### Segundo modelo XGBoost de pauta

El sistema sí generó una estimación de pauta, pero el video **no queda habilitado para invertir** porque la probabilidad principal fue de **{safe_float(xgb.get('logistic_probability')):.1%}**, menor al umbral de **{safe_float(xgb.get('gate_threshold'), GATE_THRESHOLD):.0%}**.

| Variable | Estimación de referencia |
|---|---:|
| Nicho detectado | {xgb.get('ad_niche', '—')} |
| Score de eficiencia pagada | {safe_float(xgb.get('predicted_paid_performance_score')):.2f}/100 |
| CPM dinámico calibrado | ${safe_float(xgb.get('predicted_cpm')):.2f} |
| Impresiones estimadas por dólar | {safe_float(xgb.get('estimated_impressions_per_dollar')):.2f} |
| Vistas pagadas estimadas por dólar | {safe_float(xgb.get('estimated_paid_views_per_dollar')):.2f} |
| Impresiones estimadas con presupuesto | {safe_float(xgb.get('estimated_budget_impressions')):,.0f} |
| Views pagadas estimadas | {safe_float(xgb.get('estimated_paid_views')):,.0f} |
| Clics estimados | {safe_float(xgb.get('estimated_clicks')):,.0f} |
| Estado del modelo | {xgb.get('model_status', '—')} |

**Lectura:** {xgb.get('recommendation', 'No se recomienda pautar todavía.')}  
**Método:** {xgb.get('calibration_method', 'CPM calibrado dinámicamente.')}  
**Advertencia:** {xgb.get('methodological_warning') or xgb.get('warning') or 'Usar como estimación, no como garantía de ROI.'}
""".strip()

    return f"""
### Segundo modelo XGBoost de pauta

El video pasó el gate de candidatura publicitaria de la regresión logística (**{safe_float(xgb.get('logistic_probability')):.1%} ≥ {safe_float(xgb.get('gate_threshold'), GATE_THRESHOLD):.0%}**) y fue evaluado para impulso pagado.

| Variable | Estimación |
|---|---:|
| Nicho detectado | {xgb.get('ad_niche', '—')} |
| Score de eficiencia pagada estimada | {safe_float(xgb.get('predicted_paid_performance_score')):.2f}/100 |
| CPM dinámico calibrado | ${safe_float(xgb.get('predicted_cpm')):.2f} |
| Impresiones estimadas por dólar | {safe_float(xgb.get('estimated_impressions_per_dollar')):.2f} |
| Vistas pagadas estimadas por dólar | {safe_float(xgb.get('estimated_paid_views_per_dollar')):.2f} |
| Impresiones estimadas con presupuesto | {safe_float(xgb.get('estimated_budget_impressions')):,.0f} |
| Views pagadas estimadas | {safe_float(xgb.get('estimated_paid_views')):,.0f} |
| Clics estimados | {safe_float(xgb.get('estimated_clicks')):,.0f} |
| Estado del modelo | {xgb.get('model_status', '—')} |

**Lectura:** {xgb.get('recommendation', 'Validar con prueba real de bajo presupuesto.')}  
**Método:** {xgb.get('calibration_method', 'XGBoost clasificador + CPM calibrado dinámicamente.')}  
**Advertencia:** {xgb.get('methodological_warning') or xgb.get('warning') or 'Usar como estimación, no como garantía de ROI.'}
""".strip()
