"""Segundo modelo: estimación calibrada de pauta pagada.

Este módulo conserva el contrato metodológico de la tesis:
1. Primero decide el modelo principal de clasificación.
2. Solo si la probabilidad supera el gate de 0.51, se estima pauta pagada.
3. Si existe un modelo XGBoost entrenado, se usa como señal.
4. Para evitar salidas rígidas o poco realistas, la predicción se calibra con
   reglas transparentes basadas en nicho, engagement, retención, views/hora y
   presupuesto.

Motivo del cambio:
- El modelo XGBoost anterior podía devolver CPM casi constante, por ejemplo
  12.38, porque muchas variables categóricas llegaban como "unknown".
- Esta versión evita resultados repetidos y devuelve una estimación más
  sensible al video analizado.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd

from .config import MODELS_DIR
from .features import safe_float

XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_paid_ads.joblib"
XGBOOST_METADATA_PATH = MODELS_DIR / "xgboost_paid_ads_metadata.json"
GATE_THRESHOLD = 0.51

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
    "entretenimiento": 3.8,
    "gastronomía": 4.6,
    "educación": 5.4,
    "retail/ecommerce": 6.2,
    "salud/bienestar": 7.0,
    "servicios": 7.6,
    "tecnología": 8.8,
    "inmobiliario": 9.8,
    "general/branding": 5.6,
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
    """Normaliza la categoría elegida en la UI para XGBoost.

    - auto / vacío: permite inferencia por texto.
    - otros: permite fallback general si no hay señales textuales claras.
    - categorías soportadas: fuerzan el nicho para calibrar CPM y vistas.
    """
    raw = str(value or "").strip().lower()
    if not raw or raw in {"auto", "automático", "automatico"}:
        return None
    if raw in {"otro", "otros", "other"}:
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


def infer_ad_niche(title: str = "", description: str = "", transcript_text: str = "", ocr_text: str = "", selected_category: Any = None) -> str:
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


def _stable_jitter(text: str, span: float = 0.16) -> float:
    """Variación determinista pequeña para evitar empates sin usar aleatoriedad."""
    digest = hashlib.sha256((text or "general").encode("utf-8", errors="ignore")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return 1.0 + (raw - 0.5) * 2 * span


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_paid_ad_feature_row(features: Dict[str, Any], metadata: Dict[str, Any] | None = None, ops: Dict[str, Any] | None = None) -> Dict[str, Any]:
    metadata = metadata or {}
    ops = ops or {}
    title = metadata.get("title", "") or features.get("title", "")
    description = metadata.get("description", "") or features.get("description", "")
    selected_category = metadata.get("xgboost_category") or metadata.get("ad_category") or features.get("xgboost_category") or features.get("ad_category")
    niche = infer_ad_niche(title, description, features.get("transcript_text", ""), features.get("ocr_text", ""), selected_category=selected_category)
    now = datetime.utcnow()
    month = now.month
    weekday = now.isoweekday()
    engagement = safe_float(features.get("engagement_rate", 0.0))
    retention = safe_float(ops.get("retention_rate", 0.0))
    return {
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
    }


def _rules_estimate(features: Dict[str, Any], metadata: Dict[str, Any], ops: Dict[str, Any], probability: float, budget: float) -> Dict[str, Any]:
    title = metadata.get("title", "") or features.get("title", "")
    description = metadata.get("description", "") or features.get("description", "")
    selected_category = metadata.get("xgboost_category") or metadata.get("ad_category") or features.get("xgboost_category") or features.get("ad_category")
    niche = infer_ad_niche(title, description, features.get("transcript_text", ""), features.get("ocr_text", ""), selected_category=selected_category)

    views = max(safe_float(metadata.get("views", features.get("views", 0))), 0.0)
    likes = max(safe_float(metadata.get("likes", features.get("likes", 0))), 0.0)
    comments = max(safe_float(metadata.get("comments", features.get("comments", 0))), 0.0)
    engagement_rate = safe_float(features.get("engagement_rate", 0.0))
    if engagement_rate <= 0 and views > 0:
        engagement_rate = (likes + comments) / views
    retention = _clip(safe_float(ops.get("retention_rate", 0.0)), 0.0, 1.0)
    hours = max(safe_float(ops.get("hours_since_publication", 24)), 1.0)
    views_per_hour = views / hours if views else 0.0
    duration = max(safe_float(metadata.get("duration_seconds", features.get("duration_seconds", 30))), 1.0)

    base_cpm = BASE_CPM_BY_NICHE.get(niche, BASE_CPM_BY_NICHE["general/branding"])

    quality = (
        _clip(probability, 0, 1) * 0.46
        + _clip(engagement_rate / 0.08, 0, 1) * 0.22
        + _clip(retention / 0.65, 0, 1) * 0.20
        + _clip(views_per_hour / 700.0, 0, 1) * 0.08
        + _clip((60 - min(duration, 60)) / 60, 0, 1) * 0.04
    )

    # CPM efectivo: nichos caros suben, buen engagement/retención lo baja un poco.
    cpm = base_cpm
    cpm *= 1.22 - 0.34 * _clip(quality, 0, 1)
    if budget >= 100:
        cpm *= 1.06
    elif 0 < budget <= 20:
        cpm *= 0.96
    if views < 100:
        cpm *= 1.08
    cpm *= _stable_jitter(f"{title}|{description}|{niche}", span=0.13)
    cpm = _clip(cpm, 2.2, 18.0)

    # VTR/CTR estimados, no causal. Varían con calidad y nicho.
    vtr = _clip(0.11 + 0.32 * quality, 0.08, 0.46)
    ctr = _clip(0.004 + engagement_rate * 0.38 + quality * 0.018, 0.003, 0.08)

    impressions = (max(budget, 0.0) / cpm) * 1000 if budget > 0 else 0.0
    paid_views = impressions * vtr
    clicks = impressions * ctr
    score = _clip(35 + probability * 42 + quality * 28 - (cpm / 18.0) * 8, 0, 100)

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


def _predict_model(row: Dict[str, Any]) -> Tuple[float | None, float | None, str]:
    model = _load_model()
    meta = _load_metadata()
    if model is None:
        return None, None, "modelo_no_disponible"
    input_cols = meta.get("feature_columns") or [k for k in row.keys() if k != "ad_niche"]
    X = pd.DataFrame([{c: row.get(c, 0) for c in input_cols}])
    try:
        pred = model.predict(X)[0]
        targets = getattr(model, "target_labels_", meta.get("targets", ["paid_performance_score", "cpm"]))
        pred_map = {str(t): float(v) for t, v in zip(targets, pred)}
        return pred_map.get("paid_performance_score"), pred_map.get("cpm"), "modelo_xgboost"
    except Exception as exc:
        return None, None, f"error_modelo:{type(exc).__name__}"


def _paid_recommendation(score: float, cpm: float, quality: float) -> str:
    if score >= 70 and cpm <= 8:
        return "Impulso viable: buen score estimado y CPM controlado. Probar con presupuesto pequeño y escalar si CTR/CPM reales confirman."
    if score >= 58:
        return "Impulso moderado: lanzar prueba A/B con presupuesto limitado y optimizar hook/segmentación según primeros resultados."
    if quality >= 55:
        return "Potencial creativo aceptable, pero eficiencia incierta. Mejorar segmentación o pieza antes de invertir más presupuesto."
    return "Riesgo de eficiencia: conviene ajustar creativo, mensaje o audiencia antes de pautar."


def predict_paid_ad_boost(
    *,
    features: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
    operational_metrics: Dict[str, Any] | None = None,
    model_probability: float,
    budget: float = 0.0,
    manual_cpm: float = 5.0,
) -> Dict[str, Any]:
    probability = _clip(safe_float(model_probability, 0.0), 0.0, 1.0)
    metadata = metadata or {}
    operational_metrics = operational_metrics or {}
    row = build_paid_ad_feature_row(features, metadata, operational_metrics)
    meta = _load_metadata()

    if probability < GATE_THRESHOLD:
        return {
            "eligible_for_paid_xgboost": False,
            "gate_threshold": GATE_THRESHOLD,
            "logistic_probability": round(probability, 4),
            "ad_niche": row["ad_niche"],
            "selected_ad_category": row.get("selected_ad_category", "auto"),
            "reason": "No pasa al segundo modelo porque la probabilidad de candidatura publicitaria es menor a 51%.",
            "model_metadata": meta,
        }

    rules = _rules_estimate(features, metadata, operational_metrics, probability, safe_float(budget, 0.0))
    model_score, model_cpm, model_status = _predict_model(row)

    # Si el XGBoost existe, se usa como señal, pero se calibra con reglas para
    # evitar CPM casi constantes por falta de variación en features categóricas.
    if model_score is not None and model_cpm is not None:
        raw_cpm = _clip(float(model_cpm), 1.0, 25.0)
        raw_score = _clip(float(model_score), 0.0, 100.0)
        calibrated_cpm = _clip(raw_cpm * 0.40 + rules["rules_cpm"] * 0.60, 2.2, 18.0)
        calibrated_score = _clip(raw_score * 0.45 + rules["rules_paid_performance_score"] * 0.55, 0.0, 100.0)
        method = "XGBoost disponible + calibración por señales del video/nicho."
        model_available = True
    else:
        calibrated_cpm = rules["rules_cpm"]
        calibrated_score = rules["rules_paid_performance_score"]
        method = "Estimación calibrada por reglas porque XGBoost no está disponible o no pudo inferir."
        model_available = False

    budget_v = max(safe_float(budget, 0.0), 0.0)
    impressions = (budget_v / calibrated_cpm) * 1000 if budget_v > 0 else 0.0
    paid_views = impressions * safe_float(rules.get("estimated_view_through_rate", 0.18))
    clicks = impressions * safe_float(rules.get("estimated_ctr", 0.012))

    warning = ""
    if model_status != "modelo_xgboost":
        warning = "XGBoost no disponible o no cargado; se usó estimación calibrada por reglas."
    elif abs(safe_float(model_cpm, 0) - calibrated_cpm) > 2.5:
        warning = "El CPM bruto del XGBoost fue calibrado porque podía ser poco sensible al video actual."

    return {
        "eligible_for_paid_xgboost": True,
        "model_available": model_available,
        "model_status": model_status,
        "gate_threshold": GATE_THRESHOLD,
        "logistic_probability": round(probability, 4),
        "ad_niche": rules["ad_niche"],
        "selected_ad_category": rules.get("selected_ad_category", "auto"),
        "paid_ad_feature_row": row,
        "raw_xgboost_paid_performance_score": round(model_score, 4) if model_score is not None else None,
        "raw_xgboost_cpm": round(model_cpm, 4) if model_cpm is not None else None,
        "rules_cpm": rules["rules_cpm"],
        "rules_quality_score": rules["rules_quality_score"],
        "predicted_paid_performance_score": round(calibrated_score, 2),
        "predicted_cpm": round(calibrated_cpm, 4),
        "estimated_cost_per_1000_impressions": round(calibrated_cpm, 4),
        "estimated_impressions_per_dollar": round(1000.0 / calibrated_cpm, 2),
        "estimated_paid_views_per_dollar": round((1000.0 / calibrated_cpm) * safe_float(rules.get("estimated_view_through_rate", 0.18)), 2),
        "estimated_budget_impressions": round(impressions, 0),
        "estimated_paid_views": round(paid_views, 0),
        "estimated_clicks": round(clicks, 0),
        "estimated_view_through_rate": rules["estimated_view_through_rate"],
        "estimated_ctr": rules["estimated_ctr"],
        "recommendation": _paid_recommendation(calibrated_score, calibrated_cpm, rules["rules_quality_score"]),
        "calibration_method": method,
        "warning": warning,
        "model_metadata": meta,
        "methodological_warning": "Estimación no causal. Validar con campañas reales antes de decidir presupuestos altos.",
    }


def render_paid_ad_boost_markdown(xgb: Dict[str, Any]) -> str:
    if not xgb:
        return ""
    if not xgb.get("eligible_for_paid_xgboost"):
        return f"""
### Segundo modelo XGBoost de pauta

El video **no pasó** al segundo modelo porque la regresión logística estimó una probabilidad de **{safe_float(xgb.get('logistic_probability')):.1%}**, menor al umbral de **{safe_float(xgb.get('gate_threshold'), GATE_THRESHOLD):.0%}**.

**Nicho detectado:** {xgb.get('ad_niche', '—')}  
**Motivo:** {xgb.get('reason', 'Gate no superado.')}
""".strip()

    return f"""
### Segundo modelo XGBoost de pauta

El video pasó el gate de candidatura publicitaria de la regresión logística (**{safe_float(xgb.get('logistic_probability')):.1%} ≥ {safe_float(xgb.get('gate_threshold'), GATE_THRESHOLD):.0%}**) y fue evaluado para impulso pagado.

| Variable | Estimación |
|---|---:|
| Nicho detectado | {xgb.get('ad_niche', '—')} |
| Score de eficiencia pagada estimada | {safe_float(xgb.get('predicted_paid_performance_score')):.2f}/100 |
| CPM estimado calibrado | ${safe_float(xgb.get('predicted_cpm')):.2f} |
| Impresiones estimadas por dólar | {safe_float(xgb.get('estimated_impressions_per_dollar')):.2f} |
| Vistas pagadas estimadas por dólar | {safe_float(xgb.get('estimated_paid_views_per_dollar')):.2f} |
| Impresiones estimadas con presupuesto | {safe_float(xgb.get('estimated_budget_impressions')):,.0f} |
| Views pagadas estimadas | {safe_float(xgb.get('estimated_paid_views')):,.0f} |
| Clics estimados | {safe_float(xgb.get('estimated_clicks')):,.0f} |

**Lectura:** {xgb.get('recommendation', 'Validar con prueba real de bajo presupuesto.')}  
**Método:** {xgb.get('calibration_method', 'XGBoost calibrado con señales del video.')}  
**Advertencia:** {xgb.get('methodological_warning') or xgb.get('warning') or 'Usar como estimación, no como garantía de ROI.'}
""".strip()
