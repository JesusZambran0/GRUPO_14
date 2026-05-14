"""Motor de recomendación: score híbrido + acción final.

El score híbrido combina la probabilidad del modelo predictivo con métricas
operativas del video (engagement, retención, views/hora) y el
riesgo publicitario detectado por ``policy_evaluator``.

Las métricas que pueden no haber sido parte del entrenamiento del modelo
(p.ej. ``retention_rate``, ``average_watch_time``) se usan
**aquí, en el score híbrido**, no como predictores del modelo. Eso queda
documentado explícitamente.

Acciones finales posibles:
- IMPULSAR
- AJUSTAR ANTES DE IMPULSAR
- MONITOREAR
- NO IMPULSAR
- REVISIÓN HUMANA
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import DEFAULT_CPM, POTENTIAL_MULTIPLIERS, RECOMMENDATION_CHOICES
from .features import safe_float

# Acciones finales canónicas. Estas son LAS que la app expone al usuario.
FINAL_ACTIONS = {
    "IMPULSAR",
    "AJUSTAR ANTES DE IMPULSAR",
    "MONITOREAR",
    "NO IMPULSAR",
    "REVISIÓN HUMANA",
}


def calculate_reach_per_dollar(level: str, cpm: float = DEFAULT_CPM) -> Dict[str, float]:
    """Estimación parametrizada de alcance por dólar.

    No representa ROI causal real: solo proyecta impresiones promedio dada la
    fórmula ``1000 / CPM`` ajustada por el multiplicador de potencial predicho.
    """
    cpm = max(safe_float(cpm, DEFAULT_CPM), 0.01)
    base = 1000.0 / cpm
    multiplier = POTENTIAL_MULTIPLIERS.get(level, 1.0)
    adjusted = base * multiplier
    return {
        "cpm_estimado": round(cpm, 4),
        "multiplicador_potencial": multiplier,
        "alcance_base_por_dolar": round(base, 2),
        "alcance_estimado_por_dolar": round(adjusted, 2),
    }


# ---------------------------------------------------------------------------
# Score híbrido
# ---------------------------------------------------------------------------

def _normalize(value: float, lo: float, hi: float) -> float:
    """Min-max simple con clip."""
    if hi <= lo:
        return 0.0
    return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))


def compute_hybrid_score(
    model_probability: float,
    engagement_rate: float = 0.0,
    share_rate: float = 0.0,
    retention_rate: float = 0.0,
    views_per_hour: float = 0.0,
    policy_risk_level: str = "bajo",
) -> Dict[str, Any]:
    """Score híbrido de decisión.

    Pesos (suman 1.0 antes de la penalización por políticas):
    - 0.50 model_probability
    - 0.18 engagement_rate (normalizada en [0, 0.15])
    - 0.17 retention_rate (en [0, 1])
    - 0.15 views_per_hour (log-normalizado en [0, 1000])

    Penalización por riesgo de políticas:
    - bajo: ×1.00
    - medio: ×0.85
    - alto: ×0.50
    - revisión humana: ×0.30 (y la acción final se marca como REVISIÓN HUMANA)
    """
    p = safe_float(model_probability)
    p = max(0.0, min(1.0, p))
    er = _normalize(safe_float(engagement_rate), 0.0, 0.15)
    sr = 0.0  # shares queda fuera del score: no aporta suficiente valor predictivo en esta versión.
    rr = _normalize(safe_float(retention_rate), 0.0, 1.0)
    # views_per_hour log-normalizado: 0 vph → 0, 1000 vph → 1.
    import math
    vph_raw = max(safe_float(views_per_hour), 0.0)
    vph = _normalize(math.log1p(vph_raw), 0.0, math.log1p(1000.0))

    base = 0.50 * p + 0.18 * er + 0.17 * rr + 0.15 * vph

    policy_penalty_map = {"bajo": 1.00, "medio": 0.85, "alto": 0.50, "revisión humana": 0.30}
    penalty = policy_penalty_map.get(policy_risk_level, 0.85)
    final = base * penalty

    return {
        "score_base": round(base, 4),
        "policy_penalty": penalty,
        "hybrid_score": round(final, 4),
        "components": {
            "model_probability": round(p, 4),
            "engagement_rate_norm": round(er, 4),
            "share_rate_norm": round(sr, 4),
            "retention_rate_norm": round(rr, 4),
            "views_per_hour_norm": round(vph, 4),
        },
    }


# ---------------------------------------------------------------------------
# Acción final
# ---------------------------------------------------------------------------

def determine_final_action(
    hybrid_score: float,
    policy_risk_level: str,
    requires_human_review_due_to_missing_transcript: bool = False,
    requires_adjustments: bool = False,
    has_evaluable_text: bool = True,
) -> str:
    """Mapea el score híbrido + riesgo a una de las 5 acciones canónicas.

    Args:
        hybrid_score: score híbrido 0-1.
        policy_risk_level: "bajo" | "medio" | "alto" | "revisión humana".
        requires_human_review_due_to_missing_transcript: legacy. Solo escala a
            REVISIÓN HUMANA si NO hay texto evaluable (ni título ni descripción).
        requires_adjustments: si hay ajustes textuales pendientes.
        has_evaluable_text: si hay título/descripción suficientes para evaluación parcial.

    Política nueva:
    - Solo se escala a REVISIÓN HUMANA cuando el evaluador de políticas marca
      explícitamente "revisión humana" (3+ categorías de alta severidad).
    - La transcripción ausente NO bloquea por sí sola; el modelo y el screening
      parcial siguen funcionando con título + descripción.
    """
    if policy_risk_level == "revisión humana":
        return "REVISIÓN HUMANA"
    if requires_human_review_due_to_missing_transcript and not has_evaluable_text:
        # Solo si NO tenemos NI transcripción NI metadata textual usable.
        return "REVISIÓN HUMANA"
    if policy_risk_level == "alto":
        return "NO IMPULSAR"
    s = safe_float(hybrid_score)
    if s >= 0.65 and not requires_adjustments and policy_risk_level == "bajo":
        return "IMPULSAR"
    if s >= 0.55 and policy_risk_level in {"bajo", "medio"}:
        return "AJUSTAR ANTES DE IMPULSAR"
    if s >= 0.42:
        return "AJUSTAR ANTES DE IMPULSAR"
    return "NO IMPULSAR"


# ---------------------------------------------------------------------------
# Reglas de ajustes (texto + duración + saturación)
# ---------------------------------------------------------------------------

def _detect_required_adjustments(features: Dict[str, Any]) -> List[str]:
    adjustments: List[str] = []
    if not features.get("cta_flag") and not features.get("ocr_cta_flag"):
        adjustments.append("Agregar un llamado a la acción claro en los primeros segundos del video.")
    try:
        if safe_float(features.get("ocr_word_count", 0)) > 35 and safe_float(features.get("duration_seconds", 0)) <= 10:
            adjustments.append("Reducir la cantidad de texto visible para evitar saturación en un video corto.")
    except (TypeError, ValueError):
        pass
    if not features.get("benefit_flag"):
        adjustments.append("Explicitar el beneficio principal para el usuario antes de impulsar el contenido.")
    if safe_float(features.get("duration_seconds", 0)) > 60:
        adjustments.append("Evaluar una versión más corta para pauta, priorizando el gancho inicial.")
    if not features.get("transcript_text") and not features.get("ocr_text"):
        adjustments.append(
            "Agregar transcripción manual u optimizar texto en pantalla para evaluar mejor el mensaje."
        )
    return adjustments


# ---------------------------------------------------------------------------
# Recomendación final (contrato completo)
# ---------------------------------------------------------------------------

def _level_to_legacy_recommendation(level: str, requires_adjustments: bool) -> Dict[str, str]:
    """Mantiene la salida `recomendacion_impulso` (4 valores) para compatibilidad."""
    if level in {"alto", "muy_alto"} and not requires_adjustments:
        return {"recomendacion_impulso": "impulsar", "nivel_prioridad": "alta"}
    if level in {"alto", "muy_alto"} and requires_adjustments:
        return {"recomendacion_impulso": "ajustar antes de impulsar", "nivel_prioridad": "alta"}
    if level == "medio":
        return {"recomendacion_impulso": "ajustar antes de impulsar", "nivel_prioridad": "media"}
    return {"recomendacion_impulso": "no impulsar", "nivel_prioridad": "baja"}


def _build_justification(
    level: str,
    probability: float,
    features: Dict[str, Any],
    requires_adjustments: bool,
    policy: Optional[Dict[str, Any]] = None,
    hybrid: Optional[Dict[str, Any]] = None,
) -> str:
    pieces: List[str] = []
    pieces.append(
        f"Modelo predictivo: nivel `{level}` con probabilidad estimada {probability:.2%}."
    )
    if features.get("transcript_text"):
        pieces.append("Se aprovechó la transcripción del audio para reforzar el análisis textual.")
    if features.get("ocr_text"):
        pieces.append("Se incorporó el texto visible detectado por OCR.")
    if requires_adjustments:
        pieces.append("Se detectaron ajustes recomendados antes de pautar.")
    else:
        pieces.append("No se detectaron ajustes críticos según las reglas locales.")
    if policy and policy.get("policy_risk_level"):
        pieces.append(
            f"Riesgo publicitario estimado: {policy['policy_risk_level']} "
            f"(estado YouTube Ads: {policy.get('youtube_ad_status_estimate', 'desconocido')})."
        )
    if hybrid:
        pieces.append(
            f"Score híbrido de decisión: {hybrid['hybrid_score']:.2f} "
            f"(penalización por políticas: ×{hybrid['policy_penalty']})."
        )
    pieces.append(
        "El alcance por dólar se estima con CPM parametrizado y multiplicador de potencial; "
        "no representa ROI causal real."
    )
    return " ".join(pieces)


def build_final_recommendation(
    prediction: Dict[str, Any],
    features: Dict[str, Any],
    cpm: float,
    budget: float = 0.0,
    operational_metrics: Optional[Dict[str, Any]] = None,
    policy_block: Optional[Dict[str, Any]] = None,
    requires_human_review_due_to_missing_transcript: bool = False,
    has_evaluable_text: bool = True,
) -> Dict[str, Any]:
    """Construye la recomendación final completa.

    Args:
        prediction: salida de ``src.predict.predict_from_features``.
        features: fila de features de ``src.features.build_feature_row``.
        cpm: CPM estimado en USD.
        budget: presupuesto en USD.
        operational_metrics: dict opcional con ``retention_rate``,
            ``average_watch_time``, ``hours_since_publication``, ``followers_count``,
            ``avg_channel_reach``, ``share_rate``, ``views_per_hour``.
            Estas variables pueden NO haber sido parte del entrenamiento del modelo;
            se usan SOLO en el score híbrido.
        policy_block: salida de ``policy_evaluator.evaluate_youtube_ad_policy_risk``.
        requires_human_review_due_to_missing_transcript: si True fuerza acción REVISIÓN HUMANA.
    """
    operational_metrics = operational_metrics or {}
    policy_block = policy_block or {}

    level = prediction.get("level", "medio")
    prob = safe_float(prediction.get("probability", 0))

    reach = calculate_reach_per_dollar(level, cpm)
    budget_v = max(safe_float(budget, 0), 0)

    adjustments = _detect_required_adjustments(features)
    requires_adj = bool(adjustments)

    legacy = _level_to_legacy_recommendation(level, requires_adj)
    assert legacy["recomendacion_impulso"] in RECOMMENDATION_CHOICES, "Recomendación fuera del contrato"

    # Score híbrido
    hybrid = compute_hybrid_score(
        model_probability=prob,
        engagement_rate=safe_float(features.get("engagement_rate", 0)),
        share_rate=safe_float(operational_metrics.get("share_rate", 0)),
        retention_rate=safe_float(operational_metrics.get("retention_rate", 0)),
        views_per_hour=safe_float(operational_metrics.get("views_per_hour", 0)),
        policy_risk_level=policy_block.get("policy_risk_level", "bajo"),
    )

    # Acción final canónica (5 valores)
    final_action = determine_final_action(
        hybrid_score=hybrid["hybrid_score"],
        policy_risk_level=policy_block.get("policy_risk_level", "bajo"),
        requires_human_review_due_to_missing_transcript=requires_human_review_due_to_missing_transcript,
        requires_adjustments=requires_adj,
        has_evaluable_text=has_evaluable_text,
    )
    assert final_action in FINAL_ACTIONS, f"Acción final fuera del contrato: {final_action}"

    alcance_total = round(reach["alcance_estimado_por_dolar"] * budget_v, 2) if budget_v else 0
    justification = _build_justification(level, prob, features, requires_adj, policy=policy_block, hybrid=hybrid)

    return {
        "prediccion_rendimiento": level,
        "probabilidad_rendimiento": round(prob, 4),
        "recomendacion_impulso": legacy["recomendacion_impulso"],
        "requiere_ajustes": requires_adj,
        "nivel_prioridad": legacy["nivel_prioridad"],
        "ajustes_sugeridos": adjustments,
        **reach,
        "alcance_estimado_total": alcance_total,
        # Nuevos campos
        "accion_final": final_action,
        "score_hibrido": hybrid["hybrid_score"],
        "score_hibrido_detalle": hybrid,
        "policy_risk_level": policy_block.get("policy_risk_level", "bajo"),
        "youtube_ad_status_estimate": policy_block.get("youtube_ad_status_estimate", "apto"),
        "justificacion": justification,
        "nota_metodologica": (
            "El modelo fue entrenado con videos públicos de YouTube en trending (Kaggle), "
            "lo que introduce sesgo de selección hacia contenido exitoso. Las probabilidades "
            "tienden a ser altas; las reglas de ajuste y el score híbrido compensan ese sesgo. "
            "La estimación de alcance por dólar usa CPM parametrizado y potencial predicho; "
            "no representa ROI causal real."
        ),
    }
