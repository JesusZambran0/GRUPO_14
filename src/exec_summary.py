"""Resumen ejecutivo orientado a marketing.

Convierte la salida estructurada del modelo + score híbrido + políticas +
composición visual en un texto **claro y accionable** para alguien que no es
data scientist: marketers, gerentes de cuenta, dueños de canal.

Reglas de redacción:
- Frase principal de UNA línea con la acción.
- 3 razones concretas en bullets cortos.
- Métricas clave en formato amigable (porcentajes, no decimales).
- Nada de "score_hibrido 0.6325" — usar "rendimiento esperado: alto (63 sobre 100)".
- Sin jerga técnica innecesaria. Sin referencias a "feature engineering" ni "logistic regression".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# Glosario de equivalencias humanas para los niveles del modelo.
LEVEL_HUMAN = {
    "muy_alto": "muy alto",
    "alto": "alto",
    "medio": "medio",
    "bajo": "bajo",
}

ACTION_PHRASES = {
    "IMPULSAR": {
        "headline": "✅ Adelante: el video está listo para pauta.",
        "tone": "positivo",
        "color": "#22c55e",
    },
    "AJUSTAR ANTES DE IMPULSAR": {
        "headline": "🛠️ Casi listo: con ajustes puntuales rinde mejor.",
        "tone": "neutro_positivo",
        "color": "#facc15",
    },
    "MONITOREAR": {
        "headline": "🔎 No invertir aún: dale tiempo de probarse en orgánico.",
        "tone": "neutro",
        "color": "#60a5fa",
    },
    "NO IMPULSAR": {
        "headline": "⛔ No conviene pautar este video.",
        "tone": "negativo",
        "color": "#f87171",
    },
    "REVISIÓN HUMANA": {
        "headline": "🧑‍⚖️ Mejor lo revisa una persona antes de decidir.",
        "tone": "advertencia",
        "color": "#a78bfa",
    },
}

POLICY_HUMAN = {
    "bajo": "✅ Sin alertas de política",
    "medio": "🟡 Algunas frases requieren cuidado",
    "alto": "🔴 Riesgo alto de política",
    "revisión humana": "🧑‍⚖️ Múltiples temas sensibles detectados",
}


def _to_percent(value: float) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except Exception:
        return "—"


def _format_money(usd: float) -> str:
    try:
        return f"${float(usd):,.0f}"
    except Exception:
        return "—"


def _format_int(value) -> str:
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return "—"


def build_executive_summary(
    *,
    final_recommendation: Dict[str, Any],
    prediction: Dict[str, Any],
    features: Dict[str, Any],
    operational_metrics: Dict[str, Any],
    policy_block: Dict[str, Any],
    visual_analysis: Optional[Dict[str, Any]] = None,
    script_analysis: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    cpm: float = 5.0,
    budget: float = 0.0,
) -> Dict[str, Any]:
    """Construye un resumen ejecutivo en lenguaje claro.

    Devuelve un dict con varias piezas:
    - ``headline``: una línea con la acción.
    - ``score_global_0_100``: score híbrido en escala 0-100.
    - ``probabilidad_pct``: probabilidad de buen rendimiento en %.
    - ``policy_status_human``: estado de política en frase clara.
    - ``por_que``: lista de 3 razones cortas.
    - ``que_hacer_ahora``: lista de pasos accionables.
    - ``forecast``: dict con métricas de pauta (impresiones esperadas, vistas, etc).
    - ``markdown``: render completo en markdown para pegar en la UI.
    """
    action = final_recommendation.get("accion_final", "MONITOREAR")
    action_meta = ACTION_PHRASES.get(action, ACTION_PHRASES["MONITOREAR"])

    prob = float(final_recommendation.get("probabilidad_rendimiento", 0))
    level = final_recommendation.get("prediccion_rendimiento", "medio")
    score_hib = float(final_recommendation.get("score_hibrido", 0))
    score_0_100 = int(round(score_hib * 100))
    prob_pct = int(round(prob * 100))

    policy_lvl = policy_block.get("policy_risk_level", "bajo")
    policy_status_human = POLICY_HUMAN.get(policy_lvl, policy_lvl)
    policy_cats = policy_block.get("policy_risk_categories", []) or []

    # Razones positivas y negativas, intercaladas
    razones: List[str] = []

    # Modelo
    razones.append(
        f"📊 Potencial de rendimiento estimado: **{LEVEL_HUMAN.get(level, level)}** ({prob_pct}% de probabilidad de funcionar)."
    )

    # Política
    if policy_lvl == "bajo":
        razones.append("🛡️ El contenido **no activa alertas de políticas** publicitarias.")
    elif policy_lvl == "medio":
        razones.append(
            f"⚠️ Hay **señales moderadas** de política (`{', '.join(policy_cats[:3])}`). "
            "Probablemente apto pero con monetización limitada."
        )
    elif policy_lvl == "alto":
        razones.append(
            f"🚫 **Categorías sensibles detectadas**: `{', '.join(policy_cats[:3])}`. "
            "YouTube probablemente NO monetizaría este video con anuncios estándar."
        )
    else:  # revisión humana
        razones.append(
            f"🚨 Múltiples temas delicados ({len(policy_cats)} categorías). "
            "Revisar con alguien antes de invertir."
        )

    # Composición visual (si disponible)
    if visual_analysis and visual_analysis.get("visual_ok"):
        v_score = float(visual_analysis.get("composition_score", 0))
        if v_score >= 0.55:
            razones.append(f"🎬 **Composición visual sólida** ({int(v_score*100)} sobre 100). Encuadre y balance correctos.")
        elif v_score >= 0.35:
            razones.append(f"🎬 Composición visual aceptable ({int(v_score*100)} sobre 100) pero con margen para mejorar.")
        else:
            razones.append(f"🎬 **Composición visual débil** ({int(v_score*100)} sobre 100). Revisa encuadre y foco.")

    # Engagement actual
    er = float(features.get("engagement_rate", 0))
    if er > 0.05:
        razones.append(f"❤️ Engagement actual ALTO ({er*100:.1f}% likes+comments / views).")
    elif er > 0.02:
        razones.append(f"❤️ Engagement actual saludable ({er*100:.1f}%).")
    elif er > 0:
        razones.append(f"❤️ Engagement actual bajo ({er*100:.1f}%). El video aún no resuena con su audiencia.")

    # Ajustes sugeridos → "qué hacer ahora"
    que_hacer = list(final_recommendation.get("ajustes_sugeridos", []))[:5]
    if visual_analysis and visual_analysis.get("visual_ok"):
        # Tomar 2 recomendaciones visuales si hay
        for r in visual_analysis.get("visual_recommendations", [])[:2]:
            if r not in que_hacer:
                que_hacer.append(r)
    if not que_hacer:
        que_hacer.append("No hay ajustes críticos pendientes. Mantener el contenido como está.")

    # Forecast
    reach_per_dollar = float(final_recommendation.get("alcance_estimado_por_dolar", 0))
    reach_total = float(final_recommendation.get("alcance_estimado_total", 0))

    forecast = {
        "cpm_usd": float(final_recommendation.get("cpm_estimado", cpm)),
        "budget_usd": float(budget),
        "impresiones_esperadas": int(reach_total),
        "impresiones_por_dolar": int(reach_per_dollar),
        "multiplicador_potencial": float(final_recommendation.get("multiplicador_potencial", 1.0)),
        "current_views": int(float(features.get("views", 0))),
        "current_likes": int(float(features.get("likes", 0))),
        "current_comments": int(float(features.get("comments", 0))),
    }

    headline = action_meta["headline"]
    color = action_meta["color"]

    # Construir markdown
    title_str = (metadata or {}).get("title") or "Sin título"
    md_parts: List[str] = []
    md_parts.append(f"# {headline}")
    md_parts.append(f"**Video analizado:** {title_str}")
    md_parts.append("")
    md_parts.append(f"| Métrica clave | Valor |")
    md_parts.append(f"|---|---|")
    md_parts.append(f"| **Rendimiento esperado** | {LEVEL_HUMAN.get(level, level)} ({prob_pct}% de probabilidad) |")
    md_parts.append(f"| **Score global** | **{score_0_100} / 100** |")
    md_parts.append(f"| **Política publicitaria** | {policy_status_human} |")
    if forecast["budget_usd"] > 0:
        md_parts.append(
            f"| **Con {_format_money(forecast['budget_usd'])} pautados** | ≈ {_format_int(forecast['impresiones_esperadas'])} impresiones |"
        )
    md_parts.append("")
    md_parts.append("### Por qué")
    for r in razones[:5]:
        md_parts.append(f"- {r}")
    md_parts.append("")
    md_parts.append("### Qué hacer ahora")
    for q in que_hacer[:5]:
        md_parts.append(f"- {q}")
    md_parts.append("")
    md_parts.append(
        "> *Este sistema es una herramienta de apoyo. El alcance esperado por dólar se calcula con "
        "el CPM ingresado y un multiplicador de potencial; no representa ROI real garantizado.*"
    )

    return {
        "headline": headline,
        "action": action,
        "color": color,
        "score_global_0_100": score_0_100,
        "probabilidad_pct": prob_pct,
        "nivel_humano": LEVEL_HUMAN.get(level, level),
        "policy_status_human": policy_status_human,
        "policy_categories": policy_cats,
        "por_que": razones[:5],
        "que_hacer_ahora": que_hacer[:5],
        "forecast": forecast,
        "markdown": "\n".join(md_parts),
    }
