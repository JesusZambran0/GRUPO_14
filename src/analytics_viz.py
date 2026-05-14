"""Gráficos estadísticos con estética oscura para la demo Gradio.

Genera 4 figuras:
1. score_chart: diagnóstico general.
2. projection_chart: rendimiento actual vs esperado tras pauta.
3. policy_chart: señales de política publicitaria.
4. xgboost_chart: CPM calibrado, impresiones y score estimado de pauta.
"""
from __future__ import annotations

from typing import Any, Dict

from .config import RUNTIME_CACHE_DIR

PALETTE = {
    "bg": "#0b0d12",
    "panel": "#161a22",
    "grid": "#262b36",
    "text": "#e6e9ef",
    "muted": "#8b91a3",
    "primary": "#a78bfa",
    "accent": "#22d3ee",
    "success": "#22c55e",
    "warning": "#facc15",
    "danger": "#f87171",
    "bar_actual": "#475569",
    "bar_expected": "#a78bfa",
}


def _setup_dark(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["panel"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text"],
        "axes.titlecolor": PALETTE["text"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "grid.color": PALETTE["grid"],
        "text.color": PALETTE["text"],
        "axes.grid": True,
        "grid.alpha": 0.5,
        "font.size": 10,
    })


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def create_analysis_charts(result: Dict[str, Any]) -> Dict[str, str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return {"score_chart": "", "projection_chart": "", "policy_chart": "", "xgboost_chart": ""}

    _setup_dark(plt)
    out_dir = RUNTIME_CACHE_DIR / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / "score_chart.png"
    projection_path = out_dir / "projection_chart.png"
    policy_path = out_dir / "policy_chart.png"
    xgboost_path = out_dir / "xgboost_chart.png"

    metrics = result.get("metricas", {}) or {}
    operational = metrics.get("operational", {}) or {}
    script = result.get("analisis_guion", {}) or {}
    policy_risk = str(result.get("policy_risk_level", "bajo"))
    risk_score = {"bajo": 18, "medio": 50, "alto": 80, "revisión humana": 95}.get(policy_risk, 50)

    # 1) Diagnóstico general
    labels = ["Potencial", "Retención", "Engagement", "Guion", "Riesgo política"]
    values = [
        _safe_float(result.get("probabilidad_rendimiento", 0)) * 100,
        _safe_float(operational.get("retention_rate", 0)) * 100,
        min(_safe_float(metrics.get("engagement_rate", 0)) * 1000, 100),
        _safe_float(script.get("script_quality_score", 0)),
        risk_score,
    ]

    def _bar_color(label: str, value: float) -> str:
        if label == "Riesgo política":
            return PALETTE["danger"] if value >= 60 else PALETTE["warning"] if value >= 40 else PALETTE["success"]
        return PALETTE["success"] if value >= 60 else PALETTE["warning"] if value >= 40 else PALETTE["danger"]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=[_bar_color(l, v) for l, v in zip(labels, values)], edgecolor="none", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=PALETTE["text"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Puntaje normalizado 0-100", color=PALETTE["muted"])
    ax.set_title("Diagnóstico del video", color=PALETTE["text"], fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(min(value + 1.5, 95), bar.get_y() + bar.get_height() / 2, f"{value:.0f}", va="center", color=PALETTE["text"], fontsize=10, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.patch.set_facecolor(PALETTE["bg"])
    plt.tight_layout()
    plt.savefig(score_path, dpi=140, facecolor=PALETTE["bg"])
    plt.close(fig)

    # 2) Actual vs esperado con pauta, sin shares
    proj = result.get("proyeccion_pauta", {}) or {}
    actual = {
        "Views": int(_safe_float(metrics.get("views", 0))),
        "Likes": int(_safe_float(metrics.get("likes", 0))),
        "Comments": int(_safe_float(metrics.get("comments", 0))),
    }
    expected = {
        "Views": int(_safe_float(proj.get("projected_views_after_boost", 0))),
        "Likes": int(_safe_float(proj.get("projected_likes_after_boost", 0))),
        "Comments": int(_safe_float(proj.get("projected_comments_after_boost", 0))),
    }
    if all(v == 0 for v in expected.values()):
        mult = _safe_float(result.get("multiplicador_potencial", 1.3), 1.3)
        expected = {k: int(v * mult * 1.5) for k, v in actual.items()}

    cats = list(actual.keys())
    x = np.arange(len(cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    b1 = ax.bar(x - width / 2, [actual[c] for c in cats], width, label="Actual", color=PALETTE["bar_actual"], edgecolor="none")
    b2 = ax.bar(x + width / 2, [expected[c] for c in cats], width, label="Esperado con pauta", color=PALETTE["bar_expected"], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, color=PALETTE["text"])
    ax.set_ylabel("Volumen", color=PALETTE["muted"])
    ax.set_title("Rendimiento actual vs esperado con pauta", color=PALETTE["text"], fontsize=13, fontweight="bold", pad=14)
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    max_val = max(max(actual.values()) if actual else 1, max(expected.values()) if expected else 1, 1)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + max_val * 0.015, f"{int(h):,}", ha="center", color=PALETTE["text"], fontsize=8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.patch.set_facecolor(PALETTE["bg"])
    plt.tight_layout()
    plt.savefig(projection_path, dpi=140, facecolor=PALETTE["bg"])
    plt.close(fig)

    # 3) Política
    cats_detected = result.get("analisis_politicas", {}).get("policy_risk_categories", []) or []
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    if cats_detected:
        y = np.arange(len(cats_detected))
        ax.barh(y, [100] * len(cats_detected), color=[PALETTE["danger"]] * len(cats_detected), height=0.55, edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels(cats_detected, color=PALETTE["text"])
        ax.set_xlim(0, 100)
        ax.set_xticks([])
        ax.set_title(f"Señales de política detectadas ({len(cats_detected)})", color=PALETTE["text"], fontsize=13, fontweight="bold", pad=14)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "Sin señales de política\nApto para screening estándar", ha="center", va="center", color=PALETTE["success"], fontsize=14, fontweight="bold", transform=ax.transAxes)
        ax.set_title("Riesgo publicitario", color=PALETTE["text"], fontsize=13, fontweight="bold", pad=14)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.patch.set_facecolor(PALETTE["bg"])
    plt.tight_layout()
    plt.savefig(policy_path, dpi=140, facecolor=PALETTE["bg"])
    plt.close(fig)

    # 4) XGBoost pauta
    xgb = result.get("xgboost_pauta") or result.get("analisis_xgboost_pauta") or metrics.get("xgboost_pauta") or {}
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    if xgb and xgb.get("eligible_for_paid_xgboost"):
        labels = ["Score\npagado", "CPM\n(USD)", "Imp./$\n÷10", "Views\npagadas\n÷100"]
        values = [
            _safe_float(xgb.get("predicted_paid_performance_score", 0)),
            _safe_float(xgb.get("predicted_cpm", 0)),
            _safe_float(xgb.get("estimated_impressions_per_dollar", 0)) / 10.0,
            _safe_float(xgb.get("estimated_paid_views", 0)) / 100.0,
        ]
        colors = [PALETTE["primary"], PALETTE["warning"], PALETTE["accent"], PALETTE["success"]]
        bars = ax.bar(labels, values, color=colors, edgecolor="none", width=0.58)
        ax.set_title(f"XGBoost pauta · {xgb.get('ad_niche', 'nicho general')}", color=PALETTE["text"], fontsize=13, fontweight="bold", pad=14)
        ax.set_ylabel("Escala visual normalizada", color=PALETTE["muted"])
        ax.grid(axis="y", alpha=0.3)
        max_val = max(values + [1])
        raw_labels = [
            f"{_safe_float(xgb.get('predicted_paid_performance_score', 0)):.1f}/100",
            f"${_safe_float(xgb.get('predicted_cpm', 0)):.2f}",
            f"{_safe_float(xgb.get('estimated_impressions_per_dollar', 0)):.1f}",
            f"{_safe_float(xgb.get('estimated_paid_views', 0)):,.0f}",
        ]
        for bar, txt in zip(bars, raw_labels):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + max_val * 0.03, txt, ha="center", color=PALETTE["text"], fontsize=9, fontweight="bold")
        method = xgb.get("calibration_method") or xgb.get("methodological_warning") or "Estimación de apoyo."
        ax.text(0.5, -0.24, method[:130], ha="center", va="top", color=PALETTE["muted"], fontsize=8, transform=ax.transAxes)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        probability = _safe_float(xgb.get("logistic_probability", result.get("probabilidad_rendimiento", 0))) if xgb else _safe_float(result.get("probabilidad_rendimiento", 0))
        ax.text(0.5, 0.55, "XGBoost no ejecutado", ha="center", va="center", color=PALETTE["warning"], fontsize=15, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.40, f"Probabilidad base: {probability:.1%}\nEl segundo modelo solo corre si supera el gate de 51%.", ha="center", va="center", color=PALETTE["muted"], fontsize=10, transform=ax.transAxes)
        ax.set_title("XGBoost de pauta", color=PALETTE["text"], fontsize=13, fontweight="bold", pad=14)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.patch.set_facecolor(PALETTE["bg"])
    plt.tight_layout()
    plt.savefig(xgboost_path, dpi=140, facecolor=PALETTE["bg"])
    plt.close(fig)

    return {
        "score_chart": str(score_path),
        "projection_chart": str(projection_path),
        "policy_chart": str(policy_path),
        "xgboost_chart": str(xgboost_path),
    }
