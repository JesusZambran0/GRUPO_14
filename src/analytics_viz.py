"""Gráficos estadísticos con estética oscura para la demo Gradio.

Genera 3 figuras:
1. ``score_chart``: tarjeta con métricas clave normalizadas (potencial,
   retención, engagement, guion, riesgo).
2. ``projection_chart``: comparación **rendimiento actual vs esperado tras pauta**
   por dólar invertido — views, likes, comments, shares.
3. ``policy_chart``: barra de categorías sensibles detectadas (si hay).

Todas usan el mismo color palette: fondo `#0b0d12`, acentos morado/cyan/coral.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .config import RUNTIME_CACHE_DIR

# Paleta consistente con la UI dark.
PALETTE = {
    "bg": "#0b0d12",
    "panel": "#161a22",
    "grid": "#262b36",
    "text": "#e6e9ef",
    "muted": "#8b91a3",
    "primary": "#a78bfa",   # violeta
    "accent": "#22d3ee",    # cyan
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


def create_analysis_charts(result: Dict[str, Any]) -> Dict[str, str]:
    """Genera los 3 gráficos en PNG y devuelve sus rutas."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return {"score_chart": "", "projection_chart": "", "policy_chart": ""}

    _setup_dark(plt)
    out_dir = RUNTIME_CACHE_DIR / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / "score_chart.png"
    projection_path = out_dir / "projection_chart.png"
    policy_path = out_dir / "policy_chart.png"

    metrics = result.get("metricas", {}) or {}
    operational = metrics.get("operational", {}) or {}
    script = result.get("analisis_guion", {}) or {}
    policy_risk = str(result.get("policy_risk_level", "bajo"))
    risk_score = {"bajo": 18, "medio": 50, "alto": 80, "revisión humana": 95}.get(policy_risk, 50)

    # =====================================================================
    # 1) Diagnóstico — barras horizontales con color por puntaje
    # =====================================================================
    labels = ["Potencial", "Retención", "Engagement", "Guion", "Riesgo política"]
    values = [
        float(result.get("probabilidad_rendimiento", 0)) * 100,
        float(operational.get("retention_rate", 0) or 0) * 100,
        min(float(metrics.get("engagement_rate", 0) or 0) * 1000, 100),
        float(script.get("script_quality_score", 0) or 0),
        risk_score,
    ]

    def _bar_color(label: str, v: float) -> str:
        if label == "Riesgo política":
            return PALETTE["danger"] if v >= 60 else PALETTE["warning"] if v >= 40 else PALETTE["success"]
        return PALETTE["success"] if v >= 60 else PALETTE["warning"] if v >= 40 else PALETTE["danger"]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y_pos = np.arange(len(labels))
    colors = [_bar_color(lab, v) for lab, v in zip(labels, values)]
    bars = ax.barh(y_pos, values, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=PALETTE["text"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Puntaje (0 a 100)", color=PALETTE["muted"])
    ax.set_title("Diagnóstico del video", color=PALETTE["text"], fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, values):
        ax.text(min(v + 1.5, 95), bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}", va="center", color=PALETTE["text"], fontsize=10, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.patch.set_facecolor(PALETTE["bg"])
    plt.tight_layout()
    plt.savefig(score_path, dpi=140, facecolor=PALETTE["bg"])
    plt.close(fig)

    # =====================================================================
    # 2) Comparación rendimiento actual vs esperado tras pauta
    # =====================================================================
    proj = result.get("proyeccion_pauta", {}) or {}
    actual = {
        "Views": int(float(metrics.get("views", 0) or 0)),
        "Likes": int(float(metrics.get("likes", 0) or 0)),
        "Comments": int(float(metrics.get("comments", 0) or 0)),
        "Shares": int(float(operational.get("shares", 0) or 0)),
    }
    expected = {
        "Views": int(float(proj.get("projected_views_after_boost", 0) or 0)),
        "Likes": int(float(proj.get("projected_likes_after_boost", 0) or 0)),
        "Comments": int(float(proj.get("projected_comments_after_boost", 0) or 0)),
        "Shares": int(float(proj.get("projected_shares_after_boost", 0) or 0)),
    }
    # Si los expected son 0 (no se pasó budget), creamos un proxy basado en multiplicador.
    if all(v == 0 for v in expected.values()):
        mult = float(result.get("multiplicador_potencial", 1.3))
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
    ax.set_title("Rendimiento actual vs esperado con $ de publicación", color=PALETTE["text"],
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    # Etiquetas sobre las barras
    max_val = max(max(actual.values()) if actual else 1, max(expected.values()) if expected else 1, 1)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + max_val * 0.015, f"{int(h):,}",
                    ha="center", color=PALETTE["text"], fontsize=8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.patch.set_facecolor(PALETTE["bg"])
    plt.tight_layout()
    plt.savefig(projection_path, dpi=140, facecolor=PALETTE["bg"])
    plt.close(fig)

    # =====================================================================
    # 3) Categorías de política detectadas (barra horizontal)
    # =====================================================================
    cats_detected = result.get("analisis_politicas", {}).get("policy_risk_categories", []) or []
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    if cats_detected:
        y = np.arange(len(cats_detected))
        vals = [100] * len(cats_detected)
        colors_p = [PALETTE["danger"]] * len(cats_detected)
        ax.barh(y, vals, color=colors_p, height=0.55, edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels(cats_detected, color=PALETTE["text"])
        ax.set_xlim(0, 100)
        ax.set_xticks([])
        ax.set_title(f"Señales de política detectadas ({len(cats_detected)})",
                     color=PALETTE["text"], fontsize=13, fontweight="bold", pad=14)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "Sin senales de politica\nApto para pauta estandar",
                ha="center", va="center", color=PALETTE["success"], fontsize=14, fontweight="bold",
                transform=ax.transAxes)
        ax.set_title("Riesgo publicitario", color=PALETTE["text"], fontsize=13, fontweight="bold", pad=14)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.patch.set_facecolor(PALETTE["bg"])
    plt.tight_layout()
    plt.savefig(policy_path, dpi=140, facecolor=PALETTE["bg"])
    plt.close(fig)

    return {"score_chart": str(score_path), "projection_chart": str(projection_path), "policy_chart": str(policy_path)}
