"""Análisis de sentimiento de comentarios de YouTube.

Motor ligero y autónomo: usa lexicones español/inglés y heurísticas para clasificar
comentarios en positivo / neutro / negativo sin depender de Gemini ni de modelos
pesados. Además genera insumos visuales para la app:
- porcentajes por sentimiento,
- gráfico de distribución,
- nubes de palabras positivas, neutras y negativas sin stopwords,
- lectura narrativa para recolectar insights del público.

Para obtener comentarios reales se necesita YouTube Data API v3 (opcional).
Si no hay key, acepta comentarios pegados manualmente.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lexicones simples (ampliables sin tocar la lógica)
# ---------------------------------------------------------------------------

POSITIVE_TERMS = {
    "excelente", "genial", "increíble", "increible", "gracias", "perfecto", "buenísimo",
    "buenisimo", "fenomenal", "me encanta", "me encantó", "me encanto", "fascinante",
    "útil", "util", "aprendo", "aprendí", "aprendi", "recomiendo", "lo mejor", "top",
    "great", "amazing", "excellent", "love", "helpful", "thanks", "awesome",
    "fantastic", "brilliant", "perfect", "wonderful", "like", "best",
    "bravo", "super", "espectacular", "clarísimo", "clarisimo", "claro", "entendí",
    "entendi", "buen contenido", "sigue así", "sigue asi", "suscrito", "me suscribí",
    "me suscribi", "vale la pena", "funciona", "me sirvió", "me sirvio", "aporta",
}

NEGATIVE_TERMS = {
    "malo", "pésimo", "pesimo", "horrible", "terrible", "no sirve", "no funciona",
    "no entiendo", "confuso", "aburrido", "click bait", "clickbait", "mentira",
    "estafa", "engaño", "engano", "basura", "desperdicio", "pérdida de tiempo",
    "perdida de tiempo", "bad", "terrible", "horrible", "useless", "boring", "wrong",
    "misleading", "fake", "scam", "waste", "dislike", "nope", "awful", "poor",
    "no me gustó", "no me gusto", "decepcionante", "decepciona", "deficiente",
    "caro", "lento", "falló", "fallo", "problema", "fraude", "molesto", "queja",
}

INTENSIFIERS = {
    "muy", "super", "demasiado", "extremadamente", "completamente",
    "totalmente", "absolutamente", "increíblemente", "increiblemente", "realmente",
    "bastante", "quite", "very", "so", "extremely", "absolutely",
}

NEGATORS = {"no", "ni", "nunca", "jamás", "jamas", "tampoco", "sin", "not", "never", "barely"}

STOPWORDS_ES = {
    "de", "la", "el", "y", "en", "a", "que", "del", "los", "las", "un", "una",
    "por", "para", "con", "su", "al", "se", "es", "son", "como", "más", "mas",
    "pero", "sin", "ya", "muy", "sobre", "entre", "hay", "ser", "fue", "han",
    "ha", "lo", "le", "les", "me", "te", "nos", "mi", "tu", "sus", "ese",
    "esa", "eso", "esto", "esta", "estos", "estas", "desde", "hasta", "donde",
    "cuando", "también", "tambien", "porque", "https", "http", "com", "www",
    "rt", "tco", "amp", "video", "imagen", "marca", "contenido", "publicación",
    "publicacion", "post", "reel", "comentario", "comentarios", "youtube", "canal",
    "ver", "veo", "hacer", "hace", "haces", "solo", "asi", "así", "ahi", "ahí",
    "este", "esta", "estos", "estas", "uno", "dos", "tres", "parte", "tema",
}

# ---------------------------------------------------------------------------
# Limpieza y scoring
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\sáéíóúüñ]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _tokens_sin_stopwords(text: str, stopwords_extra: Optional[List[str]] = None) -> List[str]:
    stopwords = set(STOPWORDS_ES)
    if stopwords_extra:
        stopwords.update(str(w).lower().strip() for w in stopwords_extra if str(w).strip())

    clean = (text or "").lower()
    clean = re.sub(r"https?://\S+", " ", clean)
    clean = re.sub(r"www\.\S+", " ", clean)
    clean = re.sub(r"[@#]\w+", " ", clean)
    clean = re.sub(r"[^a-záéíóúüñ\s]", " ", clean, flags=re.UNICODE)
    clean = re.sub(r"\s+", " ", clean).strip()

    return [token for token in clean.split() if len(token) >= 3 and token not in stopwords]


def _top_words(comments: List[str], n: int = 20) -> List[Tuple[str, int]]:
    counter: Counter[str] = Counter()
    for comment in comments:
        counter.update(_tokens_sin_stopwords(comment))
    return counter.most_common(n)


def _score_comment(text: str) -> float:
    """Devuelve un score de -1 (muy neg) a +1 (muy pos). 0 = neutro."""
    clean = _clean(text)
    words = clean.split()
    if not words:
        return 0.0

    pos = sum(1 for t in POSITIVE_TERMS if t in clean)
    neg = sum(1 for t in NEGATIVE_TERMS if t in clean)

    # Detectar negaciones que invierten positivos (ej: "no me encanta")
    for i, w in enumerate(words):
        if w in NEGATORS:
            context = " ".join(words[i + 1: i + 5])
            if any(t in context for t in POSITIVE_TERMS):
                neg += 1
                pos = max(0, pos - 1)

    # Intensificadores amplían la magnitud
    intensity = 1.0 + 0.3 * sum(1 for w in words if w in INTENSIFIERS)
    raw = (pos - neg) * intensity
    return max(-1.0, min(1.0, raw / max(pos + neg, 1)))


def classify_comment(text: str) -> str:
    score = _score_comment(text)
    if score > 0.2:
        return "positivo"
    if score < -0.2:
        return "negativo"
    return "neutro"

# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def analyze_comments_sentiment(
    comments: List[str],
    max_comments: int = 100,
) -> Dict[str, Any]:
    """Analiza el sentimiento de una lista de comentarios.

    Returns dict con porcentajes, ejemplos, palabras frecuentes y resultados por
    comentario. La app solo muestra markdown + gráficos; este JSON queda para API/debug.
    """
    empty = {
        "total": 0,
        "positivos": 0,
        "neutros": 0,
        "negativos": 0,
        "pct_positivo": 0.0,
        "pct_neutro": 0.0,
        "pct_negativo": 0.0,
        "score_promedio": 0.0,
        "sentimiento_dominante": "neutro",
        "ejemplos": {"positivos": [], "neutros": [], "negativos": []},
        "palabras_frecuentes": {"positivas": [], "neutras": [], "negativas": []},
        "resultados": [],
        "warning": "Sin comentarios para analizar.",
    }
    if not comments:
        return empty

    filtered = [c.strip() for c in comments if (c or "").strip()][:max_comments]
    if not filtered:
        return empty

    results: List[Dict[str, Any]] = []
    for c in filtered:
        score = _score_comment(c)
        label = classify_comment(c)
        results.append({"text": c[:500], "score": round(score, 3), "label": label})

    pos = [r for r in results if r["label"] == "positivo"]
    neg = [r for r in results if r["label"] == "negativo"]
    neu = [r for r in results if r["label"] == "neutro"]
    total = len(results)
    avg_score = round(sum(float(r["score"]) for r in results) / total, 3)

    counts = {"positivo": len(pos), "negativo": len(neg), "neutro": len(neu)}
    max_count = max(counts.values())
    tied = [k for k, v in counts.items() if v == max_count]
    dominant = "neutro" if "neutro" in tied and len(tied) > 1 else tied[0]

    ejemplos = {
        "positivos": [r["text"] for r in sorted(pos, key=lambda x: -float(x["score"]))[:3]],
        "negativos": [r["text"] for r in sorted(neg, key=lambda x: float(x["score"]))[:3]],
        "neutros": [r["text"] for r in neu[:3]],
    }

    pos_texts = [r["text"] for r in pos]
    neu_texts = [r["text"] for r in neu]
    neg_texts = [r["text"] for r in neg]

    palabras_frecuentes = {
        "positivas": _top_words(pos_texts),
        "neutras": _top_words(neu_texts),
        "negativas": _top_words(neg_texts),
    }

    return {
        "total": total,
        "positivos": len(pos),
        "neutros": len(neu),
        "negativos": len(neg),
        "pct_positivo": round(len(pos) / total * 100, 1),
        "pct_neutro": round(len(neu) / total * 100, 1),
        "pct_negativo": round(len(neg) / total * 100, 1),
        "score_promedio": avg_score,
        "sentimiento_dominante": dominant,
        "ejemplos": ejemplos,
        "palabras_frecuentes": palabras_frecuentes,
        "resultados": results,
        "warning": "",
    }


def fetch_and_analyze_comments(
    video_id: str,
    api_key: Optional[str] = None,
    max_results: int = 50,
) -> Dict[str, Any]:
    """Obtiene comentarios de YouTube Data API y analiza el sentimiento."""
    if not api_key:
        return {**analyze_comments_sentiment([]), "warning": "Sin YOUTUBE_API_KEY: no se pudieron obtener comentarios."}

    import json as _json
    import urllib.parse
    import urllib.request

    url = (
        "https://www.googleapis.com/youtube/v3/commentThreads?"
        + urllib.parse.urlencode({
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(max_results, 100),
            "textFormat": "plainText",
            "key": api_key,
        })
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        comments = [
            item["snippet"]["topLevelComment"]["snippet"].get("textDisplay", "")
            for item in data.get("items", [])
        ]
        result = analyze_comments_sentiment(comments, max_comments=max_results)
        result["source"] = "youtube_api"
        return result
    except Exception as exc:
        return {
            **analyze_comments_sentiment([]),
            "warning": f"Error obteniendo comentarios de YouTube: {exc}",
        }

# ---------------------------------------------------------------------------
# Markdown + visualizaciones para Gradio
# ---------------------------------------------------------------------------

def build_sentiment_markdown(sentiment: Dict[str, Any]) -> str:
    total = int(sentiment.get("total", 0) or 0)
    source = sentiment.get("source") or "manual/local"

    if total <= 0:
        warning = sentiment.get("warning") or "No hay comentarios suficientes para analizar."
        return f"""
### Análisis de sentimiento del público

No se encontraron comentarios suficientes para construir una lectura confiable.

**Fuente:** {source}  
**Estado:** {warning}

### Cómo usar este apartado

Pega comentarios manualmente, uno por línea, o configura `YOUTUBE_API_KEY` para traer comentarios reales desde YouTube.
""".strip()

    pct_pos = float(sentiment.get("pct_positivo", 0) or 0)
    pct_neu = float(sentiment.get("pct_neutro", 0) or 0)
    pct_neg = float(sentiment.get("pct_negativo", 0) or 0)
    dominant = sentiment.get("sentimiento_dominante", "neutro")
    avg = float(sentiment.get("score_promedio", 0) or 0)

    palabras = sentiment.get("palabras_frecuentes", {}) or {}
    top_pos = ", ".join([w for w, _ in palabras.get("positivas", [])[:10]]) or "sin señales claras"
    top_neu = ", ".join([w for w, _ in palabras.get("neutras", [])[:10]]) or "sin señales claras"
    top_neg = ", ".join([w for w, _ in palabras.get("negativas", [])[:10]]) or "sin señales claras"

    if dominant == "positivo":
        lectura = (
            "La conversación tiene inclinación favorable. El contenido está generando señales de aceptación, "
            "utilidad o afinidad emocional. Conviene reforzar los temas que aparecen en la nube positiva."
        )
    elif dominant == "negativo":
        lectura = (
            "La conversación muestra fricción. Antes de escalar pauta, conviene revisar si las críticas se "
            "concentran en la promesa, claridad, credibilidad, precio, formato o expectativa del contenido."
        )
    else:
        lectura = (
            "La conversación es mayoritariamente neutral. Esto puede indicar baja emoción, comentarios informativos "
            "o una audiencia que todavía no expresa intención clara. El reto es convertir atención en reacción."
        )

    ejemplos = sentiment.get("ejemplos", {}) or {}
    ejemplo_pos = "\n".join(f"- {x}" for x in ejemplos.get("positivos", [])[:2]) or "- Sin ejemplos positivos destacados."
    ejemplo_neu = "\n".join(f"- {x}" for x in ejemplos.get("neutros", [])[:2]) or "- Sin ejemplos neutros destacados."
    ejemplo_neg = "\n".join(f"- {x}" for x in ejemplos.get("negativos", [])[:2]) or "- Sin ejemplos negativos destacados."

    return f"""
### Resumen general

Se analizaron **{total} comentarios** con un motor local de sentimiento. El sentimiento dominante es **{dominant}** y el score promedio es **{avg:.2f}** en una escala de -1 a +1.

{lectura}

### Distribución del sentimiento

| Sentimiento | Porcentaje | Menciones |
|---|---:|---:|
| Positivo | {pct_pos:.1f}% | {int(sentiment.get("positivos", 0) or 0)} |
| Neutral | {pct_neu:.1f}% | {int(sentiment.get("neutros", 0) or 0)} |
| Negativo | {pct_neg:.1f}% | {int(sentiment.get("negativos", 0) or 0)} |

### Insights del público por territorio emocional

- **Territorio positivo:** {top_pos}
- **Territorio neutral:** {top_neu}
- **Territorio negativo:** {top_neg}

### Ejemplos representativos

**Positivos**
{ejemplo_pos}

**Neutros**
{ejemplo_neu}

**Negativos**
{ejemplo_neg}

### Recomendación accionable

Usa el gráfico para medir el clima general y las nubes para encontrar conceptos repetidos. Las palabras positivas indican qué reforzar, las neutras qué explicar mejor y las negativas qué corregir antes de invertir más pauta.
""".strip()


def create_sentiment_visuals(sentiment: Dict[str, Any]) -> Dict[str, str]:
    """Genera gráfico de porcentajes y nubes de palabras por sentimiento.

    Devuelve rutas de imágenes para componentes gr.Image. No rompe la app si falta
    matplotlib/wordcloud; en ese caso devuelve placeholders vacíos o de aviso.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        from .config import RUNTIME_CACHE_DIR
    except Exception:
        return {
            "sentiment_bar_chart": "",
            "wordcloud_positive": "",
            "wordcloud_neutral": "",
            "wordcloud_negative": "",
        }

    out_dir = RUNTIME_CACHE_DIR / "sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    bar_path = out_dir / "sentiment_distribution.png"
    wc_pos_path = out_dir / "wordcloud_positive.png"
    wc_neu_path = out_dir / "wordcloud_neutral.png"
    wc_neg_path = out_dir / "wordcloud_negative.png"

    def _placeholder(path: Path, title: str, message: str) -> str:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axis("off")
        ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=15, fontweight="bold")
        ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, wrap=True)
        plt.tight_layout()
        plt.savefig(path, dpi=140, facecolor="white")
        plt.close(fig)
        return str(path)

    total = int(sentiment.get("total", 0) or 0)
    if total <= 0:
        msg = "Sin comentarios suficientes"
        return {
            "sentiment_bar_chart": _placeholder(bar_path, "Sentimiento", msg),
            "wordcloud_positive": _placeholder(wc_pos_path, "Palabras positivas", msg),
            "wordcloud_neutral": _placeholder(wc_neu_path, "Palabras neutras", msg),
            "wordcloud_negative": _placeholder(wc_neg_path, "Palabras negativas", msg),
        }

    labels = ["Positivo", "Neutral", "Negativo"]
    values = [
        float(sentiment.get("pct_positivo", 0) or 0),
        float(sentiment.get("pct_neutro", 0) or 0),
        float(sentiment.get("pct_negativo", 0) or 0),
    ]
    colors = ["#22c55e", "#94a3b8", "#f87171"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Porcentaje (%)")
    ax.set_title("Distribución porcentual del sentimiento", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=140, facecolor="white")
    plt.close(fig)

    resultados = sentiment.get("resultados", []) or []
    textos_pos = [r.get("text", "") for r in resultados if r.get("label") == "positivo"]
    textos_neu = [r.get("text", "") for r in resultados if r.get("label") == "neutro"]
    textos_neg = [r.get("text", "") for r in resultados if r.get("label") == "negativo"]

    def _wordcloud(textos: List[str], path: Path, title: str) -> str:
        tokens: List[str] = []
        for texto in textos:
            tokens.extend(_tokens_sin_stopwords(texto))
        if not tokens:
            return _placeholder(path, title, "Sin palabras suficientes")
        wc = WordCloud(
            width=900,
            height=500,
            background_color="white",
            max_words=100,
            collocations=False,
            stopwords=STOPWORDS_ES,
        ).generate(" ".join(tokens))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=140, facecolor="white")
        plt.close(fig)
        return str(path)

    return {
        "sentiment_bar_chart": str(bar_path),
        "wordcloud_positive": _wordcloud(textos_pos, wc_pos_path, "Palabras positivas"),
        "wordcloud_neutral": _wordcloud(textos_neu, wc_neu_path, "Palabras neutras"),
        "wordcloud_negative": _wordcloud(textos_neg, wc_neg_path, "Palabras negativas"),
    }
