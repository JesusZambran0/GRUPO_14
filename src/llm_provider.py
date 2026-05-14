"""Proveedor LLM: Gemini · modelo open source local · rules.

Cambios clave:
- El modelo local por defecto ahora coincide con el adaptador QLoRA incluido: Qwen/Qwen2.5-0.5B-Instruct.
- El LLM local usa un prompt compacto; los modelos pequeños se degradan con prompts largos.
- Se valida la salida del LLM local. Si habla incoherencias, repite frases, devuelve texto corto
  o se sale del dominio, la app cae automáticamente a Gemini y luego a reglas.
- La app principal NO importa torch/transformers al arrancar. El LLM local es lazy loading.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# IMPORTANTE: el adaptador que entrenaste es de Qwen, no de SmolLM.
# Si se deja SmolLM por defecto, el adapter puede no cargar y el modelo base hablará basura.
LOCAL_MODEL_ID = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LOCAL_LORA_PATH = os.getenv("LOCAL_LORA_PATH", "models/qwen_marketing_qlora")
LOCAL_MAX_NEW_TOKENS = int(os.getenv("LLM_LOCAL_MAX_NEW_TOKENS", "420"))
LOCAL_TEMPERATURE = float(os.getenv("LLM_LOCAL_TEMPERATURE", "0.20"))
LOCAL_MIN_WORDS = int(os.getenv("LLM_LOCAL_MIN_WORDS", "70"))
ALLOW_BAD_LOCAL_LLM = os.getenv("ALLOW_BAD_LOCAL_LLM", "false").lower() in {"1", "true", "yes", "on"}

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/" f"{GEMINI_MODEL}:generateContent"
GEMINI_TIMEOUT_S = float(os.getenv("GEMINI_TIMEOUT_S", "150"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2100"))

SYSTEM_PROMPT = (
    "Eres un analista senior de contenido para YouTube, marketing, pauta digital y narrativa audiovisual. "
    "Interpreta los datos sin asumir que todo video vende un producto. "
    "Respeta la intención detectada: humor/entretenimiento, educativo, informativo, branding o comercial. "
    "Si el video es humorístico, evalúa setup, remate, ritmo, claridad del chiste y compartibilidad; no fuerces CTA de compra. "
    "Si es comercial, evalúa beneficio, oferta, prueba y acción. "
    "Responde en español natural, con recomendaciones accionables. NO uses JSON. No inventes cifras."
)

ACTION_HEADLINES = {
    "IMPULSAR": "Adelante: el video tiene señales suficientes para pautar.",
    "AJUSTAR ANTES DE IMPULSAR": "Casi listo: con ajustes puntuales rinde mejor antes de invertir.",
    "MONITOREAR": "No pautar todavía: optimiza el creativo y vuelve a evaluar con nuevas señales.",
    "NO IMPULSAR": "No conviene pautar este video con la información disponible.",
    "REVISIÓN HUMANA": "Mejor que una persona del equipo lo revise antes de decidir.",
}


def _clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:markdown|json|text)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fmt(v: Any, sfx: str = "") -> str:
    return f"{v}{sfx}" if v not in (None, "", 0) else "—"


def construir_prompt(diagnostic: Dict[str, Any]) -> str:
    """Prompt completo para Gemini. Gemini sí soporta contexto largo."""
    d = diagnostic or {}
    pub = d.get("metricas_publicas", {}) or {}
    priv = d.get("metricas_privadas", {}) or {}
    pol = d.get("politica", {}) or {}
    guion = d.get("guion", {}) or {}
    visual = d.get("visual", {}) or {}
    proj = d.get("proyeccion", {}) or {}
    sent = d.get("sentimiento", {}) or {}
    ocr = d.get("ocr", {}) or {}
    transcript = d.get("transcripcion", {}) or {}
    intent = d.get("content_intent") or guion.get("content_intent") or ocr.get("content_intent") or "general/branding"

    cats = pol.get("categorias") or []
    cats_str = ", ".join(cats) if cats else "ninguna"
    script_recs = "; ".join((guion.get("mejoras") or [])[:4]) or "—"
    transcript_preview = (transcript.get("preview") or d.get("transcript_text") or "—")[:1200]
    ocr_text = (ocr.get("text") or d.get("ocr_text") or "—")[:800]

    return (
        "Redacta una recomendación final DETALLADA para el equipo de contenido/marketing. "
        "No respondas solo con el veredicto. Explica qué entendiste del video y qué se debe mejorar. "
        "Adapta la recomendación al tipo de contenido detectado. Si es humor, no recomiendes CTA de compra ni producto; "
        "habla de setup, remate, claridad, ritmo, retención, subtítulos y compartibilidad. "
        "Si es comercial, sí puedes hablar de oferta, CTA, beneficio y conversión.\n\n"
        f"**Título:** {d.get('titulo') or '—'}\n"
        f"**Tipo de contenido detectado:** {intent}\n"
        f"**Acción recomendada por el sistema:** {d.get('accion') or '—'}\n"
        f"**Probabilidad de aptitud publicitaria:** {_fmt(d.get('probabilidad_pct'), '%')}\n"
        f"**Score global 0-100:** {_fmt(d.get('score_0_100'))}\n\n"
        "### Métricas públicas\n"
        f"- Views: {_fmt(pub.get('views'))} | Likes: {_fmt(pub.get('likes'))} | Comentarios: {_fmt(pub.get('comments'))} | Engagement: {_fmt(pub.get('engagement_pct'), '%')}\n\n"
        "### Métricas privadas / operativas\n"
        f"- Shares: {_fmt(priv.get('shares'))} | Retención: {_fmt(priv.get('retention_pct'), '%')} | Watch time prom.: {_fmt(priv.get('average_watch_time'), 's')} | Horas desde publicación: {_fmt(priv.get('hours_since_publication'))}\n\n"
        "### Política publicitaria\n"
        f"- Nivel: {_fmt(pol.get('nivel'))} | Estado YT Ads: {_fmt(pol.get('estado'))}\n- Categorías: {cats_str}\n\n"
        "### Guion / transcripción\n"
        f"- Calidad: {_fmt(guion.get('score'))}/100 | Tono: {_fmt(guion.get('tono'))} | Hook: {_fmt(guion.get('hook_type'))}\n"
        f"- Interpretación: {_fmt(guion.get('interpretacion'))}\n"
        f"- Mejoras detectadas: {script_recs}\n"
        f"- Transcripción analizada: {transcript_preview}\n\n"
        "### OCR / texto en pantalla\n"
        f"- Tipo de texto: {_fmt(ocr.get('role'))} | Relevancia OCR: {_fmt(ocr.get('relevance_pct'), '%')} | Conexión contextual: {_fmt(ocr.get('overlap_pct'), '%')}\n"
        f"- Interpretación OCR: {_fmt(ocr.get('interpretation'))}\n"
        f"- Texto detectado: {ocr_text}\n\n"
        "### Composición visual\n"
        f"- Score: {_fmt(visual.get('score_0_100'))}/100 | Resumen: {_fmt(visual.get('resumen'))}\n- Sugerencias: {'; '.join(visual.get('sugerencias') or []) or '—'}\n\n"
        "### Sentimiento\n"
        f"- Positivos: {_fmt(sent.get('pct_positivo'), '%')} | Neutros: {_fmt(sent.get('pct_neutro'), '%')} | Negativos: {_fmt(sent.get('pct_negativo'), '%')} | Dominante: {_fmt(sent.get('sentimiento_dominante'))}\n\n"
        "### Proyección con pauta\n"
        f"- CPM: USD {_fmt(proj.get('cpm'))} | Presupuesto: USD {_fmt(proj.get('presupuesto'))} | Views esperadas: {_fmt(proj.get('views_esperadas'))}\n\n"
        "Estructura obligatoria:\n"
        "### Veredicto\n"
        "### Qué está funcionando\n"
        "### Qué se debe mejorar\n"
        "### Lectura de transcripción y OCR\n"
        "### Recomendación accionable\n"
        "### Límite metodológico\n"
        "Extensión objetivo: 350 a 550 palabras. No uses JSON."
    )


def construir_prompt_local(diagnostic: Dict[str, Any]) -> str:
    """Prompt compacto para Qwen/QLoRA.

    Los modelos pequeños tienden a alucinar si reciben prompts largos con demasiadas tablas.
    Este prompt se parece más al dataset SFT: instruction + input + output esperado.
    """
    d = diagnostic or {}
    pub = d.get("metricas_publicas", {}) or {}
    priv = d.get("metricas_privadas", {}) or {}
    guion = d.get("guion", {}) or {}
    ocr = d.get("ocr", {}) or {}
    visual = d.get("visual", {}) or {}
    sent = d.get("sentimiento", {}) or {}
    pol = d.get("politica", {}) or {}
    transcript = d.get("transcripcion", {}) or {}
    intent = d.get("content_intent") or guion.get("content_intent") or ocr.get("content_intent") or "general/branding"

    transcript_preview = (transcript.get("preview") or d.get("transcript_text") or "")[:700]
    ocr_text = (ocr.get("text") or d.get("ocr_text") or "")[:400]

    return (
        "Genera una recomendación de marketing en español para decidir si este video debe pautarse, mejorarse o monitorearse. "
        "No uses JSON. No inventes cifras. Respeta el tipo de contenido: si es humor o entretenimiento, no hables de venta, CTA de compra ni producto.\n\n"
        f"Título: {d.get('titulo') or '—'}\n"
        f"Tipo de contenido: {intent}\n"
        f"Decisión sugerida: {d.get('accion') or '—'}\n"
        f"Score global: {_fmt(d.get('score_0_100'))}/100. Probabilidad: {_fmt(d.get('probabilidad_pct'), '%')}.\n"
        f"Métricas: views={_fmt(pub.get('views'))}, likes={_fmt(pub.get('likes'))}, comentarios={_fmt(pub.get('comments'))}, engagement={_fmt(pub.get('engagement_pct'), '%')}, retención={_fmt(priv.get('retention_pct'), '%')}.\n"
        f"Sentimiento: positivo={_fmt(sent.get('pct_positivo'), '%')}, neutral={_fmt(sent.get('pct_neutro'), '%')}, negativo={_fmt(sent.get('pct_negativo'), '%')}, dominante={_fmt(sent.get('sentimiento_dominante'))}.\n"
        f"Guion: score={_fmt(guion.get('score'))}/100, tono={_fmt(guion.get('tono'))}, hook={_fmt(guion.get('hook_type'))}, interpretación={_fmt(guion.get('interpretacion'))}.\n"
        f"OCR: relevancia={_fmt(ocr.get('relevance_pct'), '%')}, rol={_fmt(ocr.get('role'))}, texto='{ocr_text}'.\n"
        f"Transcripción: {transcript_preview or 'sin transcripción suficiente'}.\n"
        f"Composición visual: score={_fmt(visual.get('score_0_100'))}/100, resumen={_fmt(visual.get('resumen'))}.\n"
        f"Políticas: riesgo={_fmt(pol.get('nivel'))}, estado={_fmt(pol.get('estado'))}.\n\n"
        "Responde con estas secciones exactas:\n"
        "### Veredicto\n"
        "### Hallazgos principales\n"
        "### Qué mejorar\n"
        "### Lectura de transcripción y OCR\n"
        "### Conclusión\n"
    )


def _rules_recommendation(diagnostic: Dict[str, Any]) -> str:
    d = diagnostic or {}
    action = d.get("accion") or "MONITOREAR"
    headline = ACTION_HEADLINES.get(action, "Revisar el diagnóstico antes de decidir.")
    prob = d.get("probabilidad_pct")
    pol = d.get("politica", {}) or {}
    visual = d.get("visual", {}) or {}
    guion = d.get("guion", {}) or {}
    proj = d.get("proyeccion", {}) or {}
    sent = d.get("sentimiento", {}) or {}
    intent = d.get("content_intent") or guion.get("content_intent") or "general/branding"

    razones = []
    if prob is not None:
        razones.append(f"El modelo estima {prob}% de probabilidad de aptitud publicitaria, no ROI ni vistas garantizadas.")
    cats = pol.get("categorias") or []
    if cats:
        razones.append(f"El screening de políticas detectó señales en: {', '.join(cats[:3])} (nivel {pol.get('nivel') or '—'}).")
    else:
        razones.append("El screening de políticas no detectó categorías sensibles.")
    if visual.get("score_0_100") is not None:
        razones.append(f"Composición visual: {visual.get('score_0_100')}/100. {visual.get('resumen') or ''}".strip())
    if sent.get("sentimiento_dominante"):
        razones.append(f"Sentimiento: {sent.get('sentimiento_dominante')} ({sent.get('pct_positivo', 0):.0f}% positivos, {sent.get('pct_negativo', 0):.0f}% negativos).")

    if "humor" in str(intent).lower() or "entreten" in str(intent).lower():
        acciones = [
            "Revisar que el remate se entienda sin depender demasiado del contexto externo.",
            "Mantener subtítulos o texto en pantalla solo si ayudan al chiste o al ritmo.",
            "Probar una versión más corta si la retención cae antes del remate.",
        ]
    else:
        acciones = list((guion.get("mejoras") or [])[:3])
        if not acciones:
            acciones = [
                "Aclarar el beneficio principal en los primeros segundos.",
                "Reducir fricción visual y reforzar el mensaje central.",
                "Probar una variante con hook más directo.",
            ]

    presupuesto = proj.get("presupuesto")
    views_esp = proj.get("views_esperadas")
    proy_str = f" Con USD {presupuesto} pautados, la proyección parametrizada es ≈ {int(views_esp):,} views." if presupuesto and views_esp else ""

    return "\n".join([
        "### Veredicto",
        headline,
        "",
        "### Razones",
        *[f"- {r}" for r in razones[:4]],
        "",
        "### Próximos pasos",
        *[f"- {a}" for a in acciones[:4]],
        "",
        "### Límite metodológico",
        "Esta recomendación combina un modelo predictivo, reglas de negocio, análisis visual y lectura de texto. No garantiza viralidad, ROI ni aprobación oficial de YouTube Ads." + proy_str,
    ])


def _generate_with_gemini(diagnostic: Dict[str, Any]) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": construir_prompt(diagnostic)}]}],
        "generationConfig": {"temperature": 0.35, "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS, "topP": 0.9},
    }
    try:
        req = urllib.request.Request(
            f"{GEMINI_API_URL}?key={api_key}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        parts = (body.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
        text = _clean_text("".join(p.get("text", "") for p in parts))
        if text and len(text.split()) >= 45:
            return text
        return None
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError, Exception):
        return None



def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extrae un JSON objeto de una respuesta LLM, tolerando markdown."""
    if not text:
        return None
    clean = _clean_text(text)
    try:
        data = json.loads(clean)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _heuristic_ocr_interpretation(first_ocr_text: str, full_ocr_text: str = "", title: str = "", description: str = "", transcript_text: str = "") -> Dict[str, Any]:
    raw = (full_ocr_text or first_ocr_text or "").strip()
    clean = re.sub(r"\s+", " ", raw)
    lowered = " ".join([title or "", description or "", transcript_text or "", clean]).lower()
    if any(x in lowered for x in ["jaja", "humor", "chiste", "meme", "comedia", "pov", "cuando ", "broma"]):
        intent = "humor/entretenimiento"
        role = "caption_contexto_o_remate"
        meaning = "El texto en pantalla parece funcionar como caption, contexto o remate del contenido; no debe evaluarse como oferta o CTA de venta."
    elif any(x in lowered for x in ["compra", "oferta", "descuento", "precio", "agenda", "reserva", "whatsapp", "cotiza"]):
        intent = "comercial/promocional"
        role = "oferta_beneficio_o_cta"
        meaning = "El texto visible parece reforzar una intención comercial: oferta, beneficio, prueba o llamado a la acción."
    elif any(x in lowered for x in ["tutorial", "aprende", "paso", "tips", "consejo", "clase", "cómo", "como hacer"]):
        intent = "educativo/tutorial"
        role = "apoyo_explicativo"
        meaning = "El texto visible parece apoyar una explicación, paso o aprendizaje."
    else:
        intent = "general/branding"
        role = "contexto_visual"
        meaning = "El texto visible aporta contexto general, énfasis o subtítulo, pero requiere comparación con la transcripción para una lectura más precisa."
    return {
        "corrected_text": clean,
        "meaning": meaning,
        "content_intent": intent,
        "ocr_role": role,
        "confidence": 0.35 if clean else 0.0,
        "source": "rules",
        "warning": "Gemini no disponible; se usó interpretación heurística del OCR.",
    }


def interpret_ocr_text_with_llm(
    *,
    first_ocr_text: str,
    full_ocr_text: str = "",
    title: str = "",
    description: str = "",
    transcript_text: str = "",
    mode: str = "auto",
) -> Dict[str, Any]:
    """Normaliza e interpreta OCR ruidoso con Gemini, con fallback de reglas.

    Devuelve datos estructurados internos. No se muestra JSON al usuario.
    """
    first = (first_ocr_text or "").strip()
    full = (full_ocr_text or "").strip()
    if not first and not full:
        return {
            "corrected_text": "",
            "meaning": "No se detectó texto suficiente para interpretar OCR.",
            "content_intent": "sin_texto",
            "ocr_role": "sin_texto_detectado",
            "confidence": 0.0,
            "source": "none",
            "warning": "Sin texto OCR.",
        }

    mode_req = (mode or "auto").lower().strip()
    use_gemini = mode_req in {"auto", "gemini", "local_open_source"} and bool(os.getenv("GEMINI_API_KEY", "").strip())
    if not use_gemini:
        return _heuristic_ocr_interpretation(first, full, title, description, transcript_text)

    prompt = f"""
Eres corrector de OCR y analista de contenido audiovisual en español.
El OCR viene de frames de un video y puede tener errores. Tu tarea es reconstruir de forma conservadora qué dice probablemente el texto en pantalla y explicar qué función cumple.

REGLAS:
- No inventes frases largas que no estén respaldadas por el OCR.
- Si el video parece humor, música o entretenimiento, NO lo conviertas en anuncio ni CTA comercial.
- Si el texto es ilegible, deja corrected_text vacío o muy corto.
- Devuelve SOLO JSON válido, sin markdown.

Contexto del video:
Título: {title[:300] or "—"}
Descripción: {description[:500] or "—"}
Transcripción: {transcript_text[:900] or "—"}

Primer resultado OCR:
{first[:500] or "—"}

OCR completo agregado:
{full[:1200] or "—"}

Formato exacto:
{{
  "corrected_text": "texto corregido y normalizado",
  "meaning": "qué significa o qué función cumple el texto en pantalla",
  "content_intent": "humor/entretenimiento | musical | comercial/promocional | educativo/tutorial | informativo/noticioso | general/branding",
  "ocr_role": "subtitulo | caption_contexto_o_remate | letra_musical | oferta_beneficio_o_cta | apoyo_explicativo | contexto_visual | texto_no_confiable",
  "confidence": 0.0
}}
""".strip()
    payload = {
        "system_instruction": {"parts": [{"text": "Corrige OCR de videos de forma conservadora. No inventes. Devuelve JSON válido."}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 450, "topP": 0.8},
    }
    try:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        req = urllib.request.Request(
            f"{GEMINI_API_URL}?key={api_key}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=min(GEMINI_TIMEOUT_S, 18)) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        parts = (body.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
        raw_text = "".join(p.get("text", "") for p in parts)
        parsed = _extract_json_object(raw_text) or {}
        corrected = _clean_text(str(parsed.get("corrected_text") or full or first))
        meaning = _clean_text(str(parsed.get("meaning") or ""))
        content_intent = _clean_text(str(parsed.get("content_intent") or "general/branding"))
        ocr_role = _clean_text(str(parsed.get("ocr_role") or "contexto_visual"))
        try:
            confidence = float(parsed.get("confidence", 0.55))
        except Exception:
            confidence = 0.55
        return {
            "corrected_text": corrected,
            "meaning": meaning or "El texto en pantalla fue normalizado con LLM para mejorar la lectura del OCR.",
            "content_intent": content_intent,
            "ocr_role": ocr_role,
            "confidence": round(max(0.0, min(confidence, 1.0)), 3),
            "source": "gemini_ocr",
            "warning": "",
        }
    except Exception as exc:
        out = _heuristic_ocr_interpretation(first, full, title, description, transcript_text)
        out["warning"] = f"Gemini OCR falló ({type(exc).__name__}); se usó interpretación heurística."
        return out


@lru_cache(maxsize=1)
def load_local_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    lora_path = Path(LOCAL_LORA_PATH)
    if lora_path.exists() and (lora_path / "adapter_config.json").exists():
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, str(lora_path))

    model.eval()
    return tokenizer, model


def _local_output_quality(text: str) -> Tuple[bool, str]:
    """Filtro anti-mamadas para modelos pequeños.

    El objetivo no es juzgar calidad perfecta, sino impedir que la app muestre
    outputs incoherentes en una demo.
    """
    text = _clean_text(text)
    if ALLOW_BAD_LOCAL_LLM:
        return True, "validación desactivada"
    if not text:
        return False, "salida vacía"

    lower = text.lower()
    words = re.findall(r"\w+", lower, flags=re.UNICODE)
    if len(words) < LOCAL_MIN_WORDS:
        return False, f"salida demasiado corta ({len(words)} palabras)"

    forbidden = [
        "entrada de 2030",
        "entorno de 2030",
        "clave de entrada",
        "rellenar las traducciones",
        "no utilice json",
        "se ha crecido una proteccion",
        "cuando haces una entrevista en 2030",
    ]
    for bad in forbidden:
        if bad in lower:
            return False, f"texto incoherente detectado: {bad}"

    # Señales mínimas del dominio esperado.
    domain_terms = [
        "video", "contenido", "youtube", "retención", "transcripción", "ocr",
        "sentimiento", "pauta", "marketing", "guion", "visual", "recomendación",
        "humor", "métrica", "comentarios", "audiencia",
    ]
    if sum(1 for term in domain_terms if term in lower) < 2:
        return False, "la respuesta no habla del dominio del proyecto"

    # Repetición excesiva de una misma frase corta.
    fourgrams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    if fourgrams:
        unique_ratio = len(set(fourgrams)) / max(len(fourgrams), 1)
        if unique_ratio < 0.72:
            return False, "repetición excesiva de frases"

    return True, "ok"


def generate_local_recommendation(diagnostic: Dict[str, Any]) -> str:
    import torch  # type: ignore
    tokenizer, model = load_local_model()
    prompt = construir_prompt_local(diagnostic)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        inputs = {"input_ids": input_ids}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=LOCAL_MAX_NEW_TOKENS,
            do_sample=LOCAL_TEMPERATURE > 0,
            temperature=max(LOCAL_TEMPERATURE, 0.01),
            top_p=0.85,
            repetition_penalty=1.18,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return _clean_text(tokenizer.decode(new_tokens, skip_special_tokens=True))


def generate_recommendation_with_llm(diagnostic: Dict[str, Any], mode: str = "auto") -> Dict[str, Any]:
    started = time.time()
    mode_req = (mode or "auto").lower().strip()
    aliases = {
        "local_smol": "local_open_source",
        "local_qwen": "local_open_source",
        "open_source": "local_open_source",
    }
    mode_eff_req = aliases.get(mode_req, mode_req)
    if mode_eff_req not in {"auto", "gemini", "local_open_source", "rules"}:
        mode_eff_req = "auto"

    has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip())
    local_enabled = os.getenv("ENABLE_LOCAL_LLM", "false").lower() in {"1", "true", "yes", "on"}
    warning = ""

    effective = mode_eff_req
    if mode_eff_req == "auto":
        # Para demo estable, Gemini primero. El LLM local queda como opción técnica explícita.
        if has_gemini:
            effective = "gemini"
        elif local_enabled:
            effective = "local_open_source"
        else:
            effective = "rules"

    if effective == "gemini":
        text = _generate_with_gemini(diagnostic)
        if text:
            return {
                "recomendacion": text,
                "source": "gemini",
                "model": GEMINI_MODEL,
                "warning": "",
                "mode_requested": mode_req,
                "elapsed_s": round(time.time() - started, 3),
            }
        warning = "Gemini no respondió o la key es inválida; se usaron reglas."

    elif effective == "local_open_source":
        try:
            text = generate_local_recommendation(diagnostic)
            ok, reason = _local_output_quality(text)
            if ok:
                return {
                    "recomendacion": text,
                    "source": "local_open_source",
                    "model": LOCAL_MODEL_ID,
                    "lora_path": LOCAL_LORA_PATH,
                    "warning": "",
                    "mode_requested": mode_req,
                    "elapsed_s": round(time.time() - started, 3),
                }

            # Si el local habló mamadas, no lo mostramos: intentamos Gemini y luego rules.
            warning = f"El LLM local produjo una respuesta descartada ({reason})."
            if has_gemini:
                gemini_text = _generate_with_gemini(diagnostic)
                if gemini_text:
                    return {
                        "recomendacion": gemini_text,
                        "source": "gemini_after_local_fallback",
                        "model": GEMINI_MODEL,
                        "warning": warning,
                        "mode_requested": mode_req,
                        "elapsed_s": round(time.time() - started, 3),
                    }
        except ImportError as e:
            warning = f"El LLM local requiere torch + transformers + peft: {e}. Instala requirements-llm.txt o usa Gemini/rules."
        except MemoryError:
            warning = "El LLM local no cargó por falta de memoria; se usaron reglas."
        except Exception as e:
            warning = f"El LLM local falló ({type(e).__name__}: {e}); se usaron reglas."

    text = _rules_recommendation(diagnostic)
    return {
        "recomendacion": text,
        "source": "rules",
        "model": "rules",
        "warning": warning,
        "mode_requested": mode_req,
        "elapsed_s": round(time.time() - started, 3),
    }
