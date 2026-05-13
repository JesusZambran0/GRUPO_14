"""Análisis semántico del guion/transcripción para videos de YouTube.

Esta capa es determinista y ligera: no depende de Gemini ni de APIs externas.
A diferencia de versiones anteriores, primero detecta la intención del contenido
(humor/entretenimiento, educativo, comercial, informativo, institucional) y luego
aplica criterios adecuados a esa intención. Así un video de humor no es castigado
por no vender un producto ni tener CTA comercial.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .script_dataset_builder import (
    BENEFIT_TERMS, CTA_TERMS, EXAGGERATED_TERMS, GENERIC_OPENINGS,
    TRUST_TERMS, URGENCY_TERMS, default_patterns_reference,
)

ROOT = Path(__file__).resolve().parents[1]
PATTERNS_JSON = ROOT / "data" / "processed" / "script_patterns_reference.json"

HUMOR_TERMS = {
    "jaja", "jajaja", "jeje", "risa", "risa", "humor", "broma", "bromas",
    "chiste", "chistes", "meme", "memes", "parodia", "sketch", "comedia",
    "cómico", "comico", "gracioso", "graciosa", "sarcasmo", "irónico", "ironico",
    "troll", "fails", "fail", "absurdo", "absurda", "random", "reacción", "reaccion",
}
EDU_TERMS = {
    "tutorial", "aprende", "aprender", "guía", "guia", "paso a paso", "cómo",
    "como hacer", "explico", "explicar", "consejos", "tips", "clase", "curso",
    "lección", "leccion", "método", "metodo", "herramienta", "tutoriales",
}
COMMERCIAL_TERMS = {
    "compra", "comprar", "precio", "descuento", "oferta", "promoción", "promocion",
    "cupón", "cupon", "envío", "envio", "agenda", "reserva", "inscríbete", "inscribete",
    "servicio", "producto", "tienda", "catálogo", "catalogo", "cliente", "clientes",
    "venta", "ventas", "pago", "gratis", "whatsapp", "link en bio", "cotiza", "cotizar",
}
NEWS_TERMS = {
    "noticia", "última hora", "ultima hora", "comunicado", "denuncia", "informe",
    "actualidad", "política", "politica", "gobierno", "elecciones", "declaró", "declaro",
}


def _load_patterns() -> Dict[str, Any]:
    try:
        if PATTERNS_JSON.exists():
            return json.loads(PATTERNS_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default_patterns_reference()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?¿¡]+", text or "") if s.strip()]


def _count_terms(text: str, terms) -> int:
    low = (text or "").lower()
    return sum(1 for term in terms if str(term).lower() in low)


def _first_words(text: str, n: int = 40) -> str:
    return " ".join((text or "").split()[:n])


def _clip(text: str, max_chars: int = 520) -> str:
    text = _clean(text)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def infer_content_intent(title: str, description: str, transcript: str, category: Optional[str] = None, topic: Optional[str] = None) -> str:
    text = " ".join([title or "", description or "", transcript or "", category or "", topic or ""]).lower()
    commercial = _count_terms(text, COMMERCIAL_TERMS) + _count_terms(text, BENEFIT_TERMS) + _count_terms(text, CTA_TERMS)
    humor = _count_terms(text, HUMOR_TERMS)
    educational = _count_terms(text, EDU_TERMS)
    news = _count_terms(text, NEWS_TERMS)

    # Si hay señales explícitas de humor, prioriza entretenimiento salvo que haya venta fuerte.
    if humor >= 1 and commercial < 3:
        return "humor/entretenimiento"
    if commercial >= 3:
        return "comercial/promocional"
    if educational >= 2:
        return "educativo/tutorial"
    if news >= 1:
        return "informativo/noticioso"
    if any(x in text for x in ["vlog", "historia", "experiencia", "reacción", "reaccion", "reto", "challenge"]):
        return "entretenimiento/lifestyle"
    return "general/branding"


def _tone(text: str, cta_count: int, urgency_count: int, exaggerated_count: int, benefit_count: int, trust_count: int, intent: str) -> str:
    low = text.lower()
    if intent.startswith("humor"):
        if any(x in low for x in ["sarcasmo", "irónico", "ironico", "parodia"]):
            return "humorístico/irónico"
        return "humorístico/entretenimiento"
    if exaggerated_count >= 2 or urgency_count >= 3:
        return "sensacionalista/agresivo"
    if any(x in low for x in ["testimonio", "case study", "caso real", "experiencia", "review"]):
        return "testimonial"
    if intent.startswith("educativo"):
        return "educativo"
    if cta_count > 0 and benefit_count > 0:
        return "promocional"
    if trust_count > 0:
        return "informativo/confiable"
    return "informativo"


def _hook_type(hook: str, intent: str) -> str:
    low = hook.lower()
    if "?" in hook or low.startswith(("cómo", "como", "por qué", "porque", "qué", "que", "sabías", "sabias")):
        return "pregunta/curiosidad"
    if intent.startswith("humor") and any(t in low for t in HUMOR_TERMS):
        return "setup humorístico"
    if any(t in low for t in ["no vas a creer", "mira", "cuando", "pov", "imagina", "nadie"]):
        return "situación/expectativa"
    if _count_terms(hook, BENEFIT_TERMS) > 0:
        return "beneficio directo"
    if _count_terms(hook, COMMERCIAL_TERMS) > 0:
        return "oferta/comercial"
    return "contextual"


def _summarize_transcript(transcript: str, intent: str) -> str:
    if not transcript.strip():
        return "No hay transcripción suficiente para interpretar el guion."
    sents = _sentences(transcript)
    if not sents:
        return _clip(transcript, 360)
    core = " ".join(sents[:3])
    if intent.startswith("humor"):
        return f"La transcripción parece construir una situación de humor o entretenimiento. Fragmento inicial: {_clip(core, 420)}"
    if intent.startswith("comercial"):
        return f"La transcripción contiene señales de comunicación comercial o de oferta. Fragmento inicial: {_clip(core, 420)}"
    if intent.startswith("educativo"):
        return f"La transcripción parece explicar o enseñar un tema. Fragmento inicial: {_clip(core, 420)}"
    return f"La transcripción entrega contexto narrativo o informativo. Fragmento inicial: {_clip(core, 420)}"


def analyze_video_script(
    title: str,
    description: str,
    transcript: str,
    category: Optional[str] = None,
    topic: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Evalúa la transcripción según la intención real del video.

    Si el contenido es humor/entretenimiento, evalúa setup, claridad narrativa,
    ritmo y remate. Si es comercial, evalúa propuesta de valor y CTA. Si es
    educativo, evalúa explicación, estructura y aprendizaje.
    """
    patterns = _load_patterns()
    title = _clean(title)
    description = _clean(description)
    transcript = _clean(transcript)
    combined = _clean(" ".join([title, description, transcript]))
    words = transcript.split() if transcript else combined.split()
    transcript_word_count = len(transcript.split())
    combined_word_count = len(combined.split())
    hook = _first_words((transcript or title or description), 45)
    low_hook = hook.lower()
    intent = infer_content_intent(title, description, transcript, category=category, topic=topic)

    sentence_parts = _sentences(transcript or combined)
    avg_sentence_len = (sum(len(s.split()) for s in sentence_parts) / max(len(sentence_parts), 1)) if sentence_parts else 0

    cta_count = _count_terms(combined, CTA_TERMS)
    benefit_count = _count_terms(combined, BENEFIT_TERMS)
    urgency_count = _count_terms(combined, URGENCY_TERMS)
    trust_count = _count_terms(combined, TRUST_TERMS)
    commercial_count = _count_terms(combined, COMMERCIAL_TERMS)
    humor_count = _count_terms(combined, HUMOR_TERMS)
    edu_count = _count_terms(combined, EDU_TERMS)
    exaggerated_count = _count_terms(combined, EXAGGERATED_TERMS)
    transcript_available = bool(transcript)
    generic_opening = any(p in low_hook[:180] for p in GENERIC_OPENINGS)

    # Si no hay transcripción, NO inventamos score ni lectura de guion.
    # Para videos musicales/sin voz esto es esperado: se evalúan otros apartados.
    if not transcript_available or transcript_word_count == 0:
        return {
            "script_quality_score": None,
            "hook_score": None,
            "clarity_score": None,
            "intent_fit_score": None,
            "value_proposition_score": None,
            "cta_score": None,
            "cta_clarity": "no evaluable sin transcripción",
            "policy_claim_risk": "no_evaluable",
            "tone": "no evaluable sin transcripción",
            "content_intent": infer_content_intent(title, description, "", category=category, topic=topic),
            "hook_type": "no evaluable",
            "transcript_word_count": 0,
            "transcript_preview": "",
            "transcript_summary": "No hay transcripción suficiente para interpretar el guion.",
            "transcript_interpretation": "Sin transcripción no se evalúa estructura verbal, hook hablado, ritmo narrativo ni claridad del mensaje oral.",
            "main_strengths": [],
            "main_weaknesses": ["No hay transcripción disponible; no se puede analizar el guion."],
            "recommended_script_improvements": ["Cargar una transcripción manual si el video tiene voz o evaluar la pieza desde visuales/OCR si es musical o sin diálogo."],
            "strengths": [],
            "script_strengths": [],
            "weaknesses": ["No hay transcripción disponible; no se puede analizar el guion."],
            "recommendations": ["Cargar una transcripción manual si el video tiene voz o evaluar la pieza desde visuales/OCR si es musical o sin diálogo."],
            "what_to_improve": ["Cargar una transcripción manual si el video tiene voz o evaluar la pieza desde visuales/OCR si es musical o sin diálogo."],
            "suggested_hook": "No evaluable sin transcripción.",
            "suggested_cta": "No evaluable sin transcripción.",
            "suggested_ad_copy": "No evaluable sin transcripción.",
            "transcript_required": True,
            "transcript_available": False,
            "analysis_mode": "sin_transcripcion",
            "benchmark_note": "No se aplicó benchmark de guion porque no hay transcripción.",
            "explanation": "Este bloque se desactiva para evitar recomendaciones inventadas cuando el video no tiene voz o la transcripción falló.",
        }

    # Hook: para humor/entretenimiento no exige promesa de venta; exige situación, curiosidad o setup.
    hook_score = 52
    if "?" in hook or low_hook.startswith(("cómo", "como", "por qué", "porque", "qué", "que", "sabías", "sabias")):
        hook_score += 16
    if intent.startswith("humor") and (humor_count > 0 or any(x in low_hook for x in ["cuando", "pov", "nadie", "mira", "imagina"])):
        hook_score += 22
    elif intent.startswith("comercial") and (benefit_count > 0 or commercial_count > 0):
        hook_score += 20
    elif intent.startswith("educativo") and edu_count > 0:
        hook_score += 18
    if generic_opening:
        hook_score -= 12
    if transcript_word_count < 10:
        hook_score -= 12
    hook_score = max(0, min(100, hook_score))

    clarity_score = 76
    if transcript_word_count == 0:
        clarity_score -= 28
    elif transcript_word_count < 12:
        clarity_score -= 16
    elif transcript_word_count > 600 and (duration_seconds or 0) <= 60:
        clarity_score -= 12
    if avg_sentence_len > 24:
        clarity_score -= 12
    if combined.count("!") > 9:
        clarity_score -= 8
    clarity_score = max(0, min(100, clarity_score))

    # Score de ajuste a la intención del contenido.
    intent_fit = 55
    if intent.startswith("humor"):
        intent_fit += 18 if humor_count > 0 else 6
        if any(x in combined.lower() for x in ["cuando", "pov", "remate", "plot twist", "jaj", "meme", "broma"]):
            intent_fit += 10
        if transcript_word_count > 0 and transcript_word_count <= 120:
            intent_fit += 8
    elif intent.startswith("comercial"):
        intent_fit += min(commercial_count, 4) * 8 + min(benefit_count, 3) * 6 + min(trust_count, 2) * 4
    elif intent.startswith("educativo"):
        intent_fit += min(edu_count, 4) * 8
        if any(x in combined.lower() for x in ["paso", "ejemplo", "primero", "segundo", "explico"]):
            intent_fit += 8
    else:
        if transcript_word_count >= 20:
            intent_fit += 10
    intent_fit = max(0, min(100, intent_fit))

    # CTA no debe penalizar fuerte a humor o entretenimiento.
    if intent.startswith("comercial"):
        cta_score = max(0, min(100, 35 + min(cta_count, 3) * 20 + min(commercial_count, 2) * 8))
        cta_clarity = "claro" if cta_score >= 70 else "débil o ausente"
    elif intent.startswith("humor") or intent.startswith("entretenimiento"):
        cta_score = 68 if cta_count == 0 else 78
        cta_clarity = "no crítico para humor/entretenimiento" if cta_count == 0 else "presente"
    else:
        cta_score = 55 + min(cta_count, 2) * 12
        cta_clarity = "presente" if cta_count > 0 else "opcional según objetivo"
    cta_score = max(0, min(100, cta_score))

    policy_claim_risk = "alto" if exaggerated_count >= 2 else ("medio" if exaggerated_count == 1 else "bajo")
    script_quality_score = int(round(0.28 * hook_score + 0.26 * clarity_score + 0.30 * intent_fit + 0.16 * cta_score))

    strengths: List[str] = []
    weaknesses: List[str] = []
    improvements: List[str] = []

    if transcript_available:
        strengths.append(f"La transcripción fue detectada y permite analizar el guion sobre {transcript_word_count} palabras reales.")
    else:
        weaknesses.append("No hay transcripción suficiente; el análisis no puede evaluar ritmo, intención ni mensaje real del audio.")
        improvements.append("Verificar que el audio del MP4 sea legible o cargar una transcripción manual.")

    if hook_score >= 70:
        strengths.append("El inicio ofrece una entrada reconocible para captar atención.")
    else:
        weaknesses.append("El inicio puede ser más claro o más fuerte en los primeros segundos.")
        if intent.startswith("humor"):
            improvements.append("Abrir con una situación más reconocible, contraste o setup humorístico antes del remate.")
        elif intent.startswith("comercial"):
            improvements.append("Abrir con el beneficio principal o problema que resuelve la oferta.")
        else:
            improvements.append("Abrir con una pregunta, tensión o promesa concreta de valor.")

    if intent.startswith("humor"):
        if intent_fit >= 68:
            strengths.append("El contenido tiene señales de humor/entretenimiento; no necesita CTA comercial para ser evaluable.")
        else:
            weaknesses.append("La intención humorística no queda totalmente clara en la transcripción.")
            improvements.append("Asegurar que el texto/audio prepare el chiste y que el remate sea entendible sin contexto externo.")
        if clarity_score < 65:
            improvements.append("Reducir ruido verbal y hacer más evidente la relación entre setup y remate.")
    elif intent.startswith("comercial"):
        if intent_fit >= 70:
            strengths.append("La propuesta comercial o de valor aparece en la transcripción o metadatos.")
        else:
            weaknesses.append("La propuesta de valor no queda suficientemente explícita.")
            improvements.append("Explicar con claridad qué gana el usuario y por qué debería actuar.")
        if cta_score < 70:
            weaknesses.append("El llamado a la acción comercial es débil o ausente.")
            improvements.append("Agregar un CTA coherente con el objetivo: comprar, agendar, visitar, comentar o conocer más.")
    elif intent.startswith("educativo"):
        strengths.append("El contenido parece orientado a explicar o enseñar, por lo que se prioriza claridad y secuencia.")
        if clarity_score < 70:
            improvements.append("Ordenar la explicación en pasos y resumir la idea principal al inicio.")
    else:
        if clarity_score >= 70:
            strengths.append("La estructura textual es razonablemente clara para una lectura inicial.")
        else:
            weaknesses.append("El mensaje puede sentirse ambiguo o poco ordenado.")
            improvements.append("Definir una idea central y repetirla con menos ruido narrativo.")

    if policy_claim_risk in {"medio", "alto"}:
        weaknesses.append("Se detectan afirmaciones absolutas o exageradas que pueden elevar riesgo publicitario.")
        improvements.append("Sustituir promesas absolutas por formulaciones verificables y moderadas.")

    if not improvements:
        if intent.startswith("humor"):
            improvements.append("Probar una variante con remate más temprano y subtítulo que refuerce el contexto del chiste.")
        elif intent.startswith("comercial"):
            improvements.append("Probar una variante A/B con hook de beneficio y CTA más visible.")
        else:
            improvements.append("Mantener la estructura y probar una versión con inicio más directo.")

    suggested_hook = _suggest_hook(title, topic or category or "el contenido", intent)
    suggested_cta = _suggest_cta(category or topic or "contenido", intent)
    suggested_ad_copy = _suggest_ad_copy(title, topic or category or "contenido", policy_claim_risk, intent)
    tone = _tone(combined, cta_count, urgency_count, exaggerated_count, benefit_count, trust_count, intent)
    hook_type = _hook_type(hook, intent)
    benchmark_note = (
        f"Referencia de patrones: {patterns.get('source')} con {patterns.get('n_transcripts', 0)} transcripciones; "
        f"p75 de calidad={patterns.get('quality_score_p75', 'N/A')}."
    )

    return {
        "script_quality_score": script_quality_score,
        "hook_score": int(hook_score),
        "clarity_score": int(clarity_score),
        "intent_fit_score": int(intent_fit),
        "value_proposition_score": int(intent_fit),
        "cta_score": int(cta_score),
        "cta_clarity": cta_clarity,
        "policy_claim_risk": policy_claim_risk,
        "tone": tone,
        "content_intent": intent,
        "hook_type": hook_type,
        "transcript_word_count": transcript_word_count,
        "transcript_preview": _clip(transcript, 900),
        "transcript_summary": _summarize_transcript(transcript, intent),
        "transcript_interpretation": _interpret_transcript(intent, script_quality_score, hook_score, clarity_score),
        "main_strengths": strengths,
        "main_weaknesses": weaknesses or ["No se detectaron debilidades críticas con las reglas actuales."],
        "recommended_script_improvements": improvements,
        "strengths": strengths,
        "script_strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": improvements,
        "what_to_improve": improvements,
        "suggested_hook": suggested_hook,
        "suggested_cta": suggested_cta,
        "suggested_ad_copy": suggested_ad_copy,
        "transcript_required": True,
        "transcript_available": transcript_available,
        "analysis_mode": "intencion_detectada_y_transcripcion",
        "benchmark_note": benchmark_note,
        "explanation": (
            "El puntaje combina gancho inicial, claridad, ajuste a la intención del contenido y cierre. "
            "La evaluación distingue videos comerciales de humor, educación, información o branding para evitar recomendaciones genéricas de CTA/producto cuando no corresponden."
        ),
    }


def _interpret_transcript(intent: str, score: int, hook_score: int, clarity_score: int) -> str:
    if intent.startswith("humor"):
        if score >= 70:
            return "La transcripción sugiere una pieza de humor entendible. La prioridad no es vender, sino que el setup y el remate se entiendan rápido."
        return "La pieza parece de humor/entretenimiento, pero necesita hacer más evidente el setup, el contexto o el remate para que el público capte la intención."
    if intent.startswith("comercial"):
        if score >= 70:
            return "El guion comunica una intención comercial razonablemente clara. Conviene reforzar beneficio, prueba y cierre si el objetivo es conversión."
        return "El guion tiene intención comercial, pero la promesa de valor o el cierre de acción todavía no son suficientemente claros."
    if intent.startswith("educativo"):
        return "El guion debe priorizar claridad, secuencia y aprendizaje. El rendimiento dependerá de que el usuario entienda rápido qué va a aprender."
    if clarity_score < 60:
        return "La transcripción existe, pero el mensaje puede resultar ambiguo. Conviene ordenar la idea central y reducir ruido verbal."
    return "La transcripción permite una lectura inicial del mensaje; la mejora principal debe alinearse con el objetivo real del contenido."


def _suggest_hook(title: str, topic: str, intent: str) -> str:
    topic = _clean(topic) or "este contenido"
    title = _clean(title)
    if intent.startswith("humor"):
        return "Abrir con el contexto mínimo del chiste y adelantar una tensión visual o verbal que prepare el remate."
    if intent.startswith("educativo"):
        return f"En 3 segundos, dejar claro qué aprenderá la audiencia sobre {topic}."
    if intent.startswith("comercial"):
        if title:
            return f"Mostrar primero el problema o beneficio principal de {title.lower()} antes de explicar detalles."
        return f"Abrir con una promesa concreta relacionada con {topic}."
    if title:
        return f"Convertir el inicio de '{title}' en una pregunta o tensión que invite a seguir viendo."
    return f"Abrir con una pregunta clara sobre {topic}."


def _suggest_cta(category: str, intent: str) -> str:
    if intent.startswith("humor") or intent.startswith("entretenimiento"):
        return "No forzar CTA comercial. Si se requiere cierre, usar una invitación suave: comentar, compartir o ver la segunda parte."
    if intent.startswith("educativo"):
        return "Invitar a guardar, comentar dudas o ver el recurso completo."
    if intent.startswith("comercial"):
        return "Cerrar con una acción específica: comprar, agendar, visitar, cotizar o conocer más."
    return "Usar una acción coherente con el objetivo: comentar, guardar, suscribirse o visitar más información."


def _suggest_ad_copy(title: str, topic: str, policy_claim_risk: str, intent: str) -> str:
    title = _clean(title) or _clean(topic) or "tu video"
    if intent.startswith("humor"):
        return f"Convierte el humor de {title} en una pieza de alcance: remate claro, subtítulo breve y cierre ligero para compartir."
    if policy_claim_risk in {"medio", "alto"}:
        return f"Mejora el rendimiento de {title} con una propuesta clara, verificable y sin promesas absolutas."
    if intent.startswith("comercial"):
        return f"Presenta {title} con un beneficio claro, prueba concreta y una acción fácil de seguir."
    return f"Refuerza la idea central de {title} con un inicio más claro y una conclusión fácil de recordar."
