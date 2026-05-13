"""Capa LLM **redactora**, no predictora.

Reglas del proyecto:
- El LLM NO predice rendimiento ni decide la acción.
- El LLM solo **redacta** una recomendación en lenguaje natural a partir de la
  salida estructurada del modelo predictivo y del análisis de políticas.
- Si el LLM local no carga, se usa ``rule_based_recommendation`` que produce
  una redacción determinista a partir de las mismas señales.

LLM local soportado (opcional):
- Qwen2.5-1.5B-Instruct (default)
- SmolLM2-1.7B-Instruct

Se carga con transformers + torch. Si transformers/torch no están instalados,
si el modelo no se ha descargado, o si la carga falla, se cae limpiamente al
fallback de reglas. **La demo debe poder correr sin red.**
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .config import LLM_LOCAL_MAX_NEW_TOKENS, LLM_LOCAL_MODEL_ID, LLM_LOCAL_TIMEOUT_S

# Claves canónicas del bloque ``analisis_llm`` en el contrato.
LLM_OUTPUT_KEYS = [
    "claridad_mensaje",
    "fuerza_cta",
    "coherencia_semantica",
    "complejidad_sintactica",
    "tipo_contenido",
    "riesgo_comunicacional",
    "fortalezas",
    "debilidades",
    "ajustes_sugeridos",
    "recomendacion_llm",
    "justificacion",
]

# Cache del modelo local cargado en memoria (singleton por proceso).
_LOCAL_LLM_CACHE: Dict[str, Any] = {"loaded": False, "tokenizer": None, "model": None, "error": ""}


# ---------------------------------------------------------------------------
# Reglas locales (siempre disponibles, sin red, sin LLM)
# ---------------------------------------------------------------------------

def rule_based_recommendation(
    features: Dict[str, Any],
    final_recommendation: Dict[str, Any],
    policy_block: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Redacción determinista basada en reglas, sin LLM.

    Es el camino que la app usa por defecto en demo académica. Produce TODAS
    las claves del contrato ``LLM_OUTPUT_KEYS`` a partir de las señales del
    feature row, la salida del modelo y, si está disponible, el bloque de
    políticas de YouTube Ads.
    """
    text_total = features.get("text_total", "") or ""
    words = len(text_total.split())

    claridad = 70
    if words < 6:
        claridad = 45
    elif words > 140:
        claridad = 58
    if features.get("benefit_flag"):
        claridad += 8
    if features.get("cta_flag") or features.get("ocr_cta_flag"):
        claridad += 7
    claridad = max(0, min(100, claridad))

    fuerza_cta = 30 + 35 * int(bool(features.get("cta_flag"))) + 25 * int(bool(features.get("ocr_cta_flag")))
    fuerza_cta = max(0, min(100, fuerza_cta))

    coherencia = 60
    if features.get("transcript_text") and features.get("ocr_text"):
        coherencia = 75
    elif features.get("transcript_text") or features.get("ocr_text"):
        coherencia = 65

    if words > 130:
        complejidad = "alta"
    elif words > 45:
        complejidad = "media"
    else:
        complejidad = "baja"

    if features.get("promo_flag") or features.get("ocr_promo_flag"):
        tipo_contenido = "promocional"
    elif features.get("trust_flag") or features.get("ocr_trust_flag"):
        tipo_contenido = "testimonial"
    elif features.get("benefit_flag"):
        tipo_contenido = "informativo/promocional"
    else:
        tipo_contenido = "informativo"

    debilidades: List[str] = []
    if not (features.get("cta_flag") or features.get("ocr_cta_flag")):
        debilidades.append("El llamado a la acción no es suficientemente explícito.")
    if not features.get("benefit_flag"):
        debilidades.append("El beneficio principal no queda claramente reforzado en el texto.")
    try:
        if float(features.get("ocr_word_count", 0)) > 35 and float(features.get("duration_seconds", 0)) <= 10:
            debilidades.append("El texto visual puede generar saturación cognitiva.")
    except (TypeError, ValueError):
        pass
    try:
        if float(features.get("duration_seconds", 0)) > 60:
            debilidades.append("La duración extensa puede penalizar la retención en formatos pagos.")
    except (TypeError, ValueError):
        pass

    fortalezas: List[str] = []
    if features.get("trust_flag") or features.get("ocr_trust_flag"):
        fortalezas.append("Incluye señales de confianza o legitimidad.")
    if features.get("promo_flag") or features.get("ocr_promo_flag"):
        fortalezas.append("Incluye elementos promocionales útiles para pauta.")
    if features.get("benefit_flag"):
        fortalezas.append("Presenta una propuesta de valor reconocible.")
    if features.get("cta_flag") or features.get("ocr_cta_flag"):
        fortalezas.append("Incluye un llamado a la acción detectable por reglas.")

    nivel = final_recommendation.get("prediccion_rendimiento", "medio")
    if debilidades and nivel in {"bajo", "medio"}:
        riesgo = "alto"
    elif debilidades:
        riesgo = "medio"
    else:
        riesgo = "bajo"

    policy_block = policy_block or {}
    if policy_block.get("policy_risk_level") in {"alto", "revisión humana"}:
        riesgo = "alto"
        debilidades.append(f"Riesgo de políticas de YouTube Ads detectado: {policy_block.get('policy_risk_level')}.")

    justificacion = (
        "Redacción generada por el motor de reglas locales (sin LLM). "
        "Las puntuaciones provienen de heurísticas léxicas y métricas derivadas; "
        "no reemplazan al modelo predictivo."
    )

    return {
        "llm_available": False,
        "llm_engine": "rules",
        "claridad_mensaje": claridad,
        "fuerza_cta": fuerza_cta,
        "coherencia_semantica": coherencia,
        "complejidad_sintactica": complejidad,
        "tipo_contenido": tipo_contenido,
        "riesgo_comunicacional": riesgo,
        "fortalezas": fortalezas or ["El contenido tiene información suficiente para una evaluación inicial."],
        "debilidades": debilidades or ["No se detectaron debilidades críticas con las reglas locales."],
        "ajustes_sugeridos": final_recommendation.get("ajustes_sugeridos", []),
        "recomendacion_llm": final_recommendation.get("recomendacion_impulso", "monitorear"),
        "justificacion": justificacion,
    }


# ---------------------------------------------------------------------------
# LLM local (Qwen2.5-1.5B-Instruct / SmolLM2-1.7B-Instruct), opcional
# ---------------------------------------------------------------------------

def _load_local_llm() -> Dict[str, Any]:
    """Carga perezosa del LLM local pequeño. Cachea en memoria.

    Devuelve dict con ``loaded`` (bool), ``error`` (str) y opcionalmente
    ``tokenizer`` / ``model``. NUNCA lanza excepción: las captura todas.
    """
    if _LOCAL_LLM_CACHE["loaded"]:
        return _LOCAL_LLM_CACHE
    if _LOCAL_LLM_CACHE.get("error"):
        # Ya fallamos antes; no reintentar en cada request.
        return _LOCAL_LLM_CACHE
    try:
        # Imports perezosos: si transformers/torch no están, fallback inmediato.
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        model_id = LLM_LOCAL_MODEL_ID
        # local_files_only=True para evitar descargas en runtime de la demo.
        # Si el usuario quiere descarga inicial, debe correr el script aparte.
        prefer_offline = os.getenv("LLM_LOCAL_OFFLINE", "0") == "1"

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=prefer_offline)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=prefer_offline,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
        _LOCAL_LLM_CACHE.update({"loaded": True, "tokenizer": tokenizer, "model": model, "error": ""})
        return _LOCAL_LLM_CACHE
    except Exception as exc:
        _LOCAL_LLM_CACHE["error"] = f"{type(exc).__name__}: {exc}"
        return _LOCAL_LLM_CACHE


def _generate_with_local_llm(prompt: str) -> Optional[str]:
    """Genera texto con el LLM local. Devuelve None si falla."""
    cache = _load_local_llm()
    if not cache.get("loaded"):
        return None
    try:
        import torch  # type: ignore

        tokenizer = cache["tokenizer"]
        model = cache["model"]
        messages = [
            {"role": "system", "content": "Eres un redactor profesional de recomendaciones publicitarias para YouTube. Respondes siempre en JSON válido."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=LLM_LOCAL_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception:
        return None


def _build_llm_prompt(
    title: str,
    description: str,
    transcript_text: str,
    ocr_text: str,
    features: Dict[str, Any],
    final_recommendation: Dict[str, Any],
    policy_block: Optional[Dict[str, Any]] = None,
) -> str:
    compact = {
        "duration_seconds": features.get("duration_seconds", 0),
        "engagement_rate": features.get("engagement_rate", 0),
        "views_per_day": features.get("views_per_day", 0),
        "cta_flag": features.get("cta_flag", 0),
        "benefit_flag": features.get("benefit_flag", 0),
        "promo_flag": features.get("promo_flag", 0),
        "trust_flag": features.get("trust_flag", 0),
    }
    decision = {
        "prediccion_rendimiento": final_recommendation.get("prediccion_rendimiento"),
        "probabilidad_rendimiento": final_recommendation.get("probabilidad_rendimiento"),
        "recomendacion_impulso": final_recommendation.get("recomendacion_impulso"),
    }
    policy = policy_block or {}
    return f"""IMPORTANTE: No predigas rendimiento. Solo redacta una recomendación a partir de la salida estructurada que te paso. Devuelve SOLO un JSON válido, sin texto adicional.

Título:
{(title or '')[:400]}

Descripción:
{(description or '')[:1000]}

Transcripción del audio:
{(transcript_text or '')[:2000]}

Texto OCR visible:
{(ocr_text or '')[:800]}

Variables calculadas:
{json.dumps(compact, ensure_ascii=False)}

Decisión del modelo (NO la cambies):
{json.dumps(decision, ensure_ascii=False)}

Riesgo de políticas de YouTube Ads:
{json.dumps({k: policy.get(k) for k in ['policy_risk_level', 'youtube_ad_status_estimate', 'policy_risk_categories']}, ensure_ascii=False)}

Devuelve un JSON con estas claves exactas:
- claridad_mensaje: entero 0-100
- fuerza_cta: entero 0-100
- coherencia_semantica: entero 0-100
- complejidad_sintactica: "baja" | "media" | "alta"
- tipo_contenido: "educativo" | "promocional" | "testimonial" | "entretenimiento" | "informativo" | "informativo/promocional"
- riesgo_comunicacional: "bajo" | "medio" | "alto"
- fortalezas: lista de strings
- debilidades: lista de strings
- ajustes_sugeridos: lista de strings
- recomendacion_llm: usa exactamente el valor de "recomendacion_impulso" del modelo
- justificacion: párrafo breve y profesional en español
"""


def _coerce_to_contract(raw: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Garantiza que el dict del LLM cumpla el contrato."""
    out: Dict[str, Any] = {}
    for k in LLM_OUTPUT_KEYS:
        out[k] = raw.get(k, fallback.get(k))
    for k in ("claridad_mensaje", "fuerza_cta", "coherencia_semantica"):
        try:
            out[k] = max(0, min(100, int(round(float(out.get(k, 50))))))
        except Exception:
            out[k] = int(fallback.get(k, 50) or 50)
    for k in ("fortalezas", "debilidades", "ajustes_sugeridos"):
        if not isinstance(out.get(k), list):
            out[k] = fallback.get(k, []) if isinstance(fallback.get(k), list) else []
    return out


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def analyze_with_llm(
    text_total: str,
    features: Dict[str, Any],
    final_recommendation: Dict[str, Any],
    prefer: str = "auto",
    title: str = "",
    description: str = "",
    transcript_text: str = "",
    ocr_text: str = "",
    policy_block: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Devuelve el bloque ``analisis_llm`` cumpliendo el contrato.

    Modos:
    - ``rules``: fuerza el fallback determinista (recomendado para demo CPU).
    - ``local``: intenta cargar el LLM local pequeño; si falla, cae a reglas.
    - ``auto``: alias de ``local`` con caída transparente.

    El LLM no decide la acción; ``recomendacion_llm`` debe replicar
    ``recomendacion_impulso`` del modelo.
    """
    fallback = rule_based_recommendation(features, final_recommendation, policy_block=policy_block)
    if prefer == "rules":
        return fallback

    prompt = _build_llm_prompt(
        title=title, description=description,
        transcript_text=transcript_text, ocr_text=ocr_text,
        features=features, final_recommendation=final_recommendation,
        policy_block=policy_block,
    )
    raw_text = _generate_with_local_llm(prompt)
    if not raw_text:
        # No se pudo cargar/generar → fallback transparente.
        fallback["llm_engine"] = "rules_fallback"
        fallback["justificacion"] = (
            "El LLM local no estuvo disponible; se usó el motor de reglas. "
            f"Detalle: {_LOCAL_LLM_CACHE.get('error', 'sin detalle')}"
        )
        return fallback

    # Intentar parsear el JSON devuelto por el LLM.
    parsed: Optional[Dict[str, Any]] = None
    try:
        # Si el LLM envolvió en code fences, los limpiamos.
        cleaned = raw_text.strip()
        for fence in ("```json", "```"):
            if cleaned.startswith(fence):
                cleaned = cleaned[len(fence):].lstrip()
            if cleaned.endswith("```"):
                cleaned = cleaned[: -3].rstrip()
        # Tomar solo el primer objeto JSON detectado.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(cleaned[start: end + 1])
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        coerced = _coerce_to_contract(parsed, fallback)
        coerced["llm_available"] = True
        coerced["llm_engine"] = f"local:{LLM_LOCAL_MODEL_ID}"
        # Forzamos que recomendacion_llm respete la decisión del modelo.
        coerced["recomendacion_llm"] = final_recommendation.get("recomendacion_impulso", coerced.get("recomendacion_llm"))
        return coerced
    fallback["llm_engine"] = "rules_fallback_parse"
    fallback["justificacion"] = "El LLM local respondió pero no se pudo parsear JSON; se usaron las reglas."
    return fallback
