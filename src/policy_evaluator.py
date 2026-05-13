"""Evaluación de riesgo de políticas de YouTube Ads.

`evaluate_youtube_ad_policy_risk(text)` revisa título + descripción + transcripción
y detecta categorías sensibles según las políticas de monetización y publicidad
de YouTube (resumen público, ver
https://support.google.com/youtube/answer/6162278).

La función NO reemplaza el proceso oficial de revisión de YouTube. Es un
**screening previo** para que el equipo creativo pueda anticipar problemas.

Categorías cubiertas:
- clickbait
- afirmaciones exageradas
- lenguaje inapropiado
- violencia
- contenido adulto
- drogas
- armas
- apuestas
- salud o pérdida de peso
- claims financieros
- odio/lenguaje ofensivo
- desinformación
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Vocabularios por categoría
# ---------------------------------------------------------------------------
# Las listas son base; pueden ampliarse desde el README sin tocar la lógica.
# Cada término se busca como palabra completa cuando es posible.

POLICY_LEXICONS: Dict[str, List[str]] = {
    "clickbait": [
        "no creerás", "no vas a creer", "increíble", "shocking", "te volará",
        "100% verdadero", "esto cambia todo", "te hará llorar", "antes de que lo borren",
        "secreto revelado", "no quieren que sepas", "click aquí",
    ],
    "afirmaciones_exageradas": [
        "garantizado 100", "garantizado al 100", "resultados garantizados",
        "sin fallar", "infalible", "milagroso", "milagrosa", "milagro",
        "el mejor del mundo", "el único método", "siempre funciona",
        "nunca falla", "100% efectivo", "100% seguro",
    ],
    "lenguaje_inapropiado": [
        "mierda", "puta", "puto", "joder", "carajo", "pendejo", "pendeja",
        "cabrón", "cabrona", "verga", "coño", "imbécil", "estúpido", "idiota",
        "fuck", "shit", "bitch", "asshole",
    ],
    "violencia": [
        "matar", "asesinar", "asesinato", "homicidio", "torturar", "tortura",
        "disparar", "balear", "ejecutar a", "golpear hasta", "sangrando",
        "decapitar", "apuñalar", "secuestro", "secuestrar",
    ],
    "contenido_adulto": [
        "porno", "pornografía", "porn", "xxx", "sexo explícito", "desnudo total",
        "nudes", "onlyfans", "cam show", "contenido para adultos", "+18 explícito",
        "escena sexual",
    ],
    "drogas": [
        "cocaína", "cocaina", "marihuana", "weed", "heroína", "heroina",
        "metanfetamina", "crystal meth", "lsd", "éxtasis", "extasis", "mdma",
        "drogarse", "drogas duras", "consumir droga",
    ],
    "armas": [
        "arma de fuego", "pistola ilegal", "rifle de asalto", "ak-47", "ak47",
        "granada", "explosivo casero", "bomba casera", "munición ilegal",
        "cómo fabricar arma", "como fabricar arma",
    ],
    "apuestas": [
        "apostar online", "apuestas online", "casino online", "ruleta gratis",
        "tragamonedas", "slots gratis", "betting", "bono de casino", "ganar apostando",
        "1xbet", "stake casino",
    ],
    "salud_o_perdida_de_peso": [
        "pierde peso rápido", "pierde peso en una semana", "baja 10 kilos",
        "baja 20 kilos", "dieta milagrosa", "quema grasa garantizado",
        "cura el cáncer", "cura cancer", "remedio milagroso",
        "tratamiento garantizado", "elimina diabetes",
    ],
    "claims_financieros": [
        "hazte rico", "haz dinero fácil", "gana dinero fácil", "ingresos pasivos garantizados",
        "trabaja desde casa y gana miles", "duplica tu dinero", "inversión 100% segura",
        "rentabilidad garantizada", "sin riesgo financiero",
        "criptomoneda 100x", "shitcoin x100",
    ],
    "odio_lenguaje_ofensivo": [
        "odio a los", "muerte a los", "hay que eliminar a",
        "raza inferior", "subhumano", "supremacía",
    ],
    "desinformacion": [
        "la tierra es plana", "vacunas causan autismo",
        "5g causa", "covid es mentira", "no existe el cambio climático",
        "no existe el covid", "los medicamentos no funcionan",
    ],
}

# Etiquetas legibles para el output.
CATEGORY_LABELS = {
    "clickbait": "clickbait",
    "afirmaciones_exageradas": "afirmaciones exageradas",
    "lenguaje_inapropiado": "lenguaje inapropiado",
    "violencia": "violencia",
    "contenido_adulto": "contenido adulto",
    "drogas": "drogas",
    "armas": "armas",
    "apuestas": "apuestas",
    "salud_o_perdida_de_peso": "salud o pérdida de peso",
    "claims_financieros": "claims financieros",
    "odio_lenguaje_ofensivo": "odio/lenguaje ofensivo",
    "desinformacion": "desinformación",
}

# Severidad de cada categoría. "alto" implica riesgo de no-monetización.
CATEGORY_SEVERITY = {
    "clickbait": "medio",
    "afirmaciones_exageradas": "medio",
    "lenguaje_inapropiado": "medio",
    "violencia": "alto",
    "contenido_adulto": "alto",
    "drogas": "alto",
    "armas": "alto",
    "apuestas": "alto",
    "salud_o_perdida_de_peso": "alto",
    "claims_financieros": "alto",
    "odio_lenguaje_ofensivo": "alto",
    "desinformacion": "alto",
}


def _detect_category(text_lower: str, terms: List[str]) -> Tuple[bool, List[str]]:
    """Devuelve (hit, hits_found)."""
    hits: List[str] = []
    for term in terms:
        # Coincidencia case-insensitive como substring.
        # Los términos son frases o palabras separadas por espacio; substring es suficiente.
        if term.lower() in text_lower:
            hits.append(term)
    return (len(hits) > 0, hits)


def evaluate_youtube_ad_policy_risk(text: str, has_transcript: bool = True) -> Dict[str, Any]:
    """Evalúa el riesgo publicitario de un texto (title + description + transcript).

    Args:
        text: cadena combinada con todo el texto verbal del video.
        has_transcript: indica si el ``text`` incluye transcripción real. Si es
            False, la evaluación se marca como **parcial** (basada solo en
            metadata textual) pero NO se escala automáticamente a revisión
            humana — se reporta como riesgo `bajo`/`medio`/`alto` con bandera.

    Returns:
        dict con:
        - ``policy_risk_level``: "bajo" | "medio" | "alto" | "revisión humana"
        - ``policy_risk_categories``: lista de etiquetas legibles detectadas
        - ``policy_explanation``: explicación en español
        - ``youtube_ad_status_estimate``: "apto" | "apto limitado" | "no apto" | "revisión humana"
        - ``hits``: detalle por categoría con términos encontrados
        - ``partial_evaluation``: True si se evaluó solo con metadata sin transcripción
    """
    if not text or not text.strip():
        # SIN texto evaluable: lo razonable es "no se pudo evaluar" pero no
        # bloquear el flujo entero. Reportamos riesgo desconocido y dejamos
        # que la app decida si pedir más datos.
        return {
            "policy_risk_level": "bajo",
            "policy_risk_categories": [],
            "policy_explanation": (
                "No hay texto evaluable (ni título, ni descripción, ni transcripción). "
                "El screening de políticas no detectó señales, pero la cobertura es nula."
            ),
            "youtube_ad_status_estimate": "apto",
            "hits": {},
            "partial_evaluation": True,
            "evaluation_coverage": "ninguna",
        }

    text_lower = text.lower()
    text_lower = re.sub(r"[^\w\sñáéíóúü+]", " ", text_lower, flags=re.UNICODE)
    text_lower = re.sub(r"\s+", " ", text_lower).strip()

    hits_by_cat: Dict[str, List[str]] = {}
    severities: List[str] = []
    for cat, terms in POLICY_LEXICONS.items():
        hit, hits = _detect_category(text_lower, terms)
        if hit:
            hits_by_cat[cat] = hits
            severities.append(CATEGORY_SEVERITY.get(cat, "medio"))

    # Nivel agregado. Más permisivo: solo 3+ categorías alta severidad → revisión humana.
    n_high = severities.count("alto")
    if not hits_by_cat:
        risk_level = "bajo"
    elif n_high >= 3:
        risk_level = "revisión humana"
    elif n_high >= 1:
        risk_level = "alto"
    else:
        risk_level = "medio"

    if risk_level == "bajo":
        ad_status = "apto"
    elif risk_level == "medio":
        ad_status = "apto limitado"
    elif risk_level == "alto":
        ad_status = "no apto"
    else:
        ad_status = "revisión humana"

    pretty = [CATEGORY_LABELS.get(c, c) for c in hits_by_cat.keys()]
    if not hits_by_cat:
        explanation = (
            "El screening no detectó términos en las 12 categorías sensibles. "
            "Si hay transcripción, la cobertura es buena; si no, considera complementar con texto del video."
            if has_transcript else
            "Sin transcripción, el screening se hizo solo sobre título y descripción. No se detectaron riesgos en metadata."
        )
    else:
        base = "Se detectaron señales en: " + ", ".join(pretty) + "."
        if risk_level == "revisión humana":
            base += " Como hay 3+ categorías de alta severidad, conviene revisión humana antes de pautar."
        elif risk_level == "alto":
            base += " El nivel de riesgo es alto; YouTube probablemente limitará la monetización."
        elif risk_level == "medio":
            base += " Riesgo moderado; el inventario publicitario puede ser restringido."
        explanation = base + " Este screening no reemplaza la revisión oficial de YouTube."

    return {
        "policy_risk_level": risk_level,
        "policy_risk_categories": pretty,
        "policy_explanation": explanation,
        "youtube_ad_status_estimate": ad_status,
        "hits": {CATEGORY_LABELS.get(c, c): h for c, h in hits_by_cat.items()},
        "partial_evaluation": not has_transcript,
        "evaluation_coverage": "completa" if has_transcript else "parcial_sin_transcripcion",
    }


def policy_block_for_human_review(reason: str = "") -> Dict[str, Any]:
    """Bloque a usar cuando NO hay transcripción y debe marcarse revisión humana."""
    return {
        "policy_risk_level": "revisión humana",
        "policy_risk_categories": [],
        "policy_explanation": reason or (
            "No se evaluaron políticas: falta transcripción. Revisión humana obligatoria."
        ),
        "youtube_ad_status_estimate": "revisión humana",
        "hits": {},
    }
