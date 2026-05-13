"""Construcción ligera de patrones de guion a partir de transcripciones públicas.

Este módulo no descarga datos en runtime. Lee transcripciones ya cargadas en:
    data/raw/youtube_transcripts_900/Transcripts/*.json

También deja preparado el punto de extensión para Hugging Face:
    from datasets import load_dataset
    dataset = load_dataset("ZelonPrograms/Youtube")

En la demo pública no se descarga ZelonPrograms/Youtube para evitar fallos por red,
tiempo o cambios externos. La referencia queda documentada y el pipeline funciona
con el dataset Kaggle de transcripciones incluido en el repositorio.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_TRANSCRIPTS_DIR = ROOT / "data" / "raw" / "youtube_transcripts_900" / "Transcripts"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_CSV = PROCESSED_DIR / "youtube_transcripts_script_features.csv"
PATTERNS_JSON = PROCESSED_DIR / "script_patterns_reference.json"

CTA_TERMS = {
    "subscribe", "like", "comment", "share", "click", "download", "visit", "buy", "try", "watch",
    "learn more", "sign up", "follow", "save", "join", "comenta", "suscríbete", "comparte", "guarda",
    "haz clic", "visita", "compra", "descarga", "regístrate", "sígueme", "conoce más",
}
BENEFIT_TERMS = {
    "learn", "improve", "discover", "avoid", "reduce", "increase", "optimize", "how to", "why", "tips",
    "strategy", "mistake", "secret", "guide", "aprende", "mejora", "descubre", "evita", "reduce",
    "aumenta", "optimiza", "cómo", "por qué", "consejos", "estrategia", "errores", "guía",
}
URGENCY_TERMS = {
    "now", "today", "limited", "urgent", "last chance", "hoy", "ahora", "urgente", "última oportunidad", "limitado"
}
TRUST_TERMS = {
    "official", "verified", "study", "evidence", "expert", "safe", "guarantee", "case study", "oficial",
    "verificado", "estudio", "evidencia", "experto", "seguro", "caso real",
}
EXAGGERATED_TERMS = {
    "guaranteed", "100%", "without effort", "miracle", "instant", "get rich", "lose weight fast",
    "garantizado", "100 %", "sin esfuerzo", "milagroso", "resultado inmediato", "gana dinero rápido",
    "pierde peso rápido", "cura definitiva",
}
GENERIC_OPENINGS = {
    "hello", "hi guys", "hey guys", "welcome", "today we are going", "in this video",
    "hola", "hola amigos", "bienvenidos", "en este video", "hoy vamos",
}


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text


def _count_terms(text: str, terms: Iterable[str]) -> int:
    low = text.lower()
    return sum(1 for t in terms if t.lower() in low)


def _first_words(text: str, n: int = 30) -> str:
    return " ".join(text.split()[:n])


def _read_transcript_json(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if isinstance(data, list):
        return _normalize_text(" ".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in data))
    if isinstance(data, dict):
        if "transcript" in data:
            return _normalize_text(data.get("transcript", ""))
        if "text" in data:
            return _normalize_text(data.get("text", ""))
        if "segments" in data and isinstance(data["segments"], list):
            return _normalize_text(" ".join(str(x.get("text", "")) for x in data["segments"] if isinstance(x, dict)))
    return ""


def _script_quality_features(transcript: str) -> Dict[str, Any]:
    transcript = _normalize_text(transcript)
    words = transcript.split()
    hook = _first_words(transcript, 30)
    low_hook = hook.lower()
    low_text = transcript.lower()

    word_count = len(words)
    sentence_parts = [s.strip() for s in re.split(r"[.!?]+", transcript) if s.strip()]
    avg_sentence_len = (sum(len(s.split()) for s in sentence_parts) / max(len(sentence_parts), 1)) if sentence_parts else 0

    hook_has_question = "?" in hook or any(low_hook.startswith(x) for x in ["how", "why", "what", "when", "where", "cómo", "por qué", "qué"])
    hook_has_benefit = _count_terms(hook, BENEFIT_TERMS) > 0
    generic_opening = any(p in low_hook[:160] for p in GENERIC_OPENINGS)
    cta_count = _count_terms(transcript, CTA_TERMS)
    benefit_count = _count_terms(transcript, BENEFIT_TERMS)
    urgency_count = _count_terms(transcript, URGENCY_TERMS)
    trust_count = _count_terms(transcript, TRUST_TERMS)
    exaggerated_count = _count_terms(transcript, EXAGGERATED_TERMS)

    hook_score = 50 + 20 * int(hook_has_question) + 25 * int(hook_has_benefit) - 20 * int(generic_opening)
    hook_score = max(0, min(100, hook_score))
    cta_score = max(0, min(100, 35 + min(cta_count, 3) * 20))
    value_score = max(0, min(100, 45 + min(benefit_count, 4) * 12 + min(trust_count, 2) * 8))
    clarity_score = 78
    if word_count < 25:
        clarity_score -= 15
    if word_count > 2500:
        clarity_score -= 12
    if avg_sentence_len > 28:
        clarity_score -= 15
    if transcript.count("!") > 6:
        clarity_score -= 6
    clarity_score = max(0, min(100, clarity_score))
    policy_claim_risk = "alto" if exaggerated_count >= 2 else ("medio" if exaggerated_count == 1 else "bajo")
    script_quality_score = int(round(0.30 * hook_score + 0.25 * clarity_score + 0.25 * value_score + 0.20 * cta_score))

    return {
        "transcript_word_count": word_count,
        "hook_first_30_words": hook,
        "hook_has_question": int(hook_has_question),
        "hook_has_benefit": int(hook_has_benefit),
        "generic_opening": int(generic_opening),
        "cta_count": cta_count,
        "benefit_count": benefit_count,
        "urgency_count": urgency_count,
        "trust_count": trust_count,
        "exaggerated_claim_count": exaggerated_count,
        "avg_sentence_length": round(avg_sentence_len, 2),
        "hook_score": hook_score,
        "cta_score": cta_score,
        "value_proposition_score": value_score,
        "clarity_score": clarity_score,
        "policy_claim_risk": policy_claim_risk,
        "script_quality_score": script_quality_score,
    }


def build_from_local_transcripts(raw_dir: Path = RAW_TRANSCRIPTS_DIR) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        transcript = _read_transcript_json(path)
        if not transcript:
            continue
        feats = _script_quality_features(transcript)
        rows.append({
            "video_id": path.stem,
            "source_dataset": "kaggle_youtube_trending_videos_transcripts_900",
            "title": "",
            "views": None,
            "transcript": transcript,
            **feats,
        })
    return pd.DataFrame(rows)


def build_patterns_reference(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return default_patterns_reference()
    top = df.sort_values("script_quality_score", ascending=False).head(min(150, len(df)))
    hook_terms = Counter()
    cta_terms = Counter()
    benefit_terms = Counter()
    for _, row in top.iterrows():
        hook = str(row.get("hook_first_30_words", "")).lower()
        transcript = str(row.get("transcript", "")).lower()
        for term in BENEFIT_TERMS:
            if term in hook:
                hook_terms[term] += 1
            if term in transcript:
                benefit_terms[term] += 1
        for term in CTA_TERMS:
            if term in transcript:
                cta_terms[term] += 1
    return {
        "source": "youtube_trending_videos_transcripts_900",
        "n_transcripts": int(len(df)),
        "quality_score_mean": float(round(df["script_quality_score"].mean(), 2)),
        "quality_score_p75": float(round(df["script_quality_score"].quantile(0.75), 2)),
        "top_hook_patterns": [x for x, _ in hook_terms.most_common(20)] or sorted(list(BENEFIT_TERMS))[:20],
        "top_cta_patterns": [x for x, _ in cta_terms.most_common(20)] or sorted(list(CTA_TERMS))[:20],
        "top_benefit_patterns": [x for x, _ in benefit_terms.most_common(20)] or sorted(list(BENEFIT_TERMS))[:20],
        "risk_terms": sorted(list(EXAGGERATED_TERMS)),
        "generic_openings": sorted(list(GENERIC_OPENINGS)),
        "notes": "Patrones extraídos de transcripciones públicas de videos populares. No garantizan viralidad; se usan como referencia semántica para mejorar guion.",
    }


def default_patterns_reference() -> Dict[str, Any]:
    return {
        "source": "default_rules",
        "n_transcripts": 0,
        "quality_score_mean": 0,
        "quality_score_p75": 70,
        "top_hook_patterns": sorted(list(BENEFIT_TERMS))[:20],
        "top_cta_patterns": sorted(list(CTA_TERMS))[:20],
        "top_benefit_patterns": sorted(list(BENEFIT_TERMS))[:20],
        "risk_terms": sorted(list(EXAGGERATED_TERMS)),
        "generic_openings": sorted(list(GENERIC_OPENINGS)),
        "notes": "Patrones por defecto; ejecutar script_dataset_builder.py para usar transcripciones reales.",
    }


def build_and_save() -> Dict[str, Any]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_from_local_transcripts()
    df.to_csv(FEATURES_CSV, index=False)
    patterns = build_patterns_reference(df)
    PATTERNS_JSON.write_text(json.dumps(patterns, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"rows": int(len(df)), "features_csv": str(FEATURES_CSV), "patterns_json": str(PATTERNS_JSON)}


if __name__ == "__main__":
    print(json.dumps(build_and_save(), ensure_ascii=False, indent=2))
