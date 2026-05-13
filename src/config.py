"""Configuración central del proyecto YouTube Boost AI."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
    _HAS_DOTENV = True
except Exception:  # dotenv es opcional
    _HAS_DOTENV = False

    def load_dotenv(*args, **kwargs):  # noqa: D401
        return False

ROOT_DIR = Path(__file__).resolve().parents[1]
if _HAS_DOTENV:
    load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEMO_DATA_DIR = DATA_DIR / "demo"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DEMO_CACHE_DIR = ROOT_DIR / "demo_cache"
RUNTIME_CACHE_DIR = ROOT_DIR / ".cache"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

# LLM local pequeño (opcional). Si no se carga, se usa rule_based_recommendation.
# Modelos recomendados:
#   - "Qwen/Qwen2.5-1.5B-Instruct"  (default, ~3 GB)
#   - "HuggingFaceTB/SmolLM2-1.7B-Instruct"
LLM_LOCAL_MODEL_ID = os.getenv("LLM_LOCAL_MODEL_ID", "HuggingFaceTB/SmolLM2-135M-Instruct").strip()
LLM_LOCAL_MAX_NEW_TOKENS = int(os.getenv("LLM_LOCAL_MAX_NEW_TOKENS", "512"))
LLM_LOCAL_TIMEOUT_S = int(os.getenv("LLM_LOCAL_TIMEOUT_S", "60"))

# Transcripción con faster-whisper (local, opcional).
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny").strip()  # tiny | base | small
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip()
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu").strip()

DEFAULT_CPM = float(os.getenv("DEFAULT_CPM", "5.0"))

# Limites de procesamiento del video en modo demo (Hugging Face Spaces gratuito).
MAX_VIDEO_DURATION_SECONDS = float(os.getenv("MAX_VIDEO_DURATION_SECONDS", "60"))
MAX_OCR_FRAMES = int(os.getenv("MAX_OCR_FRAMES", "20"))

POTENTIAL_MULTIPLIERS = {
    "bajo": 0.70,
    "medio": 1.00,
    "alto": 1.30,
    "muy_alto": 1.60,
}

# Vocabulario base para detección de señales textuales.
CTA_TERMS = {
    "compra", "comprar", "agenda", "agendar", "suscríbete", "suscribete", "clic", "click",
    "visita", "regístrate", "registrate", "descarga", "aprovecha", "llama", "contáctanos",
    "contactanos", "reserva", "prueba", "conoce", "entra", "mira", "descubre",
}
URGENCY_TERMS = {
    "hoy", "ahora", "últimos", "ultimos", "limitado", "solo", "oferta", "descuento",
    "promoción", "promocion", "gratis", "exclusivo", "imperdible", "ya",
}
TRUST_TERMS = {
    "oficial", "garantía", "garantia", "certificado", "seguro", "recomendado", "clientes",
    "testimonio", "verificado", "calidad", "expertos", "años", "experiencia",
}
BENEFIT_TERMS = {
    "beneficio", "mejora", "ahorra", "rápido", "rapido", "fácil", "facil", "resultado",
    "solución", "solucion", "crece", "gana", "aprende", "optimiza", "convierte",
}
PROMO_TERMS = {"oferta", "descuento", "gratis", "2x1", "promoción", "promocion", "cupón", "cupon"}
PRICE_TERMS = {"$", "usd", "precio", "desde", "mensual", "pago", "dólar", "dolar"}

# Conjunto canónico de valores que la app puede devolver en `recomendacion_impulso`.
RECOMMENDATION_CHOICES = {
    "impulsar",
    "no impulsar",
    "ajustar antes de impulsar",
    "monitorear",
}

DEFAULT_OUTPUT_FIELDS = [
    "prediccion_rendimiento",
    "probabilidad_rendimiento",
    "recomendacion_impulso",
    "requiere_ajustes",
    "nivel_prioridad",
    "alcance_estimado_por_dolar",
    "alcance_estimado_total",
    "metricas",
    "analisis_transcripcion",
    "analisis_ocr",
    "analisis_llm",
    "ajustes_sugeridos",
    "justificacion",
]


def ensure_dirs() -> None:
    """Asegura que existan los directorios usados en runtime."""
    for path in [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, DEMO_DATA_DIR,
        MODELS_DIR, OUTPUTS_DIR, DEMO_CACHE_DIR, RUNTIME_CACHE_DIR,
    ]:
        path.mkdir(exist_ok=True, parents=True)
