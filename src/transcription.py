"""Transcripción de video/audio usando Google Speech Recognition.

Estrategia:
1. Convertir el archivo fuente a WAV mono 16kHz con ffmpeg (el formato que
   Google Speech acepta mejor).
2. Si el audio supera 60s, dividirlo en chunks de 55s y transcribir cada uno.
3. Usar ``speech_recognition.Recognizer.recognize_google`` (API gratuita sin key).
4. Si SpeechRecognition no está instalado o Google falla, devolver texto vacío
   con un warning claro — nunca romper el pipeline.

Transcripción manual sigue teniendo prioridad sobre la automática.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RUNTIME_CACHE_DIR, WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL_SIZE
from .video_processing import hash_file, validate_video_path

TRANSCRIBABLE_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi",
    ".wav", ".mp3", ".m4a", ".ogg", ".flac",
}
CHUNK_SECONDS = 55          # Google Speech acepta hasta 60s por chunk
SAMPLE_RATE   = 16000       # 16 kHz — óptimo para reconocimiento de voz
MAX_TOTAL_SECONDS = 300     # no procesar más de 5 min de audio
LANGUAGE_FALLBACKS = ["es-ES", "en-US", "en-GB"]
MIN_TRANSCRIPT_WORDS = 8

def normalize_video_input(video: Any) -> Optional[str]:
    if video is None:
        return None
    if isinstance(video, str):
        return video
    if isinstance(video, dict):
        return video.get("path") or video.get("name") or video.get("orig_name")
    if isinstance(video, (list, tuple)) and video:
        return normalize_video_input(video[0])
    return str(video) if video else None


# ---------------------------------------------------------------------------
# Conversión a WAV con ffmpeg
# ---------------------------------------------------------------------------

def convert_to_wav(src: str | Path, dst: str | Path) -> Dict[str, Any]:
    """Convierte cualquier video/audio a WAV mono 16 kHz con ffmpeg.

    Returns dict con ``ok``, ``path`` y ``warning``.
    """
    src, dst = str(src), str(dst)
    cmd = [
    "ffmpeg", "-y",
    "-i", src,
    "-vn",
    "-acodec", "pcm_s16le",
    "-ar", str(SAMPLE_RATE),
    "-ac", "1",
    "-af", "loudnorm,highpass=f=80,lowpass=f=8000",
    "-f", "wav",
    dst,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and Path(dst).exists() and Path(dst).stat().st_size > 0:
            return {"ok": True, "path": dst, "warning": ""}
        stderr = proc.stderr[-300:] if proc.stderr else ""
        return {"ok": False, "path": None, "warning": f"ffmpeg falló: {stderr}"}
    except Exception as exc:
        return {"ok": False, "path": None, "warning": f"ffmpeg error: {exc}"}


def get_wav_duration(wav_path: str) -> float:
    """Duración del WAV en segundos con ffprobe."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
            capture_output=True, text=True, timeout=20,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return float(proc.stdout.strip())
    except Exception:
        pass
    return 0.0


def split_wav_into_chunks(wav_path: str, chunk_s: int, out_dir: str) -> List[str]:
    """Divide un WAV largo en chunks de ``chunk_s`` segundos. Devuelve lista de rutas."""
    chunks: List[str] = []
    duration = get_wav_duration(wav_path)
    if duration <= 0:
        return [wav_path]
    start = 0.0
    idx = 0
    while start < min(duration, MAX_TOTAL_SECONDS):
        chunk_path = str(Path(out_dir) / f"chunk_{idx:03d}.wav")
        cmd = [
            "ffmpeg", "-y", "-i", wav_path,
            "-ss", str(start), "-t", str(chunk_s),
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            chunk_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
        if proc.returncode == 0 and Path(chunk_path).exists():
            chunks.append(chunk_path)
        start += chunk_s
        idx += 1
    return chunks or [wav_path]



# ---------------------------------------------------------------------------
# Faster-Whisper local opcional
# ---------------------------------------------------------------------------

def _language_to_whisper(language: str) -> Optional[str]:
    if not language:
        return None
    lang = language.lower().split("-")[0].strip()
    return lang or None


def _transcribe_wav_faster_whisper(wav_path: str, language: str = "es-ES") -> Dict[str, Any]:
    """Transcribe con faster-whisper si está instalado.

    Es el motor preferido para MP4 porque no depende de servicios externos. Si no
    está disponible o falla, el pipeline cae a Google Speech Recognition.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:
        return {
            "ok": False,
            "text": "",
            "language_used": None,
            "warning": f"faster-whisper no disponible: {exc}",
        }

    try:
        model = WhisperModel(
            WHISPER_MODEL_SIZE or "tiny",
            device=WHISPER_DEVICE or "cpu",
            compute_type=WHISPER_COMPUTE_TYPE or "int8",
        )
        segments, info = model.transcribe(
            wav_path,
            language=_language_to_whisper(language),
            vad_filter=True,
            beam_size=1,
        )
        parts = []
        for seg in segments:
            txt = (getattr(seg, "text", "") or "").strip()
            if txt:
                parts.append(txt)
        text = " ".join(parts).strip()
        if text and len(text.split()) >= 3:
            return {
                "ok": True,
                "text": text,
                "language_used": getattr(info, "language", None) or _language_to_whisper(language),
                "warning": "",
            }
        return {
            "ok": False,
            "text": "",
            "language_used": getattr(info, "language", None) if info else None,
            "warning": "faster-whisper no detectó palabras suficientes.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "text": "",
            "language_used": None,
            "warning": f"faster-whisper falló: {exc}",
        }

# ---------------------------------------------------------------------------
# Google Speech Recognition
# ---------------------------------------------------------------------------

def _transcribe_wav_google(wav_path: str, language: str = "es-ES") -> Dict[str, Any]:
    """Transcribe un WAV con Google Speech Recognition.

    Prueba primero el idioma solicitado y luego fallbacks español/inglés.
    Esto ayuda con comerciales en inglés como Old Spice.
    """
    try:
        import speech_recognition as sr  # type: ignore
    except ImportError:
        return {
            "ok": False,
            "text": "",
            "language_used": None,
            "warning": (
                "SpeechRecognition no está instalado. "
                "Instala con: pip install SpeechRecognition."
            ),
        }

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio_data = recognizer.record(source)
    except Exception as exc:
        return {
            "ok": False,
            "text": "",
            "language_used": None,
            "warning": f"No se pudo leer el audio WAV: {exc}",
        }

    languages = []
    if language:
        languages.append(language)
    for lang in LANGUAGE_FALLBACKS:
        if lang not in languages:
            languages.append(lang)

    warnings = []

    for lang in languages:
        try:
            text = recognizer.recognize_google(audio_data, language=lang)
            text = (text or "").strip()
            if text and len(text.split()) >= 3:
                return {
                    "ok": True,
                    "text": text,
                    "language_used": lang,
                    "warning": "",
                }
        except sr.UnknownValueError:
            warnings.append(f"{lang}: no entendió el audio")
        except sr.RequestError as exc:
            warnings.append(f"{lang}: error de red {exc}")
        except Exception as exc:
            warnings.append(f"{lang}: error {exc}")

    return {
        "ok": False,
        "text": "",
        "language_used": None,
        "warning": "Google Speech no pudo entender el audio. " + " | ".join(warnings[:3]),
    }

def transcribe_video(
    video_path: str,
    language: str = "es-ES",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Convierte video/audio a WAV y transcribe con Google Speech Recognition.

    1. Valida el archivo.
    2. Convierte a WAV mono 16 kHz (ffmpeg).
    3. Si dura > CHUNK_SECONDS, divide en chunks y transcribe cada uno.
    4. Concatena los textos.
    5. Cachea por hash de archivo en ``.cache/transcription/``.

    Args:
        video_path: ruta al MP4, WAV, MP3, etc.
        language: código de idioma para Google Speech (``es-ES``, ``en-US``, etc.).
        use_cache: si True, usa caché por hash de archivo.

    Returns:
        dict con ``ok``, ``transcript_text``, ``engine``, ``language``, ``warning``,
        ``chunks_processed``.
    """
    # Validación
    validation = validate_video_path(video_path)
    if not validation.get("ok"):
        return {
            "ok": False, "transcript_text": "", "engine": "google-speech",
            "language": language, "warning": validation.get("warning", "Archivo no válido."),
            "chunks_processed": 0,
        }
    p = Path(str(video_path))
    if p.suffix.lower() not in TRANSCRIBABLE_EXTENSIONS:
        return {
            "ok": False, "transcript_text": "", "engine": "google-speech",
            "language": language,
            "warning": f"Extensión '{p.suffix}' no soportada.",
            "chunks_processed": 0,
        }

    # Caché
    cache_key = f"{hash_file(video_path) or 'nohash'}_{language}"
    cache_file: Optional[Path] = None
    if use_cache:
        cache_dir = RUNTIME_CACHE_DIR / "transcription"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                cached["warning"] = (cached.get("warning") or "") + " [caché]"
                return cached
            except Exception:
                pass

    with tempfile.TemporaryDirectory(prefix="ytboost_trans_") as tmp:
        wav_path = str(Path(tmp) / "audio.wav")
        conv = convert_to_wav(video_path, wav_path)
        if not conv["ok"]:
            return {
                "ok": False, "transcript_text": "", "engine": "google-speech",
                "language": language,
                "warning": f"No se pudo convertir a WAV: {conv['warning']}",
                "chunks_processed": 0,
            }

        duration = get_wav_duration(wav_path)

        # 1) Motor local preferido: faster-whisper. Funciona con MP4 convertido a WAV
        # y evita depender de una API externa para transcribir.
        whisper_result = _transcribe_wav_faster_whisper(wav_path, language=language)
        if whisper_result.get("ok") and whisper_result.get("text"):
            result = {
                "ok": True,
                "transcript_text": whisper_result.get("text", ""),
                "engine": f"faster-whisper/{WHISPER_MODEL_SIZE}",
                "language": language,
                "language_used": whisper_result.get("language_used"),
                "warning": "",
                "chunks_processed": 1,
            }
            if cache_file:
                try:
                    cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
            return result

        # 2) Respaldo liviano: SpeechRecognition + Google Speech por chunks.
        if duration > CHUNK_SECONDS:
            chunks = split_wav_into_chunks(wav_path, CHUNK_SECONDS, tmp)
        else:
            chunks = [wav_path]

        texts: List[str] = []
        warnings: List[str] = []
        languages_used: List[str] = []
        if whisper_result.get("warning"):
            warnings.append(whisper_result.get("warning"))

        for chunk in chunks:
            r = _transcribe_wav_google(chunk, language)
            if r["ok"] and r["text"]:
                texts.append(r["text"])
                if r.get("language_used"):
                    languages_used.append(r["language_used"])
            elif r["warning"]:
                warnings.append(r["warning"])

    full_text = " ".join(t.strip() for t in texts if t.strip()).strip()
    result = {
        "ok": bool(full_text),
        "transcript_text": full_text,
        "engine": "google-speech",
        "language": language,
        "language_used": languages_used[0] if languages_used else None,
        "warning": " | ".join(set(warnings)) if warnings and not full_text else "",
        "chunks_processed": len(chunks),
    }
    if cache_file and result["ok"]:
        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Resolver transcripción: manual > automática > vacía
# ---------------------------------------------------------------------------

def resolve_transcript(
    manual_transcript: str,
    video_path: Optional[str],
    language: str = "es-ES",
) -> Dict[str, Any]:
    """Resuelve la transcripción.

    Regla:
    - Si hay video, se intenta transcripción automática.
    - La transcripción manual solo se usa si NO hay video.
    - Nunca se usa texto demo como fallback.
    """

    # 1. Si hay video, SIEMPRE intentar automático
    if video_path:
        result = transcribe_video(video_path, language=language)
        transcript = (result.get("transcript_text") or "").strip()
        word_count = len(transcript.split())

        if result.get("ok") and word_count >= MIN_TRANSCRIPT_WORDS:
            return {
                "transcript_text": transcript,
                "source": result.get("engine", "google-speech"),
                "language": result.get("language_used") or result.get("language") or language,
                "warning": result.get("warning", ""),
                "ok": True,
                "requires_human_review": False,
                "word_count": word_count,
            }

        return {
            "transcript_text": "",
            "source": "failed",
            "language": language,
            "warning": (
                result.get("warning")
                or "No se pudo generar una transcripción automática suficiente."
            ),
            "ok": False,
            "requires_human_review": True,
            "word_count": word_count,
        }

    # 2. Si NO hay video, usar manual si existe
    manual = (manual_transcript or "").strip()
    if manual:
        return {
            "transcript_text": manual,
            "source": "manual",
            "language": language,
            "warning": "Transcripción ingresada manualmente porque no se subió video.",
            "ok": True,
            "requires_human_review": False,
            "word_count": len(manual.split()),
        }

    # 3. Sin video y sin manual
    return {
        "transcript_text": "",
        "source": "missing",
        "language": language,
        "warning": (
            "No se cargó video ni transcripción. "
            "La evaluación completa de políticas y guion queda bloqueada."
        ),
        "ok": False,
        "requires_human_review": True,
        "word_count": 0,
    }