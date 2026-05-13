"""Precarga el modelo faster-whisper la primera vez.

Ejecuta este script una sola vez con internet para descargar los pesos.
Después la app puede correr sin internet.

Uso:
    python scripts/download_whisper_model.py [tiny|base|small]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    size = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    print(f"[whisper] Descargando modelo '{size}'...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(size, device="cpu", compute_type="int8")
        # Forzar la carga real
        del model
        print(f"[whisper] OK: modelo '{size}' descargado/disponible.")
        return 0
    except ImportError as exc:
        print(f"[whisper] faster-whisper no está instalado: {exc}")
        print("           Ejecuta: pip install faster-whisper")
        return 1
    except Exception as exc:
        print(f"[whisper] Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
