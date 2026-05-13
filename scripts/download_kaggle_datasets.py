"""Descarga datasets reales de YouTube desde Kaggle.

Requisitos:
1. Cuenta en https://www.kaggle.com/.
2. API token: en tu perfil → Settings → API → "Create New Token". Esto descarga
   un archivo ``kaggle.json``.
3. Guardarlo en ``~/.kaggle/kaggle.json`` con permisos ``600``:
       mkdir -p ~/.kaggle
       mv kaggle.json ~/.kaggle/kaggle.json
       chmod 600 ~/.kaggle/kaggle.json
4. Instalar el cliente:
       pip install kaggle
5. Ejecutar este script:
       python scripts/download_kaggle_datasets.py

Los CSVs se descomprimen en ``data/raw/`` y son usados automáticamente por
``python -m src.train``. Después del entrenamiento, ``models/model_metadata.json``
indicará ``"dataset_kind": "real_or_mixed"``.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Datasets candidatos. Si alguno cambia de nombre o desaparece, agrega o ajusta.
# Verifica el slug en https://www.kaggle.com/datasets antes de ejecutar.
DATASETS = [
    # Trending diario, multilingüe (regiones).
    "rsrishav/youtube-trending-video-dataset",
    # Trending US clásico (1 millón+ de filas históricas).
    "datasnaek/youtube-new",
    # Estadísticas globales de canales.
    "nelgiriyewithana/global-youtube-statistics-2023",
]


def _kaggle_available() -> bool:
    return shutil.which("kaggle") is not None


def _kaggle_credentials_present() -> bool:
    home_token = Path.home() / ".kaggle" / "kaggle.json"
    return home_token.exists()


def _download_one(slug: str) -> bool:
    print(f"[kaggle] Descargando {slug} → {RAW}")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(RAW), "--unzip"],
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"[kaggle] ⚠ {slug} falló (rc={result.returncode}). Detalle:")
            print(result.stderr.strip() or result.stdout.strip())
            return False
        print(f"[kaggle] ✅ {slug} OK")
        return True
    except subprocess.TimeoutExpired:
        print(f"[kaggle] ⚠ {slug} timeout (10 min).")
        return False
    except Exception as exc:
        print(f"[kaggle] ⚠ Error con {slug}: {exc}")
        return False


def main() -> int:
    if not _kaggle_available():
        print("[kaggle] El cliente 'kaggle' no está instalado. Ejecuta: pip install kaggle")
        return 1
    if not _kaggle_credentials_present():
        print("[kaggle] Falta ~/.kaggle/kaggle.json. Sigue las instrucciones del docstring.")
        return 1

    if not DATASETS:
        print("[kaggle] No hay datasets configurados. Edita la lista DATASETS.")
        return 1

    print(f"[kaggle] Datasets a descargar: {len(DATASETS)}")
    successes = 0
    for slug in DATASETS:
        if _download_one(slug):
            successes += 1

    csvs = sorted(RAW.glob("*.csv"))
    print(f"\n[kaggle] CSVs disponibles en data/raw/: {len(csvs)}")
    for c in csvs:
        size_kb = c.stat().st_size // 1024
        print(f"  - {c.name} ({size_kb} KB)")

    if successes == 0:
        print("\n[kaggle] Ningún dataset se pudo descargar. Verifica credenciales y conexión.")
        return 1

    print("\n[kaggle] Listo. Siguiente paso:")
    print("    python -m src.train")
    print("Esto re-entrenará el modelo con los datos reales y lo registrará en models/model_metadata.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
