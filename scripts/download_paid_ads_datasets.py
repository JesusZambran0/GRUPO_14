#!/usr/bin/env python
"""Prepara datasets públicos para XGBoost.

Los CSV ya vienen incluidos en la versión FULL generada bajo data/raw/paid_ads_xgboost.
Este script documenta fuentes y descarga la fuente GitHub pública cuando hay URL directa.
Para Kaggle se deja el comando porque requiere credenciales del usuario (~/.kaggle/kaggle.json).
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/raw/paid_ads_xgboost'
OUT.mkdir(parents=True, exist_ok=True)

FACEBOOK_KAG_URL = 'https://raw.githubusercontent.com/mGalarnyk/Python_Tutorials/master/Kaggle/Facebook/KAG_conversion_data.csv'
KAGGLE_SOURCES = [
    'minalchoudhary/marketing-campaign-dataset',
    'mirzayasirabdullah07/marketing-campaign-performance-dataset',
    'aashwinkumar/ppc-campaign-performance-data',
]

def main():
    fb = OUT/'facebook_kag_conversion.csv'
    if not fb.exists():
        print('Descargando Facebook KAG conversion...')
        urllib.request.urlretrieve(FACEBOOK_KAG_URL, fb)
    print('Kaggle requiere credenciales. Puedes descargar datasets adicionales con:')
    for ds in KAGGLE_SOURCES:
        print(f'  kaggle datasets download -d {ds} -p {OUT} --unzip')

if __name__ == '__main__':
    main()
