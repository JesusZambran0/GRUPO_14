# Reporte ejecutivo PDF

La aplicación incluye un botón visible en la cabecera para generar un PDF ejecutivo de una página después de ejecutar el análisis del video.

## Flujo

1. El usuario carga una URL, un MP4 o un caso demo.
2. Ejecuta **ANALIZAR VIDEO**.
3. La app habilita el botón **Generar PDF ejecutivo**.
4. El PDF se genera desde el JSON de diagnóstico ya calculado por la aplicación.

El botón inicia deshabilitado para evitar reportes vacíos. Solo se activa cuando `analyze_video` devuelve datos válidos.

## Diseño

El reporte usa fondo blanco, tipografía sans serif y estilo de dashboard ejecutivo. La primera línea muestra en grande la decisión principal:

- `RECOMENDACIÓN: PAUTAR`
- `RECOMENDACIÓN: AJUSTAR ANTES DE PAUTAR`
- `RECOMENDACIÓN: NO PAUTAR`
- `RECOMENDACIÓN: MONITOREAR, NO PAUTAR AÚN`
- `RECOMENDACIÓN: REVISIÓN HUMANA`

## Módulos consolidados

El PDF resume los módulos principales del software:

- métricas públicas del video;
- proyección y estimación de pauta;
- análisis de sentimiento;
- OCR;
- transcripción;
- análisis visual y capturas de frames;
- análisis de guion;
- evaluación de políticas;
- recomendación redactada por LLM/reglas.

## Archivos

- `src/report_pdf.py`: generador del PDF.
- `app.py`: botón de cabecera, habilitación posterior al análisis y descarga del PDF.
- `outputs/reports/.gitkeep`: carpeta de salida de PDFs generados en runtime.
- `requirements.txt`: agrega `reportlab`.

Los PDFs generados son salidas de ejecución. No es necesario commitearlos.
