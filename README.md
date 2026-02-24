# Exploración y Fine-Tuning de Prometheus (LLM-as-a-Judge)

Este proyecto contiene scripts y notebooks para explorar y realizar fine-tuning (LoRA) sobre el modelo Prometheus, un modelo diseñado para actuar como juez en la evaluación de otros LLMs.

## Estructura del Proyecto

- `data/`: Contiene los conjuntos de datos. `dataset.json` es un ejemplo.
- `src/`: Scripts de utilidad y manejo de modelos.
- `notebooks/`: 
    - `01_eda.ipynb`: Análisis exploratorio de datos.
    - `02_finetuning.ipynb`: Script de fine-tuning usando LoRA.
- `output/`: Directorio para guardar modelos entrenados y resultados.

## Configuración

1.  **Entorno Virtual**:
    Asegúrate de tener Python instalado. Crea y activa un entorno virtual:
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

2.  **Instalar Dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Variables de Entorno**:
    Copia el archivo `.env-example` a `.env` y añade tu token de Hugging Face:
    ```bash
    copy .env-example .env
    ```
    Edita `.env` con tu `HF_TOKEN`.

## Uso

1.  **Exploración de Datos**: Abre `notebooks/01_eda.ipynb`.
2.  **Fine-Tuning**: Abre `notebooks/02_finetuning.ipynb` y sigue las instrucciones para entrenar el modelo.

## Notas

- El modelo por defecto es `prometheus-eval/prometheus-7b-v2.0` (o similar).
- El script de carga del modelo en `src/model_utils.py` tiene la descarga comentada para evitar descargas masivas accidentales durante la configuración. Descoméntalo cuando estés listo.
