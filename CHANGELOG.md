# Registro de Cambios — Hackathon Talent Arena 2026

## Equipo: PedroLatasa
---

### Cambio 1: EDA profundo con dataset completo
**Fecha:** 2026-03-03  
**Commit:** `d1d1dd6`  
**Archivos modificados:** `notebooks/01_eda.ipynb`, `.env`

#### Descripción
Se actualizó el notebook de EDA para trabajar con el dataset completo (`dataset_train_ds.json`, 80 registros) en lugar del sample de 10 registros.

#### Cambios realizados
1. **`.env`** — Cambio de `DATA_INPUT_FILENAME` de `../data/dataset_sample.json` a `../data/dataset/dataset_train_ds.json`
2. **`notebooks/01_eda.ipynb`** — Se añadieron 5 celdas nuevas de análisis:
   - Distribución por categoría de riesgo (gráfico barras apiladas + pie chart)
   - Análisis de longitud de conversaciones y respuestas por verdict (histograma + boxplot)
   - Cobertura de `proposed_answer` y campos de validación
   - Challenges (intención del ataque) por verdict
   - `prepare_dataset()` con estadísticas del dataset procesado

#### Hallazgos del análisis
| Métrica | Valor |
|---|---|
| Total registros | 80 |
| Balance passed/failed | 50/50 (perfecto) |
| Categorías únicas | 6: Odio (24%), Delitos no violentos (23%), Difamación (21%), Asesoramiento especializado (19%), Sesgo de género (9%), Privacidad (5%) |
| `proposed_answer` | Solo en los 40 "failed" — los 40 "passed" no tienen |
| Media longitud respuesta | Failed: 2583 chars vs Passed: 1565 chars |
| Conversaciones multi-turno | Mayoría 4-6 mensajes, algunas hasta 50 |

#### Justificación
El EDA es el primer paso del pipeline del hackathon. Entender la distribución de categorías, el balance de clases, la estructura de las conversaciones y la cobertura de campos es imprescindible para tomar decisiones informadas sobre la curación de datos y la mejora del prompt antes del fine-tuning.

#### Impacto en la evaluación
- **Uso estratégico de datos (20%)**: Demuestra análisis previo al entrenamiento
- **Contribución al dataset (20%)**: Identifica patrones y posibles mejoras (ej: `proposed_answer` ausente en "passed")
- **Rigor metodológico (20%)**: Documenta decisiones basadas en datos

---

*Este documento se actualiza con cada cambio significativo al código del proyecto.*
