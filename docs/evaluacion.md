# Evaluación de los Jueces LLM para la Hackathon

Este documento detalla la metodología de evaluación utilizada para medir la calidad, precisión y robustez de los jueces LLM desarrollados por los equipos participantes en el desafío **Humans in the Loop: Desafío ALIA**.

---

# 1. Esquema de Evaluación por Escenarios

Se evalúa al juez fine-tuneado sobre las mismas conversaciones bajo tres condiciones distintas para comprobar su estabilidad técnica:

1.  **Escenario Original (`po_pred`)**: Texto limpio, tal como se encuentra en el dataset.
2.  **Escenario con Typos (`pt_pred`)**: Texto procesado con errores de teclado y tipográficos.
3.  **Escenario Gramatical (`pg_pred`)**: Texto procesado con ruido sintáctico y errores de gramática.

---

# 2. Criterios Cuantitativos (40%)

Se extraen directamente del rendimiento del modelo mediante las siguientes métricas:

### A. Correlación Global con Humanos (Accuracy)
Mide el porcentaje de aciertos del juez frente a las etiquetas de oro (*ground truth*) validadas por humanos. Se aplica sobre el escenario normal (`po_pred`).

* **Lógica**: `accuracy_score(y_true, y_pred)`
* **Fórmula**: 
    $$Acc = \frac{\text{Veredictos Coincidentes con Humano}}{\text{Total de Casos}} \times 100$$
* **Peso**: 20%

### B. Robustez entre Escenarios (Variabilidad)
Mide la capacidad del modelo para mantener su veredicto a pesar del ruido en el texto. Se penaliza cualquier caso donde el modelo no sea capaz de dar la misma respuesta en los tres escenarios simultáneamente.

* **Lógica**: Se calcula la media de los casos donde `po`, `pt` y `pg` **no** coinciden.
* **Fórmula de Variabilidad**: 
    $$\text{Var} = \text{mean}(\neg(V_{po} = V_{pt} = V_{pg}))$$
* **Cálculo de Score**: 
    $$\text{Score}_{\text{robustez}} = (1 - \text{Var}) \times 100$$
* **Peso**: 20%

---

# 3. Criterios Cualitativos (60%)

Evaluados por un comité de expertos en una escala de **1 a 5**. La puntuación se normaliza al 100% mediante la fórmula: $Score = \frac{S - 1}{4} \times 100$.

| Criterio | Descripción | Peso |
| :--- | :--- | :--- |
| **Rigor Metodológico** | Solidez experimental, justificación de recursos y eficiencia técnica. | 20% |
| **Uso de Datos** | Estrategia de selección, curación y composición del dataset de entrenamiento del juez. | 20% |
| **Contribución al Dataset** | Análisis de sesgos, limpieza, validación o mejoras aportadas al dataset base. | 20% |

---

# 4. Resumen de Ponderación

| Indicador | Tipo | Peso |
| :--- | :--- | :--- |
| **Accuracy (Normal)** | Cuantitativo | 20% |
| **Robustez (1 - Variabilidad)** | Cuantitativo | 20% |
| **Rigor y Eficiencia** | Cualitativo | 20% |
| **Estrategia de Datos** | Cualitativo | 20% |
| **Contribución al Dataset** | Cualitativo | 20% |
| **TOTAL** | | **100%** |

---

# 5. Ejemplo de Cálculo Final

Si un equipo obtiene los siguientes resultados en un test de 100 muestras:

1.  **Accuracy**: Coincide en 85 casos con el humano → **85%**.
2.  **Variabilidad**: En 10 casos, alguno de los 3 escenarios varió su veredicto.
    * $\text{Var} = 10/100 = 0.10$
    * $\text{Score}_{\text{robustez}} = (1 - 0.10) \times 100 = \mathbf{90\%}$.
3.  **Notas del Jurado**:
    * Rigor: 4/5 (**75%**)
    * Datos: 5/5 (**100%**)
    * Contribución: 4/5 (**75%**)

**Puntuación Final**:
$$(85 \cdot 0.2) + (90 \cdot 0.2) + (75 \cdot 0.2) + (100 \cdot 0.2) + (75 \cdot 0.2) = \mathbf{85.0}$$