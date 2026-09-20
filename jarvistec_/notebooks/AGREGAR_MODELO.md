# Cómo agregar un modelo a JarvisTEC

Cada modelo necesita 3 cosas: un notebook, un artefacto entrenado y un archivo en el backend.
No hay que tocar `main.py`, los routers ni el router de comandos: el modelo se registra solo.

## 1. Notebook (`notebooks/NN_nombre.ipynb`)

Copiar `notebooks/plantilla_modelo.ipynb` y renombrarla. Ya trae las 6 secciones de la rúbrica en orden:
1. Análisis del problema (0.5 pt)
2. Entendimiento de los datos (0.5 pt)
3. Exploración de los datos (0.5 pt)
4. Modelo (2 pts)
5. Evaluación (1 pt)
6. Conclusión (0.5 pt)

Pasos:
1. Poner el CSV en `data/` (no se sube a git).
2. Editar **solo la celda de Configuración** (nombre del modelo, ruta, columna objetivo, tipo de tarea, si es serie temporal).
3. Ejecutar de arriba abajo y completar los textos marcados con **COMPLETAR**. La rúbrica califica el análisis y las conclusiones, no solo el código.
4. La última celda de código guarda `artifacts/<nombre>.joblib` (un `Pipeline` completo con preprocesamiento) y `artifacts/<nombre>.meta.json` con las métricas y un `default_input` listo para copiar al backend.

Notas:
- Para series temporales (bitcoin, aguacate, etc.) poner `ES_SERIE_TEMPORAL = True` y `COLUMNA_FECHA`: el split será cronológico. Para predecir "mañana" suele hacer falta crear variables de rezago (valores de días anteriores) antes del modelo; añadan esa celda en la sección 4.
- La plantilla es un punto de partida: agreguen otros modelos, ajuste de hiperparámetros o gráficos si el problema lo pide.
- El recomendador de películas no encaja en esta plantilla (no es supervisado clásico); habrá que adaptar las secciones 4 y 5.

## 2. Archivo del backend (`backend/models/vino.py`)

Copiar `toy_wine.py` y cambiar lo necesario. Ejemplo con modelo real:

```python
import joblib
import pandas as pd
from typing import Any
from .base import BaseModel, ARTIFACTS_DIR


class Vino(BaseModel):
    name = "vino"                      # único, sin espacios
    title = "Calidad del vino"
    description = "Clasifica la calidad de un vino según sus propiedades químicas."
    task_type = "clasificacion"
    keywords = ["calidad del vino", "clasificar vino"]   # frases de comando de voz
    input_fields = {"alcohol": "float, % de alcohol", "volatile_acidity": "float"}
    default_input = {"alcohol": 11.5, "volatile_acidity": 0.4}   # usado por comandos de voz

    def load(self):
        self.pipeline = joblib.load(ARTIFACTS_DIR / "vino.joblib")

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        df = pd.DataFrame([inputs])
        return {"quality": str(self.pipeline.predict(df)[0])}

    def speak(self, result):
        return f"La calidad estimada del vino es {result['quality']}."
```

Reglas:
- `name` debe ser único (la API falla al iniciar si hay duplicados).
- `predict` recibe y devuelve diccionarios serializables a JSON (convertir tipos numpy con `float()`/`int()`/`str()`).
- `default_input` permite que un comando de voz funcione sin dar datos (ej. "JarvisTEC, tipo de cambio para mañana").
- `keywords`: 2-3 frases naturales que diría un usuario. Incluir el nombre del tema.
- Nunca subir datasets pesados ni `.joblib` grandes al repo (están en `.gitignore`); documenten cómo generarlos.

## 3. Probar

```bash
python -m pytest -q
uvicorn backend.main:app --reload
```

En http://localhost:8000/docs probar `POST /models/vino/predict` y `POST /command`
con `{"text": "JarvisTEC calidad del vino"}`.

## 4. Entregables por modelo (checklist)

- [ ] Notebook con las 6 secciones
- [ ] Artefacto entrenado en `artifacts/` (o script para regenerarlo)
- [ ] Archivo en `backend/models/`
- [ ] 2-3 frases de comando
- [ ] 1-2 artículos científicos para la bibliografía
- [ ] Sección redactada en Overleaf
