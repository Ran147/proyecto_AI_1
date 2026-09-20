# Cómo agregar un modelo a JarvisTEC

Cada modelo necesita 3 cosas: un notebook, un artefacto entrenado y un archivo en el backend.
No hay que tocar `main.py`, los routers ni el router de comandos: el modelo se registra solo.

## 1. Notebook (`notebooks/NN_nombre.ipynb`)

Sigue las secciones de la rúbrica, en este orden:
1. Análisis del problema (0.5 pt)
2. Entendimiento de los datos (0.5 pt)
3. Exploración de los datos (0.5 pt)
4. Modelo (2 pts)
5. Evaluación (1 pt)
6. Conclusión (0.5 pt)

Al final, guardar el modelo entrenado:

```python
import joblib
joblib.dump(pipeline, "../artifacts/vino.joblib")   # incluir el preprocesamiento en el pipeline
```

Consejo: guarden un `sklearn.Pipeline` (escalado + codificación + modelo) para que el backend
solo reciba los datos crudos y llame a `predict`.

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
