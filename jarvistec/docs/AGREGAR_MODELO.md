# Cómo agregar un modelo a JarvisTEC

Cada modelo tiene dos mitades separadas:

| Mitad | Dónde | Qué contiene |
|---|---|---|
| Entrenamiento (offline) | `training/<nombre>/` | notebook, ficha con dataset, configuración y evaluación |
| Inferencia (backend) | `backend/models/<nombre>/` | adaptador (`adapter.py`) y esquema de entradas (`schema.py`) |

El puente entre ambas es el artefacto `artifacts/<nombre>.joblib`. El backend nunca importa nada de `training/`.

Los routers y el enrutador de comandos **no se tocan**: el orquestador (`backend/core/orchestrator.py`)
es el único que registra y ejecuta modelos. Solo hay que agregar la clase a `MODEL_CLASSES`.

```
petición HTTP / comando de voz
        │
routers/models.py, routers/command.py  ── solo traducen HTTP
        │
core/orchestrator.py  ── busca el modelo, lo carga (una vez), valida con su esquema,
        │                ejecuta predict() y arma la respuesta
core/registry.py      ── una instancia por `name`; rechaza duplicados al iniciar
        │
models/<nombre>/adapter.py  ── carga artifacts/<nombre>.joblib y predice
```

## 1. Entrenamiento (`training/<nombre>/`)

Copiar `training/_plantilla/` como `training/<nombre>/`. El notebook ya trae las 6 secciones de la rúbrica en orden:
1. Análisis del problema (0.5 pt)
2. Entendimiento de los datos (0.5 pt)
3. Exploración de los datos (0.5 pt)
4. Modelo (2 pts)
5. Evaluación (1 pt)
6. Conclusión (0.5 pt)

Pasos:
1. Poner el CSV en `data/` (no se sube a git) y anotar la fuente en `training/<nombre>/README.md`.
2. Editar **solo la celda de Configuración** (nombre del modelo, ruta `../../data/...`, columna objetivo, tipo de tarea, si es serie temporal).
3. Ejecutar de arriba abajo y completar los textos marcados con **✍️ COMPLETAR**. La rúbrica califica el análisis y las conclusiones, no solo el código.
4. La última celda de código guarda `artifacts/<nombre>.joblib` (un `Pipeline` completo con preprocesamiento) y `artifacts/<nombre>.meta.json` con las métricas y un `default_input` listo para copiar al backend.
5. Copiar las métricas reales del `meta.json` a la sección Evaluación de la ficha.

Notas:
- Para series temporales (bitcoin, aguacate, etc.) poner `ES_SERIE_TEMPORAL = True` y `COLUMNA_FECHA`: el split será cronológico. Para predecir "mañana" suele hacer falta crear variables de rezago (valores de días anteriores) antes del modelo; añadan esa celda en la sección 4.
- La plantilla es un punto de partida: agreguen otros modelos, ajuste de hiperparámetros o gráficos si el problema lo pide.
- El recomendador de películas no encaja en esta plantilla (no es supervisado clásico); habrá que adaptar las secciones 4 y 5.

## 2. Inferencia (`backend/models/<nombre>/`)

Crear el paquete con tres archivos (se puede copiar `backend/models/demo_vino/` como punto de partida).

`schema.py`: las entradas que acepta el modelo. Se valida **antes** de llamar a `predict()`
(una entrada inválida responde 422) y de aquí se generan los `input_fields` que muestra `GET /models`.

```python
from pydantic import BaseModel as Schema, Field


class VinoInputs(Schema):
    alcohol: float = Field(description="% de alcohol")
    volatile_acidity: float = Field(description="acidez volátil")
```

`adapter.py`: la clase que el orquestador ejecuta.

```python
from typing import Any

import joblib
import pandas as pd

from ..base import BaseModel
from .schema import VinoInputs


class Vino(BaseModel):
    name = "vino"                      # único, minúsculas, sin espacios (se usa en /models/{name})
    title = "Calidad del vino"
    description = "Clasifica la calidad de un vino según sus propiedades químicas."
    task_type = "clasificacion"
    keywords = ["calidad del vino", "clasificar vino"]   # frases de comando de voz
    input_schema = VinoInputs
    default_input = {"alcohol": 11.5, "volatile_acidity": 0.4}   # del meta.json; usado por comandos de voz

    def load(self):
        self.pipeline = joblib.load(self.artifact_path())   # artifacts/vino.joblib

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        df = pd.DataFrame([inputs])
        return {"quality": str(self.pipeline.predict(df)[0])}

    def speak(self, result):
        return f"La calidad estimada del vino es {result['quality']}."
```

`__init__.py`: exporta la clase.

```python
from .adapter import Vino

__all__ = ["Vino"]
```

## 3. Registrar

Agregar la clase a `MODEL_CLASSES` en `backend/models/__init__.py`:

```python
from .vino import Vino

MODEL_CLASSES = [..., Vino]
```

Si reemplaza a un demo con el mismo `name` (p. ej. `vino`), quitar el demo de la lista y borrar su paquete
`demo_vino/`; la API no inicia con dos modelos con el mismo nombre.

Reglas (el registro las verifica al iniciar la API):
- `name` único, en minúsculas, sin espacios.
- Una misma frase de `keywords` no puede estar en dos modelos (el comando sería ambiguo).

Otras reglas:
- `predict` recibe las entradas ya validadas y devuelve un diccionario serializable a JSON (convertir tipos numpy con `float()`/`int()`/`str()`).
- `load` se ejecuta una sola vez, en la primera predicción. Si falla (p. ej. falta el `.joblib`), la API responde 503 para ese modelo y los demás siguen funcionando.
- `default_input` permite que un comando de voz funcione sin dar datos (ej. "JarvisTEC, tipo de cambio para mañana").
- `keywords`: 2-3 frases naturales que diría un usuario. Incluir el nombre del tema.
- `is_demo = True` solo para lógica de juguete; los modelos entrenados lo dejan en `False`.
- Nunca subir datasets pesados ni `.joblib` grandes al repo (están en `.gitignore`); documentar en la ficha cómo regenerarlos.

## 4. Probar

```bash
python -m pytest -q
uvicorn backend.main:app --reload
```

En http://localhost:8000/docs probar `POST /models/vino/predict` y `POST /command`
con `{"text": "JarvisTEC calidad del vino"}`.

Para probar un adaptador sin artefacto, `tests/test_orchestrator.py` muestra cómo construir un
`ModelOrchestrator` con clases propias y sustituirlo en la API con `app.dependency_overrides`.

## 5. Entregables por modelo (checklist)

- [ ] `training/<nombre>/notebook.ipynb` con las 6 secciones
- [ ] `training/<nombre>/README.md` con dataset, reproducción y métricas reales
- [ ] Artefacto entrenado en `artifacts/` (o instrucciones para regenerarlo)
- [ ] Paquete `backend/models/<nombre>/` registrado en `MODEL_CLASSES`
- [ ] 2-3 frases de comando
- [ ] 1-2 artículos científicos para la bibliografía
- [ ] Sección redactada en Overleaf
