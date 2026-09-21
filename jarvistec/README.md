# JarvisTEC

Asistente personal con reconocimiento de emociones (imagen) y comandos de voz que ejecutan modelos de machine learning.
Proyecto grupal de Inteligencia Artificial, TEC.

## Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # completar GOOGLE_API_KEY (ver docs/CONFIGURAR_GOOGLE.md)
uvicorn backend.main:app --reload
```

- Documentación interactiva (Swagger): http://localhost:8000/docs
- Estado y modelos cargados: http://localhost:8000/health
- Tests: `python -m pytest -q`

Ejecutar SIEMPRE desde la raíz del repo (la carpeta que contiene `backend/`).

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado de la API y lista de modelos |
| GET | `/models` | Metadatos de todos los modelos |
| GET | `/models/{name}` | Metadatos de un modelo |
| POST | `/models/{name}/predict` | Ejecuta un modelo. Body: `{"inputs": {...}}` |
| POST | `/command` | Texto transcrito -> modelo correcto. Body: `{"text": "JarvisTEC precio del bitcoin"}` |
| POST | `/emotion` | Imagen (campo `image`) -> emoción dominante y puntajes |
| POST | `/transcribe` | Audio (campo `audio`) -> texto transcrito |

## Estructura

```
backend/models/    un archivo por modelo (se registran solos)
backend/services/  lógica: router de comandos, (luego) rostro y voz
backend/routers/   endpoints
notebooks/         plantilla_modelo.ipynb + un notebook por dataset
artifacts/         modelos entrenados (.joblib), ignorados por git
data/              datasets, ignorados por git
frontend/          HTML + CSS + JS (la API lo sirve en http://localhost:8000/)
```

## Agregar un modelo

Ver [docs/AGREGAR_MODELO.md](docs/AGREGAR_MODELO.md).
