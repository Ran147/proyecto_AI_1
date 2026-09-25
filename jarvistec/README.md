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
backend/
  core/orchestrator.py  único punto de acceso a los modelos: catálogo, predicción y comandos
  core/registry.py      registro de modelos; falla al iniciar si hay nombres duplicados
  models/__init__.py    MODEL_CLASSES: lista explícita de modelos activos
  models/base.py        contrato BaseModel que implementa cada adaptador
  models/<nombre>/      un paquete por modelo: adapter.py + schema.py (solo inferencia)
  services/             router de intenciones, rostro y voz
  routers/              endpoints HTTP (delegan en el orquestador)
training/               entrenamiento offline, nunca importado por el backend
  _plantilla/           notebook con las 6 secciones + ficha del modelo
  <nombre>/             un directorio por modelo real
artifacts/              modelos entrenados (.joblib + .meta.json), ignorados por git
data/                   datasets, ignorados por git
frontend/               HTML + CSS + JS (la API lo sirve en http://localhost:8000/)
```

Los modelos `bitcoin` y `vino` actuales son **demos sin entrenar** (`backend/models/demo_*`,
`"demo": true` en `GET /models`); sirven para probar el flujo mientras llegan los modelos reales.

## Agregar un modelo

Ver [docs/AGREGAR_MODELO.md](docs/AGREGAR_MODELO.md).
