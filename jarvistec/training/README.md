# Entrenamiento offline

Todo lo que produce un modelo vive aquí y **nunca se importa desde el backend**.
El backend solo consume el resultado: el artefacto en `artifacts/`.

```
training/
  _plantilla/            copiar como training/<nombre>/
    notebook.ipynb       las 6 secciones de la rúbrica (configuración en la 1.ª celda de código)
    README.md            ficha: dataset, cómo reproducir, resultados de evaluación
  <nombre>/              un directorio por modelo real (vino, bitcoin, churn...)
```

`<nombre>` debe coincidir con el `name` del adaptador en `backend/models/<paquete>/`.

## Flujo

1. Descargar el dataset en `data/` (ignorado por git) y anotar su origen en `training/<nombre>/README.md`.
2. Editar la celda **Configuración** del notebook (nombre, ruta, columna objetivo, tipo de tarea).
   Las rutas son relativas a `training/<nombre>/`: `../../data/` y `../../artifacts/`.
3. Ejecutar el notebook de arriba abajo. La sección 5 (Evaluación) calcula las métricas sobre el conjunto de prueba.
4. La última celda guarda:
   - `artifacts/<nombre>.joblib`: `Pipeline` completo (preprocesamiento + modelo).
   - `artifacts/<nombre>.meta.json`: métricas de prueba, lista de features y un `default_input`.
5. Copiar las métricas reales de `meta.json` a la ficha `README.md` del modelo. Nunca escribir métricas que no salieron de una ejecución.

Luego se crea el adaptador de inferencia: ver [docs/AGREGAR_MODELO.md](../docs/AGREGAR_MODELO.md).

## Estado

Aún no hay datasets ni modelos entrenados en el repositorio. Los modelos `bitcoin` y `vino`
que expone la API son **demos sin entrenar** (`backend/models/demo_*`) y no tienen carpeta aquí.
