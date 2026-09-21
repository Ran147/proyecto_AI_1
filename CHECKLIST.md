# 📋 Checklist de Estado del Proyecto - JarvisTEC
**Curso:** Inteligencia Artificial — I Semestre 2026 (Campus Tecnológico Local San Carlos)  
**Profesor:** Ing. Efrén Jiménez Delgado  
**Última actualización:** Marzo 2026  

---

## 📊 1. Resumen Ejecutivo de Avance

> [!IMPORTANT]
> **Directriz sobre APIs comerciales de Google:**  
> **Google Cloud Vision** y **Google Cloud Speech-to-Text** **NO son utilizadas ni se van a utilizar** en la entrega del proyecto. Aunque en el código actual existen implementaciones iniciales hacia dichas APIs, estas **serán removidas por completo** para sustituirlas por **modelos propios locales** (visión por computadora para reconocimiento facial/emociones y modelo local offline para voz a texto).



---

## 🎯 2. Requerimientos de la Guía vs Estado Actual

### A. Plataforma de Software / Interfaz de Usuario (10%)
- [x] **Diseño creativo interactivo:** Implementado HUD futurista estilo reactor Arc / Iron Man (`jarvistec/frontend/index.html` y `style.css`).
- [x] **Visualización reactiva:** Animación SVG del reactor que reacciona en tiempo real a la intensidad del micrófono mediante Web Audio API (`AudioContext`).
- [x] **Captura de video en tiempo real:** Módulo WebRTC para activar/desactivar la cámara web y capturar fotogramas (`captureFrame`).
- [x] **Entrada multimodal:** Soporte para comandos por voz (micrófono con `MediaRecorder`) y por texto escrito.
- [x] **Respuesta principal en texto:** La respuesta oficial de Jarvis se presenta en texto en pantalla y en el log de interacción. *(Nota: La síntesis de voz por navegador con `speechSynthesis` es un extra cosmético opcional, no un requerimiento de la guía).*
- [x] **Historial y estado:** Panel lateral con lista dinámica de comandos reconocidos, log de mensajes y visor de JSON crudo de la API.

---

### B. Servicio API REST (10%)
- [x] **Framework FastAPI configurado:** Servidor modular con soporte CORS y montaje de estáticos para servir el frontend (`jarvistec/backend/main.py`).
- [x] **Endpoint de salud:** `GET /health` reportando estado del sistema, modelos cargados y proveedores activos.
- [x] **Endpoints de catálogo de modelos:**
  - [x] `GET /models`: Retorna metadatos de todos los modelos registrados.
  - [x] `GET /models/{name}`: Metadatos específicos de un modelo.
- [x] **Endpoint de predicción directa:** `POST /models/{name}/predict` con validación de inputs.
- [x] **Endpoint de enrutamiento de comandos:** `POST /command` con detección de *wake-word* («JarvisTEC») y coincidencia difusa (*fuzzy matching*).
- [x] **Endpoints para agentes:**
  - [x] `POST /emotion`: Recepción multipart de imagen para inferencia de sentimientos.
  - [x] `POST /transcribe`: Recepción multipart de audio para transcripción a texto.
- [x] **Arquitectura desacoplada:** Clase abstracta `BaseModel` (`jarvistec/backend/models/base.py`) con auto-registro de nuevos modelos al agregarlos a la carpeta.
- [ ] **Integración de los 10 endpoints de modelos reales:** Conectar los 10 modelos definitivos de ML entrenados.

---

### C. Alcance 1: Agente Inteligente de Reconocimiento de Sentimientos / Emociones
> ⚠️ **NO SE UTILIZA GOOGLE VISION:**  
> Las funciones y llamadas a la API de Google Vision (`_call_vision`, etc.) que están actualmente en el código **no se utilizan ni se utilizarán**, y **serán removidas por completo**. El reconocimiento de emociones se resolverá al 100% mediante un **modelo propio local**.

- [x] **Flujo end-to-end implementado en frontend:**
  - Captura de frame desde la cámara web en el frontend.
  - Envío a endpoint `POST /emotion`.
  - Normalización de salida (`happiness`, `sadness`, `anger`, `surprise`, `neutral`) y barras de probabilidad en la UI.
  - Inyección de empatía: Jarvis antepone frases de empatía según el estado anímico detectado.
- [x] **Modo simulación provisional (Mock):** Permite pruebas sin requerir ninguna clave externa mientras se conecta el modelo propio.
- [ ] **Remoción de código de Google y conexión del Modelo Propio (PENDIENTE):**
  - [ ] **Remover del código:** Eliminar `_call_vision`, `parse_vision_response` y las referencias a `GOOGLE_API_KEY` en `emotion_service.py`.
  - [ ] **Seleccionar arquitectura propia:** (Ej. red neuronal convolucional / CNN entrenada sobre FER-2013, o modelo local con OpenCV / PyTorch / ONNX).
  - [ ] **Detección facial local:** Integrar detector facial local (ej. Haar Cascade / MediaPipe) para recortar el rostro antes de la clasificación.
  - [ ] **Inferencia local:** Cargar y ejecutar el modelo de emociones directamente en el backend local.

---

### D. Alcance 2: Agente Inteligente de Voz a Texto (Speech-to-Text) y Ejecución de Comandos
> ⚠️ **NO SE UTILIZA GOOGLE SPEECH-TO-TEXT:**  
> Las funciones y llamadas a Google Cloud Speech-to-Text (`_call_speech`, etc.) que están en el código **no se utilizan ni se utilizarán**, y **serán removidas por completo**. La transcripción de voz a texto se resolverá exclusivamente con un **modelo propio local**.

- [x] **Flujo de grabación y audio:**
  - Grabación en navegador con `MediaRecorder` (`audio/webm` o `audio/wav`).
  - Envío binario a `POST /transcribe`.
- [x] **Router de comandos e intenciones:**
  - Normalización de texto (eliminación de acentos, caracteres especiales).
  - Detección flexible de palabra de activación («JarvisTEC», «Jarvis», «yarvis tek»).
  - Algoritmo de similitud (`SequenceMatcher`) para relacionar el comando con las *keywords* de cada modelo.
  - Ejecución automática del modelo correspondiente y respuesta en texto.
- [x] **Modo simulación provisional (Mock):** Permite probar comandos sin conexión externa mientras se monta el modelo propio.
- [ ] **Remoción de código de Google y conexión del Modelo Propio de STT (PENDIENTE):**
  - [ ] **Remover del código:** Eliminar `_call_speech`, `parse_speech_response` y endpoints externos de Google en `speech_service.py`.
  - [ ] **Seleccionar solución local offline:** Integrar modelo local (ej. Whisper local/faster-whisper, Vosk en español o Wav2Vec2).
  - [ ] **Inferencia local:** Procesar el audio recibido en el servidor de forma 100% offline y retornar el texto transcrito.
  - [ ] **Actualizar dependencias:** Instalar paquetes necesarios en `requirements.txt` y eliminar dependencias de APIs de Google.

---

### E. Los 10 Modelos de Machine Learning (30% Modelos + 10% API + 10% App)
*Cada modelo debe contar con notebook completo (5 pts c/u), integración en backend y soporte en la UI.*

| # | Problema / Modelo Seleccionado | Dataset | 1. Análisis Problema | 2. Entendimiento Datos | 3. EDA (Exploración) | 4. Modelo (Entrenamiento) | 5. Evaluación | 6. Conclusión | Backend `.py` | Artefacto `.joblib` | Comando Voz Asociado |


> **Nota actual sobre modelos:**  
> Actualmente el repositorio posee:
> - `jarvistec/backend/models/toy_bitcoin.py` (cálculo simulado +1%).
> - `jarvistec/backend/models/toy_wine.py` (regla fija de if/else).
> - `jarvistec/notebooks/plantilla_modelo.ipynb` (plantilla guía lista para ser clonada).
> 
> Los datasets a usar son: 

1. Calidad del vino
2. Precio de automóvil
3. Churn de telefonía
4. Stroke
5. Precio del aguacate
6. Precio de Bitcoin
7. Masa corporal (body fat)
8. Recomendar una película
9. Hepatitis C
10. Cirrosis

---

### F. Documentación Científica en Overleaf (10%)
*La guía estipula desarrollo obligatorio en Overleaf con formato de artículo científico aplicado.*

- [ ] **1. Portada** (Institución, curso, profesor, integrantes, fecha).
- [ ] **2. Solución Planteada** (Descripción integral de la arquitectura del asistente JarvisTEC).
- [ ] **3. Arquitecturas de ML** (Diagramas y descripción técnica de los modelos implementados: visión, voz y los 10 modelos predictivos).
- [ ] **4. Justificación y Explicación de los Modelos** (Fundamentación teórica y metodológica de la elección de algoritmos).
- [ ] **5. Análisis de Resultados Obtenidos** (Métricas, matrices de confusión, curvas ROC/R2/RMSE, tiempos de respuesta).
- [ ] **6. Bibliografía Científica** (Artículos de IEEE, ACM, Springer o Scopus que respalden cada enfoque).
- [ ] **7. Anexos** (Capturas del sistema funcionando, diagramas de clases, endpoints de la API).

---


## 3. Hoja de Ruta Priorizada (Próximos Pasos)

1. **Selección de los 10 problemas y datasets:**  
   Elegir los 10 problemas de la lista oficial de la guía y descargar los datos en `jarvistec/data/`.
2. **Entrenamiento de los 10 notebooks:**  
   Completar los 10 notebooks usando la estructura de `plantilla_modelo.ipynb` y exportar los pipelines a `jarvistec/artifacts/<nombre>.joblib`.
3. **Registro de modelos en backend:**  
   Crear un archivo `.py` en `jarvistec/backend/models/` por cada modelo heredando de `BaseModel`.
4. **Desarrollo del Modelo Propio de Reconocimiento Facial / Emociones:**  
   Remover código y dependencias de Google Vision e integrar el modelo local propio (ej. CNN / MobileNet sobre FER-2013) con inferencia en `emotion_service.py`.
5. **Desarrollo del Modelo Propio de Speech-to-Text (STT):**  
   Remover código y dependencias de Google STT e integrar el motor local offline propio (ej. Whisper local/faster-whisper o Vosk en español) en `speech_service.py`.
6. **Redacción en Overleaf:**  
   Iniciar el documento científico con el desglose de los experimentos y la bibliografía requerida.


## En resumen todo de reduce en:
1. Dise˜nar un modelo de agentes inteligentes que permita reconocer el sentimiento de la persona mediante el uso de una foto tomada desde una c´amara en tiempo real, puede ser un frame de un video
2. Dise˜nar un modelo de agente inteligente que partiendo de un audio lo traduce a texto y ejecuta la instrucci´on dada
3. Dise˜nar al menos 10 algoritmos de aprendizaje autom´atica de la lista proporcionada
4. Crear un conjunto de instrucciones o comandos asociados a los algortimos de aprendizaje autom´atico
seleccionados que permitan ejecutarlos, por ejemplo JarvisTEC ”tipo de cambio para ma˜nana”

Los dos primeros puntos de este texto anterior son la prioriedad inicial, las bases para comenzar a hacerlo ya estan.