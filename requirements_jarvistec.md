
# Proyecto JarvisTEC — Requerimientos Estructurados

## Contexto del proyecto

Asistente personal por voz e imagen que reconoce el rostro y la voz del usuario, interpreta comandos de aprendizaje automático en lenguaje natural (ej. "JarvisTEC, tipo de cambio para mañana") y responde ejecutando el modelo correspondiente.

---

## Etapa 0 — Aspectos administrativos (aplica durante todo el proyecto)

- Grupos de máximo 3 integrantes.
- Entrega final: semana 11 (se puede terminar y presentar antes).
- Documentación obligatoria en Overleaf (sin excepción).
- Repositorio obligatorio en Bitbucket, GitLab o GitHub con todo el proyecto.

---

## Etapa 1 — Diseño y arquitectura del sistema

**Objetivo:** definir cómo el usuario interactuará con la máquina.

1. Diseñar una plataforma/interfaz para la comunicación usuario–máquina (libertad creativa; ejemplo dado: interfaz estilo J.A.R.V.I.S.).
2. El asistente no requiere rostro real; puede usarse una animación, imagen o efecto de audio como representación visual/sonora al responder.

---

## Etapa 2 — Reconocimiento de emociones (visión)

**Objetivo:** que el sistema interprete el estado emocional del usuario.

1. Diseñar un modelo de agente inteligente que reconozca el sentimiento/emoción de la persona a partir de:
   - Una foto tomada en tiempo real desde cámara, o
   - Un frame extraído de un video.
2. **Restricción para la primera entrega (semana 10):** se permite usar APIs externas (Azure Face API o equivalentes de Google) para el reconocimiento facial.
   - Referencia: Azure Face API — https://azure.microsoft.com/es-es/services/cognitive-services/face/

---

## Etapa 3 — Conversión de voz a texto y ejecución de comandos

**Objetivo:** que el sistema entienda instrucciones habladas y las ejecute.

1. Diseñar un modelo de agente inteligente que:
   - Traduzca audio a texto (Speech-to-Text).
   - Ejecute la instrucción/comando contenido en ese texto.
2. **Restricción para la primera entrega (semana 10):** se permite usar APIs externas para voz a texto.
   - Referencias: Google Speech-to-Text — https://cloud.google.com/speech-to-text?hl=es
   - También se menciona ChatGPT (https://chat.openai.com/) como herramienta permitida en esta entrega.

---

## Etapa 4 — Selección e implementación de modelos de Machine Learning

**Objetivo:** construir al menos 10 modelos de ML funcionales.

1. Seleccionar **10 modelos de ML** de la lista provista (ver Anexo A) — o, alternativamente:
   - Proponer un dataset propio/externo ligado a una necesidad real (empresa, institución, comunidad, organización).
   - Requiere aprobación previa del profesor: presentar el problema, el origen y naturaleza del dataset (público, institucional o propio), y el modelo propuesto, **antes de iniciar el desarrollo**.
   - El dataset alternativo debe permitir el mismo tipo de análisis exigido a los 10 modelos.
   - Se puede reutilizar un mismo dataset externo para más de un modelo, siempre que cada uno resuelva una tarea distinta (ej. clasificación vs. predicción) y esté aprobado por el profesor.
2. Cada modelo debe incluir estas 6 secciones de análisis (ver criterios de evaluación en Etapa 6):
   - Análisis del problema
   - Entendimiento de los datos
   - Exploración de los datos
   - Modelo (implementación)
   - Evaluación
   - Conclusión

---

## Etapa 5 — Integración: API REST y comandos

**Objetivo:** exponer los modelos como servicios accesibles y ejecutables por comando.

1. Crear un **servicio API REST** que exponga los endpoints necesarios para cubrir todas las funcionalidades del proyecto (uno por modelo, más los de visión/voz).
2. Crear un conjunto de **instrucciones/comandos** asociados a cada algoritmo de ML seleccionado, de forma que puedan invocarse por voz (ej. `JarvisTEC "tipo de cambio para mañana"`).

---

## Etapa 6 — Documentación científica (en Overleaf)

**Objetivo:** justificar técnicamente las decisiones tomadas, con respaldo bibliográfico.

Estructura obligatoria del documento:

1. Portada
2. Solución planteada
3. Arquitecturas de ML
4. Justificación y explicación del modelo desarrollado
5. Análisis de resultados obtenidos
6. Bibliografía
7. Anexos

Enfoque: perspectiva científica aplicada, con modelos justificados mediante artículos científicos.

---

## Etapa 7 — Entrega y evaluación

### Cronograma de entrega

| Alcances incluidos                                              | Semana de entrega | Comentario                                                                           |
| --------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------ |
| 1 (interfaz), 2 (emociones), 3 (voz-texto), 4 (≥10 modelos ML) | Semana 10         | Única entrega en la que se permiten APIs de Google/Azure para emociones y voz-texto |

### Criterios de evaluación

| Ítem                | %   | Detalle                                                                                                                                                                      |
| -------------------- | --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Creación del modelo | 30% | 5 pts por modelo → distribuidos en: Análisis del problema (0.5), Entendimiento de datos (0.5), Exploración de datos (0.5), Modelo (2), Evaluación (1), Conclusión (0.5) |
| Aplicación          | 10% | 1 punto por cada modelo completado e integrado a la aplicación                                                                                                              |
| API REST             | 10% | 1 punto por cada modelo completado y expuesto vía API                                                                                                                       |
| Documento            | 10% | Completo = 10 pts; incompleto = 3 pts; no presentado = 0 pts                                                                                                                 |

---

## Anexo A — Lista de datasets/algoritmos sugeridos

| Tarea                                            | Dataset                                            |
| ------------------------------------------------ | -------------------------------------------------- |
| Predecir precio de Bitcoin                       | Kaggle: bitcoin-price-prediction                   |
| Predecir precio de un automóvil                 | GitHub: amankharwal/Website-data (car data.csv)    |
| Recomendar una película                         | Google Drive (dataset de películas)               |
| Predecir precio de acciones S&P 500              | Kaggle: camnugent/sandp500                         |
| Predecir precio de una casa                      | Kaggle: zillow-prize-1                             |
| Clasificar calidad del vino                      | Kaggle: rajyellow46/wine-quality                   |
| Predecir inventario de una compañía            | Kaggle: demand-forecasting-kernels-only            |
| Predecir tarifa de viajes en bicicleta           | Kaggle: chh-ola                                    |
| Clasificar abandono (churn) de clientes móviles | GitHub: IBM/telco-customer-churn-on-icp4d          |
| Predecir retraso (delay) de vuelos               | Kaggle: giovamata/airlinedelaycauses               |
| Clasificar riesgo de accidente cerebrovascular   | Kaggle: fedesoriano/stroke-prediction-dataset      |
| Clasificar tipo de hepatitis                     | Kaggle: fedesoriano/hepatitis-c-dataset            |
| Predecir masa corporal de un paciente            | Kaggle: fedesoriano/body-fat-prediction-dataset    |
| Clasificar tipo de cirrosis                      | Kaggle: fedesoriano/cirrhosis-prediction-dataset   |
| Predecir pacientes recuperados de COVID-19       | Kaggle: novel-corona-virus-2019-dataset            |
| Predecir índice de precios (Walmart)            | Kaggle: walmart-recruiting-store-sales-forecasting |
| Predecir precio del aguacate                     | Kaggle: neuromusic/avocado-prices                  |
| Predecir crímenes diarios en Londres            | Kaggle: jboysen/london-crime                       |
| Predecir crímenes diarios en Chicago            | Kaggle: chicago/chicago-crime                      |
| Predecir ventas de Rossmann                      | Kaggle: rossmann-store-sales                       |

*Nota: si se usa un dataset distinto a los listados, requiere validación previa del profesor (ver Etapa 4).*
