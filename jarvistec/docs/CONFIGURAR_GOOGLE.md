# Configurar Google Cloud (rostro y voz)

JarvisTEC usa dos APIs de Google en la entrega de la semana 10:

- **Cloud Vision** (detección de rostros) para leer la emoción de un frame de cámara.
- **Cloud Speech-to-Text** para convertir el audio del micrófono en texto.

Ambas se llaman por REST con una **clave de API**, sin instalar SDKs.

## Pasos

1. Entrar a https://console.cloud.google.com e iniciar sesión.
2. Crear un proyecto nuevo (ej. `jarvistec`).
3. Activar la facturación del proyecto. Google la exige aunque haya cuota gratuita; revisen los precios y cuotas gratuitas vigentes de Vision y Speech-to-Text antes de usarlas, y configuren una **alerta de presupuesto** (Facturación > Presupuestos y alertas) para evitar sorpresas.
4. En **APIs y servicios > Biblioteca**, habilitar:
   - `Cloud Vision API`
   - `Cloud Speech-to-Text API`
5. En **APIs y servicios > Credenciales > Crear credenciales > Clave de API**.
6. Editar la clave y en **Restricciones de API** limitarla solo a esas dos APIs (así, si se filtra, el daño es menor).
7. Copiar `.env.example` a `.env` y pegar la clave:
   ```
   GOOGLE_API_KEY=AIza...
   ```
8. Reiniciar `uvicorn`. El indicador de la esquina superior derecha de la interfaz ya no debe decir "voz/rostro simulados".

**Nunca** suban `.env` ni la clave al repositorio (ya está en `.gitignore`). Si la clave se filtra, elimínenla en la consola y creen otra.

## Compartir la clave con el equipo

Cada integrante puede usar la misma clave (pásenla por un canal privado, no por el repo) o crear su propia clave en el mismo proyecto.

## Modo simulado

Si `GOOGLE_API_KEY` está vacía, el backend funciona en modo simulado para desarrollar sin credenciales:

- `/emotion` devuelve una emoción aleatoria (o fija con `MOCK_EMOTION=happiness` en `.env`).
- `/transcribe` devuelve el texto de `MOCK_TRANSCRIPT` sin escuchar el audio.

Las respuestas incluyen `"provider": "mock"` y la interfaz lo indica. **No usar el modo simulado en la demo o en la evaluación.**

## Cómo se interpretan las emociones

Google Vision no da porcentajes, sino niveles de verosimilitud (`VERY_UNLIKELY` ... `VERY_LIKELY`) para cuatro emociones: alegría, tristeza, enojo y sorpresa. El backend los convierte a puntajes 0-1 y elige la emoción dominante; si ninguna llega al nivel `POSSIBLE`, se considera **neutral**. Esta limitación (4 emociones, sin miedo ni asco) debe mencionarse en el documento.

## Solución de problemas

| Síntoma | Causa probable |
|---|---|
| `Google Vision: API key not valid` | Clave mal copiada, o la API no está habilitada / no está en las restricciones de la clave |
| `Google Vision: ... billing` | Falta activar la facturación del proyecto |
| `No se detectó ningún rostro` | Poca luz, rostro de lado o muy lejos |
| `No se entendió el audio` | Micrófono muy lejos, ruido, o audio de más de ~1 minuto |
| Transcripción vacía en Safari | Safari graba en MP4, que Google no admite: usar Chrome, Edge o Firefox |
| "JarvisTEC" transcrito raro | Es normal: el router tolera variantes ("yarvis tec"); las frases de sugerencia ya se envían a Google |

## Cambiar de proveedor

`backend/services/emotion_service.py` y `speech_service.py` exponen una sola función pública (`detect_emotion`, `transcribe`) y un diccionario `PROVIDERS`. Para agregar otro proveedor (ej. un modelo propio) se escribe una función nueva y se registra ahí; el resto del sistema no cambia.
