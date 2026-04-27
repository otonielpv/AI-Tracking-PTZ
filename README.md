# AI Tracking PTZ

Sistema de auto-tracking para camaras PTZ de iglesia con captura RTSP, deteccion de personas con YOLO, tracking persistente, control PTZ, integracion MIDI para FreeShow y ruta final de despliegue en TensorRT.

## Estado actual

El proyecto ya esta consolidado en un unico CLI. Los prototipos por hito se retiraron como entrypoints independientes y ahora todo se ejecuta desde un solo comando:

```powershell
python -m ai_tracking_ptz <comando> [opciones]
```

Los modulos internos siguen separados por responsabilidad para mantener el codigo mantenible: video, tracking, control, MIDI y backends PTZ.

## Capacidades

- Captura RTSP optimizada con lectura en hilo separado.
- Fuente alternativa por archivo de video para pruebas offline.
- Deteccion y tracking de personas con Ultralytics YOLO Nano.
- Limite de inferencia configurable para contener carga de GPU.
- PTZ virtual para desarrollar y ajustar la logica sin hardware real.
- Control ONVIF PTZ autenticado para la camara final.
- PID para recentrado suave sobre el objetivo.
- Seleccion automatica de objetivo basada en persistencia, area, posicion y confianza.
- Activacion, desactivacion y reacquisition por MIDI para integracion con FreeShow.
- Exportacion a TensorRT para el PC final con RTX.

## Requisitos

- Python 3.11 o superior recomendado.
- Windows con OpenCV funcionando correctamente.
- FFmpeg disponible si vas a generar streams RTSP de prueba.
- Para ONVIF real: IP, usuario, contraseña y puerto del servicio PTZ.
- Para MIDI real: un puerto de entrada accesible desde Windows, normalmente loopMIDI o el puerto expuesto por tu flujo con FreeShow.
- Para TensorRT final: NVIDIA RTX, drivers, CUDA y TensorRT instalados en el PC definitivo.

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

## Uso rapido

### 1. Ver una fuente de video

RTSP autenticado:

```powershell
python -m ai_tracking_ptz view --rtsp-url "rtsp://usuario:password@192.168.1.120:554/stream" --rtsp-transport tcp
```

Video local:

```powershell
python -m ai_tracking_ptz view --video-file ".\samples\people.mp4" --loop-video
```

### 2. Ver tracking de personas

```powershell
python -m ai_tracking_ptz track --video-file ".\samples\people.mp4" --loop-video --model yolov8n.pt --tracker botsort.yaml --imgsz 640 --max-inference-fps 15
```

### 3. Probar PTZ virtual

```powershell
python -m ai_tracking_ptz ptz-test --ptz-backend virtual --video-file ".\samples\people.mp4" --loop-video
```

### 4. Probar ONVIF PTZ

```powershell
python -m ai_tracking_ptz ptz-test --ptz-backend onvif --host "192.168.1.120" --port 80 --username "admin" --password "tu_password"
```

### 5. Ejecutar auto-tracking

Prueba segura con PTZ virtual:

```powershell
python -m ai_tracking_ptz auto-track --video-file ".\samples\people.mp4" --loop-video --ptz-backend virtual --model yolov8n.pt --tracker botsort.yaml --imgsz 640 --max-inference-fps 15 --start-enabled
```

Uso real con RTSP + ONVIF:

```powershell
python -m ai_tracking_ptz auto-track --rtsp-url "rtsp://usuario:password@192.168.1.120:554/stream" --rtsp-transport tcp --ptz-backend onvif --host "192.168.1.120" --port 80 --username "admin" --password "tu_password" --model yolov8n.pt --tracker botsort.yaml --imgsz 640 --max-inference-fps 15 --start-enabled
```

### 6. Listar puertos MIDI disponibles

```powershell
python -m ai_tracking_ptz list-midi
```

### 7. Integrar FreeShow por MIDI

```powershell
python -m ai_tracking_ptz auto-track --video-file ".\samples\people.mp4" --loop-video --ptz-backend virtual --midi-input-name "LoopMIDI Port" --midi-channel 0 --midi-toggle-note 60 --midi-enable-note 61 --midi-disable-note 62 --midi-reacquire-note 63 --start-enabled
```

### 8. Exportar a TensorRT

```powershell
python -m ai_tracking_ptz export-engine --model yolov8n.pt --imgsz 640 --device cuda:0 --half --workspace 4
```

## Comandos

### `view`

Abre una fuente RTSP o un video local y muestra el stream con metricas basicas.

Opciones clave:

- `--rtsp-url` o `--video-file`
- `--rtsp-transport tcp|udp`
- `--loop-video`
- `--width` y `--height`

### `track`

Ejecuta tracking de personas con YOLO sobre una fuente RTSP o archivo local.

Opciones clave:

- `--model yolov8n.pt`
- `--tracker botsort.yaml`
- `--imgsz 640`
- `--max-inference-fps 15`
- `--device cpu|cuda:0`

### `ptz-test`

Permite probar movimiento PTZ manual.

Backends soportados:

- `virtual`: mueve un viewport sobre un video local.
- `onvif`: mueve la camara real mediante ONVIF.

Controles:

- `W` tilt arriba
- `S` tilt abajo
- `A` pan izquierda
- `D` pan derecha
- `Z` zoom out
- `X` zoom in
- `Space` stop
- `Q` o `Esc` salir

### `auto-track`

Es el flujo principal del sistema. Ejecuta detector, tracker, selector automatico de target, PID y control PTZ. Puede correr con backend virtual para pruebas o con ONVIF para uso real.

Comportamiento:

- Cuando tracking esta activado, el selector pondera area, cercania al centro, confianza y persistencia del tracker.
- Cuando tracking esta desactivado, la camara deja de corregir y entra en `stop`.
- El comando MIDI de `reacquire` fuerza nueva adquisicion de objetivo.
- El overlay muestra estado MIDI, target actual, lock del selector, errores de centro e informacion de inferencia.

### `list-midi`

Lista todos los puertos MIDI de entrada visibles para la aplicacion.

### `export-engine`

Exporta un modelo YOLO `.pt` a TensorRT `.engine`. Esta operacion debe ejecutarse en el PC final con la GPU definitiva.

## Parametros importantes para produccion

### Inferencia

- Usa `yolov8n.pt` o el `.engine` equivalente.
- Mantén `--imgsz 640` para limitar consumo.
- Mantén `--max-inference-fps 15` como punto de partida en el PC de streaming.

### RTSP

- En Windows, `--rtsp-transport tcp` suele ser la opcion mas estable.
- Si la camara requiere autenticacion, usa usuario y contraseña dentro de la URL RTSP.

### ONVIF

- El canal ONVIF usa autenticacion separada de RTSP: `--host`, `--port`, `--username`, `--password`.
- Si los WSDL no se resuelven automaticamente, usa `--wsdl-dir`.

### MIDI

- Si no pasas `--midi-input-name`, el flujo sigue funcionando, pero sin control externo.
- Para FreeShow en Windows, lo normal es rutear eventos a un puerto virtual como loopMIDI.

### PID y estabilidad

- `--deadzone-x` y `--deadzone-y` evitan microvibraciones cerca del centro.
- `--pid-pan-*` y `--pid-tilt-*` permiten ajustar respuesta y suavidad.
- La heuristica de target se ajusta con `--selector-*`.

## Flujo recomendado de trabajo

### Desarrollo sin hardware

1. Usa `view` o `track` con un video local.
2. Ajusta el tracking con `track`.
3. Ajusta el comportamiento del movimiento con `ptz-test --ptz-backend virtual`.
4. Valida el lazo completo con `auto-track --ptz-backend virtual`.

### Paso a hardware real

1. Verifica el stream RTSP con `view`.
2. Verifica movimiento ONVIF con `ptz-test --ptz-backend onvif`.
3. Ejecuta `auto-track` con RTSP + ONVIF.
4. Ajusta PID, deadzones y selector automatico.
5. Solo al final exporta a TensorRT y compara `.pt` contra `.engine`.

## TensorRT y RTX 3050

La exportacion a TensorRT es parte del despliegue final, no del desarrollo inicial. El archivo `.engine` depende de la GPU, los drivers, CUDA y TensorRT instalados en el equipo donde se exporta.

Objetivo real de TensorRT en este proyecto:

- reducir latencia
- mejorar uso de CUDA cores
- contener consumo de VRAM
- convivir mejor con OBS o vMix

Una vez generado el `.engine`, puedes reutilizarlo directamente en el flujo principal:

```powershell
python -m ai_tracking_ptz auto-track --rtsp-url "rtsp://usuario:password@192.168.1.120:554/stream" --ptz-backend onvif --host "192.168.1.120" --username "admin" --password "tu_password" --model yolov8n.engine --tracker botsort.yaml --imgsz 640 --max-inference-fps 15 --start-enabled
```

## Fuente RTSP de prueba

Si no tienes la camara disponible, puedes levantar un RTSP local con MediaMTX y FFmpeg:

```powershell
docker run --rm -it -p 8554:8554 bluenviron/mediamtx:latest
```

En otra terminal:

```powershell
ffmpeg -re -f lavfi -i testsrc=size=1280x720:rate=30 -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p -rtsp_transport tcp -f rtsp rtsp://127.0.0.1:8554/test
```

Luego:

```powershell
python -m ai_tracking_ptz view --rtsp-url "rtsp://127.0.0.1:8554/test" --rtsp-transport tcp
```

## Estructura del proyecto

```text
src/ai_tracking_ptz/
	cli.py                 CLI unificado
	__main__.py            Entrada unica con python -m ai_tracking_ptz
	video/                 Captura RTSP y archivo
	tracking/              YOLO y seleccion automatica de objetivo
	control/               PID
	camera/                Backends PTZ virtual, ONVIF y referencia VISCA
	midi/                  Integracion MIDI
```

## Notas finales

- El backend VISCA se mantiene solo como referencia tecnica; para esta camara el camino principal es ONVIF.
- El flujo mas seguro para ajustar heuristicas y PID sigue siendo el backend virtual.
- El flujo mas cercano a produccion es `auto-track` con RTSP + ONVIF + MIDI.
