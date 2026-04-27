# AI Tracking PTZ

Hito 1 implementa una captura RTSP optimizada para OpenCV con lectura en hilo separado para evitar acumulacion de buffer y mostrar siempre el frame mas reciente.

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecucion Hito 1

```powershell
$env:PYTHONPATH = "src"
python -m ai_tracking_ptz.apps.milestone1_rtsp_viewer --rtsp-url "rtsp://usuario:password@ip:554/stream" --rtsp-transport tcp
```

Opcionalmente puedes pasar `--width` y `--height` si quieres solicitar una resolucion especifica al stream. Por defecto el visor usa RTSP sobre TCP para reducir cortes y problemas de transporte en Windows.

Si la camara requiere autenticacion, el usuario y la contraseña deben ir dentro de la URL RTSP, por ejemplo:

```powershell
python -m ai_tracking_ptz.apps.milestone1_rtsp_viewer --rtsp-url "rtsp://usuario:password@192.168.1.120:554/stream" --rtsp-transport tcp
```

## Ejecucion Hito 2

Hito 2 integra Ultralytics con el modelo Nano, detecta solo la clase `person`, usa tracking con BoT-SORT y limita la inferencia a 15 FPS como maximo para no castigar VRAM ni GPU mientras haces streaming.

```powershell
$env:PYTHONPATH = "src"
python -m ai_tracking_ptz.apps.milestone2_person_tracking --rtsp-url "rtsp://127.0.0.1:8554/test" --rtsp-transport tcp --model yolov8n.pt --tracker botsort.yaml --imgsz 640 --max-inference-fps 15
```

Tambien puedes validar Hito 2 con un archivo local que contenga personas:

```powershell
$env:PYTHONPATH = "src"
python -m ai_tracking_ptz.apps.milestone2_person_tracking --video-file ".\samples\people.mp4" --loop-video --model yolov8n.pt --tracker botsort.yaml --imgsz 640 --max-inference-fps 15
```

Notas practicas:

- La primera ejecucion descargara `yolov8n.pt` si no existe localmente.
- `BoT-SORT` necesita `lap`, por eso ya queda fijado en `requirements.txt`. Si lo ejecutaste antes de instalarlo, simplemente relanza el comando una vez mas.
- Si usas `--video-file`, el stream no depende de RTSP y es la forma mas rapida de comprobar que aparecen cajas e IDs sobre personas reales.
- Si quieres reducir mas carga, usa un substream RTSP mas bajo y manten `--imgsz 640`.
- Si CUDA no esta disponible, Ultralytics caera a CPU automaticamente.
- En pantalla veras bounding boxes, confianza e ID del tracker para cada persona detectada.

## Ejecucion Hito 3

Hito 3 ahora usa ONVIF PTZ autenticado, que encaja con tu camara si el control requiere usuario y contraseña. El tester por teclado crea una sesion ONVIF, toma el primer perfil de media disponible y manda comandos `ContinuousMove` y `Stop`.

```powershell
$env:PYTHONPATH = "src"
python -m ai_tracking_ptz.apps.milestone3_onvif_keyboard_test --host "192.168.1.120" --port 80 --username "admin" --password "tu_password" --pan-speed 0.5 --tilt-speed 0.5 --zoom-speed 0.3
```

Controles del tester:

- `W` inclinacion arriba
- `S` inclinacion abajo
- `A` paneo izquierda
- `D` paneo derecha
- `Z` zoom out
- `X` zoom in
- `Space` stop
- `Q` salir

Notas practicas:

- Las velocidades ONVIF aqui se expresan en rango `0.0-1.0`.
- Si tu servicio ONVIF no escucha en `80`, ajusta `--port` al puerto real del dispositivo.
- Si `onvif-zeep` no encuentra los WSDL automaticamente en tu entorno, pasa `--wsdl-dir` con la ruta local correspondiente.
- El canal RTSP sigue usando credenciales dentro de la URL; el canal PTZ ONVIF usa `--username` y `--password` como autenticacion separada.
- El tester VISCA anterior se mantiene en el repo como referencia, pero ya no es el camino principal para esta camara.

## PTZ Virtual

Para desarrollar sin hardware real, el repo incluye una PTZ virtual que simula `pan`, `tilt` y `zoom` sobre un video local. No emula ONVIF en red, pero si replica la semantica de movimiento continuo y sirve para afinar la logica de seguimiento y PID antes de conectar la camara real.

```powershell
$env:PYTHONPATH = "src"
python -m ai_tracking_ptz.apps.virtual_ptz_keyboard_test --video-file ".\samples\people.mp4" --loop-video --width 1280 --height 720 --max-zoom 4.0 --pan-speed 0.5 --tilt-speed 0.5 --zoom-speed 0.4
```

Controles del emulador:

- `W` inclinacion arriba
- `S` inclinacion abajo
- `A` paneo izquierda
- `D` paneo derecha
- `Z` zoom out
- `X` zoom in
- `Space` stop
- `Q` salir

La ventana `Virtual PTZ Source` muestra el frame completo con el viewport actual y `Virtual PTZ Output` muestra lo que veria la camara virtual. Esa es la base util para desarrollar el Hito 4 sin depender aun del hardware ONVIF.

## Ejecucion Hito 4

Hito 4 une tracking y control. En esta etapa se ejecuta YOLO sobre la salida de la PTZ virtual, se selecciona como objetivo la persona mas grande en pantalla y un PID calcula velocidades de `pan` y `tilt` para recentrarla. Todavia no aplica la logica final de automatizacion por MIDI; eso queda para Hito 5.

```powershell
$env:PYTHONPATH = "src"
python -m ai_tracking_ptz.apps.milestone4_virtual_pid_tracking --video-file ".\samples\people.mp4" --loop-video --model yolov8n.pt --tracker botsort.yaml --imgsz 640 --max-inference-fps 15
```

Notas practicas:

- La ventana `Milestone 4 - Source` muestra el video fuente y el viewport actual de la PTZ virtual.
- La ventana principal muestra la salida virtual con bounding boxes, ID del tracker, deadzone central y velocidades calculadas por el PID.
- El objetivo actual se elige por area de bounding box. Es una aproximacion temporal para validar el lazo de control antes de la logica automatica definitiva del Hito 5.
- Los parametros `--pid-pan-*`, `--pid-tilt-*`, `--deadzone-x` y `--deadzone-y` estan expuestos para ajuste fino.

## Direccion Hito 5

Hito 5 ya no se planteara con seleccion manual de objetivo. El comportamiento deseado es totalmente automatico y orientado a operacion en vivo con FreeShow mediante comandos MIDI.

Objetivos funcionales del Hito 5:

- Activar o desactivar AI tracking por eventos MIDI, sin interaccion con raton ni teclado.
- Elegir automaticamente el objetivo principal con la menor configuracion manual posible.
- Mantener estabilidad del objetivo para no saltar entre personas cercanas.
- Conservar deadzones, suavidad y seguridad del movimiento antes de pasar a ONVIF real.
- Exportar y validar el modelo en TensorRT (`.engine`) para aprovechar mejor la GPU final sin comprometer OBS o vMix.

Direccion tecnica prevista:

- Un listener MIDI recibira comandos tipo `tracking on`, `tracking off` y opcionalmente `reacquire target`.
- La seleccion del objetivo dejara de depender de clics y pasara a una heuristica automatica: prioridad por persistencia del tracker, tamaño relativo, posicion central y confianza.
- Cuando AI tracking este desactivado por MIDI, la camara dejara de enviar correcciones y permanecera en stop.
- Cuando AI tracking se reactive, el sistema reacquirira automaticamente al mejor candidato visible.
- La ruta de despliegue final del detector no se quedara en `yolov8n.pt`: en el PC definitivo se exportara a TensorRT y se comparara consumo de VRAM, latencia y estabilidad frente a la version PyTorch para elegir el backend final.

Notas de despliegue para TensorRT:

- Esta parte solo tiene sentido cerrarla en el PC final con la RTX 3050, porque el `.engine` depende de la GPU, drivers, CUDA y TensorRT instalados.
- El objetivo no es solo subir FPS, sino reducir latencia y carga de GPU para convivir con OBS o vMix.
- La exportacion a `.engine` debe hacerse al final del flujo, cuando la logica de tracking y control ya este estable.

Esto encaja mejor con un flujo de iglesia automatizado, donde FreeShow actua como orquestador y el operador no tiene que intervenir en cada seguimiento.

## Ejecucion Hito 5

Hito 5 implementa el flujo automatico: estado `tracking on/off` controlado por MIDI, selector automatico de objetivo con memoria de tracker y lazo PID sobre la PTZ virtual. El modelo puede ser `.pt` o `.engine`, asi que este mismo flujo sirve para validar la transicion a TensorRT cuando llegues al PC final.

```powershell
$env:PYTHONPATH = "src"
python -m ai_tracking_ptz.apps.milestone5_midi_auto_tracking --video-file ".\samples\people.mp4" --loop-video --model yolov8n.pt --tracker botsort.yaml --imgsz 640 --max-inference-fps 15 --start-enabled
```

Opcionalmente puedes conectar un puerto MIDI de FreeShow:

```powershell
$env:PYTHONPATH = "src"
python -m ai_tracking_ptz.apps.milestone5_midi_auto_tracking --video-file ".\samples\people.mp4" --loop-video --midi-input-name "LoopMIDI Port" --midi-channel 0 --midi-toggle-note 60 --midi-enable-note 61 --midi-disable-note 62 --midi-reacquire-note 63 --start-enabled
```

Comportamiento del Hito 5:

- Cuando `tracking` esta activado, el selector automatico puntua candidatos por area, cercania al centro, confianza y persistencia del tracker.
- Cuando `tracking` esta desactivado por MIDI, el sistema deja de corregir movimiento y la camara queda en `stop`.
- Un evento MIDI de `reacquire` limpia el lock actual y obliga a buscar el mejor candidato visible otra vez.
- El overlay muestra puerto MIDI, ultimo evento MIDI, target lock actual y backend de modelo cargado.

Notas practicas:

- Si no pasas `--midi-input-name`, el flujo sigue funcionando y el estado queda local al proceso.
- Para usar MIDI real en Windows, normalmente necesitaras un puerto virtual como loopMIDI o el puerto expuesto por FreeShow.
- Este flujo aun usa PTZ virtual como backend de seguridad para afinar heuristicas y PID antes de mover la camara ONVIF real.

## Exportacion TensorRT

Cuando pases al PC final con la RTX 3050, puedes exportar el detector a TensorRT asi:

```powershell
$env:PYTHONPATH = "src"
python -m ai_tracking_ptz.apps.export_tensorrt_engine --model yolov8n.pt --imgsz 640 --device cuda:0 --half --workspace 4
```

Si la exportacion genera `yolov8n.engine`, luego puedes reutilizarlo directamente en Hito 5 cambiando `--model yolov8n.engine`.

## Prueba sin camara

La opcion mas rapida es probar con un RTSP publico. Estos endpoints a veces dejan de responder, asi que sirven para validacion puntual, no para pruebas repetibles.

```powershell
$env:PYTHONPATH = "src"
python -m ai_tracking_ptz.apps.milestone1_rtsp_viewer --rtsp-url "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov" --rtsp-transport tcp
```

Si ese stream no responde, la opcion mas estable es levantar un RTSP local de prueba con MediaMTX y FFmpeg para emitir un patron o un video de prueba hacia `rtsp://127.0.0.1:8554/test`.

Ejemplo con Docker para MediaMTX:

```powershell
docker run --rm -it -p 8554:8554 bluenviron/mediamtx:latest
```

En otra terminal, publica un video o patron de prueba con FFmpeg:

```powershell
ffmpeg -re -f lavfi -i testsrc=size=1280x720:rate=30 -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p -rtsp_transport tcp -f rtsp rtsp://127.0.0.1:8554/test
```

Luego ejecuta el visor contra ese RTSP local:

```powershell
$env:PYTHONPATH = "src"
python -m ai_tracking_ptz.apps.milestone1_rtsp_viewer --rtsp-url "rtsp://127.0.0.1:8554/test" --rtsp-transport tcp
```

Si FFmpeg devuelve `Broken pipe` o `End of file`, casi siempre significa que el servidor RTSP cerro la sesion. Las causas mas comunes aqui son: MediaMTX no estaba realmente corriendo, Docker Desktop no estaba levantado, el contenedor se cerro al cerrar la terminal, o hubo un problema de transporte. Primero valida que MediaMTX siga vivo y luego prueba este comando:

```powershell
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/test
```

Si `ffplay` tampoco abre, el problema esta en el publisher o en MediaMTX, no en OpenCV.# AI-Tracking-PTZ
