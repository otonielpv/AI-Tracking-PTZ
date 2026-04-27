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

Hito 4 une tracking y control. En esta etapa se ejecuta YOLO sobre la salida de la PTZ virtual, se selecciona como objetivo la persona mas grande en pantalla y un PID calcula velocidades de `pan` y `tilt` para recentrarla. Todavia no aplica selector manual ni logica de iglesia; eso queda para Hito 5.

```powershell
$env:PYTHONPATH = "src"
python -m ai_tracking_ptz.apps.milestone4_virtual_pid_tracking --video-file ".\samples\people.mp4" --loop-video --model yolov8n.pt --tracker botsort.yaml --imgsz 640 --max-inference-fps 15
```

Notas practicas:

- La ventana `Milestone 4 - Source` muestra el video fuente y el viewport actual de la PTZ virtual.
- La ventana principal muestra la salida virtual con bounding boxes, ID del tracker, deadzone central y velocidades calculadas por el PID.
- El objetivo actual se elige por area de bounding box. Es una aproximacion temporal para validar el lazo de control antes del selector manual del Hito 5.
- Los parametros `--pid-pan-*`, `--pid-tilt-*`, `--deadzone-x` y `--deadzone-y` estan expuestos para ajuste fino.

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
