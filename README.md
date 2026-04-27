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
