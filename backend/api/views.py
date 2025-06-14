import os
import cv2
import base64
import uuid
import numpy as np
import threading
from io import BytesIO
from django.core.files.base import ContentFile
from django.conf import settings
from rest_framework.response import Response
from rest_framework.decorators import api_view
from azure.storage.fileshare import ShareFileClient
from .models import Billete
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, "runs", "detect", "train11", "weights", "best.pt")
modelo_yolo = YOLO(RUTA_MODELO)

umbrales = {
    "benito": 0.96,
    "banco": 0.94,
    "carruaje": 0.93,
    "serie": 0.93,
    "numero": 0.93,
    "marca": 0.94,
    "texto": 0.91,
    "patron": 0.90,
    "serie2": 0.86,
    "hilo": 0.84,
}

def guardar_imagen_en_azure(imagen_np, nombre_archivo):
    try:
        _, buffer = cv2.imencode('.png', imagen_np)
        content = buffer.tobytes()

        file_client = ShareFileClient(
            account_url=f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.file.core.windows.net",
            share_name=settings.AZURE_FILE_SHARE_NAME,
            file_path=f"{settings.AZURE_FILE_DIRECTORY}/{nombre_archivo}",
            credential=settings.AZURE_STORAGE_ACCOUNT_KEY,
        )

        file_client.create_file(size=len(content))
        file_client.upload_file(content)
        print(f"[✅] Imagen subida correctamente a Azure: {nombre_archivo}")
    except Exception as e:
        print("❌ Error al subir imagen a Azure:")
        import traceback
        traceback.print_exc()

def detectar_billete(imagen):
    image_array = np.asarray(bytearray(imagen.read()), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 80, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, None

    contorno_billete = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(contorno_billete) < 5000:
        return None, None

    x, y, w, h = cv2.boundingRect(contorno_billete)
    return img, (x, y, w, h)

def recortar_billete(imagen, region):
    imagen.seek(0)
    image_array = np.asarray(bytearray(imagen.read()), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    x, y, w, h = region
    return img[y:y+h, x:x+w]

def necesita_mejora(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    contraste = cv2.calcHist([gray], [0], None, [256], [0, 256]).std()
    if contraste < 50:
        return "contraste"
    nitidez = cv2.Laplacian(gray, cv2.CV_64F).var()
    if nitidez < 100:
        return "nitidez"
    return None

def mejorar_imagen(imagen, tipo):
    if tipo == "contraste":
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    elif tipo == "nitidez":
        return cv2.filter2D(imagen, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    return imagen

def clasificar_confianza(candado, confianza):
    umbral = umbrales.get(candado, 0.9)
    if confianza >= umbral:
        return "verdadero"
    elif confianza >= umbral - 0.03:
        return "revisar"
    return "falso"

@api_view(['POST'])
def subir_imagen(request):
    imagen_subida = request.FILES.get('imagen')
    if not imagen_subida:
        return Response({"error": "No se proporcionó ninguna imagen"}, status=400)

    imagen_subida.seek(0)
    _, region = detectar_billete(imagen_subida)
    if not region:
        return Response({"error": "No se pudo detectar el billete"}, status=400)

    imagen_subida.seek(0)
    billete = recortar_billete(imagen_subida, region)

    mejora = necesita_mejora(billete)
    if mejora:
        billete = mejorar_imagen(billete, mejora)

    _, buffer_png = cv2.imencode('.png', billete)
    nombre_archivo = f"{uuid.uuid4()}.png"
    billete_modelo = Billete(imagen=ContentFile(buffer_png.tobytes(), name=nombre_archivo))
    billete_modelo.save()

    _, buffer_jpg = cv2.imencode('.jpg', billete)
    imagen_base64 = base64.b64encode(buffer_jpg).decode('utf-8')

    resultados = []
    prediccion = modelo_yolo.predict(billete, conf=0.4, save=False)[0]
    for box in prediccion.boxes:
        cls = int(box.cls)
        nombre = prediccion.names[cls]
        conf = float(box.conf[0])
        coords = list(map(int, box.xyxy[0]))
        resultados.append({
            "candado": nombre,
            "confianza": round(conf, 3),
            "coordenadas": coords,
            "clasificacion": clasificar_confianza(nombre, conf)
        })

    # Subir imagen a Azure en segundo plano
    threading.Thread(target=guardar_imagen_en_azure, args=(billete, nombre_archivo)).start()

    return Response({
        "mensaje": f"Billete {'mejorado y ' if mejora else ''}guardado exitosamente.",
        "billete_id": billete_modelo.id,
        "imagen_base64": imagen_base64,
        "nombre_archivo": nombre_archivo,
        "detecciones": resultados
    }, status=201)
