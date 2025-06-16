import cv2
import numpy as np
import os
from django.core.files.base import ContentFile
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Billete
from ultralytics import YOLO
import base64
from io import BytesIO

# Ruta al modelo YOLO entrenado
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Sube desde api/ a TT/
RUTA_MODELO = os.path.join(BASE_DIR, "runs", "detect", "train11", "weights", "best.pt")
modelo_yolo = YOLO(RUTA_MODELO)  # ✅ Cargar el modelo aquí

def detectar_billete(imagen):
    image_array = np.asarray(bytearray(imagen.read()), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_range = np.array([0, 0, 150])
    high_range = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, low_range, high_range)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
    contornos, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, None
    contorno_billete = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(contorno_billete) < 5000:
        return None, None
    x, y, w, h = cv2.boundingRect(contorno_billete)
    return img, (x, y, w, h)

def recortar_billete_original(imagen, region):
    image_array = np.asarray(bytearray(imagen.read()), dtype=np.uint8)
    img_original = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    x, y, w, h = region
    return img_original[y:y+h, x:x+w]

def necesita_mejora(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    contraste = hist.std()
    if contraste < 50:
        return "contraste"
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian < 100:
        return "nitidez"
    return None

def mejorar_imagen(imagen, tipo):
    if tipo == "contraste":
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_mejorado = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_mejorado, a, b)), cv2.COLOR_LAB2BGR)
    if tipo == "nitidez":
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(imagen, -1, kernel_sharpen)
    return imagen

def clasificar_confianza(conf):
    if conf > 0.9:
        return "verdadero"
    elif conf > 0.6:
        return "revisar"
    else:
        return "falso"

@api_view(['POST'])
def subir_imagen(request):
    if 'imagen' not in request.FILES:
        return Response({"error": "No se proporcionó ninguna imagen"}, status=400)

    imagen_original = request.FILES['imagen']
    imagen_original.seek(0)
    _, region = detectar_billete(imagen_original)

    if region is None:
        return Response({"error": "No se pudo detectar el billete"}, status=400)

    imagen_original.seek(0)
    billete_recortado = recortar_billete_original(imagen_original, region)

    tipo_mejora = necesita_mejora(billete_recortado)
    if tipo_mejora:
        billete_recortado = mejorar_imagen(billete_recortado, tipo_mejora)

    _, buffer = cv2.imencode('.png', billete_recortado)
    imagen_transformada = ContentFile(buffer.tobytes(), name=imagen_original.name)
    billete = Billete(imagen=imagen_transformada)
    billete.save()

    # Convertir imagen a base64 para enviar como string
    _, buffer = cv2.imencode('.jpg', billete_recortado)
    imagen_base64 = base64.b64encode(buffer).decode('utf-8')

    # Ejecutar inferencia con YOLO
    detecciones = modelo_yolo.predict(billete_recortado, save=False, conf=0.4)[0]
    resultados = []

    for box in detecciones.boxes:
        cls = int(box.cls)
        nombre = detecciones.names[cls]
        conf = float(box.conf[0])
        coords = list(map(int, box.xyxy[0]))
        resultado = {
            "candado": nombre,
            "confianza": round(conf, 3),
            "coordenadas": coords,
            "clasificacion": clasificar_confianza(conf)
        }
        resultados.append(resultado)

    return Response({
        "mensaje": f"Billete {'mejorado y ' if tipo_mejora else ''}guardado exitosamente.",
        "billete_id": billete.id,
        "imagen_base64": imagen_base64,
        "detecciones": resultados
    }, status=201)
