import cv2
import numpy as np
from django.core.files.base import ContentFile
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Billete

def detectar_billete(imagen):
    """ Primera detección: Encuentra el billete y lo recorta sin importar pérdida de calidad """

    # Leer la imagen con OpenCV
    image_array = np.asarray(bytearray(imagen.read()), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Convertir a HSV para segmentación basada en color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_range = np.array([0, 0, 150])   # Rango mínimo (puede ajustarse)
    high_range = np.array([180, 80, 255])  # Rango máximo
    mask = cv2.inRange(hsv, low_range, high_range)

    # Aplicar morfología para limpiar la máscara
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

    # Encontrar contornos en la máscara
    contornos, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        return None, None  # No se encontraron contornos

    # Seleccionar el contorno más grande (asumimos que es el billete)
    contorno_billete = max(contornos, key=cv2.contourArea)

    # Verificar si el contorno detectado es lo suficientemente grande
    if cv2.contourArea(contorno_billete) < 5000:
        return None, None  # Demasiado pequeño para ser un billete

    # Obtener rectángulo delimitador del billete
    x, y, w, h = cv2.boundingRect(contorno_billete)

    return img, (x, y, w, h)  # Devolvemos la imagen original y la región detectada

def recortar_billete_original(imagen, region):
    """ Segunda detección: Recortar el billete en la imagen original sin procesar """

    # Leer la imagen original nuevamente sin alteraciones
    image_array = np.asarray(bytearray(imagen.read()), dtype=np.uint8)
    img_original = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Extraer la región detectada en la primera imagen
    x, y, w, h = region
    billete_final = img_original[y:y+h, x:x+w]

    return billete_final

def necesita_mejora(imagen):
    """ Analiza si la imagen necesita mejoras de contraste, nitidez o bordes """

    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # 1️⃣ Evaluar Contraste (Si es muy bajo, aplicar CLAHE)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    contraste = hist.std()  # Desviación estándar del histograma
    if contraste < 50:  # Umbral bajo para contraste
        return "contraste"

    # 2️⃣ Evaluar Desenfoque (Si es muy borroso, aplicar nitidez)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian < 100:  # Umbral para desenfoque
        return "nitidez"

    return None  # Si la imagen ya está bien, no necesita mejoras

def mejorar_imagen(imagen, tipo):
    """ Aplica la mejora correspondiente según el análisis """

    if tipo == "contraste":
        # Aplicar CLAHE para mejorar el contraste sin distorsión
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_mejorado = clahe.apply(l)
        lab_mejorado = cv2.merge((l_mejorado, a, b))
        imagen_mejorada = cv2.cvtColor(lab_mejorado, cv2.COLOR_LAB2BGR)
        return imagen_mejorada

    if tipo == "nitidez":
        # Aplicar un filtro de nitidez para recuperar detalles
        kernel_sharpen = np.array([[-1, -1, -1], 
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
        imagen_mejorada = cv2.filter2D(imagen, -1, kernel_sharpen)
        return imagen_mejorada

    return imagen  # Si no se necesita mejora, devolver la imagen tal cual

@api_view(['POST'])
def subir_imagen(request):
    """ API que recibe una imagen, detecta el billete y mejora si es necesario """

    if 'imagen' not in request.FILES:
        return Response({"error": "No se proporcionó ninguna imagen"}, status=400)

    imagen_original = request.FILES['imagen']

    # Primera fase: Detección y obtención de la región del billete
    _, region = detectar_billete(imagen_original)
    if region is None:
        return Response({"error": "No se pudo detectar el billete"}, status=400)

    # Segunda fase: Recorte en la imagen original para conservar calidad
    imagen_original.seek(0)  # Resetear la lectura del archivo para volver a usarlo
    billete_recortado = recortar_billete_original(imagen_original, region)

    # Analizar si la imagen necesita mejoras
    tipo_mejora = necesita_mejora(billete_recortado)
    if tipo_mejora:
        billete_recortado = mejorar_imagen(billete_recortado, tipo_mejora)

    # Guardar la imagen recortada en memoria
    _, buffer = cv2.imencode('.png', billete_recortado)
    imagen_transformada = ContentFile(buffer.tobytes(), name=imagen_original.name)

    # Guardar la imagen final en el modelo
    billete = Billete(imagen=imagen_transformada)
    billete.save()

    return Response({
        "mensaje": f"Billete recortado y {'mejorado' if tipo_mejora else 'guardado sin cambios'}",
        "billete_id": billete.id  # Devolver el ID del billete guardado
    }, status=201)
