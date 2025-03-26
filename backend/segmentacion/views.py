import cv2
import numpy as np
import os
from django.core.files.base import ContentFile
from django.conf import settings
from rest_framework.response import Response
from rest_framework.decorators import api_view
from api.models import Billete  # Importar desde la app principal

# Dimensiones de referencia (imagen de ejemplo: 1572x673 píxeles)
REF_WIDTH = 1572
REF_HEIGHT = 673

# Coordenadas de referencia de los candados en la imagen de 1572x673
# Formato: (x1_ref, y1_ref, x2_ref, y2_ref)
CANDADOS_REF = {
    "Banco de México":         (427, 17, 807, 129),
    "Benito Juárez":           (811, 63, 1163, 489),
    "Hilo de seguridad":       (305, 51, 361, 591),
    "500 multicolor":          (1139, 33, 1457, 177),
    "Número de serie creciente": (441, 515, 797, 585),
    "Otro número de serie":    (1427, 219, 1517, 473),
    "Leyenda Benito Juárez":   (829, 489, 1165, 581)
}

def segmentar_billete(billete_path):
    """Carga la imagen del billete, obtiene su tamaño real y dibuja dinámicamente las marcas de segmentación."""
    # Cargar la imagen del billete ya recortado
    billete = cv2.imread(billete_path)
    if billete is None:
        return None

    # Obtener dimensiones reales de la imagen recibida
    actual_height, actual_width, _ = billete.shape

    # Calcular factores de escala respecto a la imagen de referencia
    scale_x = actual_width / REF_WIDTH
    scale_y = actual_height / REF_HEIGHT

    # Copiar la imagen para dibujar sobre ella
    billete_segmentado = billete.copy()

    # Para cada candado, calcular las coordenadas escaladas y dibujar el rectángulo
    for nombre, (x1_ref, y1_ref, x2_ref, y2_ref) in CANDADOS_REF.items():
        x1 = int(x1_ref * scale_x)
        y1 = int(y1_ref * scale_y)
        x2 = int(x2_ref * scale_x)
        y2 = int(y2_ref * scale_y)
        cv2.rectangle(billete_segmentado, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Rectángulo rojo
        cv2.putText(billete_segmentado, nombre, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Guardar la imagen segmentada en la carpeta 'carpeta_segmentados'
    carpeta_segmentados = os.path.join(settings.MEDIA_ROOT, 'carpeta_segmentados')
    os.makedirs(carpeta_segmentados, exist_ok=True)

    # Aquí usamos un nombre fijo; en producción, podrías generar un nombre único
    ruta_guardado = os.path.join(carpeta_segmentados, "billete_segmentado.png")
    cv2.imwrite(ruta_guardado, billete_segmentado)

    return ruta_guardado

@api_view(['POST'])
def segmentar_imagen(request):
    """API que recibe el ID de un billete procesado y genera la imagen segmentada con marcas dinámicas."""
    billete_id = request.data.get("billete_id")
    if not billete_id:
        return Response({"error": "Se requiere el ID del billete"}, status=400)

    try:
        billete = Billete.objects.get(id=billete_id)
    except Billete.DoesNotExist:
        return Response({"error": "Billete no encontrado"}, status=404)

    # Obtener la ruta de la imagen del billete procesado
    billete_path = billete.imagen.path

    # Generar la imagen segmentada y guardarla en 'carpeta_segmentados'
    ruta_segmentada = segmentar_billete(billete_path)
    if ruta_segmentada is None:
        return Response({"error": "No se pudo procesar la segmentación"}, status=500)

    return Response({
        "mensaje": "Imagen segmentada y guardada correctamente",
        "ruta_segmentada": ruta_segmentada
    }, status=201)
