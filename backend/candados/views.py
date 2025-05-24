import os
import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from ultralytics import YOLO
import easyocr

# Configuración OCR y modelo
os.environ['EASYOCR_MODEL_STORAGE_DIR'] = r'C:\Users\Jesus\.EasyOCR\model'
reader = easyocr.Reader(['es'])
model = YOLO("runs/detect/train7/weights/best.pt")

def clasificar_confianza(conf):
    if conf >= 0.9:
        return "verdadero"
    elif conf >= 0.6:
        return "revisar"
    else:
        return "falso"

@csrf_exempt
def subir_imagen(request):
    if request.method == 'POST' and request.FILES.get('imagen'):
        imagen = request.FILES['imagen']
        ruta = default_storage.save(f'temp/{imagen.name}', imagen)
        ruta_completa = os.path.join(default_storage.location, ruta)

        img = cv2.imread(ruta_completa)
        if img is None:
            return JsonResponse({"error": "No se pudo cargar la imagen"}, status=400)

        results = model.predict(source=ruta_completa, save=False, conf=0.4)
        candados_info = {}

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                cls_name = result.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                clasificacion = clasificar_confianza(conf)
                region = img[y1:y2, x1:x2]
                texto_detectado = None

                if cls_name in ["serie", "serie2"]:
                    roi_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                    resultados_ocr = reader.readtext(roi_rgb)
                    if resultados_ocr:
                        texto_detectado = resultados_ocr[0][1]

                candados_info[cls_name] = {
                    "confianza": round(conf, 2),
                    "coordenadas": [x1, y1, x2, y2],
                    "clasificacion": clasificacion,
                    "texto": texto_detectado
                }

        return JsonResponse(candados_info)

    return JsonResponse({"error": "Debe enviar una imagen por POST"}, status=400)
