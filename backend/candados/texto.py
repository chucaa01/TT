import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import os
import re

# Ruta a los modelos descargados de EasyOCR
os.environ['EASYOCR_MODEL_STORAGE_DIR'] = r'C:\Users\Jesus\.EasyOCR\model'

# Inicializar OCR
reader = easyocr.Reader(['es'])

# Cargar modelo YOLO
model = YOLO("runs/detect/train6/weights/best.pt")

# Ruta de imagen
image_path = "C:/Users/Jesus/Documents/GitHub/TT/pruebas/prueba_billete_2.jpg"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen en {image_path}")

# Corrección de errores comunes en OCR
def normalizar_texto(texto):
    texto = texto.upper()
    texto = texto.replace('O', '0')
    texto = texto.replace('o', '0')
    texto = texto.replace('I', '1')
    texto = texto.replace('l', '1')
    texto = re.sub(r'[^A-Z0-9]', '', texto)
    return texto

# Verificación de crecimiento de tamaño de caracteres
def validar_tamanos_en_aumento(roi, texto):
    alturas = []
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(255 - roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])

    for x, y, w, h in bounding_boxes:
        alturas.append(h)

    if len(alturas) < 2:
        return False, alturas

    # Comprobar que la lista de alturas esté en aumento (con cierta tolerancia)
    en_aumento = all(alturas[i] <= alturas[i + 1] + 1 for i in range(len(alturas) - 1))
    return en_aumento, alturas

# Diccionario para guardar resultados
resultados_finales = {}

# Inferencia
results = model.predict(source=image_path, save=False, conf=0.4)

# Extraer texto para candados serie y serie2
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls)
        cls_name = result.names[cls_id]

        if cls_name in ["serie", "serie2"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = img[y1:y2, x1:x2]

            if roi.size == 0:
                print(f"[{cls_name.upper()}] Región vacía.")
                continue

            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            mejor_texto = ""
            mejor_confianza = 0.0
            mejor_roi = roi  # para usar en validación de tamaños

            for angulo in [0, 90, 180, 270]:
                rotada = cv2.rotate(roi_rgb, {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE
                }.get(angulo, None)) if angulo != 0 else roi_rgb

                resultados_ocr = reader.readtext(rotada)
                for (_, texto, conf) in resultados_ocr:
                    texto_limpio = normalizar_texto(texto)
                    if conf > mejor_confianza and len(texto_limpio) >= 6:
                        mejor_texto = texto_limpio
                        mejor_confianza = conf
                        mejor_roi = rotada

            resultados_finales[cls_name] = {
                "texto": mejor_texto,
                "confianza": mejor_confianza,
                "roi": mejor_roi
            }

            if mejor_texto:
                print(f"[{cls_name.upper()}] Texto corregido: {mejor_texto} (confianza: {mejor_confianza:.2f})")
            else:
                print(f"[{cls_name.upper()}] No se detectó texto.")

# Comparar textos
texto_1 = resultados_finales.get("serie", {}).get("texto", "")
texto_2 = resultados_finales.get("serie2", {}).get("texto", "")
conf1 = resultados_finales.get("serie", {}).get("confianza", 0)
conf2 = resultados_finales.get("serie2", {}).get("confianza", 0)

print("\n🔎 Comparando textos:")
if texto_1 == texto_2:
    print("✅ Los textos de 'serie' y 'serie2' son IGUALES.")
else:
    print("⚠️ Los textos de 'serie' y 'serie2' son DIFERENTES.")
    # Validación automática si hay alta similitud y una mayor confianza
    if conf2 > conf1 and len(texto_2) == len(texto_1) and sum(c1 == c2 for c1, c2 in zip(texto_1, texto_2)) >= len(texto_1) - 2:
        print(f"💡 Sugerencia: usar texto de 'serie2' como válido: {texto_2}")

# Verificar aumento de tamaño en 'serie'
if "serie" in resultados_finales and resultados_finales["serie"]["texto"]:
    print("\n🔎 Verificando crecimiento de tamaño de caracteres en 'serie'...")
    roi_original = resultados_finales["serie"]["roi"]
    en_aumento, alturas = validar_tamanos_en_aumento(roi_original, texto_1)
    print(f"Alturas detectadas: {alturas}")
    if en_aumento:
        print("✅ Los caracteres van en aumento de tamaño.")
    else:
        print("⚠️ Los caracteres NO van en aumento de tamaño.")
