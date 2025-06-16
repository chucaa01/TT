import os
import json
import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# Rutas absolutas
INPUT_FOLDER = r"C:\Users\Jesus\Downloads\fotos\Fotos_billetes\billetesss"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "Fotos_billetes_procesadas")
API_URL = "http://127.0.0.1:8000/api/subir-imagen/"

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cargar fuente para texto
try:
    font = ImageFont.truetype("arial.ttf", 50)
except IOError:
    font = ImageFont.load_default()

# Iterar sobre todas las imágenes en la carpeta
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        filepath = os.path.join(INPUT_FOLDER, filename)
        print(f"📤 Enviando {filename}...")

        # Enviar imagen a la API
        with open(filepath, "rb") as f:
            response = requests.post(API_URL, files={"imagen": f})
        
        if response.status_code != 201:
            print(f"❌ Error al procesar {filename}: {response.status_code}")
            continue

        # Decodificar la respuesta JSON
        data = response.json()
        image_data = base64.b64decode(data['imagen_base64'])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Dibujar candados detectados
        for d in data['detecciones']:
            x1, y1, x2, y2 = d["coordenadas"]
            confianza = d["confianza"] * 100
            etiqueta = f"{d['candado']} ({confianza:.1f}%)"
            color = "green" if d["clasificacion"] == "verdadero" else "orange"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, max(y1 - 20, 0)), etiqueta, fill=color, font=font)

        # Guardar imagen procesada
        output_path = os.path.join(OUTPUT_FOLDER, f"procesado_{filename}")
        image.save(output_path)
        print(f"✅ Guardado: {output_path}")
