import torch
import cv2
import numpy as np
import os
from torchvision import transforms
from model import YOLO

# Clases
CLASSES = [
    "banco", "hilo", "serie", "serie2", "numero",
    "texto", "benito", "patron", "marca", "carruaje", "billete"
]

# Transformación para preprocesar
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === FUNCIÓN PARA HACER PREDICCIÓN ===
def predecir(ruta_imagen):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar modelo
    model = YOLO(grid_size=7, num_bboxes=2, num_classes=11).to(device)
    model.load_state_dict(torch.load("modelo_yolo_basico.pth", map_location=device))
    model.eval()

    # Leer imagen original
    original = cv2.imread(ruta_imagen)
    if original is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")
    
    img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # Inferencia
    with torch.no_grad():
        pred = model(img_tensor)
        pred = pred.squeeze(0).cpu().numpy()

    resultados = []
    h_img, w_img = original.shape[:2]
    cell_size_x = w_img / 7
    cell_size_y = h_img / 7

    for i in range(7):
        for j in range(7):
            celda = pred[i, j]
            for b in range(2):  # 2 cajas por celda
                offset = b * 5
                conf = celda[offset + 4]
                if conf > 0.5:  # Umbral de confianza
                    cx = celda[offset + 0] * cell_size_x + j * cell_size_x
                    cy = celda[offset + 1] * cell_size_y + i * cell_size_y
                    w = celda[offset + 2] * w_img
                    h = celda[offset + 3] * h_img
                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)
                    clase_idx = np.argmax(celda[10:])
                    clase = CLASSES[clase_idx]
                    resultados.append((x1, y1, x2, y2, clase, conf))
                    cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(original, f"{clase} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Crear carpeta de salida si no existe
    carpeta_salida = "predicciones"
    os.makedirs(carpeta_salida, exist_ok=True)

    salida_path = os.path.join(carpeta_salida, "resultado_prediccion.jpg")
    cv2.imwrite(salida_path, original)

    return salida_path, resultados

# === PRUEBA DIRECTA ===
if __name__ == "__main__":
    ruta_img = "dataset/imagenes/billete_1.jpeg"
    salida, resultados = predecir(ruta_img)
    print(f"✅ Resultados guardados en: {salida}")
    print("📦 Resultados:")
    for r in resultados:
        print(r)
