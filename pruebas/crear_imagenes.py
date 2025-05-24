import os
import cv2
import albumentations as A

# Rutas de imágenes y etiquetas
IMAGES_DIR = "C:/Users/Jesus/Documents/GitHub/TT/dataset/images/train"
LABELS_DIR = "C:/Users/Jesus/Documents/GitHub/TT/dataset/labels/train"

# Cargar el último índice
def obtener_ultimo_indice():
    indices = []
    for archivo in os.listdir(IMAGES_DIR):
        if archivo.startswith("billete_") and archivo.endswith(".jpeg"):
            nombre = archivo.split(".")[0].replace("billete_", "")
            if nombre.isdigit():
                indices.append(int(nombre))
    return max(indices) if indices else 0

ultimo_id = obtener_ultimo_indice()

# Transformaciones
transformaciones = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.3),
    A.RandomGamma(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.Rotate(limit=15, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Aumentar imagen y etiquetas
def aumentar_y_guardar(img_path, label_path, nuevo_id):
    imagen = cv2.imread(img_path)
    with open(label_path, 'r') as f:
        contenido = f.readlines()

    bboxes, clases = [], []
    for linea in contenido:
        partes = linea.strip().split()
        if len(partes) == 5:
            cls, *bbox = map(float, partes)
            clases.append(int(cls))
            bboxes.append(bbox)

    if not bboxes:
        return

    aumentado = transformaciones(image=imagen, bboxes=bboxes, class_labels=clases)
    nueva_img = aumentado['image']
    nuevas_bboxes = aumentado['bboxes']
    nuevas_clases = aumentado['class_labels']

    nuevo_nombre = f"billete_{nuevo_id}"
    cv2.imwrite(os.path.join(IMAGES_DIR, f"{nuevo_nombre}.jpeg"), nueva_img)

    with open(os.path.join(LABELS_DIR, f"{nuevo_nombre}.txt"), "w") as f:
        for cls, bbox in zip(nuevas_clases, nuevas_bboxes):
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")

# Ejecutar aumentos
conteo = 0
for archivo in os.listdir(IMAGES_DIR):
    if archivo.endswith(".jpeg"):
        nombre = os.path.splitext(archivo)[0]
        etiqueta = nombre + ".txt"
        img_path = os.path.join(IMAGES_DIR, archivo)
        label_path = os.path.join(LABELS_DIR, etiqueta)

        if os.path.exists(label_path):
            for _ in range(3):  # Aumentar 3 veces por imagen
                ultimo_id += 1
                aumentar_y_guardar(img_path, label_path, ultimo_id)
                conteo += 1

print(f"✅ Aumentos realizados: {conteo} imágenes generadas.")
