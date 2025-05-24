import os
import shutil
import random

# Rutas base
base_path = r"C:/Users/Jesus/Documents/GitHub/TT/dataset"
images_path = os.path.join(base_path, "imagenes")
labels_path = os.path.join(base_path, "labels")

# Crear carpetas de salida
for split in ["train", "val"]:
    os.makedirs(os.path.join(images_path, split), exist_ok=True)
    os.makedirs(os.path.join(labels_path, split), exist_ok=True)

# Obtener lista de imágenes
all_images = [f for f in os.listdir(images_path) if f.endswith((".jpg", ".jpeg", ".png"))]

# Mezclar aleatoriamente
random.shuffle(all_images)

# Separar en 80%/20%
split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# Función para mover imágenes y labels
def mover(imagenes, tipo):
    for img in imagenes:
        nombre_base, _ = os.path.splitext(img)
        label_file = f"{nombre_base}.txt"

        # Rutas origen
        img_src = os.path.join(images_path, img)
        lbl_src = os.path.join(labels_path, label_file)

        # Rutas destino
        img_dst = os.path.join(images_path, tipo, img)
        lbl_dst = os.path.join(labels_path, tipo, label_file)

        # Mover
        shutil.move(img_src, img_dst)
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, lbl_dst)

# Mover archivos
mover(train_images, "train")
mover(val_images, "val")

print("✅ Dataset dividido correctamente en train/val.")
