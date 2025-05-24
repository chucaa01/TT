import os

val_img_dir = r"C:/Users/Jesus/Documents/GitHub/TT/dataset/imagenes/val"
val_label_dir = r"C:/Users/Jesus/Documents/GitHub/TT/dataset/labels/val"

missing_labels = []

for img_file in os.listdir(val_img_dir):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(val_label_dir, base_name + '.txt')
        if not os.path.exists(label_path):
            missing_labels.append(label_path)
            # Crear archivo vacío si no existe
            open(label_path, 'w').close()

if missing_labels:
    print("🚫 Se crearon los siguientes archivos .txt que faltaban:")
    for f in missing_labels:
        print("  -", f)
else:
    print("✅ Todos los archivos .txt están presentes en labels/val.")
