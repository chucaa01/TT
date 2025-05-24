import os
import shutil
import subprocess

# Rutas absolutas
base_path = "C:/Users/Jesus/Documents/GitHub/TT"
runs_path = os.path.join(base_path, "runs")
cache_train = os.path.join(base_path, "dataset", "imagenes", "train.cache")
cache_val = os.path.join(base_path, "dataset", "imagenes", "val.cache")
dataset_yaml = os.path.join(base_path, "dataset", "dataset_config.yaml")

# 1. Eliminar carpeta runs (donde se guarda el entrenamiento)
if os.path.exists(runs_path):
    shutil.rmtree(runs_path)
    print("🧹 Carpeta 'runs' eliminada.")

# 2. Eliminar archivos .cache
for cache_file in [cache_train, cache_val]:
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"🧹 Cache eliminado: {cache_file}")

# 3. Eliminar pesos anteriores .pt si los hubiera
for root, _, files in os.walk(base_path):
    for f in files:
        if f.endswith(".pt") and "yolov8" in f:
            try:
                os.remove(os.path.join(root, f))
                print(f"🧹 Pesos eliminados: {f}")
            except:
                pass

# 4. Ejecutar entrenamiento desde cero
print("\n🚀 Iniciando entrenamiento desde cero...")
command = [
    "yolo",
    "detect",
    "train",
    "model=yolov8n.pt",
    f"data={dataset_yaml}",
    "epochs=100",
    "imgsz=640",
    "batch=16"
]

subprocess.run(command, shell=True)
