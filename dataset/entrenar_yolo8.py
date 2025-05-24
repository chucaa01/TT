from ultralytics import YOLO

# Cargar modelo preentrenado pequeño
modelo = YOLO('yolov8n.pt')

# Entrenar con tu dataset
modelo.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,
    name='billetes_yolo8'
)
