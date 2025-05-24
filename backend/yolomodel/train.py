import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import YOLO
from dataset import YOLODataset

# ==== Configuraciones ====
IMAGE_SIZE = 224
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 20
S = 7  # Grid size
B = 2  # Bounding boxes
C = 11 # Clases (candados)

# ==== Obtener ruta base ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
image_dir = os.path.join(BASE_DIR, "dataset", "imagenes")
label_dir = os.path.join(BASE_DIR, "dataset", "labels")

# ==== Transformaciones ====
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ==== Dataset y DataLoader ====
train_dataset = YOLODataset(
    image_dir=image_dir,
    label_dir=label_dir,
    S=S, B=B, C=C,
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==== GPU info ====
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# ==== Modelo, optimizador y pérdida ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(grid_size=S, num_bboxes=B, num_classes=C).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()  # Puedes cambiarla más adelante

# ==== Entrenamiento ====
for epoch in range(EPOCHS):
    print(f"🌀 Época {epoch+1}/{EPOCHS}")
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        preds = model(x)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"  [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "modelo_yolo_basico.pth")
print("✅ Modelo guardado como modelo_yolo_basico.pth")

