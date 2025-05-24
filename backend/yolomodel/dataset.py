import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, S=7, B=2, C=11, transform=None):
        """
        Dataset personalizado para leer imágenes y etiquetas en formato YOLO
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S  # Grid size
        self.B = B  # Bounding boxes por celda
        self.C = C  # Clases

        self.images = [file for file in os.listdir(image_dir) if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Obtener el nombre de la imagen y cargarla
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")

        # Obtener el archivo de etiquetas .txt correspondiente
        label_path = os.path.join(self.label_dir, os.path.splitext(img_filename)[0] + ".txt")

        # Inicializar matriz target con ceros [S, S, B*5 + C]
        label_matrix = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    class_label, x, y, w, h = map(float, line.strip().split())

                    i = int(self.S * y)  # Fila
                    j = int(self.S * x)  # Columna

                    x_cell = self.S * x - j
                    y_cell = self.S * y - i
                    w_cell = self.S * w
                    h_cell = self.S * h

                    if label_matrix[i, j, 4] == 0:
                        label_matrix[i, j, 0:5] = torch.tensor([x_cell, y_cell, w_cell, h_cell, 1])
                        label_matrix[i, j, 5 + int(class_label)] = 1

        if self.transform:
            image = self.transform(image)

        return image, label_matrix
