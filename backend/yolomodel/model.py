import torch
import torch.nn as nn

class YOLO(nn.Module):
    def __init__(self, grid_size=7, num_bboxes=2, num_classes=10):
        super(YOLO, self).__init__()

        self.S = grid_size        # Número de celdas en la rejilla (SxS)
        self.B = num_bboxes       # Número de bounding boxes por celda
        self.C = num_classes      # Número de clases

        # Red convolucional
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),  # (entrada RGB)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 224x224 → 112x112

            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112 → 56x56

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56 → 28x28

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 → 14x14

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14 → 7x7
        )

        # Capa completamente conectada (flatten → salida final)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*256, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))  # salida final
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)  # reshape final
        return x
