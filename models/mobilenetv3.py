import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class MobileNetV3(nn.Module):

    def __init__(self, classes=2):
        super().__init__()

        # Inicializa a arquitetura sem pesos pré-treinados
        self.model = mobilenet_v3_small(weights=None)

        # Substitui a camada final para nossas classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, classes)

    def forward(self, x):
        return self.model(x)