import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18ImageNet(nn.Module):
    """ResNet-18 com os pesos pré-treinados na ImageNet."""

    def __init__(self, classes=2):
        super().__init__()

        # Carrega a arquitetura com os pesos treinados na ImageNet (transfer learning)
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Substitui a camada final para nossas classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, classes)

    def forward(self, x):
        return self.model(x)
