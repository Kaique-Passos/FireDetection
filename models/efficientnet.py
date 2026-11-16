import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0ImageNet(nn.Module):
    """EfficientNet-B0 com os pesos pré-treinados na ImageNet."""

    def __init__(self, classes=2):
        super().__init__()

        # Carrega a arquitetura com os pesos treinados na ImageNet (transfer learning)
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Substitui a camada final para nossas classes
        # (o classificador é Sequential(Dropout(0.2), Linear(1280, 1000)))
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, classes)

    def forward(self, x):
        return self.model(x)
