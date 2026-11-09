import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3(nn.Module):

    def __init__(self, classes=2, pretrained=False):
        super().__init__()

        # pretrained=True carrega os pesos treinados na ImageNet (transfer learning);
        # pretrained=False inicializa a arquitetura com pesos aleatórios
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = mobilenet_v3_small(weights=weights)

        # Substitui a camada final para nossas classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, classes)

    def forward(self, x):
        return self.model(x)


class MobileNetV3ImageNet(MobileNetV3):
    """MobileNetV3 Small com os pesos pré-treinados na ImageNet."""

    def __init__(self, classes=2):
        super().__init__(classes=classes, pretrained=True)
