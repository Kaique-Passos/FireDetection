import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights


class ShuffleNetV2(nn.Module):

    def __init__(self, classes=2, pretrained=False):
        super().__init__()

        # pretrained=True carrega os pesos treinados na ImageNet (transfer learning);
        # pretrained=False inicializa a arquitetura com pesos aleatórios
        weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = shufflenet_v2_x1_0(weights=weights)

        # Substitui a camada final para nossas classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, classes)

    def forward(self, x):
        return self.model(x)


class ShuffleNetV2ImageNet(ShuffleNetV2):
    """ShuffleNetV2 x1.0 com os pesos pré-treinados na ImageNet."""

    def __init__(self, classes=2):
        super().__init__(classes=classes, pretrained=True)
