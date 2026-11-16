import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights


class ShuffleNetV2ImageNet(nn.Module):
    """ShuffleNetV2 x1.0 com os pesos pré-treinados na ImageNet."""

    def __init__(self, classes=2):
        super().__init__()

        # Carrega a arquitetura com os pesos treinados na ImageNet (transfer learning)
        self.model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)

        # Substitui a camada final para nossas classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, classes)

    def forward(self, x):
        return self.model(x)
