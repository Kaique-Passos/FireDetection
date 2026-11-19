import torch.nn as nn
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights


class SqueezeNetImageNet(nn.Module):
    """SqueezeNet 1.1 com os pesos pré-treinados na ImageNet."""

    def __init__(self, classes=2):
        super().__init__()

        # Carrega a arquitetura com os pesos treinados na ImageNet (transfer learning)
        self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)

        # A SqueezeNet não tem camada Linear no final: a classificação é feita
        # por uma convolução 1x1 (512 -> 1000 classes) seguida de média global.
        # Substitui essa convolução por uma com a saída para nossas classes.
        # A ReLU original depois dela é retirada para que a saída possa ter
        # valores negativos, como a Linear final das outras redes.
        in_channels = self.model.classifier[1].in_channels
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels, classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.model(x)
