import torch.nn as nn
import timm


class MobileNetV4ImageNet(nn.Module):
    """MobileNetV4 Conv Small (Qin et al., ECCV 2024) com os pesos pré-treinados na ImageNet.

    Não existe no torchvision: vem da biblioteca timm (pip install timm).
    Os pesos 'e2400_r224_in1k' foram treinados na ImageNet-1k com imagens 224x224."""

    def __init__(self, classes=2):
        super().__init__()

        # Carrega a arquitetura com os pesos treinados na ImageNet (transfer learning).
        # num_classes troca a camada final por uma nova, com a saída para nossas classes.
        self.model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k',
                                       pretrained=True, num_classes=classes)

    def forward(self, x):
        return self.model(x)
