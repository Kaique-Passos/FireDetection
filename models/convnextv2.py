import torch.nn as nn
import timm


class ConvNeXtV2AttoImageNet(nn.Module):
    """ConvNeXt V2 Atto (Woo et al., CVPR 2023) com os pesos pré-treinados na ImageNet.

    Não existe no torchvision: vem da biblioteca timm (pip install timm).
    Os pesos 'fcmae_ft_in1k' foram pré-treinados com o FCMAE (autoencoder
    mascarado) e depois ajustados na ImageNet-1k."""

    def __init__(self, classes=2):
        super().__init__()

        # Carrega a arquitetura com os pesos treinados na ImageNet (transfer learning).
        # num_classes troca a camada final por uma nova, com a saída para nossas classes.
        self.model = timm.create_model('convnextv2_atto.fcmae_ft_in1k',
                                       pretrained=True, num_classes=classes)

    def forward(self, x):
        return self.model(x)
