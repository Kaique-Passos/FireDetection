from torch.nn import CrossEntropyLoss
from torch import optim
from torchvision import transforms
from .nadam_optim import Nadam

models_conf = {
    'firenet': {
        'img_dims': (64, 64),
        'model_name': 'FireNet',
        'model_path': 'model_firenet.pth',
        'class_name': 'FireNet',
        'module_name': 'models.firenet_pt',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {'eps': 1e-6},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((64, 64)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((64, 64)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((64, 64)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': None,
        'scheduler_params': {}
    }
}

models_conf['octfiresnet'] = {
        'img_dims': (96, 96),
        'model_name': 'OctFiResNet',
        'model_path': 'model_octfiresnet.pth',
        'class_name': 'OctFiResNet',
        'module_name': 'models.octfiresnet',
        'criterion': CrossEntropyLoss(),
        'optimizer': Nadam,
        'optimizer_params': {'lr': 1e-4, 'eps': 1e-7},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((96, 96)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((96, 96)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((96, 96)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': None,
        'scheduler_params': {}
    }

models_conf['resnet'] = {
        'img_dims': (224, 224),
        'model_name': 'ResNet50',
        'model_path': 'model_resnet.pth',
        'class_name': 'resnet_sharma',
        'module_name': 'models.resnet',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((224, 224)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((224, 224)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((224, 224)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': None,
        'scheduler_params': {}
    }

models_conf['kutralnet'] = {
        'img_dims': (84, 84),
        'model_name': 'KutralNet',
        'model_path': 'model_kutralnet.pth',
        'class_name': 'KutralNet',
        'module_name': 'models.kutralnet',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': optim.lr_scheduler.StepLR,
        'scheduler_params': { 'step_size':85 }
    }

models_conf['kutralnetoct'] = {
        'img_dims': (84, 84),
        'model_name': 'KutralNet Octave',
        'model_path': 'model_kutralnetoct.pth',
        'class_name': 'KutralNetOct',
        'module_name': 'models.kutralnetoct',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': None,
        'scheduler_params': {}
    }

models_conf['kutralnet_mobile'] =  {
        'img_dims': (84, 84),
        'model_name': 'KutralNet Mobile',
        'model_path': 'model_kutralnet_mobile.pth',
        'class_name': 'KutralNetMobile',
        'module_name': 'models.kutralnet_mobile',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': None,
        'scheduler_params': {}
    }

models_conf['kutralnet_mobileoct'] = {
        'img_dims': (84, 84),
        'model_name': 'KutralNet Mobile Octave',
        'model_path': 'model_kutralnet_mobileoct.pth',
        'class_name': 'KutralNetMobileOct',
        'module_name': 'models.kutralnet_mobileoct',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': None,
        'scheduler_params': {}
    }

# MobileNetV3 Small com pesos pré-treinados na ImageNet (transfer learning).
# As imagens são normalizadas com a média e o desvio padrão da ImageNet, como
# no pré-treino, e a taxa de aprendizado é menor (1e-4) para fazer o
# fine-tuning sem destruir os pesos aprendidos.
imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

models_conf['mobilenetv3_imagenet'] = {
    'img_dims': (224, 224),
    'model_name': 'MobileNetV3 Small (ImageNet)',
    'model_path': 'model_mobilenetv3_imagenet.pth',
    'class_name': 'MobileNetV3ImageNet',
    'module_name': 'models.mobilenetv3',
    'criterion': CrossEntropyLoss(),
    'optimizer': optim.Adam,
    'optimizer_params': {'lr': 1e-4},

    'preprocess_train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'preprocess_val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'preprocess_test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'scheduler': None,
    'scheduler_params': {}
}

# ShuffleNetV2 x1.0 com pesos da ImageNet, no mesmo esquema da
# 'mobilenetv3_imagenet' (normalização da ImageNet e lr 1e-4)
models_conf['shufflenetv2_imagenet'] = {
    'img_dims': (224, 224),
    'model_name': 'ShuffleNetV2 x1.0 (ImageNet)',
    'model_path': 'model_shufflenetv2_imagenet.pth',
    'class_name': 'ShuffleNetV2ImageNet',
    'module_name': 'models.shufflenetv2',
    'criterion': CrossEntropyLoss(),
    'optimizer': optim.Adam,
    'optimizer_params': {'lr': 1e-4},

    'preprocess_train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'preprocess_val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'preprocess_test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'scheduler': None,
    'scheduler_params': {}
}

# EfficientNet-B0 com pesos da ImageNet, no mesmo esquema da
# 'mobilenetv3_imagenet' (normalização da ImageNet e lr 1e-4)
models_conf['efficientnetb0_imagenet'] = {
    'img_dims': (224, 224),
    'model_name': 'EfficientNet-B0 (ImageNet)',
    'model_path': 'model_efficientnetb0_imagenet.pth',
    'class_name': 'EfficientNetB0ImageNet',
    'module_name': 'models.efficientnet',
    'criterion': CrossEntropyLoss(),
    'optimizer': optim.Adam,
    'optimizer_params': {'lr': 1e-4},

    'preprocess_train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'preprocess_val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'preprocess_test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'scheduler': None,
    'scheduler_params': {}
}

# SqueezeNet 1.1 com pesos da ImageNet, no mesmo esquema da
# 'mobilenetv3_imagenet' (normalização da ImageNet e lr 1e-4)
models_conf['squeezenet_imagenet'] = {
    'img_dims': (224, 224),
    'model_name': 'SqueezeNet 1.1 (ImageNet)',
    'model_path': 'model_squeezenet_imagenet.pth',
    'class_name': 'SqueezeNetImageNet',
    'module_name': 'models.squeezenet',
    'criterion': CrossEntropyLoss(),
    'optimizer': optim.Adam,
    'optimizer_params': {'lr': 1e-4},

    'preprocess_train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'preprocess_val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'preprocess_test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'scheduler': None,
    'scheduler_params': {}
}

# ResNet-18 com pesos da ImageNet, no mesmo esquema da
# 'mobilenetv3_imagenet' (normalização da ImageNet e lr 1e-4).
# Diferente da 'resnet' acima (ResNet50 do artigo de Sharma et al.)
models_conf['resnet18_imagenet'] = {
    'img_dims': (224, 224),
    'model_name': 'ResNet-18 (ImageNet)',
    'model_path': 'model_resnet18_imagenet.pth',
    'class_name': 'ResNet18ImageNet',
    'module_name': 'models.resnet18',
    'criterion': CrossEntropyLoss(),
    'optimizer': optim.Adam,
    'optimizer_params': {'lr': 1e-4},

    'preprocess_train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'preprocess_val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'preprocess_test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        imagenet_normalize
    ]),

    'scheduler': None,
    'scheduler_params': {}
}

def get_config(base_model):
    return models_conf[base_model]
