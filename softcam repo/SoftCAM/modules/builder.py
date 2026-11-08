import copy
import timm
import torch
import torch.nn as nn
import bagnets.pytorchnet

from torchvision import models
from utils.func import print_msg, select_out_features
from .models import SparseBagnet, FullyConvModel

def generate_model(cfg):
    out_features = select_out_features(cfg.data.num_classes, cfg.train.criterion)

    model = build_model(cfg, out_features)

    if cfg.train.checkpoint:
        NotImplementedError('Load checkpoint: not yet implemented')

    if cfg.base.device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(cfg.base.device)
    return model


def build_model(cfg, num_classes):
    network = cfg.train.network
    pretrained = cfg.train.pretrained

    if network == 'retfound':
        # Load pre-trained model from hugging face
        model = timm.create_model("hf_hub:bitfount/RETFound_MAE", pretrained=True)
        #model.head = nn.Linear(1024, num_classes)
        model = ExplainRetFound(model, num_classes=num_classes)
    elif 'bagnet' in network:
        model = BUILDER[network](pretrained=pretrained)
    elif 'inception' in network:
        model = timm.create_model('inception_v4', pretrained=True)
    else:
        model = BUILDER[network](weights=WEIGHTS[network])

    if network == 'retfound':
        pass 
    elif cfg.train.conv_cls:
        model = FullyConvModel(cfg, model, num_classes)
    else:
        if 'resnet' in network or 'resnext' in network or 'shufflenet' in network or 'bagnet' in network:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'densenet' in network:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif 'vgg' in network:
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            ) 
        elif 'efficientnet' in network:
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.classifier[-1].in_features, num_classes))
        elif 'inception' in network:
            model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
        elif 'mobilenet' in network:
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.last_channel, num_classes))
        elif 'squeezenet' in network:
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            raise NotImplementedError('Not implemented network.')

    return model


BUILDER = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'wide_resnet50': models.wide_resnet50_2,
    'wide_resnet101': models.wide_resnet101_2,
    'resnext50': models.resnext50_32x4d,
    'resnext101': models.resnext101_32x8d,
    'mobilenet': models.mobilenet_v2,
    'squeezenet': models.squeezenet1_1,
    'shufflenet_0_5': models.shufflenet_v2_x0_5,
    'shufflenet_1_0': models.shufflenet_v2_x1_0,
    'shufflenet_1_5': models.shufflenet_v2_x1_5,
    'shufflenet_2_0': models.shufflenet_v2_x2_0,
    'efficientnet_v2_m': models.efficientnet_v2_m,
    'bagnet33': bagnets.pytorchnet.bagnet33,
    'bagnet17': bagnets.pytorchnet.bagnet17,
    'bagnet9': bagnets.pytorchnet.bagnet9
}     

WEIGHTS = {
    'resnet50': 'ResNet50_Weights.DEFAULT',
    'vgg16': 'VGG16_Weights.IMAGENET1K_V1',
    'densenet121': 'DenseNet121_Weights.DEFAULT',
    'efficientnet_v2_m': 'EfficientNet_V2_M_Weights.DEFAULT'
}