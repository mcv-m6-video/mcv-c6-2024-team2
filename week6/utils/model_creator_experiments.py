""" Functions to create models """

import torch
import torch.nn as nn
import torchvision


def create_from_torchhub(model_name: str, num_classes: int) -> nn.Module:

    if model_name == 'slow_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        model.blocks[5].proj = nn.Identity()
        new_kernel_size = (4, 5, 5)
        for name, module in model.named_modules():
            if isinstance(module, nn.AvgPool3d):
                module.kernel_size = new_kernel_size
        model = nn.Sequential(model, nn.Linear(2048, num_classes, bias=True), )
    if model_name == 'resnet':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        model.fc = nn.Identity()
        model = nn.Sequential(model, nn.Linear(2048, num_classes, bias=True), )
    if model_name == 'vgg':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        model.classifier[6] = nn.Identity()
        model = nn.Sequential(model, nn.Linear(4096, num_classes, bias=True), )
    if model_name == 'vgg_3d':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        for count, feature in enumerate(model.features):
            if isinstance(feature, nn.Conv2d):
                model.features[count] = nn.Conv3d(feature.in_channels, feature.out_channels,
                                                  kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                                  padding=(1, 1, 1), dilation=(1, 1, 1))
            if isinstance(feature, nn.MaxPool2d):
                model.features[count] = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2),
                                                  padding=0, dilation=1)
        model.avgpool = nn.AdaptiveAvgPool3d((7, 7, 1))
        model.classifier[6] = nn.Identity()
        model = nn.Sequential(model, nn.Linear(4096, num_classes, bias=True), )

    return model
