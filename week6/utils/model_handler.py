import torch
import torch.nn as nn

available_models = [
    "c2d_r50",
    "i3d_r50",
    "slow_r50",
    "slow_r50",
    "slowfast_r50",
    "slowfast_r50",
    "slowfast_r101",
    "slowfast_r101",
    "csn_r101",
    "r2plus1d_r50",
    "x3d_xs",
    "x3d_s",
    "x3d_m",
    "x3d_l",
]


def create(model_name: str, load_pretrain: bool, num_classes: int) -> nn.Module:
    if model_name in available_models:
        return create_model(
            model_name=model_name, load_pretrain=load_pretrain, num_classes=num_classes
        )
    else:
        raise ValueError(f"Model {model_name} not supported")


def create_model(model_name: str, load_pretrain: bool, num_classes: int) -> nn.Module:

    model = torch.hub.load(
        "facebookresearch/pytorchvideo", model_name, pretrained=load_pretrain
    )
    # model.blocks[5].proj = nn.Identity()
    # return nn.Sequential(
    #     model,
    #     nn.Linear(2048, num_classes, bias=True),
    # )
    model.blocks[6].add_module(
        "linear", torch.nn.Linear(model.blocks[6].proj.in_features, num_classes)
    )
    return model


