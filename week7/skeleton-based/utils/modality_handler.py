import torch
import torch.nn as nn
from torchvision.models import resnet50
from typing import List
import warnings
import os
import yaml
from models import HDGCNModel, ModalityProjectionModel


warnings.filterwarnings("ignore")


# TEMPORAL MODALITY
class Temporal_Modality(nn.Module):

    def __init__(self, model_name: str = "x3d_xs", load_pretrain: bool = True) -> None:
        super().__init__()
        # load model
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", "x3d_xs", pretrained=load_pretrain
        )
        # remove last layer to use it as feature extractor
        model.blocks[5].proj = nn.Identity()
        self.model = model

        # FREEZE # TODO: MAY BE WE CAN TRY TO TRAIN THIS AS WELL AS AN EXPERIMENT
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # output shape (batch_size, 2048)
        return self.model(x)


class Spatial_Modality(nn.Module):
    def __init__(
        self, model_name: str = "resnet-50", load_pretrain: bool = True
    ) -> None:
        super(Spatial_Modality, self).__init__()
        # LOAD PRETRAINED RESNET-50
        if model_name == "resnet-50" and load_pretrain:
            self.base_model = resnet50(pretrained=True)
            # REMOVE LINEAR LAYERS
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
        else:
            raise NotImplementedError(
                f"{model_name} not supported or load_pretrain set to False."
            )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # FREEZE # TODO: MAY BE WE CAN TRY TO TRAIN THIS AS WELL AS AN EXPERIMENT
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # INPUT SHAPE: (batch_size, num_channel, clip_length, height, width)
        batch_size, c, t, h, w = x.size()
        x = x.view(batch_size * t, c, h, w)
        x = self.base_model(x)
        x = self.global_pool(x)
        x = x.view(batch_size, t, -1)

        # AGGREGATE FEATURES
        x = x.mean(dim=1)

        return x


config_path = (
    "./config.yaml"
)
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError("config.yaml not found")

skel_model_args = config["skel_model_args"]


# MODALITY HANDLER
device = "cuda" if torch.cuda.is_available() else "cpu"
temporal_model = Temporal_Modality().to(device)
spatial_model = Spatial_Modality().to(device)
skeleton_model = HDGCNModel(**skel_model_args).to(device)
skeleton_model.load_state_dict(torch.load(f"runs-200-33600.pt"))
projector_model = ModalityProjectionModel(input_modality_shape=256, output_modality_shape=2048).to(device)

def get_modality_data(
    batched_clips: torch.Tensor,
    batched_skels: torch.Tensor,
    modality_names: List[str],
):
    aggregated_features = []
    for modality_name in modality_names:
        if modality_name == "temporal":
            aggregated_features.append(temporal_model(batched_clips))
        elif modality_name == "spatial":
            aggregated_features.append(spatial_model(batched_clips))
        elif modality_name == "skeleton":
            skels = skeleton_model(batched_skels)
            skels = projector_model(skels)
            aggregated_features.append(skels)
        else:
            raise NotImplementedError(f"{modality_name} not supported.")

    return torch.cat(aggregated_features, dim=1)


if __name__ == "__main__":

    sample_data = torch.randn(16, 3, 8, 240, 240)

    print("=====================================")
    print("Testing Temporal Modality")
    print("=====================================")
    print(f"Input shape: {sample_data.shape}")
    output = temporal_model(sample_data)
    print(f"Output shape: {output.shape}")

    print("=====================================")
    print("Testing Spatial Modality")
    print("=====================================")
    print(f"Input shape: {sample_data.shape}")
    output = spatial_model(sample_data)
    print(f"Output shape: {output.shape}")

    print("=====================================")
    print("Testing Both Modalities")
    print("=====================================")
    modality_names = ["temporal", "spatial"]
    output = get_modality_data(sample_data, modality_names)
    print(f"Output shape: {output.shape}")
