import torch
import torch.nn as nn
from torchvision.models import resnet50
from typing import List
import warnings

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


# MODALITY HANDLER
device = "cuda" if torch.cuda.is_available() else "cpu"
temporal_model = Temporal_Modality().to(device)
spatial_model = Spatial_Modality().to(device)


def get_modality_data(
    batched_clip: torch.Tensor,
    modality_names: List[str],
):
    aggregated_features = []
    for modality_name in modality_names:
        if modality_name == "temporal":
            aggregated_features.append(temporal_model(batched_clip))
        elif modality_name == "spatial":
            aggregated_features.append(spatial_model(batched_clip))
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
