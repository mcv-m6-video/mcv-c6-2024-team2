#
import torch
import torch.nn as nn


### DEFINE EMBEDDING SHAPE
FINAL_INPUT_EMBEDDING_DIM = 2048
# TODO: DEFINE EMBEDDING OF EACH MODALITY HERE
MODALITY_1_EMBEDDING_DIM = 128  # Example dimension for modality 1
MODALITY_2_EMBEDDING_DIM = 128  # Example dimension for modality 2
MODALITY_3_EMBEDDING_DIM = 128  # Example dimension for modality 3


"""
THIS IS THE MODEL FOR THE MODALITY PROJECTION. IN CASE OF MODALITY IS IN DIFFERENT DIMENSION,
WE NEED TO PROJECT IT TO THE SAME DIMENSION. FOR THAT WE USE THIS MODEL.
"""


class ModalityProjectionModel(nn.Module):
    def __init__(
        self,
        input_modality_shape: int,
        output_modality_shape: int = FINAL_INPUT_EMBEDDING_DIM,
    ):
        super(ModalityProjectionModel, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm1d(input_modality_shape),
            nn.Linear(input_modality_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_modality_shape),
        )

    def forward(self, x):
        return self.model(x)


"""
THIS IS THE MODEL FOR THE MULTIMODAL CLASSIFICATION MODEL
"""


class MultiModalModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_modalities: int,
        input_feature_shape: int = FINAL_INPUT_EMBEDDING_DIM,
    ):
        super(MultiModalModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_feature_shape * num_modalities, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


"""
THIS IS THE MODEL FOR THE AUTOENCODER FOR INTERMEDIATE FUSION. 
WHICH WILL BE USED TO LEARN THE REPRESENTATION FOR EACH MODALITY SEPARETLY. 
"""


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder part
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # Latent space
            nn.Linear(512, 256),
        )

        # Decoder part
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            # Reconstruction
            nn.Linear(1024, 2048),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MultiModalAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, num_modalities):
        super(MultiModalAttentionFusion, self).__init__()
        self.num_modalities = num_modalities
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(num_modalities * embed_dim, embed_dim)

    def forward(self, modality_features):
        batch_size = modality_features[0].shape[0]
        attn_input = torch.stack(modality_features).permute(1, 0, 2)
        attn_output, _ = self.multihead_attn(attn_input, attn_input, attn_input)
        # reshape
        combined_features = attn_output.permute(1, 0, 2).reshape(batch_size, -1)
        fused_output = self.linear(combined_features)
        return fused_output


