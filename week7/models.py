#
import torch
import torch.nn as nn


### DEFINE EMBEDDING SHAPE
FINAL_INPUT_EMBEDDING_DIM = 2048
# TODO: DEFINE EMBEDDING OF EACH MODALITY HERE
MODALITY_1_EMBEDDING_DIM = 128  # Example dimension for modality 1
MODALITY_2_EMBEDDING_DIM = 128  # Example dimension for modality 2
MODALITY_3_EMBEDDING_DIM = 128  # Example dimension for modality 3


# class TemporalPathway(nn.Module):
#     def __init__(self, input_shape: int):
#         super(TemporalPathway, self).__init__()
#         # Define the architecture for the temporal pathway
#         self.temporal_pathway = nn.Sequential(
#             nn.Linear(input_shape, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, FINAL_INPUT_EMBEDDING_DIM),
#         )

#     def forward(self, x):
#         return self.temporal_pathway(x)


# class SpatialPathway(nn.Module):
#     def __init__(self, input_shape: int):
#         super(SpatialPathway, self).__init__()
#         # Define the architecture for the spatial pathway
#         self.spatial_pathway = nn.Sequential(
#             nn.Linear(input_shape, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, FINAL_INPUT_EMBEDDING_DIM),
#         )

#     def forward(self, x):
#         return self.spatial_pathway(x)


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


# class AttenionFusedClassifier(nn.Module):
#     def __init__(self, num_classes, embed_dim, num_heads, num_modalities):
#         super(AttenionFusedClassifier, self).__init__()
#         self.fusion_model = MultiModalAttentionFusion(
#             embed_dim, num_heads, num_modalities
#         )
#         self.classifier = nn.Linear(embed_dim, num_classes)

#     def forward(self, modality_features):
#         fused_output = self.fusion_model(modality_features)
#         output = self.classifier(fused_output)
#         return output


# test code to try multi modal model
if __name__ == "__main__":

    # modality 1: dimension 750
    dim_1 = 750
    modality_1 = torch.randn(5, dim_1)
    print(f"Modality 1 shape: {modality_1.shape}")
    projector_1 = ModalityProjectionModel(dim_1)
    projected_modality_1 = projector_1(modality_1)
    print(f"Projected Modality 1 shape: {projected_modality_1.shape}")

    # modality 2: dimension 1000
    dim_2 = 1000
    modality_2 = torch.randn(5, dim_2)
    print(f"Modality 2 shape: {modality_2.shape}")
    projector_2 = ModalityProjectionModel(dim_2)
    projected_modality_2 = projector_2(modality_2)
    print(f"Projected Modality 2 shape: {projected_modality_2.shape}")

    # modality 3: dimension 500
    dim_3 = 500
    modality_3 = torch.randn(5, dim_3)
    print(f"Modality 3 shape: {modality_3.shape}")
    projector_3 = ModalityProjectionModel(dim_3)
    projected_modality_3 = projector_3(modality_3)
    print(f"Projected Modality 3 shape: {projected_modality_3.shape}")

    # multimodal model
    print("Testing multimodal model")
    ### concat all the projected modalities
    multimodal_input = torch.cat(
        (projected_modality_1, projected_modality_2, projected_modality_3), dim=1
    )
    print(f"Multimodal input shape: {multimodal_input.shape}")
    num_classes = 10
    num_modalities = 3
    multimodal_model = MultiModalModel(num_classes, num_modalities)
    output = multimodal_model(multimodal_input)
    print(f"Output shape: {output.shape}")

    # TEST FOR AUTOENCODER
    autoencoder = Autoencoder()
    input_data = torch.randn(5, 2048)
    output = autoencoder(input_data)
    print(f"Output shape: {output.shape}")

    # TEST FOR MULTI MODAL ATTENTION FUSION
    print("Testing Multi Modal Attention Fusion")
    num_heads = 4
    embed_dim = 2048
    fusion_model = MultiModalAttentionFusion(embed_dim, num_heads, num_modalities)
    fused_output = fusion_model(
        [projected_modality_1, projected_modality_2, projected_modality_3]
    )
    print(f"Fused output shape: {fused_output.shape}")

    # TEST FOR ATTENTION FUSED CLASSIFIER
    print("Testing Attention Fused Classifier")
    num_classes = 10
    classifier = AttenionFusedClassifier(
        num_classes, embed_dim, num_heads, num_modalities
    )
    output = classifier(
        [projected_modality_1, projected_modality_2, projected_modality_3]
    )
    print(f"Output shape: {output.shape}")
