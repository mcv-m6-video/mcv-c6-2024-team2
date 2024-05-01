import os
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List


# lcoal imports
from utils import get_modality_data


# early fusion handler
def early_fusion_handler(
    multi_modal_model: nn.Module,
    batched_clip_data: torch.Tensor,
    modality_names: List[str],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    eval: bool = False,
    labels: torch.Tensor = None,
):
    """

    NOTE: FOR EARLY FUSION, THE INPUTS ARE COMBINED INTO A SINGLE TENSOR BEFORE BEING PASSED TO THE MODEL
    NUMBER OF MODEL: 1
    INPUT SHAPE: (BATCH_SIZE, NUM_MODALITIES*MODALITY_FEATURES)
    OUTPUT SHAPE: (BATCH_SIZE, NUM_CLASSES)
    """

    # STEP 1: DATA PREP
    final_data = get_modality_data(batched_clip_data, modality_names)

    # STEP 2: TRAINING
    if not eval:

        optimizer.zero_grad()
        output = multi_modal_model(final_data)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

    else:
        # STEP 3: EVALUATION
        with torch.no_grad():
            outputs = multi_modal_model(final_data)

    return outputs


# late fusion handler
def late_fusion_handler(
    multi_modal_model: nn.Module,
    batched_clip_data: torch.Tensor,
    modality_names: List[str],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    eval: bool = False,
    labels: torch.Tensor = None,
):
    """

    NOTE: FOR LATE FUSION, EACH MODALITY IS PASSED THROUGH A SEPARATE MODEL AND THE OUTPUTS ARE COMBINED BEFORE BEING PASSED TO THE FINAL MODEL
    NUMBER OF MODELS: NUM_MODALITIES
    INPUT SHAPE: (BATCH_SIZE, MODALITY_FEATURES)
    OUTPUT SHAPE: (BATCH_SIZE, NUM_CLASSES)
    """

    pass


# join fusion handler

# hybrid fusion handler

# ensemble fusion handler
