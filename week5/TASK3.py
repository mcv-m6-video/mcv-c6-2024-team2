import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from UpdatedHMDB51Dataset import HMDB51Dataset
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb

trained_model_path = "../model_x3d_xs_epoch_200.pth"


# Load the pretrained model
def load_pretrained_model(model_path, num_classes=51,device="cuda"):
    # Load the model architecture
    model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_xs", pretrained=False)
    model.blocks[5].proj = nn.Identity()  
    model = nn.Sequential(
        model,
        nn.Linear(
            2048, num_classes, bias=True
        ),  
    )

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model = model.to(device)
    model.eval()
    return model


def create_evaluation_dataset(
    frames_dir: str,
    annotations_dir: str,
    split: HMDB51Dataset.Split,
    clip_length: int,
    temporal_stride: int,
    crop_size: int = None,
    batch_size: int = 1,
) -> DataLoader:
    # Here we assume the evaluation is done on split 1, and use the validation regime.
    
    dataset = HMDB51Dataset(
        videos_dir=frames_dir,
        annotations_dir=annotations_dir,
        split=split,
        regime=HMDB51Dataset.Regime.VALIDATION,
        clip_length=clip_length,
        temporal_stride=temporal_stride,
        crop_size=crop_size,
    )
    # We use a batch size of 1 to evaluate one video at a time
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn
    )
    return loader


def evaluate_multiclip(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str="cuda",
) -> None:
    model.eval()  # Set the model to evaluation mode

    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(data_loader, total=len(data_loader)):
            inputs, labels = batch["clips"].to(device), batch["labels"].to(device)
            outputs = model(inputs)  # Forward pass

            loss = loss_fn(outputs, labels)  # Compute loss

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Calculate predictions (assumes outputs are logits for each class)
            _, predicted_labels = torch.max(outputs, 1)

            # Update correct predictions count
            correct_predictions += (predicted_labels == labels).sum().item()

    # Calculate the average loss and accuracy
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    print(
        f"Evaluation complete. Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}"
    )
    return accuracy


available_clip_lengths = [8, 16, 32]
available_temporal_strides = [8, 16, 32]
available_crop_sizes = [
    # None,
                        # 20, 40, 60, 80, 100,120,140,160,
                        180]


# load the pretrained model
model = load_pretrained_model(trained_model_path)
# loss function
loss_fn = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

frame_dir  = "/ghome/group04/c6-diana/week5/frames"
annotation_dir = "/ghome/group04/c6-diana/week5/w5_source_code/data/hmdb51/testTrainMulti_601030_splits"

def objective(trail):
    wandb.init(project="c6_week5_task3_v2", config=trail.params)
    
    clip_length = trail.suggest_categorical("clip_length", available_clip_lengths)
    temporal_stride = trail.suggest_categorical("temporal_stride", available_temporal_strides)
    # crop_size = None
    crop_size = trail.suggest_categorical("crop_size", available_crop_sizes) # 3.2
    print(f"clip_length: {clip_length}\ntemporal_stride: {temporal_stride}\ncrop_size: {crop_size}")
    
    eval_loader_1 = create_evaluation_dataset(
        frames_dir=frame_dir,
        annotations_dir=annotation_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1,
        clip_length=clip_length,
        temporal_stride=temporal_stride,
        crop_size=crop_size,
        batch_size=1,
    )
    
    eval_loader_2 = create_evaluation_dataset(
       frames_dir=frame_dir,
        annotations_dir=annotation_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_2,
        clip_length=clip_length,
        temporal_stride=temporal_stride,
        crop_size=crop_size,
        batch_size=1,
    )
    
    eval_loader_3 = create_evaluation_dataset(
     frames_dir=frame_dir,
        annotations_dir=annotation_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_3,
        clip_length=clip_length,
        temporal_stride=temporal_stride,
        crop_size=crop_size,
        batch_size=1,
    )
    print("-------------------------------------")
    
    # Evaluate on all splits
    accuracy_1 = evaluate_multiclip(model, eval_loader_1, loss_fn, device)
    accuracy_2 = evaluate_multiclip(model, eval_loader_2, loss_fn, device)
    accuracy_3 = evaluate_multiclip(model, eval_loader_3, loss_fn, device)
    
    mean_accuracy = (accuracy_1 + accuracy_2 + accuracy_3) / 3
    
    wandb.log({"mean_accuracy": mean_accuracy,
               "clip_length": clip_length,
                "temporal_stride": temporal_stride,
                "crop_size": crop_size})
    
    return mean_accuracy



if __name__ == "__main__":
    
    wandb_kwargs = {"project": "FINDING_BEST_COMBINATION_C6_WEEK5_TASK3"}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)
    
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )
    study.optimize(
        objective,
        n_trials=90,
        timeout=600000000000000000,
        callbacks=[wandbc],  # weight and bias connection
    )