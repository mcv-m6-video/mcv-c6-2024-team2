""" Main script for training a video classification model on HMDB51 dataset. """

import argparse
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator

from torch.utils.data import DataLoader

# from datasets.HMDB51Dataset import HMDB51Dataset
from HMDB51Dataset import HMDB51Dataset

from utils import model_handler
from utils import model_analysis
from utils import statistics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    description: str = "",
) -> float:
    """
    Trains the given model using the provided data loader, optimizer, and loss function.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): The data loader containing the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        loss_fn (nn.Module): The loss function used to compute the training loss.
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        float: The mean accuracy over all batches.
    """
    model.train()
    pbar = tqdm(train_loader, desc=description, total=len(train_loader))
    loss_train_mean = statistics.RollingMean(window_size=len(train_loader))
    hits = count = 0  # auxiliary variables for computing accuracy
    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch["clips"].to(device), batch["labels"].to(device)
        # Forward pass
        outputs = model(clips)
        # Compute loss
        loss = loss_fn(outputs, labels)
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Update progress bar with metrics
        loss_iter = loss.item()
        hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
        hits += hits_iter
        count += len(labels)
        pbar.set_postfix(
            loss=loss_iter,
            loss_mean=loss_train_mean(loss_iter),
            acc=(float(hits_iter) / len(labels)),
            acc_mean=(float(hits) / count),
        )
    return float(hits) / count  # Return the mean accuracy over all batches


def evaluate(
    model: nn.Module,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    description: str = "",
) -> dict:
    """
    Evaluates the given model using the provided data loader and loss function,
    and returns class-wise accuracy, overall accuracy, and average validation loss.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        dict: Dictionary containing overall accuracy, class-wise accuracy, and average validation loss.
    """
    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
    hits = count = total_loss = (
        0  # auxiliary variables for computing overall accuracy and loss
    )
    class_correct = {}
    class_totals = {}

    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch["clips"].to(device), batch["labels"].to(device)
        # Forward pass
        with torch.no_grad():
            outputs = model(clips)
            # Compute loss (just for logging, not used for backpropagation)
            loss = loss_fn(outputs, labels)
            loss_iter = loss.item()
            total_loss += loss_iter * len(labels)

            # Compute metrics
            predictions = outputs.argmax(dim=1)
            correct = predictions.eq(labels)
            for label, correct_pred in zip(labels, correct):
                label = label.item()
                if label not in class_correct:
                    class_correct[label] = 0
                    class_totals[label] = 0
                class_correct[label] += correct_pred.item()
                class_totals[label] += 1

            hits_iter = correct.sum().item()
            hits += hits_iter
            count += len(labels)

            # Update progress bar with metrics
            pbar.set_postfix(
                loss=loss_iter,
                loss_mean=loss_valid_mean(loss_iter),
                acc=(float(hits_iter) / len(labels)),
                acc_mean=(float(hits) / count),
            )

    average_loss = total_loss / count if count != 0 else 0
    class_acc = {
        class_id: class_correct[class_id] / class_totals[class_id]
        for class_id in class_correct
    }
    overall_acc = float(hits) / count if count != 0 else 0

    return {
        "overall_acc": overall_acc,
        "class_acc": class_acc,
        "average_loss": average_loss,
    }


def create_datasets(
    frames_dir: str,
    annotations_dir: str,
    split: HMDB51Dataset.Split,
    clip_length: int,
    crop_size: int,
    temporal_stride: int,
) -> Dict[str, HMDB51Dataset]:
    """
    Creates datasets for training, validation, and testing.

    Args:
        frames_dir (str): Directory containing the video frames (a separate directory per video).
        annotations_dir (str): Directory containing annotation files.
        split (HMDB51Dataset.Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.

    Returns:
        Dict[str, HMDB51Dataset]: A dictionary containing the datasets for training, validation, and testing.
    """
    datasets = {}
    for regime in HMDB51Dataset.Regime:
        datasets[regime.name.lower()] = HMDB51Dataset(
            frames_dir,
            annotations_dir,
            split,
            regime,
            clip_length,
            crop_size,
            temporal_stride,
        )

    return datasets


def create_dataloaders(
    datasets: Dict[str, HMDB51Dataset],
    batch_size: int,
    batch_size_eval: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        datasets (Dict[str, HMDB51Dataset]): A dictionary containing datasets for training, validation, and testing.
        batch_size (int, optional): Batch size for the data loaders. Defaults to 8.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 2.
        pin_memory (bool, optional): Whether to pin memory in DataLoader for faster GPU transfer. Defaults to True.

    Returns:
        Dict[str, DataLoader]: A dictionary containing data loaders for training, validation, and testing datasets.
    """
    dataloaders = {}
    for key, dataset in datasets.items():
        dataloaders[key] = DataLoader(
            dataset,
            batch_size=(batch_size if key == "training" else batch_size_eval),
            shuffle=(key == "training"),  # Shuffle only for training dataset
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return dataloaders


def create_optimizer(
    optimizer_name: str, parameters: Iterator[nn.Parameter], lr: float = 1e-4
) -> torch.optim.Optimizer:
    """
    Creates an optimizer for the given parameters.

    Args:
        optimizer_name (str): Name of the optimizer (supported: "adam" and "sgd" for now).
        parameters (Iterator[nn.Parameter]): Iterator over model parameters.
        lr (float, optional): Learning rate. Defaults to 1e-4.

    Returns:
        torch.optim.Optimizer: The optimizer for the model parameters.
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")


def print_model_summary(
    model: nn.Module,
    clip_length: int,
    crop_size: int,
    print_model: bool = False,
    print_params: bool = True,
    print_FLOPs: bool = True,
) -> None:
    """
    Prints a summary of the given model.

    Args:
        model (nn.Module): The model for which to print the summary.
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        print_model (bool, optional): Whether to print the model architecture. Defaults to True.
        print_params (bool, optional): Whether to print the number of parameters. Defaults to True.
        print_FLOPs (bool, optional): Whether to print the number of FLOPs. Defaults to True.

    Returns:
        None
    """
    if print_model:
        print(model)

    if print_params:
        num_params = sum(p.numel() for p in model.parameters())
        # num_params = model_analysis.calculate_parameters(model) # should be equivalent
        print(f"Number of parameters (M): {round(num_params / 10e6, 2)}")

    if print_FLOPs:
        num_FLOPs = model_analysis.calculate_operations(
            model, clip_length, crop_size, crop_size
        )
        print(f"Number of FLOPs (G): {round(num_FLOPs / 10e9, 2)}")

    return {"num_params": num_params, "num_FLOPs": num_FLOPs}


if __name__ == "__main__":

    available_models = model_handler.available_models

    # TODO: SAVE G-FLOP, TOTAL PARAM, TEST ACCURACY

    model_data = {}

    frames_dir = "/ghome/group04/c6-diana/week5/frames"
    annotations_dir = (
        "/ghome/group04/c6-diana/week5/data/hmdb51/testTrainMulti_601030_splits"
    )
    clip_length = 8
    crop_size = 240
    temporal_stride = 8
    load_pretrain = True
    optimizer_name = "adam"
    lr = 1e-4
    epochs = 50
    batch_size = 16
    batch_size_eval = 32
    validate_every = 5
    num_workers = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create datasets
    print("\n::: Creating datasets :::")
    datasets = create_datasets(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1,  # hardcoded
        clip_length=clip_length,
        crop_size=crop_size,
        temporal_stride=temporal_stride,
    )

    # Create data loaders
    print("\n::: Creating data loaders :::")
    loaders = create_dataloaders(
        datasets,
        batch_size,
        batch_size_eval=batch_size_eval,
        num_workers=num_workers,
    )

    for model_name in available_models:

        print(f"\n::: MODEL NAME {model_name} :::")
        model_data[model_name] = {}

        # Init model, optimizer, and loss function
        print("\n::: Initializing model, optimizer, and loss function :::")
        try:
            model = model_handler.create(
                model_name, load_pretrain, datasets["training"].get_num_classes()
            )

            optimizer = create_optimizer(optimizer_name, model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()

            print("\n::: MODEL SUMMARY :::")
            model_summary = print_model_summary(model, clip_length, crop_size)

            model_data[model_name]["g_flop"] = model_summary["num_FLOPs"]
            model_data[model_name]["total_param"] = model_summary["num_params"]

            model = model.to(device)

            print("\n::: Starting training :::")

            # EARLY STOPPING
            LOSS_TOLERANCE = 0.01
            LOSS_OBSERVATION_WINDOW = 5
            loss_window = []
            best_loss = float("inf")

            for epoch in range(epochs):
                try:
                    # Validation
                    if epoch % validate_every == 0:
                        description = f"Validation [Epoch: {epoch+1}/{epochs}]"
                        output = evaluate(
                            model,
                            loaders["validation"],
                            loss_fn,
                            device,
                            description=description,
                        )
                        loss = output["average_loss"]
                        # EARLY STOPPING
                        loss_window.append(loss)
                        if len(loss_window) > LOSS_OBSERVATION_WINDOW:
                            loss_window.pop(0)
                            if all(
                                [
                                    loss - best_loss > LOSS_TOLERANCE
                                    for loss in loss_window
                                ]
                            ):
                                print(f"Early stopping at epoch {epoch+1}")
                                break

                    # Training
                    description = f"Training [Epoch: {epoch+1}/{epochs}]"
                    train_mean_acc = train(
                        model,
                        loaders["training"],
                        optimizer,
                        loss_fn,
                        device,
                        description=description,
                    )

                    if train_mean_acc > 0.9:
                        break
                except Exception as e:
                    print(f"SOMETHING WENT WRONG\n{e}")
                    break

            # save model
            torch.save(
                model.state_dict(), f"./model_{model_name}_epoch_{epoch}_{epochs}.pth"
            )

            # Testing
            evaluate(
                model,
                loaders["validation"],
                loss_fn,
                device,
                description=f"Validation [Final]",
            )
            test_result = evaluate(
                model, loaders["testing"], loss_fn, device, description=f"Testing"
            )

            model_data[model_name]["test_accuracy"] = test_result

            # export model data
            with open("model_data.json", "w") as f:
                json.dump(model_data, f)
        except Exception as e:
            continue
