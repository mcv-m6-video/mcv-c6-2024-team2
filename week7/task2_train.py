import os
import yaml
from tabulate import tabulate
from typing import List, Dict, Iterator
from tqdm import tqdm
import wandb


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# local imports
from utils import *
from utils import statistics, get_modality_data
from models import (
    MultiModalModel,
    ModalityProjectionModel,
    Autoencoder,
    MultiModalAttentionFusion,
)


def console_json_table(data, headers: List[str] = ["Parameter", "Value"]):
    data = {"keys": list(data.keys()), "values": list(data.values())}
    table = zip(data["keys"], data["values"])
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


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


def evaluate(
    modality_models: Dict[str, nn.Module],
    valid_loader: DataLoader,
    modality_loss_fn: Dict[str, nn.Module],
    device: str,
    modalities: List[str],
    fusion_type: str,
    description: str = "",
    **kwargs,
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

    # set models to evaluation mode
    for _, model in modality_models.items():
        model.eval()

    # set autoencoders to evaluation mode
    if fusion_type == "intermediate_fusion":
        for _, autoencoder in kwargs["projection_autoencoders"].items():
            autoencoder.eval()
    # set attention model to evaluation mode
    if fusion_type == "attention_fusion":
        kwargs["attention_fusion_model"].eval()

    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
    hits = count = total_loss = (
        0  # auxiliary variables for computing overall accuracy and loss
    )
    class_correct = {}
    class_totals = {}

    for batch in pbar:
        # Gather batch and move to device
        clips, keypoints, labels = batch["clips"].to(device), batch["keypoints"], batch["labels"].to(device)
        # Forward pass
        with torch.no_grad():
            # outputs = model(clips)

            if fusion_type == "early_fusion":
                model = modality_models["combined"]
                data = get_modality_data(clips, keypoints, modalities)
                outputs = model(data)
                loss = modality_loss_fn["combined"](outputs, labels)
                loss_iter = loss.item()

            elif fusion_type == "late_fusion":
                all_losses = []
                outputs = []
                for modality_name in modalities:
                    model = modality_models[modality_name]
                    data = get_modality_data(clips, keypoints, [modality_name])
                    output = model(data)
                    outputs.append(output)
                    loss = modality_loss_fn[modality_name](output, labels)
                    all_losses.append(loss)
                loss_iter = sum(all_losses) / len(all_losses)
                loss_iter = loss_iter.item()

                # argmax
                outputs = torch.stack(outputs, dim=0).mean(dim=0)
            elif fusion_type == "intermediate_fusion":
                modality_data = []
                for modality_name in modalities:
                    # pass through autoencoder
                    autoencoder = kwargs["projection_autoencoders"][modality_name]
                    data = get_modality_data(clips, keypoints, [modality_name])
                    projected_data = autoencoder.encoder(data)
                    modality_data.append(projected_data)
                data = torch.cat(modality_data, dim=1)
                model = modality_models["combined"]
                outputs = model(data)
                loss = modality_loss_fn["combined"](outputs, labels)
                loss_iter = loss.item()
            elif fusion_type == "attention_fusion":
                modality_features = []
                for modality_name in modalities:
                    model = modality_models[modality_name]
                    data = get_modality_data(clips, keypoints, [modality_name])
                    output = model(data)
                    modality_features.append(output)
                fused_feature = kwargs["attention_fusion_model"](modality_features)

                # classification model
                model = modality_models["combined"]
                outputs = model(fused_feature)
                loss = modality_loss_fn["combined"](outputs, labels)
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


def train(
    modality_models: Dict[str, nn.Module],
    train_loader: DataLoader,
    # optimizer: torch.optim.Optimizer,
    modality_optimizers: Dict[str, torch.optim.Optimizer],
    modality_loss_fn: Dict[str, nn.Module],
    device: str,
    modalities: List[str],
    fusion_type: str,
    description: str = "",
    **kwargs,
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

    for _, model in modality_models.items():
        model.train()

    pbar = tqdm(train_loader, desc=description, total=len(train_loader))
    loss_train_mean = statistics.RollingMean(window_size=len(train_loader))
    hits = count = 0  # auxiliary variables for computing accuracy
    for batch in pbar:
        # Gather batch and move to device
        clips, keypoints, labels = batch["clips"].to(device), batch["keypoints"].to(device), batch["labels"].to(device)

        if fusion_type == "early_fusion":
            model = modality_models["combined"]
            data = get_modality_data(clips, modalities)
            outputs = model(data)
            loss = modality_loss_fn["combined"](outputs, labels)
            loss_iter = loss.item()
            # Backward pass
            loss.backward()

            modality_optimizers["combined"].step()
            modality_optimizers["combined"].zero_grad()

            # DIANA: WANDB LOGGING HERE FOR TRAINING LOSS, TRAINING ACCURACY FOR EARLY FUSION CLASSIFICATION MODEL

        elif fusion_type == "late_fusion":
            all_losses = []
            for modality_name in modalities:
                model = modality_models[modality_name]
                data = get_modality_data(clips, [modality_name])
                outputs = model(data)
                loss = modality_loss_fn[modality_name](outputs, labels)
                all_losses.append(loss)
                # Backward pass
                loss.backward()
                # DIANA: WANDB LOGGING HERE FOR TRAINING LOSS, TRAINING ACCURACY FOR LATE FUSION CLASSIFICATION MODELS

                modality_optimizers[modality_name].step()
                modality_optimizers[modality_name].zero_grad()
            loss_iter = sum(all_losses) / len(all_losses)
            loss_iter = loss_iter.item()
        elif fusion_type == "intermediate_fusion":

            # DIANA: WANDB LOGGING HERE FOR TRAINING LOSS, TRAINING ACCURACY FOR INTERMEDIATE FUSION CLASSIFICATION MODEL
            """
            WE NEED OBSERVE THE LOSS AND ACCURACY FOR THE AUTOENCODER MODEL
            WE NEED TO OBSERVE THE LOSS AND ACCURACY FOR THE CLASSIFICATION MODEL
            """

            # ! NOTE: WE ARE TRAINING THE PROJECTION AUTOENCODER MODEL FOR EACH MODALITY
            modality_data = []
            for modality_name in modalities:
                # pass through autoencoder
                autoencoder = kwargs["projection_autoencoders"][modality_name]
                data = get_modality_data(clips, [modality_name])
                # forward pass
                projected_data = autoencoder.encoder(data)
                modality_data.append(projected_data)
                # loss
                loss = kwargs["projection_autoencoders_loss_fn"][modality_name](
                    projected_data, data
                )
                # backward pass
                loss.backward()
                # optimizer step

                kwargs["projection_autoencoders_optimizers"][modality_name].step()
                kwargs["projection_autoencoders_optimizers"][modality_name].zero_grad()
            data = torch.cat(modality_data, dim=1)
            model = modality_models["combined"]
            outputs = model(data)
            loss = modality_loss_fn["combined"](outputs, labels)
            loss_iter = loss.item()
            # Backward pass
            loss.backward()
            # optimizer step
            modality_optimizers["combined"].step()
            modality_optimizers["combined"].zero_grad()
        elif fusion_type == "attention_fusion":

            # DIANA: WANDB LOGGING HERE FOR TRAINING LOSS, TRAINING ACCURACY FOR ATTENTION MODEL
            # DIANA: WANDB LOGGING HERE FOR TRAINING LOSS, TRAINING ACCURACY FOR CLASSIFICATION MODEL

            modality_features = []
            # Gather features from each modality
            for modality_name in modalities:
                data = get_modality_data(clips, keypoints, [modality_name])
                modality_features.append(data)

            # fuse the modality features
            fused_feature = kwargs["attention_fusion_model"](modality_features)

            # classification on the fused feature
            model = modality_models["combined"]
            outputs = model(fused_feature)

            # compute loss
            loss = modality_loss_fn["combined"](outputs, labels)
            loss_iter = loss.item()

            # backward pass
            loss.backward()

            # update attention model
            kwargs["attention_fusion_optimizer"].step()
            kwargs["attention_fusion_optimizer"].zero_grad()

            # update classification model
            modality_optimizers["combined"].step()
            modality_optimizers["combined"].zero_grad()

        hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
        hits += hits_iter
        count += len(labels)
        pbar.set_postfix(
            loss=loss_iter,
            loss_mean=loss_train_mean(loss_iter),
            acc=(float(hits_iter) / len(labels)),
            acc_mean=(float(hits) / count),
        )
    # RETURN MEAN ACCURACY AND LOSS
    return float(hits) / count, loss_train_mean


wandb.init(project="c6_w7")

if __name__ == "__main__":
    print("========== C6 : MULTI MODAL VIDEO CLASSIFICATION ==========")
    config_path = (
        "./config.yaml"
    )
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            print("========== CONFIGURATION FROM YAML FILE ==========")
            console_json_table(config)
            if "experiment_name" not in config:
                print("ERROR: 'experiment_name' key not found in config")
            else:
                experiment_name = config["experiment_name"]
                print("Experiment Name:", experiment_name)
    else:
        raise FileNotFoundError("config.yaml not found")

    print("========== CONFIGURATION ==========")
    console_json_table(config)
    experiment_name = config["experiment_name"]
    frames_dir = config["frames_dir"]
    annotations_dir = config["annotations_dir"]
    clip_length = config["clip_length"]
    crop_size = config["crop_size"]
    temporal_stride = config["temporal_stride"]
    load_pretrain = config["load_pretrain"]
    optimizer_name = config["optimizer_name"]
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    batch_size_eval = config["batch_size_eval"]
    validate_every = config["validate_every"]
    num_workers = config["num_workers"]
    fusion_type = config["fusion_type"]
    modalities = config["modalities"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("========== DATA PREPARATION ==========")
    datasets = create_datasets(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1,  # hardcoded
        clip_length=clip_length,
        crop_size=crop_size,
        temporal_stride=temporal_stride,
    )
    loaders = create_dataloaders(
        datasets,
        batch_size,
        batch_size_eval=batch_size_eval,
        num_workers=num_workers,
    )

    print("========== MULTI-MODEL PREPARATION ==========")

    """
    MODEL INFORMATION:
        1. EARLY FUSION:
            - CLASSIFICATION MODEL: 1
        2. LATE FUSION:
            - CLASSIFICATION MODEL: NUMBER OF MODALITIES
        3. INTERMEDIATE FUSION:
            - CLASSIFICATION MODEL: 1
            - PROJECTION MODEL: NUMBER OF MODALITIES
        4. ATTENTION FUSION:
            - CLASSIFICATION MODEL: 1
            - ATTENTION MODEL: 1
        
    """

    # CLASIFICATION MODELS
    modality_models = {}
    if fusion_type in ["early_fusion", "intermediate_fusion", "attention_fusion"]:
        num_modalities = len(modalities)
        modality_models["combined"] = MultiModalModel(
            num_classes=51, num_modalities=num_modalities
        ).to(device)
    elif fusion_type == "late_fusion":
        for mod in modalities:
            modality_models[mod] = MultiModalModel(num_classes=51, num_modalities=1).to(
                device
            )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    # OPTIMIZERS
    modality_optimizers = {}
    if fusion_type in ["early_fusion", "intermediate_fusion", "attention_fusion"]:
        modality_optimizers["combined"] = create_optimizer(
            optimizer_name, modality_models["combined"].parameters(), lr=lr
        )
    elif fusion_type == "late_fusion":
        for modality_name in modalities:
            modality_optimizers[modality_name] = create_optimizer(
                optimizer_name, modality_models[modality_name].parameters(), lr=lr
            )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    # LOSS FUNCTION
    modality_loss_fn = {}
    if fusion_type in ["early_fusion", "intermediate_fusion", "attention_fusion"]:
        modality_loss_fn["combined"] = nn.CrossEntropyLoss()
    elif fusion_type == "late_fusion":
        for modality_name in modalities:
            modality_loss_fn[modality_name] = nn.CrossEntropyLoss()

    # --------------- AUTOENCODER MODEL INTERMEDIATE FUSION ----------------#

    # AUTOENCODER MODEL
    projection_autoencoders = {}
    if fusion_type == "intermediate_fusion":
        for modality_name in modalities:
            projection_autoencoders[modality_name] = Autoencoder().to(device)
    # AUTOENCODER OPTIMIZER
    projection_autoencoders_optimizers = {}
    if fusion_type == "intermediate_fusion":
        for modality_name in modalities:
            projection_autoencoders_optimizers[modality_name] = create_optimizer(
                optimizer_name,
                projection_autoencoders[modality_name].parameters(),
                lr=lr,
            )
    # AUTOENCODER LOSS FUNCTION
    projection_autoencoders_loss_fn = {}
    if fusion_type == "intermediate_fusion":
        for modality_name in modalities:
            projection_autoencoders_loss_fn[modality_name] = nn.MSELoss()

    # --------------- ATTENTION MODEL ATTENTIONO BASED FUSION ----------------#

    ### ATTENTION FUSION MODEL
    attention_fusion_model = None
    if fusion_type == "attention_fusion":
        attention_fusion_model = MultiModalAttentionFusion(
            embed_dim=2048, num_heads=4, num_modalities=len(modalities)
        ).to(device)

    ### ATTENTION FUSION OPTIMIZER
    attention_fusion_optimizer = None
    if fusion_type == "attention_fusion":
        attention_fusion_optimizer = create_optimizer(
            optimizer_name, attention_fusion_model.parameters(), lr=lr
        )

    ### ATTENTION FUSION LOSS FUNCTION
    attention_fusion_loss_fn = None
    if fusion_type == "attention_fusion":
        attention_fusion_loss_fn = nn.CrossEntropyLoss()

    print("========== TRAINING ==========")
    wandb.log({"message": "Training started"})

    # EARLY STOPPING VARIABLES
    LOSS_TOLERANCE = 0.01
    LOSS_OBSERVATION_WINDOW = 5
    loss_window = []
    best_loss = float("inf")

    for epoch in tqdm(range(epochs)):
        # Validation
        if epoch % validate_every == 0:
            description = f"Validation [Epoch: {epoch+1}/{epochs}]"
            output = evaluate(
                modality_models=modality_models,
                valid_loader=loaders["validation"],
                modality_loss_fn=modality_loss_fn,
                device=device,
                description=description,
                modalities=modalities,
                fusion_type=fusion_type,
                # AUTOENCODER VARIABLES
                projection_autoencoders=projection_autoencoders,
                projection_autoencoders_loss_fn=projection_autoencoders_loss_fn,
                projection_autoencoders_optimizers=projection_autoencoders_optimizers,
                # ATTENTION FUSION VARIABLES
                attention_fusion_model=attention_fusion_model,
                attention_fusion_loss_fn=attention_fusion_loss_fn,
            )

            # Log validation metrics to wandb
            wandb.log(
                {
                    "epoch": epoch,
                    "validation_loss": output["average_loss"],
                    "validation_accuracy": output["overall_acc"],
                }
            )

        # Training
        description = f"Training [Epoch: {epoch+1}/{epochs}]"
        train_mean_acc, train_mean_loss = train(
            modality_models=modality_models,
            train_loader=loaders["training"],
            modality_optimizers=modality_optimizers,
            modality_loss_fn=modality_loss_fn,
            device=device,
            description=description,
            modalities=modalities,
            fusion_type=fusion_type,
            # AUTOENCODER VARIABLES
            projection_autoencoders=projection_autoencoders,
            projection_autoencoders_loss_fn=projection_autoencoders_loss_fn,
            projection_autoencoders_optimizers=projection_autoencoders_optimizers,
            # ATTENTION FUSION VARIABLES
            attention_fusion_model=attention_fusion_model,
            attention_fusion_loss_fn=attention_fusion_loss_fn,
            attention_fusion_optimizer=attention_fusion_optimizer,
        )

        # Log training metrics to wandb
        wandb.log(
            {
                "epoch": epoch,
                "training_accuracy": train_mean_acc,
            }
        )

        if train_mean_acc > 0.9:
            print("========== EARLY STOPPING ==========")
            print("Training accuracy is above 90%")
            break

    wandb.log({"message": "Training finished"})

    print("========== SAVING MODELS ==========")
    # save model
    for modality_name, model in modality_models.items():
        torch.save(
            model.state_dict(),
            f"{fusion_type}_{experiment_name}_{modality_name}_epoch_{epoch}.pth",
        )

    # TESTING
    print("========== FINAL VALIDATION ==========")
    evaluate(
        modality_models,
        loaders["validation"],
        modality_loss_fn,
        device,
        modalities,
        fusion_type,
        description=f"Validation [Final]",
    )
    print("========== TESTING ==========")

    wandb.log({"message": "Testing started"})

    test_result = evaluate(
        modality_models=modality_models,
        valid_loader=loaders["testing"],
        modality_loss_fn=modality_loss_fn,
        device=device,
        description=description,
        modalities=modalities,
        fusion_type=fusion_type,
    )

    console_json_table(
        data={
            "overall_acc": test_result["overall_acc"],
            "average_loss": test_result["average_loss"],
        },
        headers=["Parameter", "Value"],
    )
    console_json_table(data=test_result["class_acc"], headers=["Class", "Accuracy"])

    wandb.log(
        {
            "testing_accuracy": test_result["overall_acc"],
            "testing_loss": test_result["average_loss"],
        }
    )

    wandb.log({"message": "Testing finished"})
