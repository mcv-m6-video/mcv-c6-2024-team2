""" Main script for training a video classification model on HMDB51 dataset. """

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator
import numpy as np

from torch.utils.data import DataLoader

from datasets.HMDB51Dataset import HMDB51Dataset
from models import model_creator
from utils import model_analysis
from utils import statistics

import wandb
import optuna

parser = argparse.ArgumentParser(description='Train a video classification model on HMDB51 dataset.')
parser.add_argument('frames_dir', type=str, 
                    help='Directory containing video files')
parser.add_argument('--annotations-dir', type=str, default="/ghome/group04/c6-diana/week5/data/hmdb51/testTrainMulti_601030_splits",
                    help='Directory containing annotation files')
parser.add_argument('--clip-length', type=int, default=4,
                    help='Number of frames of the clips')
parser.add_argument('--crop-size', type=int, default=182,
                    help='Size of spatial crops (squares)')
parser.add_argument('--temporal-stride', type=int, default=4,
                    help='Receptive field of the model will be (clip_length * temporal_stride) / FPS')
parser.add_argument('--model-name', type=str, default='x3d_xs',
                    help='Model name as defined in models/model_creator.py')
parser.add_argument('--load-pretrain', action='store_true', default=False,
                help='Load pretrained weights for the model (if available)')
parser.add_argument('--optimizer-name', type=str, default="adam",
                    help='Optimizer name (supported: "adam" and "sgd" for now)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Batch size for the training data loader')
parser.add_argument('--batch-size-eval', type=int, default=16,
                    help='Batch size for the evaluation data loader')
parser.add_argument('--validate-every', type=int, default=5,
                    help='Number of epochs after which to validate the model')
parser.add_argument('--num-workers', type=int, default=2,
                    help='Number of worker processes for data loading')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use for training (cuda or cpu)')
parser.add_argument('--segment-number', type=str, default=3,
                    help='The amount of segments to divide the video in')


def train(
        model: nn.Module,
        train_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: nn.Module,
        device: str,
        segment_number: int,
        description: str = ""
    ) -> None:
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
        None
    """
    model.train()
    pbar = tqdm(train_loader, desc=description, total=len(train_loader))
    loss_train_mean = statistics.RollingMean(window_size=len(train_loader))
    hits = count = 0 # auxiliary variables for computing accuracy
    
    for batch in pbar:
        sum = None
        for i in range(segment_number):
            # Gather batch and move to device
            clips, labels = torch.tensor(np.array(batch['clips']))[:, i].to(device), batch['labels'].to(device)

            # Forward pass
            outputs = model(clips)

            if sum is None:
                sum = torch.zeros_like(outputs, device='cuda:0')

            sum += outputs

        outputs = sum / segment_number

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
            acc_mean=(float(hits) / count)
        )

    wandb.log({"train_mean_accuracy": float(hits) / count,
            "train_accuracy": float(hits_iter) / len(labels),
            "train_loss_mean": loss_train_mean(loss_iter),
            "train_crop_size": loss_iter})

    return loss_iter


def evaluate(
        model: nn.Module, 
        valid_loader: DataLoader, 
        loss_fn: nn.Module,
        device: str,
        segment_number: int,
        description: str = "", 
    ) -> None:
    """
    Evaluates the given model using the provided data loader and loss function.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        None
    """
    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
    hits = count = 0 # auxiliary variables for computing accuracy

    for batch in pbar:
        sum = None
        for i in range(segment_number):
            # Gather batch and move to device
            clips, labels = torch.tensor(np.array(batch['clips']))[:, i].to(device), batch['labels'].to(device)

            # Forward pass
            outputs = model(clips)

            if sum is None:
                sum = torch.zeros_like(outputs, device='cuda:0')
 
            sum += outputs

        # # Gather batch and move to device
        # clips, labels = batch['clips'].to(device), batch['labels'].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(clips)
            # Compute loss (just for logging, not used for backpropagation)
            loss = loss_fn(outputs, labels) 
            # Compute metrics
            loss_iter = loss.item()
            hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
            hits += hits_iter
            count += len(labels)
            # Update progress bar with metrics
            pbar.set_postfix(
                loss=loss_iter,
                loss_mean=loss_valid_mean(loss_iter),
                acc=(float(hits_iter) / len(labels)),
                acc_mean=(float(hits) / count)
            )

    wandb.log({"val_mean_accuracy": float(hits) / count,
        "val_accuracy": float(hits_iter) / len(labels),
        "val_loss_mean": loss_valid_mean(loss_iter),
        "val_crop_size": loss_iter})


def create_datasets(
        frames_dir: str,
        annotations_dir: str,
        split: HMDB51Dataset.Split,
        clip_length: int,
        crop_size: int,
        temporal_stride: int,
        segment_number: int
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
            segment_number
        )
    
    return datasets


def create_dataloaders(
        datasets: Dict[str, HMDB51Dataset],
        batch_size: int,
        batch_size_eval: int = 8,
        num_workers: int = 2,
        pin_memory: bool = True
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
            batch_size=(batch_size if key == 'training' else batch_size_eval),
            shuffle=(key == 'training'),  # Shuffle only for training dataset
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
            
    return dataloaders


def create_optimizer(optimizer_name: str, parameters: Iterator[nn.Parameter], lr: float = 1e-4) -> torch.optim.Optimizer:
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
        print_model: bool = True,
        print_params: bool = True,
        print_FLOPs: bool = True
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
        #num_params = model_analysis.calculate_parameters(model) # should be equivalent
        print(f"Number of parameters (M): {round(num_params / 10e6, 2)}")

    if print_FLOPs:
        num_FLOPs = model_analysis.calculate_operations(model, clip_length, crop_size, crop_size)
        print(f"Number of FLOPs (G): {round(num_FLOPs / 10e9, 2)}")


def objective(trial):
    args = parser.parse_args()

    params = {
        'temporal_stride': trial.suggest_categorical('temporal_stride', [4, 8]),
        'clip_length': trial.suggest_categorical('clip_length', [4]),
        'epochs': trial.suggest_categorical('epochs', [30]),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
        'segment_number': trial.suggest_categorical('segment_number', [3, 5]),
        'lr': trial.suggest_categorical('lr', [0.001, 0.1]),
        'optimizer_name': trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    }
    
    config = dict(trial.params)
    config['trial.number'] = trial.number

    wandb.init(
        project='c6_more epochs',
        config=config,
        reinit=True,
    )


    # Create datasets
    datasets = create_datasets(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=params['clip_length'],
        crop_size=args.crop_size,
        temporal_stride=params['temporal_stride'],
        segment_number=params['segment_number']
    )

    # Create data loaders
    loaders = create_dataloaders(
        datasets,
        params['batch_size'],
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )


    # try:
    # Initialize model, optimizer, and loss function
    model = model_creator.create(args.model_name, args.load_pretrain, datasets["training"].get_num_classes())
    optimizer = create_optimizer(params['optimizer_name'], model.parameters(), lr=params['lr'])
    loss_fn = nn.CrossEntropyLoss()

    # Print model summary
    print_model_summary(model, params['clip_length'], args.crop_size)

    model = model.to(args.device)

    # Define early stopping parameters
    patience = 200
    min_delta = 0.001
    best_val_loss = np.Inf
    current_patience = 0

    segment_number = params['segment_number']

    epochs = params['epochs']
    for epoch in range(epochs):
        # Validation
        if epoch % args.validate_every == 0:
            description = f"Validation [Epoch: {epoch+1}/{epochs}]"
            evaluate(model, loaders['validation'], loss_fn, args.device, segment_number, description=description)
        # Training
        description = f"Training [Epoch: {epoch+1}/{epochs}]"
        train_loss = train(model, loaders['training'], optimizer, loss_fn, args.device, segment_number, description=description)

        if train_loss < best_val_loss - min_delta:
            best_val_loss = train_loss
            current_patience = 0

            # Save the best model
            print("Best model. Saving weights")
            
            model_path = './weights_more.pth'
            torch.save(model.state_dict(), model_path)
        else:
            current_patience += 1
            if current_patience > patience:
                print("Early stopping.")
                break

    # Testing
    val_loss = evaluate(model, loaders['validation'], loss_fn, args.device, segment_number, description=f"Validation [Final]")
    test_loss = evaluate(model, loaders['testing'], loss_fn, args.device, segment_number, description=f"Testing")

    return val_loss
    
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name='c6')
    study.optimize(objective, n_trials=100)

    exit()
