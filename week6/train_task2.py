""" Main script for training a video classification model on HMDB51 dataset. """

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator

from torch.utils.data import DataLoader

from datasets.HMDB51Dataset import HMDB51Dataset
from utils import model_creator
from utils import model_analysis
from utils import statistics

import numpy
import matplotlib.pyplot as plt
import wandb

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str,
        description: str = ""
):
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
    hits = count = 0  # auxiliary variables for computing accuracy
    losses = []

    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        # Forward pass
        if args.model_name == 'resnet' or args.model_name == 'vgg':
            clips = clips.view(-1, 3, args.crop_size, args.crop_size)
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

        loss_mean = loss_train_mean(loss_iter)
        losses.append(loss_mean)
        pbar.set_postfix(
            loss=loss_iter,
            loss_mean=loss_mean,
            acc=(float(hits_iter) / len(labels)),
            acc_mean=(float(hits) / count)
        )
    true_loss_mean = numpy.mean(losses)
    acc_mean = (float(hits) / count)
    return true_loss_mean, acc_mean


def evaluate(
        model: nn.Module,
        valid_loader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        description: str = "",
        best_acc: float = 0.0):
    """
    Evaluates the given model using the provided data loader and loss function.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".
        best_acc (float, optional)
        acc_per_class (array)

    Returns:
        None
    """

    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
    hits = count = 0  # auxiliary variables for computing accuracy
    correct_preds_per_class = numpy.zeros(len(HMDB51Dataset.CLASS_NAMES))
    class_count = numpy.zeros(len(HMDB51Dataset.CLASS_NAMES))

    losses = []

    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        # Forward pass
        with torch.no_grad():
            if args.model_name == 'resnet' or args.model_name == 'vgg':
                clips = clips.view(-1, 3, args.crop_size, args.crop_size)
            outputs = model(clips)
            # Compute loss (just for logging, not used for backpropagation)
            loss = loss_fn(outputs, labels)
            # Compute metrics
            loss_iter = loss.item()
            hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
            hits += hits_iter
            count += len(labels)

            # Populate bar chart
            for out, label in zip(outputs.argmax(dim=1), labels):
                class_count[label.item()] += 1
                if out == label:
                    correct_preds_per_class[out] = correct_preds_per_class[out] + 1

            loss_mean = loss_valid_mean(loss_iter)
            losses.append(loss_mean)
            # Update progress bar with metrics
            pbar.set_postfix(
                loss=loss_iter,
                loss_mean=loss_valid_mean(loss_iter),
                acc=(float(hits_iter) / len(labels)),
                acc_mean=(float(hits) / count)
            )

    acc_per_class = (correct_preds_per_class / class_count) * 100
    true_loss_mean = numpy.mean(losses)
    print(true_loss_mean)
    acc_mean = (float(hits) / count)
    if acc_mean > best_acc:
        torch.save(model.state_dict(), f"{args.model_name}_{args.patience}_{args.min_delta}.pth")
    return true_loss_mean, acc_mean, acc_per_class


def create_datasets(
        frames_dir: str,
        annotations_dir: str,
        split: HMDB51Dataset.Split,
        clip_length: int,
        crop_size: int,
        temporal_stride: int
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
            temporal_stride
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


def create_optimizer(optimizer_name: str, parameters: Iterator[nn.Parameter],
                     lr: float = 1e-4) -> torch.optim.Optimizer:
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
        # num_params = model_analysis.calculate_parameters(model) # should be equivalent
        print(f"Number of parameters (M): {round(num_params / 10e6, 2)}")

    if print_FLOPs:
        num_FLOPs = model_analysis.calculate_operations(model, clip_length, crop_size, crop_size)
        print(f"Number of FLOPs (G): {round(num_FLOPs / 10e9, 2)}")


def bar_chart(acc_per_class, type):
    plt.figure(figsize=(10, 6))
    plt.tight_layout()
    plt.rcParams["font.size"] = "8"
    plt.xticks(rotation=90, ha='right')
    plt.bar(HMDB51Dataset.CLASS_NAMES, acc_per_class, align="edge")
    plt.xlabel("Classes")
    plt.ylabel("Accuracy(%)")
    plt.title(f"Accuracy per class on {type}")
    plt.savefig(f'./plots/acc_per_class_{args.model_name}_{args.patience}_{args.min_delta}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a video classification model on HMDB51 dataset.')
    parser.add_argument('--frames_dir', type=str,
                        help='Directory containing video files')
    parser.add_argument('--annotations-dir', type=str, default="data/hmdb51/testTrainMulti_601030_splits",
                        help='Directory containing annotation files')
    parser.add_argument('--clip-length', type=int, default=4,
                        help='Number of frames of the clips')
    parser.add_argument('--crop-size', type=int, default=182,
                        help='Size of spatial crops (squares)')
    parser.add_argument('--temporal-stride', type=int, default=12,
                        help='Receptive field of the model will be (clip_length * temporal_stride) / FPS')
    parser.add_argument('--model-name', type=str, default='x3d_xs',
                        help='Model name as defined in models/model_creator_experiments.py')
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
    parser.add_argument('--es-start-epoch', type=int, default=100,
                        help='Number of epochs to wait before starting to monitor improvement for early stopping.')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.01)

    args = parser.parse_args()

    # Create datasets
    datasets = create_datasets(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1,  # hardcoded
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        temporal_stride=args.temporal_stride
    )

    # Create data loaders
    loaders = create_dataloaders(
        datasets,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )

    config = {"model_name": args.model_name, "patience": args.patience, "min_delta": args.min_delta}
    wandb.init(project="c6-w6-task2",
               config=config,
               group=f"{args.model_name}_{args.patience}_{args.min_delta}",
               reinit=True)

    # Init model, optimizer, and loss function
    model = model_creator_experiments.create_from_torchhub(args.model_name, True,
                                 datasets["training"].get_num_classes())
    optimizer = create_optimizer(args.optimizer_name, model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    wandb.watch(model, log="all", log_freq=10)

    print_model_summary(model, args.clip_length, args.crop_size)

    model = model.to(args.device)
    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    best_val_acc = 0.
    for epoch in range(args.epochs):
        wandb.log({"epoch": epoch + 1})
        # Validation
        if epoch % args.validate_every == 0:
            description = f"Validation [Epoch: {epoch + 1}/{args.epochs}]"
            avg_loss, val_acc, _ = evaluate(model, loaders['validation'], loss_fn, args.device,
                                            description=description, best_acc=best_val_acc)
            wandb.log({"val_acc": val_acc})
            wandb.log({"val_loss": avg_loss})

            print(best_val_acc, val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if epoch > args.es_start_epoch and early_stopper.early_stop(avg_loss):
                break

        # Training
        description = f"Training [Epoch: {epoch + 1}/{args.epochs}]"
        train_loss, train_acc = train(model, loaders['training'], optimizer, loss_fn, args.device, description=description)
        wandb.log({"train_acc": train_acc})
        wandb.log({"train_loss": train_loss})

    # Reload best model
    model.load_state_dict(torch.load(f"{args.model_name}_{args.patience}_{args.min_delta}.pth"))

    # Testing
    val_loss, acc_val, val_acc_per_class = evaluate(model, loaders['validation'], loss_fn, args.device, description=f"Validation [Final]")
    test_loss, acc_test, test_acc_per_class = evaluate(model, loaders['testing'], loss_fn, args.device, description=f"Testing")

    print(f"Validation finished with: acc = {acc_val}, loss = {test_loss}")
    print(f"Testing finished with: acc = {acc_test}, loss = {test_loss}")

    bar_chart(val_acc_per_class, "validation dataset")
    bar_chart(test_acc_per_class, "testing dataset")

    wandb.finish()
    exit()
