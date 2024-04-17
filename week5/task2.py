""" Main script for training a video classification model on HMDB51 dataset. """

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import wandb
import numpy
import matplotlib.pyplot as plt
from train import *


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
        true_loss_mean (float): Mean of mean losses for all batches
        acc_mean (float): Mean of accuracies for all batches
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
    print(true_loss_mean)
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
        best_acc (float, optional): Highest most recent accuracy to compare with the current one

    Returns:
        true_loss_mean (float): Mean of mean losses for all batches
        acc_mean (float): Mean of accuracies for all batches
        acc_per_class (float): Array containg accuracies per class
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
    acc_mean = (float(hits) / count)
    if acc_mean > best_acc:
        torch.save(model.state_dict(), f"model_{args.patience}_{args.min_delta}.pth")
    return true_loss_mean, acc_mean, acc_per_class


def bar_chart(acc_per_class, type):
    plt.figure(figsize=(10, 6))
    plt.tight_layout()
    plt.rcParams["font.size"] = "8"
    plt.xticks(rotation=90, ha='right')
    plt.bar(HMDB51Dataset.CLASS_NAMES, acc_per_class, align="edge")
    plt.xlabel("Classes")
    plt.ylabel("Accuracy(%)")
    plt.title(f"Accuracy per class on {type}")
    plt.savefig(f'./acc_per_class_{type}_{args.patience}_{args.min_delta}.png')


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

    config = {"patience": args.patience, "min_delta": args.min_delta}
    wandb.init(project="c6-w5-early-stopping",
               config=config,
               group=f"{args.patience}_{args.min_delta}",
               reinit=True)

    # Init model, optimizer, and loss function
    model = model_creator.create(args.model_name, args.load_pretrain,
                                 datasets["training"].get_num_classes())
    optimizer = create_optimizer(args.optimizer_name, model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model = model.to(args.device)
    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)
    wandb.watch(model, log="all", log_freq=10)

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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if epoch > args.es_start_epoch and early_stopper.early_stop(avg_loss):
                break
        # Training
        description = f"Training [Epoch: {epoch + 1}/{args.epochs}]"
        train_loss, train_acc = train(model, loaders['training'], optimizer, loss_fn, args.device, description=description)
        wandb.log({"train_acc": train_acc})
        wandb.log({"train_loss": train_loss})

    # Reload best model for evaluation
    model.load_state_dict(torch.load(f"model_{args.patience}_{args.min_delta}.pth"))

    # Testing
    val_loss, acc_val, val_acc_per_class = evaluate(model, loaders['validation'], loss_fn, args.device, description=f"Validation [Final]")
    test_loss, acc_test, test_acc_per_class = evaluate(model, loaders['testing'], loss_fn, args.device, description=f"Testing")

    print(f"Validation finished with: acc = {acc_val}, loss = {test_loss}")
    print(f"Testing finished with: acc = {acc_test}, loss = {test_loss}")

    bar_chart(val_acc_per_class, "validation dataset")
    bar_chart(test_acc_per_class, "testing dataset")

    wandb.finish()
    exit()
