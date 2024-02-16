"""
David de Oliveira Lima
Jan, 2024
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from typing import *

from utils.metrics import *
from utils.helpers import *
from sklearn.metrics import accuracy_score, f1_score

"""
TODO:
 - Choose which metric to watch and save each checkpoint.
 - Pass a list of metrics to be calculated each epoch.
"""

def alpha_coefficient(t: int, T1:int = 100, T2:int = 600, alpha_f = 3) -> float:
    if t < T1:
        return 0
    elif t < T2:
        return (t-T1)*alpha_f/(T2-T1)
    else:
        return alpha_f


def supervised_training(
        model: nn.Module,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        criterion: torch.nn.modules.loss._Loss,
        validation_loader: Optional[DataLoader] = None,
        checkpoint_name: Optional[str] = None,
        device: Optional[str] = None,
) -> nn.Module:
    """
    Function for training a model through a regular supervised learning approach.

    Returns: torch.nn.Module model.

    Parameters:
     - model: Model that will be trained.
     - epochs: How many epochs the model will be trained for.
     - train_loader: torch.utils.data.DataLoader object to be used for the
                     training.
     - criterion: Loss function to be used.
     - test_loader: torch.utils.data.DataLoader object for evaluation of
                    the model.
     - metrics: Array of metrics to be calculated on each epoch.
     - validation_loader: Optional. torch.utils.data.DataLoader object used
                          as validation during training.
     - checkpoint_name: Name of the model's checkpoint file. Leave as `None` to
                        not save any checkpoints.
                        NOTE: currently saves checkpoints based on loss, acc and f1.
     - device: Which device the training will take place on. Leave as `None`
               to detect automatically.
    """

    if not device:
        device = get_device()

    model.to(device)
    best_loss = np.inf
    best_acc = -np.inf
    best_f1 = -np.inf

    for epoch in range(0, epochs):
        total = 0
        running_loss = 0
        running_acc = 0
        running_f1 = 0

        model.train()
        supervised_bar = tqdm(train_loader)
        for x, y in supervised_bar:
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)

            running_loss += loss.item()*x.size(0)
            running_acc += calculate_accuracy(y_hat, y)*x.size(0)
            running_f1 += calculate_f1_score(y_hat, y)*x.size(0)
            total += x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            supervised_bar.set_description(f"Loss {running_loss/total:.2f} Acc {running_acc/total:.2f} F1 {running_f1/total:.2f}")

        print(f"[Epoch {epoch}/{epochs}] Loss {running_loss/total:.2f} Accuracy: {running_acc/total:.2f} F1-Score: {running_f1/total:.2f}", end=' ')

        val_loss, val_acc, val_f1 = evaluate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        val_f1 = val_f1[0] # HACK
        print(f"Val Loss {val_loss:.2f} Val Acc {val_acc:.2f} Val F1 {val_f1:.2f}")

        if val_loss < best_loss:
            print(f"[!] New Best Loss: {best_loss} -> {val_loss}. ", end='')
            if checkpoint_name is not None:
                save_checkpoint(
                    filename=checkpoint_name + "-best_loss",
                    model=model,
                    optimizer=optimizer,
                    current_epoch=epoch
                )
            best_loss = val_loss
        if val_acc > best_acc:
            print(f"[!] New Best Accuracy: {best_acc} -> {val_acc}. ", end='')
            if checkpoint_name is not None:
                save_checkpoint(
                    filename=checkpoint_name + "-best_accuracy",
                    model=model,
                    optimizer=optimizer,
                    current_epoch=epoch
                )
            best_acc = val_acc
        if val_f1 > best_f1:
            print(f"[!] New Best F1 Score: {best_f1} -> {val_f1}. ", end='')
            if checkpoint_name is not None:
                save_checkpoint(
                    filename=checkpoint_name + "-best_f1-score",
                    model=model,
                    optimizer=optimizer,
                    current_epoch=epoch
                )
            best_f1 = val_f1

    return model

def semisupervised_training(
        model: nn.Module,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        labelled_loader: DataLoader,
        unlabelled_loader: DataLoader,
        criterion: torch.nn.modules.loss._Loss,
        supervised_step: int,
        validation_loader: Optional[DataLoader] = None,
        checkpoint_name: Optional[str] = None,
        device: Optional[str] = None,
        unlabelled_weight: Optional[Callable] = alpha_coefficient,
) -> nn.Module:
    """
    Function for training a model using labelled and unlaballed samples through pseudo-labels.

    Returns: torch.nn.Module model.

    Parameters:
     - model: Model that will be trained.
     - epochs: How many epochs the model will be trained for.
     - optimizer: Optimizer to be used during training.
     - labelled_loader: torch.utils.data.DataLoader object for loading labelled
                        samples of the dataset.
     - unlabelled_loader: torch.utils.data.DataLoader object for loading unlabelled
                        samples of the dataset.
     - criterion: Loss function to be used.
     - supervised_step: Number of epochs to wait until the labelled set forward pass happens.
     - validation_loader: Optional. torch.utils.data.DataLoader object used
                          as validation during training.
     - checkpoint_name: Name of the model's checkpoint file. Leave as `None` to
                        not save any checkpoints.
                        NOTE: currently saves checkpoints based on loss, acc and f1.
     - device: Which device the training will take place on. Leave as `None`
               to detect automatically.
     - unlabelled_weight: Function to be used to weight the unlabelled data loss during
                          training. Must accept the current epoch as the only obligatory
                          parameter. (Default: alpha_coefficient(t))
    """

    if not device:
        device = get_device()

    model.to(device)
    best_loss = np.inf
    best_acc = -np.inf
    best_f1 = -np.inf

    for epoch in range(0, epochs):
        total = 0
        running_loss = 0
        running_acc = 0
        running_f1 = 0

        model.train()
        unlabelled_bar = tqdm(unlabelled_loader)
        for x in unlabelled_bar:
            x = x.to(device)

            model.eval()
            _, pseudolabel = torch.max(model(x), 1)
            model.train()

            y_hat = model(x)
            unlabelled_loss =  unlabelled_weight(epoch) * criterion(y_hat, pseudolabel)

            running_loss += unlabelled_loss.item()*x.size(0)
            running_acc += calculate_accuracy(y_hat, pseudolabel)*x.size(0)
            running_f1 += calculate_f1_score(y_hat, pseudolabel)*x.size(0)
            total += x.size(0)

            optimizer.zero_grad()
            unlabelled_loss.backward()
            optimizer.step()

            unlabelled_bar.set_description(f"Loss {running_loss/total:.2f} Acc {running_acc/total:.2f} F1 {running_f1/total:.2f}")

        # One forward pass through the labelled dataset every `supervised_step` epochs
        if epoch % supervised_step == 0:
            for x, y in tqdm(labelled_loader, desc="[Supervised forward pass]"):
                x, y = x.to(device), y.to(device)

                y_hat = model(x)
                labelled_loss = criterion(y_hat, y)

                optimizer.zero_grad()
                labelled_loss.backward()
                optimizer.step()

        print(f"[Epoch {epoch}/{epochs}] Loss {running_loss/total:.2f} Accuracy: {running_acc/total:.2f} F1-Score: {running_f1/total:.2f}", end=' ')

        val_loss, val_acc, val_f1 = evaluate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        val_f1 = val_f1[0] # HACK
        print(f"Val Loss {val_loss:.2f} Val Acc {val_acc:.2f} Val F1 {val_f1:.2f}")

        if val_loss < best_loss:
            print(f"[!] New Best Loss: {best_loss} -> {val_loss}. ", end='')
            if checkpoint_name is not None:
                save_checkpoint(
                    filename=checkpoint_name + "-best_loss",
                    model=model,
                    optimizer=optimizer,
                    current_epoch=epoch
                )
            best_loss = val_loss
        if val_acc > best_acc:
            print(f"[!] New Best Accuracy: {best_acc} -> {val_acc}. ", end='')
            if checkpoint_name is not None:
                save_checkpoint(
                    filename=checkpoint_name + "-best_accuracy",
                    model=model,
                    optimizer=optimizer,
                    current_epoch=epoch
                )
            best_acc = val_acc
        if val_f1 > best_f1:
            print(f"[!] New Best F1 Score: {best_f1} -> {val_f1}. ", end='')
            if checkpoint_name is not None:
                save_checkpoint(
                    filename=checkpoint_name + "-best_f1-score",
                    model=model,
                    optimizer=optimizer,
                    current_epoch=epoch
                )
            best_f1 = val_f1

    return model

def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: torch.nn.modules.loss._Loss,
        device: Optional[str] = None,
        metrics: Optional[list[str]] = None,
):
    """
    Evaluates a given model and returns its loss, accuracy and f1-score performance of a given
    Dataset.

    Returns: Tuple containing loss, accuracy and specified metrics.

    Parameters:
     - model: Model that will be trained
     - dataloader: torch.utils.data.DataLoader object for loading custom dataset.
     - criterion: Loss function to be used.
     - device: Which device the training will take place on. Leave as `None`
               to detect automatically.
     - metrics: List of metric functions to calculate during evaluation.
    """
    if not device:
        device = get_device()

    model.to(device)
    model.eval()

    val_total = 0
    val_running_loss = 0

    if metrics is None:
        metrics = [accuracy_score, lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro")]
    else:
        metrics = [accuracy_score] + metrics

    running_metrics = [0] * len(metrics)

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        _, y_hat = torch.max(y_hat, 1)

        # Update running metrics
        for i in range(len(metrics)):
            running_metrics[i] += metrics[i](y_hat.cpu().detach().numpy(), y.cpu().detach().numpy()) * x.size(0)
        val_running_loss += loss.item()*x.size(0)
        val_total += x.size(0)

    # Get metrics' mean
    val_running_loss /= val_total
    for i in range(len(running_metrics)):
        running_metrics[i] /= val_total

    val_accuracy = running_metrics.pop(0)
    model.train()
    return val_running_loss, val_accuracy, running_metrics
