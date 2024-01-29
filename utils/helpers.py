"""
David de Oliveira Lima,
2024
"""

import torch
import torch.nn as nn
import torch.utils.data
from typing import Optional

import torchvision.models as models
import os

from utils.metrics import *

def load_checkpoint(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[str] = None,
) -> None:
    if device is None:
        device = get_device()
        model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

    # load variables
    try:
        model.load_state_dict(checkpoint['state_dict']())
    except:
        correct_state_dict = {}
        for key, val in checkpoint['state_dict']().items():
            correct_state_dict[key[len('module.'):]] = val
        model.load_state_dict(correct_state_dict)
    ## add option to pass no optimizer (e.g. in inference)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optmizer']())

def save_checkpoint(
        filename: str,
        model: torch.nn.modules.module.Module,
        optimizer: torch.optim.Optimizer,
        current_epoch: int # In order to know how many epochs the model has been trained for
    ) -> None:

    if not filename.endswith(".pt"):
        filename += ".pt"

    print(f"Saving checkpoing to file '{filename}'...", end='')
    checkpoint = {
        "state_dict": model.state_dict,
        "optmizer": optimizer.state_dict,
        "epoch": current_epoch
    }
    torch.save(checkpoint, filename)
    print("Saved.")

def print_dict(d):
    for key, value in d.items():
        print(f'{key}: {value}')

def get_model(model_str: str, num_classes: int) -> nn.Module:
    match (model_str):
        case "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            preprocess = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
        case _:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            preprocess = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    for params in model.parameters():
        params.requires_grad = True

    return model, preprocess

def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", end=' ')
    if device == "cuda":
        print(torch.cuda.get_device_name())
    else:
        print(device)

    return device
