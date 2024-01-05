import torch
import os

def load_checkpoint(checkpoint_path, model, optimizer, device):
    # TODO: add and check device
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

    # load variables
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        correct_state_dict = {}
        for key, val in checkpoint['state_dict'].items():
            correct_state_dict[key[len('module.'):]] = val
        model.load_state_dict(correct_state_dict)
    ## add option to pass no optimizer (e.g. in inference)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']

    return step

def save_checkpoint(
        filename: str,
        model: torch.nn.modules.module.Module,
        optimizer: torch.optim.Optimizer,
        current_epoch: int # In order to know how many epochs the model has been trained for
    ):

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
    print()
    for key, value in d.items():
        print(f'{key}: {value}')
    print()
