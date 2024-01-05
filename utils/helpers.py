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

def handle_config_arguments(argv):
    defaults = {
        "epochs": 10,
        "labelled_batch_size": 128,
        "unlabelled_batch_size": 256,
        "num_classes": 2,
        "lr": 1e-5,
        "T1": 1,
        "T2": 6,
        "alpha_f": .03,
    }
    if len(argv) < 1:
        print("[!] Using the default configuration.")
        return defaults
    else:
        raise NotImplementedError()


def print_dict(d):
    for key, value in d.items():
        print(f'{key}: {value}')
