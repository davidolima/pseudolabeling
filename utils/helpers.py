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

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint...")
    directories_path = '/'.join(filename.split('/')[:-1])
    os.makedirs(directories_path, exist_ok = True)
    torch.save(state, filename)

def print_dict(d):
    print()
    for key, value in d.items():
        print(f'{key}: {value}')
    print()
