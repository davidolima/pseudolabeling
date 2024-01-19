#!/usr/bin/env python3

"""
David de Oliveira Lima
Jan, 2024
"""

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchvision.transforms as T

from utils.metrics import *

import sys
import argparse

from utils.data import LabelledSet, UnlabelledSet #, FullRadiographDataset
from utils.helpers import *
from train import *

if __name__ == "__main__":
    # Setup
    root = "/datasets/pan-radiographs/"
    criterion = nn.CrossEntropyLoss()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Semi-supervised training for the OdontoAI dataset using pseudo-labels.")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="Input batch size. (default: 64)")
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="Number of epochs. (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-5, metavar="LR",
                        help="Adjust the learning rate. (default: 1e-5)")
    parser.add_argument("--num_classes", type=int, default=2, metavar="N",
                        help="Number of classes. (default: 2)")
    configs = parser.parse_args().__dict__

    T_train = T.Compose([ # Transformations, model and optimizer from Hougaz et al. (2023).
        T.Resize((224,224), antialias=True),
        T.RandomHorizontalFlip(.5),
        T.ToTensor(),
        T.Normalize((.5, .5, .5), (.5, .5, .5), inplace=True),
    ])

    T_test = T.Compose([
        T.Resize((224,224), antialias=True),
        T.ToTensor()
    ])

    print("Labelled set ", end='')
    labelled_set   = LabelledSet  (root, list(range( 1,20)), T_train)
    print("Validation set ", end='')
    validation_set = LabelledSet  (root, list(range( 20,21)), T_train)
    print("Test set ", end='')
    test_set       = LabelledSet  (root, list(range(21,31)), T_test)

    print(f"[!] {sum(map(len, [labelled_set,validation_set,test_set]))} images were loaded in total.")

    labelled_loader = DataLoader(
        labelled_set,
        batch_size=configs["batch_size"],
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=configs["batch_size"],
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=configs["batch_size"],
        shuffle=True,
    )

    model = get_model("efficientnet_b0", configs["num_classes"])
    opt = AdamW(model.parameters(), lr=configs['lr'])

    print("-- Current configuration --------------")
    [print(f"{key}: {value}") for key, value in configs.items()]
    print("---------------------------------------")

    # Supervised learning
    model = supervised_training(
        model=model,
        epochs=configs["epochs"],
        optimizer=opt,
        train_loader=labelled_loader,
        criterion=criterion,
        validation_loader=validation_loader,
        checkpoint_name="supervised_model",
    )

    print("[!] Beginning evaluation...")
    test_loader = DataLoader(
        test_set,
        batch_size=configs["batch_size"],
        shuffle=True,
    )

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
    print(f"Evaluation Results: Loss: {test_loss} Accuracy: {test_acc} F1-Score: {test_f1}")

    # Proceed to training using pseudolabels
    model = get_model("efficientnet_b0", configs["num_classes"])
    model = load_checkpoint("supervised_model.pt", model, opt, device=None)
