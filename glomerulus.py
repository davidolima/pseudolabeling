#!./bin/python3

"""
David de Oliveira Lima
Fev, 2024
"""

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchvision.transforms as T

from utils.metrics import *

import sys
import argparse

from utils.generic_dataset import GenericDataset
from utils.helpers import *
from train import *

if __name__ == "__main__":
    # Setup
    root = "/datasets/glomerulus-kaggle"
    criterion = nn.CrossEntropyLoss()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Semi-supervised training for the glomerulus dataset using pseudo-labels.")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="Input batch size. (default: 64)")
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="Number of epochs. (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-5, metavar="LR",
                        help="Adjust the learning rate. (default: 1e-5)")
    parser.add_argument("--num_classes", type=int, default=4, metavar="N",
                        help="Number of classes. (default: 4)")
    parser.add_argument("--skip_supervised", type=bool, default=False, metavar="B",
                        help="Skips to the training using pseudolabels. (default: False)")
    parser.add_argument("--skip_supervised_evaluation", type=bool, default=False, metavar="B",
                        help="Skips evaluation of the model after supervised training and before semisupervised training. (default: False)")
    configs = parser.parse_args().__dict__

    model, preprocess = get_model("efficientnet_b0", configs["num_classes"])
    opt = AdamW(model.parameters(), lr=configs['lr'])
    
    T_train = T.Compose([
        T.Resize((224,224), antialias=True),
        T.ToTensor(),
        T.RandomHorizontalFlip(.5),
        T.Normalize((.5, .5, .5), (.5, .5, .5), inplace=True),
    ])

    T_test = T.Compose([
        T.Resize((224,224), antialias=True),
        T.ToTensor(),
        T.Normalize((.5, .5, .5), (.5, .5, .5), inplace=True),
    ])

    print("-- Current configuration --------------")
    [print(f"{key}: {value}") for key, value in configs.items()]
    print("---------------------------------------")

    # Load data
    labelled_set   = GenericDataset(root, ["train"], transforms=T_train)
    labelled_set, validation_set = labelled_set.split(.5, shuffle=True)
    validation_set, test_set = validation_set.split(.5, shuffle=True)

    unlabelled_set = GenericDataset(root, ["test"], ignore_unlabelled=False, transforms=T_train)
    print(f"[!] {sum(map(len, [labelled_set,validation_set,test_set]))} labelled images were loaded in total.")

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

    if not configs["skip_supervised"]:
        # Supervised learning
        model = supervised_training(
            model=model,
            epochs=configs["epochs"],
            optimizer=opt,
            train_loader=labelled_loader,
            criterion=criterion,
            validation_loader=validation_loader,
            checkpoint_name="glomerulus-supervised_model",
        )

    print("[!] Beginning evaluation...")
    test_loader = DataLoader(
        test_set,
        batch_size=configs["batch_size"],
        shuffle=True,
    )

    # Evaluate model over test set.
    model, _ = get_model("efficientnet_b0", configs["num_classes"])
    load_checkpoint("checkpoints/glomerulus-supervised_model-best_loss.pt", model, opt, device=None)

    if not configs["skip_supervised_evaluation"]:
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
        print(f"[-] Evaluation Results: Loss: {test_loss} Accuracy: {test_acc} F1-Score: {test_f1}")

    ## Semisupervised training - Pseudolabels
    # Proceed to training using pseudolabels
    unlabelled_loader = DataLoader(
        unlabelled_set,
        batch_size=configs["batch_size"],
        shuffle=True,
    )
    print("[!] Beginning Semisupervised Training...")
    model = semisupervised_training(
        model=model,
        epochs=configs["epochs"],
        optimizer=opt,
        labelled_loader=labelled_loader,
        unlabelled_loader=unlabelled_loader,
        criterion=criterion,
        supervised_step=50,
        validation_loader = validation_loader,
        checkpoint_name = "glomerulus-semisupervised_model",
    )
    
    # Evaluate model over test set.
    model, _ = get_model("efficientnet_b0", configs["num_classes"])
    load_checkpoint("glomerulus-semisupervised_model-best_loss.pt", model, opt, device=None)
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
    print(f"[-] Evaluation Results: Loss: {test_loss} Accuracy: {test_acc} F1-Score: {test_f1}")
