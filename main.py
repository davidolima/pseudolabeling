#!/bin/python3

# %% [markdown]
# # Imports

# %%
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import accuracy_score, f1_score

import numpy as np
import datetime as dt
from tqdm import tqdm
from train import *
from utils.data import LabelledSet, UnlabelledSet #, FullRadiographDataset
from utils.helpers import *

# %%
# Dataset radiografias
root = "/datasets/pan-radiographs/"
T_train = T.Compose([ # Transformations, model and optimizer from Hougaz, 2022
    T.Resize((224,224), antialias=True),
    T.RandomHorizontalFlip(.5),
    T.ToTensor(),
    T.Normalize((.5, .5, .5), (.5, .5, .5), inplace=True),
])

T_test = T.Compose([
    T.Resize((224,224), antialias=True),
    T.ToTensor()
])

# In this case we are removing labelled data to test the reliability of pseudolabelling,
# but I am separating the labelled and the unlabelled set so that it's easier to
# to reutilize this code to other datasets.
labelled_set = LabelledSet(root, list(range(1,20)), T_train)
test_set = LabelledSet(root, list(range(20,30)), T_test)
unlabelled_set = UnlabelledSet(root, list(range(30,31)), T_train)

print(f"[!] {sum(map(len, [labelled_set,test_set,unlabelled_set]))} images were loaded in total.")

# %% [markdown]
# # Setup

# %%

# Configuration
configs = {
    "epochs": 10,
    "labelled_batch_size": 128,
    "unlabelled_batch_size": 256,
    "num_classes": 2,
    "lr": 1e-5,
    "T1": 1,
    "T2": 6,
    "alpha_f": .03,
}
criterion = nn.BCELoss()
device = "cuda" if torch.cuda.is_available() else 'cpu'

print("-- Current configuration --------------")
[print(f"{key}: {value}") for key, value in configs.items()]
print("---------------------------------------")

# %%
def alpha(t, T1=100, T2=600, alpha_f=3):
    if t < T1:
        return 0
    elif t < T2:
        return alpha_f*(t-T1)/(T2-T1)
    else: # T2 <= t
        return alpha_f

# %%

labelled_loader = DataLoader(
    labelled_set,
    batch_size=configs["labelled_batch_size"],
    shuffle=True,
)

unlabelled_loader = DataLoader(
    unlabelled_set,
    batch_size=configs["unlabelled_batch_size"],
    shuffle=False,
)

# %%
# def get_state_dict(self, *args, **kwargs):
#     kwargs.pop("check_hash")
#     return load_state_dict_from_url(self.url, *args, **kwargs)
# WeightsEnum.get_state_dict = get_state_dict

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, configs['num_classes'])

opt = AdamW(model.parameters(), lr=configs['lr'])

# %% [markdown]
# # Learn over annotated dataset

# %%

metrics = [accuracy_score, f1_score]

# %%
# Train the model using pseudolabels

#torch.autograd.set_detect_anomaly(True)
multiple_gpus = False
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        print(f"Running on {torch.cuda.device_count()} GPUs.")
        multiple_gpus = True
        device = "cuda:0"
        device_unlabelled = "cuda:1"
    else:
        print("Running on " + torch.cuda.get_device_name() + ".")
        device_unlabelled = device 
else:
    print("GPU not detected. Running on CPU.")
    device_unlabelled = device

if multiple_gpus:
    model = nn.DataParallel(model, device_ids=[0,1])
model.to(device)
model.train()
pseudolabels = [torch.zeros(configs["unlabelled_batch_size"], device=device_unlabelled)] * (len(unlabelled_set)//configs['unlabelled_batch_size']) + [torch.zeros(len(unlabelled_set)%configs["unlabelled_batch_size"],device=device_unlabelled)]
best_loss = np.inf
best_acc = -np.inf
best_f1 = -np.inf

for epoch in range(0, configs["epochs"]): # FIXME: Remove range starting from 1. For testing only.
    total = 0
    running_loss = 0
    running_metrics = [0] * len(metrics)
    bar = tqdm(labelled_loader)
    for x,y in bar:
        x,y = x.to(device), y.to(device).float()
        opt.zero_grad()

        # Calculate labelled loss
        y_hat = model(x.float()).argmax(axis=1).float()
        labelled_loss = criterion(y_hat, y)

        # Calculate unlabelled loss
        unlabelled_loss = 0
        
        # When t <= T1, we only fill pseudolabels with our predictions for the i-th batch.
        # After that we calculate the unlabelled loss applying our criterion to the curr_predictions
        # using pseudolabels as the ground truth.
        for i, x_prime in enumerate(unlabelled_loader):
            x_prime = x_prime.to(device_unlabelled).float()
            curr_predictions = model(x_prime).argmax(axis=1).to(device_unlabelled).float()
            if epoch >= configs["T1"]:
                #print(curr_predictions.squeeze(), pseudolabels[i])
                unlabelled_loss += criterion(curr_predictions, torch.as_tensor(pseudolabels[i], device=device_unlabelled)).item()*x_prime.size(0)
            pseudolabels[i] = curr_predictions
        del curr_predictions

        unlabelled_loss /= len(unlabelled_set)

        # Calculate final loss
        # alpha() will assure the loss won't be affected by the unlabelled data until current the epoch is >= T1.
        loss = labelled_loss + alpha(epoch)*unlabelled_loss # Equation 15 - Lee, 2013

        running_loss += loss.item()*x.size(0)
        total += x.size(0)

        for i, metric in enumerate(metrics):
            running_metrics[i] += metric(y_hat.cpu(), y.cpu().detach().numpy())*x.size(0)

        loss.backward()
        opt.step()
        metrics_text = f"[Epoch {epoch}/{configs['epochs']}] Labelled Loss: {labelled_loss.item():.5f} Unlabelled Loss: {unlabelled_loss:.5f} Running Loss: {running_loss/total:.5f} "
        for i, metric in enumerate(metrics):
            metrics_text += f"{metric.__name__}: {running_metrics[i]/total:.3f} "

        bar.set_description(metrics_text)

    curr_loss = running_loss/total
    metrics_text = f"[Epoch {epoch}/{configs['epochs']}] Loss: {curr_loss:.5f} "
    for i, metric in enumerate(metrics):
        metrics_text += f"{metric.__name__}: {running_metrics[i]/total:.3f} "
        print(metrics_text)

    curr_acc = running_metrics[0]/total
    curr_f1 = running_metrics[1]/total
    if curr_loss < best_loss:
        filename= f"best_loss-{curr_loss:.3f}-" + dt.datetime.now().strftime('%d-%m-%Y_%H-%M')
        print(f"[!] New Best Loss: {best_loss} -> {curr_loss}. ", end='')
        save_checkpoint(
            filename=filename,
            model=model,
            optimizer=opt,
            current_epoch=epoch
        )
        best_loss = curr_loss
    if curr_acc > best_acc:
        filename= f"best_acc-{running_loss:.3f}-" + dt.datetime.now().strftime('%d-%m-%Y_%H-%M')
        print(f"[!] New best accuracy: {best_acc} -> {curr_acc}. ", end='')
        save_checkpoint(
            filename=filename,
            model=model,
            optimizer=opt,
            current_epoch=epoch
        )
        best_acc = curr_acc
    if curr_f1 > best_f1:
        filename= f"best_f1-{curr_f1:.3f}-" + dt.datetime.now().strftime('%d-%m-%Y_%H-%M')
        print(f"[!] New best F1 score: {best_f1} -> {curr_f1}. ", end='')
        save_checkpoint(
            filename=filename,
            model=model,
            optimizer=opt,
            current_epoch=epoch
        )
        best_f1 = curr_f1

    del curr_loss, curr_acc, curr_f1
# %%
test_loader = DataLoader(
    test_set,
    batch_size=configs["labelled_batch_size"],
    shuffle=True,
)

evaluate (
    model, 
    configs["epochs"],
    criterion,
    test_loader,
    [accuracy_score, f1_score],
)

# %% [markdown]
# # Learn using pseudo-labels
