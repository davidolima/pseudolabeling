#!./bin/python3

# %% [markdown]
# # Imports

# %%
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader,Dataset

import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from skimage.color import gray2rgb

import numpy as np
from tqdm import tqdm
from train import *
from utils.data import LabelledSet, UnlabelledSet #, FullRadiographDataset

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
test_set = LabelledSet(root, list(range(20,29)), T_test)
unlabelled_set = UnlabelledSet(root, list(range(29,31)), T_train)

print(f"[!] {sum(map(len, [labelled_set,test_set,unlabelled_set]))} images were loaded in total.")

# %% [markdown]
# # Setup

# %%

# Configuration
epochs = 10
#labelled_only_epochs = 10
batch_size = 32
num_classes = 2
lr = 1e-5
T1 = 1
T2 = 6
alpha_f = .03
criterion = nn.BCELoss()
device = "cuda" if torch.cuda.is_available() else 'cpu'

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
    batch_size=batch_size,
    shuffle=True,
)

unlabelled_loader = DataLoader(
    unlabelled_set,
    batch_size=batch_size,
    shuffle=True,
)

# %%
# def get_state_dict(self, *args, **kwargs):
#     kwargs.pop("check_hash")
#     return load_state_dict_from_url(self.url, *args, **kwargs)
# WeightsEnum.get_state_dict = get_state_dict

model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

opt = AdamW(model.parameters(), lr=lr)

# %% [markdown]
# # Learn over annotated dataset

# %%

metrics = [accuracy_score, f1_score]

# %%
# Train the model using pseudolabels

#torch.autograd.set_detect_anomaly(True)

model.to(device)
model.train()
for epoch in range(1, epochs): # FIXME: Remove range starting from 1. For testing only.
    total = 0
    running_loss = 0
    running_metrics = [0] * len(metrics)
    pseudolabels = torch.zeros(batch_size, device=device)
    bar = tqdm(labelled_loader)
    for x,y in bar:
        x,y = x.to(device), y.to(device).float()
        opt.zero_grad()

        # Calculate loss
        y_hat = model(x.float()).argmax(axis=1).float()
        labelled_loss = criterion(y_hat, y)
        unlabelled_loss = 0
        for x_prime, _ in unlabelled_loader:
            curr_predictions = model(x_prime.to(device).float()).argmax(axis=1).float()
            unlabelled_loss += criterion(curr_predictions, pseudolabels).item()*x_prime.size(0)
            pseudolabels = curr_predictions
        # alpha() will assure the loss won't be affected
        # by the unlabelled data until current the epoch >= T1.
        loss = labelled_loss + alpha(epoch)*unlabelled_loss # Equation 15 - Lee, 2013

        running_loss += loss.item()*x.size(0)
        total += x.size(0)

        for i, metric in enumerate(metrics):
            running_metrics[i] += metric(y_hat.cpu(), y.cpu().detach().numpy()) * x.size(0)

        loss.backward()
        opt.step()
        metrics_text = f"[Epoch {epoch}/{epochs}] Loss: {running_loss/total:.5f} "
        for i, metric in enumerate(metrics):
            metrics_text += f"{metric.__name__}: {running_metrics[i]/total:.3f} "

        bar.set_description(metrics_text)

    metrics_text = f"[Epoch {epoch}/{epochs}] Loss: {running_loss/total:.5f} "
    for i, metric in enumerate(metrics):
        metrics_text += f"{metric.__name__}: {running_metrics[i]/total:.3f} "
        print(metrics_text)

# %%
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True,
)

evaluate (
    model, 
    epochs,
    criterion,
    test_loader,
    [accuracy_score, f1_score],
)

# %% [markdown]
# # Learn using pseudo-labels
