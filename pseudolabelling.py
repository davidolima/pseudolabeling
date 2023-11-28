# %% [markdown]
# # Imports

# %%
import torch
import torch.nn as nn
from torch.optim import Adam
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
from utils.data import LabelledSet, UnlabelledSet,FullRadiographDataset

# %%
# Dataset radiografias
# ds = FullRadiographDataset("/datasets/pan-radiographs/", [1,2,3,4,5], None)

# %% [markdown]
# # Setup

# %%
epochs = 1000
batch_size = 128
num_classes = 10
device = "cuda" if torch.cuda.is_available() else 'cpu'

def alpha(t, T1=100, T2=600, alpha_f=3):
    if t < T1:
        return 0
    elif t < T2:
        return alpha_f*(t-T1)/(T2-T1)
    else: # T2 <= t
        return alpha_f

# %%
transforms = T.Compose([
    gray2rgb,
    T.ToTensor()
])

train_ds, test_ds = train_test_split(
    MNIST("/Tera/datasets/", download=True, transform=transforms),
    test_size=.2,
    train_size=.8,
    shuffle=True
)

labelled, unlabelled = train_test_split(
    train_ds,
    test_size=.2,
    train_size=.8,
    shuffle=True
) 

labelled_loader = DataLoader(
    labelled,
    batch_size=batch_size,
    shuffle=True,
)

unlabelled_loader = DataLoader(
    unlabelled,
    batch_size=batch_size,
    shuffle=True,
)

criterion = nn.CrossEntropyLoss()

# %%
# def get_state_dict(self, *args, **kwargs):
#     #kwargs.pop("check_hash")
#     return load_state_dict_from_url(self.url, *args, **kwargs)
# WeightsEnum.get_state_dict = get_state_dict

model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

opt = Adam(model.parameters(), lr=1e-3)

# %% [markdown]
# # Learn over annotated dataset

# %%

metrics = [accuracy_score, lambda x,y: f1_score(x,y, average='macro')]

# %%

print("Training on", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
torch.autograd.set_detect_anomaly(True)
model.to(device)
model.train()
for epoch in range(epochs):
    total = 0
    running_loss = 0
    running_metrics = [0] * len(metrics)
    bar = tqdm(labelled_loader)
    for (x,y), (unlabelled,_) in zip(bar,unlabelled_loader):
        x,y, unlabelled = x.to(device), y.to(device), unlabelled.to(device)
        y_hat = model(x)

        # pseudo-label loss is calculated only after first epoch
        if epoch == 0:
            loss = criterion(y_hat, y)
        else:
            loss = criterion(y_hat,y) + alpha(epoch)*criterion(model(unlabelled), pseudo_labels.detach())

        opt.zero_grad()
        running_loss += loss.item()*x.size(0)

        pseudo_labels = model(unlabelled) # Calculate pseudo-labels for the next batch

        loss.backward(retain_graph=True)

        total += x.size(0)

        for i, metric in enumerate(metrics):
            running_metrics[i] += metric(y_hat.cpu().argmax(axis=1),y.cpu())*x.size(0)

        opt.step()
        bar.set_description(f"[Epoch {epoch}]")

    metrics_text = f"[Epoch {epoch}] Loss: {running_loss/total:.5f} "
    for i, metric in enumerate(metrics):
        metrics_text += f"{metric.__name__}: {running_metrics[i]/total:.3f} "
    print(metrics_text)

# %%
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

evaluate (
    model, 
    epochs,
    criterion,
    test_loader,
    [accuracy_score, lambda x,y: f1_score(x,y, average='macro')],
)

# %% [markdown]
# # Learn using pseudo-labels

