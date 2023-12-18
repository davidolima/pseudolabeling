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
from utils.data import LabelledSet, UnlabelledSet #, FullRadiographDataset

# %%
# Dataset radiografias
root = "/datasets/pan-radiographs/"
transforms = T.Compose([
    T.PILToTensor(),
    T.Resize((224,224)),
    # T.ToTensor(),
])
# ds = FullRadiographDataset(root, [1,2,3,4,5], transforms)
labelled_set = LabelledSet(root, list(range(1,20)), transforms)
test_set = LabelledSet(root, list(range(20,29)), transforms)
unlabelled_set = UnlabelledSet(root, list(range(29,31)), transforms)

print(f"[!] {sum(map(len, [labelled_set,test_set,unlabelled_set]))} images were loaded in total.")

# %% [markdown]
# # Setup

# %%
epochs = 10
batch_size = 8
num_classes = 2
lr = 1e-5
T1 = 100
T2 = 600
alpha_f = 3
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
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

model = torchvision.models.efficientnet_b0(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

opt = Adam(model.parameters(), lr=lr)

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
    for x,y in bar:
        x,y = x.to(device), y.to(device)
        y_hat = model(x.float())

        # pseudo-label loss is calculated only after first epoch
        opt.zero_grad()
        if epoch == 0:
            loss = criterion(y_hat.argmax(1).float(), y.float())
            pseudo_labels = []
            for unlabelled_x, _ in unlabelled_loader:
                pseudo_labels += model(unlabelled_x.to(device).float()).to('cpu')
        else:
            pseudo_loss = 0
            total_pseudo = 0
            new_pseudo_labels = []
            for (unlabelled_x, _), y_prime in zip(unlabelled_loader, pseudo_labels):
                y_unlabel_predict = model(unlabelled_x)
                new_pseudo_labels += y_unlabel_predict
                pseudo_loss += criterion(y_unlabel_predict, y_prime)*unlabelled_x.size(0)
                total_pseudo += unlabelled_x.size(0)
                print(pseudo_loss, total_pseudo, unlabelled_x.size(0))
            pseudo_labels = new_pseudo_labels

            loss = criterion(y_hat,y) + alpha(epoch,T1=T1,T2=T2,alpha_f=alpha_f)*(pseudo_loss/total_pseudo)

        running_loss += loss.item()*x.size(0)
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
    [accuracy_score, f1_score],
)

# %% [markdown]
# # Learn using pseudo-labels

