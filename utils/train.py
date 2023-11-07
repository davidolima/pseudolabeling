import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
from tqdm import tqdm
import typing

def train(model:nn.Module, 
          epochs:int,
          criterion,
          dataloader:DataLoader,
          opt:torch.optim.Optimizer,
          metrics: list,
          val_loader:DataLoader=None,
          device='cuda',
          history=False):
    
    if history or val_loader is not None:
        raise NotImplemented()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for x,y in tqdm(dataloader):
            x,y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            running_loss += loss.item()*x.size(0)

            opt.zero_grad()
            loss.backward()
            opt.step()
        
        running_loss = running_loss/len(dataloader)
        epoch_text = f"[Epoch {epoch}/{epochs}] Loss: {running_loss:.5f} "

        for metric in metrics:
            epoch_text += f"{metric.__name__}: {metric(y_hat.argmax(axis=1).detach().numpy(),y)} " 

        print(epoch_text)


def crossValidation(model,
                    epochs,
                    n_splits,
                    x_train,
                    y_train,
                    opt:torch.optim.Optimizer,
                    metrics: list,
                    val_loader:DataLoader=None,
                    device='cuda',
                    history=False):
    kf = StratifiedKFold(n_splits, shuffle=True)
    x = kf.split(x_train,y_train)


