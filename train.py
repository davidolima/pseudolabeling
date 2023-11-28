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
          device=None,
          history=False):
    
    if history or val_loader is not None:
        raise NotImplemented()
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        
    print("Training on", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")

    model.to(device)
    model.train()
    for epoch in range(epochs):
        total = 0
        running_loss = 0
        running_metrics = [0] * len(metrics)
        bar = tqdm(dataloader)
        for x,y in bar:
            x,y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            running_loss += loss.item()*x.size(0)
            total += x.size(0)
            
            for i, metric in enumerate(metrics):
                running_metrics[i] += metric(y_hat.cpu().argmax(axis=1),y.cpu())*x.size(0)

            opt.zero_grad()
            loss.backward()
            opt.step()
            bar.set_description(f"[Epoch {epoch}]")
        
        metrics_text = f"[Epoch {epoch}] Loss: {running_loss/total:.5f} "
        for i, metric in enumerate(metrics):
            metrics_text += f"{metric.__name__}: {running_metrics[i]/total:.3f} " 
        print(metrics_text)

def evaluate(
        model:nn.Module, 
        epochs:int,
        criterion,
        test_loader:DataLoader,
        metrics: list,
        device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        
    print("Testing on", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")

    test_loader = tqdm(test_loader)

    running_loss = 0
    running_metrics = [0] * len(metrics)
    total = 0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            total += len(y)
        
            running_loss += loss.item()*x.size(0)

            for i, metric in enumerate(metrics):
                running_metrics[i] += metric(y_hat.cpu().argmax(axis=1),y.cpu())*x.size(0)

            test_loader.set_description(f"Loss: {running_loss/total:.5f}")
        
        metrics_text = f"[Test Results] Loss: {running_loss/total:.5f} "
        for i, metric in enumerate(metrics):
            metrics_text += f"{metric.__name__}: {running_metrics[i]/total:.3f} " 
        print(metrics_text)

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


