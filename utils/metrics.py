"""
David de Oliveira Lima,
Jan, 2024
"""

import torch

def calculate_accuracy(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
    _, y_hat = torch.max(y_hat, 1)
    return torch.div(torch.sum(y_hat == y_true), len(y_true)).item()

def calculate_f1_score(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
    assert len(y_hat) == len(y_true)
    _, y_hat = torch.max(y_hat, 1)
    metrics = {"tp":0,"tn":0,"fp":0,"fn":0}
    for i in range(len(y_hat)):
        if y_hat[i]:
            if y_true[i]:
                metrics["tp"]+=1
            else:
                metrics["fp"]+=1
        else: # !y_hat[i]
            if not y_true[i]:
                metrics["tn"]+=1
            else:
                metrics["fn"]+=1

    precision = metrics["tp"]/(metrics["tp"]+metrics["fp"]) if (metrics["tp"]+metrics["fp"]) != 0 else 0
    recall = metrics["tp"]/(metrics["tp"]+metrics["fn"]) if (metrics["tp"]+metrics["fn"]) != 0 else 0
    return 2*precision*recall/(precision+recall) if (precision+recall) != 0 else 0

def class_specific_f1_score(cls: int, y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
    _, y_hat = torch.max(y_hat, 1)
    metrics = {"tp":0,"tn":0,"fp":0,"fn":0}
    for i in range(len(y_hat)):
        if y_hat[i] == cls:
            if y_true[i] == cls:
                metrics["tp"] += 1
            else:
                metrics["fp"] += 1
        else:
            if y_true[i] != cls:
                metrics["tn"] += 1
            else:
                metrics["fn"] += 1
    recall = metrics["tp"]/(metrics["tp"]+metrics["fn"]) if (metrics["tp"]+metrics["fn"]) != 0 else 0
    precision = metrics["tp"]/(metrics["tp"]+metrics["fp"]) if (metrics["tp"]+metrics["fp"]) != 0 else 0
    return 2*precision*recall/(precision+recall) if (precision+recall) != 0 else 0

def calculate_classwise_f1_score(y_hat: torch.Tensor, y_true: torch.Tensor) -> list[float]:
    assert len(y_hat) == len(y_true)
    out = [] # Each index of this list corresponds to a class
    for i in range(max(torch.max(y_true).item(), torch.max(y_hat).item())+1):
        out[i] = class_specific_f1_score(i, y_hat, y_true)

    return out
