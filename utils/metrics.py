"""
David de Oliveira Lima,
2024
"""

import torch

def calculate_accuracy(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.div(torch.sum(y_hat == y_true), len(y_true)).item()

def calculate_f1_score(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
    assert len(y_hat) == len(y_true)
    metrics = {"tp":0,"tn":0,"fp":0,"fn":0}
    for i in range(len(y_hat)):
        if y_hat[i] == y_true[i]:
           if y_hat[i]:
               metrics["tp"]+=1
           else:
               metrics["tn"]+=1
        elif y_hat[i] != y_true[i]:
            if y_hat[i]:
                metrics["fp"]+=1
            else:
                metrics["fn"]+=1

    precision = metrics["tp"]/(metrics["tp"]+metrics["fp"]) if (metrics["tp"]+metrics["fp"]) != 0 else 0
    recall = metrics["tp"]/(metrics["tp"]+metrics["fn"]) if (metrics["tp"]+metrics["fn"]) != 0 else 0
    return 2*precision*recall/(precision+recall) if (precision+recall) != 0 else 0
