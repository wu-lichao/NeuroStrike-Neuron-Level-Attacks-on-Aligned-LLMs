import torch
import torch.nn as nn
import torch.optim as optim
# from torchmetrics.classification import BinaryAUROC
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import zscore
import util

# Define a simple logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def safety_probe(activations_tensor, labels_tensor, device, num_runs=1, num_epochs=5000):
    importances_all = []
    # assert not torch.isnan(activations_tensor).any(), "NaN detected in the activations_tensor"
    # Remove rows with NaN activations and corresponding labels
    nan_mask = torch.isnan(activations_tensor).any(dim=1)
    if nan_mask.any():
        activations_tensor = activations_tensor[~nan_mask]
        labels_tensor = labels_tensor[~nan_mask]
    assert torch.all((labels_tensor == 0) | (labels_tensor == 1)), "Labels must be 0 or 1"
    for run in range(num_runs):
        torch.manual_seed(1234+run)
        activations, labels = util.shuffle(activations_tensor, labels_tensor)
        clf = LogisticRegressionModel(activations_tensor.size(1)).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-3)
        for epoch in range(num_epochs):
            clf.train()
            optimizer.zero_grad()
            outputs = clf(activations)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 500 == 0:
                clf.eval()
                with torch.no_grad():
                    preds = clf(activations)
                auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, ROC-AUC: {auc:.3f}")
        run_importance = clf.linear.weight.data.cpu().numpy().flatten()
        importances_all.append(run_importance)

    importances_avg = np.mean(importances_all, axis=0)
    return importances_avg

