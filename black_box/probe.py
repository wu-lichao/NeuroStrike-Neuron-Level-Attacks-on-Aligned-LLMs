import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

    @torch.jit.export
    def predict(self, x):
        # self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            # predictions = (probabilities >= 0.5).int()
        return probabilities

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPModel, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x
    
    @torch.jit.export
    def predict(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).int()
        return predictions


def safety_probe(activations, labels, device, num_runs=5):
    num_epochs = 20
    for run in range(num_runs):
        torch.manual_seed(1234+run)
        X_train, X_val, y_train, y_val = train_test_split(
            activations, labels, test_size=0.2, random_state=42
        )
        
        # Convert numpy arrays back to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

        clf = LogisticRegressionModel(X_train.size(1)).to(device)
        # clf = MLPModel(X_train.size(1), int(X_train.size(1)/2)).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(clf.parameters(), lr=0.0001, weight_decay=1e-3)

        for epoch in range(num_epochs):
            # Training phase
            clf.train()
            optimizer.zero_grad()
            outputs = clf(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Validation phase
            if (epoch + 1) % 500 == 0:
                clf.eval()
                with torch.no_grad():
                    val_outputs = clf(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    val_preds = torch.sigmoid(val_outputs).cpu().numpy()
                    val_labels = y_val.cpu().numpy()
                    val_preds_binary = (val_preds >= 0.5).astype(int)
                    val_accuracy = (val_preds_binary == val_labels).mean()
                    val_auc = roc_auc_score(val_labels, val_preds)
                
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Training Loss: {loss.item():.4f}, "
                      f"Validation Loss: {val_loss:.4f}, "
                      f"Validation Accuracy: {val_accuracy:.3f}, "
                      f"Validation ROC-AUC: {val_auc:.3f}")

    return clf

