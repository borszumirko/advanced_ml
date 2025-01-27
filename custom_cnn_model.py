from read_data import read_data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Just a skeleton for now
# Needs testing and proper matching to the data format

activations = {}

class CustomCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=12, 
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(64)
    
        self.fc = nn.Linear(64 * 30, num_classes)
    
    def forward(self, x):
        """
        x shape: (batch_size, 12, 30)
        """
        x = F.relu(self.bn1(self.conv1(x)))  
        x = F.relu(self.bn2(self.conv2(x))) 
        
        # Flatten
        x = x.view(x.size(0), -1)   
        
        logits = self.fc(x)   
        return logits
    
def get_activation(name):
    """Returns a hook function that saves activation to 'activations' dict."""
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def train_model(model, train_loader, num_epochs=30, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_scores = []
    acc_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            
            # Compute training accuracy
            _, predicted = torch.max(logits, dim=1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += X_batch.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = 100.0 * correct_train / total_train
        loss_scores.append(epoch_loss)
        acc_scores.append(epoch_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_acc:.2f}%")
    
    return loss_scores, acc_scores

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(X_batch)
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += X_batch.size(0)
    acc = 100.0 * correct / total
    return acc


def get_predictions_and_labels(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)               # shape: (batch_size, num_classes)
            _, preds = torch.max(logits, dim=1)   # shape: (batch_size,)

            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(y_batch.cpu().numpy()) 

    return np.array(all_preds), np.array(all_labels)


def register_activations(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))

    sample_batch, label_batch = next(iter(test_loader))
    single_sample = sample_batch[0:1].to(device)
    single_label = label_batch[0].item()

    # 6) Forward pass
    _ = model(single_sample)

    conv1_act = activations['conv1']  # shape: (1, 32, 30)
    conv2_act = activations['conv2']  # shape: (1, 64, 30)
    return conv1_act, conv2_act

#def get_activations()