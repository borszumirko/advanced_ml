import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from read_data import read_data
from preprocessing import preprocess_data
from vowelDataset import VowelDataset
from torch.utils.data import DataLoader, Subset
from custom_cnn_model import train_model, evaluate_model
from sklearn.model_selection import StratifiedKFold
from itertools import product

class CNNForSVD(nn.Module):
    def __init__(self, num_classes=9):
        super(CNNForSVD, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(32 * 5 * 10, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, 5, 10)
        x = x.unsqueeze(1)
        # Output shape: (batch_size, 1, 5, 10)
        
        x = self.conv1(x)  
        x = self.bn1(x)
        x = F.relu(x)
        # Output shape: (batch_size, 16, 5, 10)

        x = self.conv2(x)  
        x = self.bn2(x)
        x = F.relu(x)
        # Output shape: (batch_size, 32, 5, 10)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  
        # Output shape: (batch_size, 32 * 5 * 10)

        x = self.fc(x)
        # Output shape: (batch_size, num_classes)
        return x



def run_cross_validation(train_inputs, train_labels, model):
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    weight_decays = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3]

    # Generate all combinations
    hyperparams = list(product(learning_rates, weight_decays))

    param_search = []
    for lr, wd in hyperparams:
        accuracies = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold = 1
        
        for train_index, val_index in kf.split(train_inputs, np.argmax(train_labels, axis=1)):
            y_train, y_val = train_labels[train_index], train_labels[val_index]

            X_train = [train_inputs[i] for i in train_index]
            X_val = [train_inputs[i] for i in val_index]
                    
            X_train_transformed, X_val_transformed = preprocess_data(X_train, X_val)

            training = VowelDataset(X_train_transformed, y_train)
            validation = VowelDataset(X_val_transformed, y_val)

            train_loader = DataLoader(training, batch_size=16, shuffle=True)  
            val_loader = DataLoader(validation, batch_size=16, shuffle=False)
            _, _ = train_model(model, train_loader, num_epochs=30, lr=lr, weight_decay=wd)
            fold += 1
            val_acc = evaluate_model(model, val_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            # print(f"Fold: {fold}, Val Accuracy: {val_acc:.2f}%")
            accuracies.append(val_acc)
        print(f"Mean Validation Accuracy: {np.mean(accuracies):.2f}% for lr: {lr}, wd: {wd}")
        param_search.append({'acc': np.mean(accuracies), 'lr': lr, 'wd': wd})

    return find_best_params_CNN(param_search)



def train_CNN(CNNModel):
    train_inputs, test_inputs, train_labels, test_labels = read_data()

    X_train, X_test = preprocess_data(train_inputs, test_inputs, 5, 10)
    y_test = np.array(test_labels)                   
    y_train = np.array(train_labels)                 

    train_dataset = VowelDataset(X_train, y_train)
    test_dataset = VowelDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = CNNModel()

    max_acc, lr, wd = run_cross_validation(train_inputs, y_train, model)
    print(f"Best Validation Accuracy: {max_acc:.2f}%, lr: {lr}, wd: {wd}")

    loss_scores, acc_scores = train_model(model, train_loader, num_epochs=30, lr=lr, weight_decay=wd)

    test_acc = evaluate_model(model, test_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Final Test Accuracy: {test_acc:.2f}%")

def find_best_params_CNN(cv_stats):
    max_acc = 0
    lr = None
    wd = None

    for dict in cv_stats:
        if dict['acc'] > max_acc:
            max_acc = dict['acc']
            lr = dict['lr']
            wd = dict['wd']
    
    return max_acc, lr, wd

