from baseline_model import perform_baseline_model
from svd_2D import run_2dsvd
from read_data import read_data
from plotting import plot_datapoint, plot_multiple_datapoints
from custom_cnn_model import CustomCNN, train_model, evaluate_model
from vowelDataset import VowelDataset, pad_sequences
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


def main():
    train_inputs, test_inputs, train_labels, test_labels = read_data()

    plot_datapoint(train_inputs[3], 4)
    plot_multiple_datapoints(train_inputs)

    ## Perform the baseline model
    best_k, best_score, test_score = perform_baseline_model()
    print("Baseline model(kMM): ")
    print(f'Best k: {best_k}, Validation score: {best_score}, Test score: {test_score}')

    ## Perform SVD model
    run_2dsvd()

    ## Perform custom CNN model
    X_train = pad_sequences(train_inputs, max_len=30)
    y_train = np.array(train_labels)                 
    X_test = pad_sequences(test_inputs, max_len=30)    
    y_test = np.array(test_labels)                   

    train_dataset = VowelDataset(X_train, y_train)
    test_dataset = VowelDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = CustomCNN(num_classes=9)
    train_model(model, train_loader, num_epochs=30, lr=1e-3)

    # test_acc = evaluate_model(model, test_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # print(f"Final Test Accuracy: {test_acc:.2f}%")
    

if __name__ == "__main__":
    main()