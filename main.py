from baseline_model import perform_baseline_model
from svd_2D import run_2dsvd, best_params_2dsvd
from read_data import read_data
from plotting import plot_datapoint, plot_multiple_datapoints, heatmap_plot, plot_loss_accuracy, activations_plot
from custom_cnn_model import CustomCNN, train_model, evaluate_model
from vowelDataset import VowelDataset, pad_sequences
from torch.utils.data import Dataset, DataLoader
from preprocessing import preprocess_data
from svd_2d_CNN import train_CNN
import torch
import numpy as np


def main():
    train_inputs, test_inputs, train_labels, test_labels = read_data()

    plot_datapoint(train_inputs[3], 4)
    plot_multiple_datapoints(train_inputs)

    ## Perform the baseline model
    best_k, best_score, test_score = perform_baseline_model()
    print("Baseline model(kNN): ")
    print(f'Best k: {best_k}, Validation score: {best_score}, Test score: {test_score}')

    ## Perform SVD model
    # run_2dsvd()

    ## Perform custom CNN model
    X_train = pad_sequences(train_inputs, max_len=30)
    y_train = np.array(train_labels)                 
    X_test = pad_sequences(test_inputs, max_len=30)    
    y_test = np.array(test_labels)                   

    # X_train, X_test = preprocess_data(train_inputs, test_inputs, 5, 10)

    train_dataset = VowelDataset(X_train, y_train)
    test_dataset = VowelDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = CustomCNN(num_classes=9, input_channels=12, input_seq_length=30)
    # model = CNNForSVD()
    loss_scores, acc_scores = train_model(model, train_loader, num_epochs=30, lr=1e-4)

    test_acc = evaluate_model(model, test_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # heatmap_plot(model, test_loader)

    # plot_loss_accuracy(loss_scores, acc_scores)

    # activations_plot(model, test_loader)

def experiment():
    # kNN
    # max_acc, best_k, best_row_ev, best_col_ev = run_2dsvd()
    # print(f"best val acc: {max_acc}, best_k: {best_k}, best_row_ev: {best_row_ev}, best_col_ev: {best_col_ev}")
    # best val f1: 0.9662996262996263, best_k: 3, best_row_ev: 5, best_col_ev: 10
    # best_params_2dsvd(3, 5, 10)
    # Test accuracy: 96.75675675675676%, Test error rate 3.24%, F1-score: 0.967813, k=3, row_evs=5, col_evs=10
    
    # CNN
    # Best Validation Weigthed F1: 0.98, lr: 0.001, wd: 0.001
    # Confusion matrix saved to figures/confusion_matrix_CNN_2dsvd.pdf
    # Final Test Error: 2.43%
    train_CNN(CustomCNN)

if __name__ == "__main__":
    main()
    # experiment()