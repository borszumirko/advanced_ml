import matplotlib
# matplotlib.use('Agg')  # For non-interactive plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from custom_cnn_model import get_predictions_and_labels, register_activations

def plot_datapoint(data, label):
    T = data.shape[0] 
    plt.figure(figsize=(10, 6))
    
    for ch in range(12):
        plt.plot(data[:, ch])
    
    plt.xlim(0, T-1)
    
    plt.xlabel("Time step")
    plt.ylabel("Coefficient value")
    plt.title("12-channel time series")
    plt.grid(True)
    plt.savefig(f'figures/datapoint_{label}')


def plot_multiple_datapoints(data):
    selected_speakers = [1, 3, 5, 7]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    for ax, idx in zip(axes.flat, selected_speakers):
        data_i = data[idx]
        
        for ch in range(12):
            ax.plot(data_i[:, ch])
        
        ax.set_title(f"Speaker {idx+1}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Coefficient value")
        ax.grid(True)

    fig.tight_layout()
    plt.savefig("figures/datapoints.png")


def heatmap_plot(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_pred, y_true = get_predictions_and_labels(model, dataset, device)
    conf_matrix = confusion_matrix(y_true, y_pred)

    class_names = [f"Speaker {i+1}" for i in range(9)]  

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.savefig("figures/heatmap.png")


def plot_loss_accuracy(loss_scores, acc_scores):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_scores, label='Train Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(acc_scores, label='Train Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("figures/loss_accuracy.png")


def activations_plot(model, test_loader):
    conv1_act, conv2_act = register_activations(model, test_loader)

    conv1_act = conv1_act[0] 
    conv2_act = conv2_act[0] 

    n_channels_to_plot = 3

    # First convolutional layer
    plt.figure(figsize=(10, 6))
    for c in range(n_channels_to_plot):
        channel_data = conv1_act[c].cpu().numpy()
        plt.plot(channel_data, label=f"Channel {c}")

    plt.title("Conv1 Activation - First 3 Channels")
    plt.xlabel("Time step")
    plt.ylabel("Activation")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig("figures/activation_plot_conv1.png")

    # Second convolutional layer
    plt.figure(figsize=(10, 6))
    for c in range(n_channels_to_plot):
        channel_data = conv2_act[c].cpu().numpy()
        plt.plot(channel_data, label=f"Channel {c}")

    plt.title("Conv2 Activation - First 3 Channels")
    plt.xlabel("Time step")
    plt.ylabel("Activation")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig("figures/activation_plot_conv2.png")