import matplotlib
matplotlib.use('Agg')  # For non-interactive plots
import matplotlib.pyplot as plt


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
    plt.show()