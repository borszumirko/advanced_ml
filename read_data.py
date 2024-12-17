import numpy as np
import matplotlib.pyplot as plt

def read_blocks(filepath):
    ae_data = np.loadtxt(filepath)
    inputs = []

    block = [[] for _ in range(12)]
    for row in ae_data:
        if np.array_equal(row, np.ones(12)):
            inputs.append(np.array(block))
            block = [[] for _ in range(12)]
            continue

        for idx, value in enumerate(row):
            block[idx].append(value)

    return inputs


def read_data():
    return read_blocks('data/ae.train'), read_blocks('data/ae.test'), create_train_labels(), create_test_labels()

def create_train_labels():
    labels = []
    for i in range(270):
        label = np.zeros(9)
        label[i // 30] = 1
        labels.append(label)
    return labels

def create_test_labels():
    num_utterances = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    labels = []
    for idx, speaker in enumerate(num_utterances):
        for _ in range(speaker):
            label = np.zeros(9)
            label[idx] = 1
            labels.append(label)
    return labels


def plot_datapoint(data):
    
    plt.figure(figsize=(10, 6))

    for i in range(12):
        plt.plot(data[i], label=f'Line {i+1}')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('12 Lines Plot')
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()
