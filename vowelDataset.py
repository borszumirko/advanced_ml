import numpy as np
import torch

class VowelDataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # shape is (30, 12)
        x_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        # Transpose to (12, 30)
        x_tensor = x_tensor.transpose(0, 1)

        # Convert one-hot label to single integer
        y_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        y_int = torch.argmax(y_tensor).long()

        return x_tensor, y_int


def pad_sequences(sequences, max_len=30):
    """
    Zero-pad each sequence in 'sequences' to length max_len along the time axis.
    Each sequence is expected to have shape (T, 12).
    Returns a single numpy array of shape (num_sequences, max_len, 12).
    """
    num_sequences = len(sequences)
    n_channels = 12
    
    padded = np.zeros((num_sequences, max_len, n_channels), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        length = seq.shape[0]  # T
        copy_len = min(length, max_len)
        padded[i, :copy_len, :] = seq[:copy_len, :]
    
    return padded