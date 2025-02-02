import numpy as np
from read_data import read_data
from preprocessing import unify_lengths, create_covariance_matrices, compute_eigenvectors, get_eigenvectors
import matplotlib.pyplot as plt

def elementwise_difference(matrix1, matrix2):
    """
    Calculate the sum of elementwise square differences between two matrices.
    """
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape.")
    
    return np.sum((matrix1 - matrix2) ** 2)

train_inputs, test_inputs, train_labels, test_labels = read_data()

train_extended, test_extended, max_train_length = unify_lengths(train_inputs, test_inputs)
row_cov, col_cov = create_covariance_matrices(train_extended, max_train_length)

# Find s for 98% variance explained
s = compute_eigenvectors(col_cov)
sse_s = []
# Loop through all possible r
for i in range(1, 12):
    V = get_eigenvectors(col_cov, s)
    U = get_eigenvectors(row_cov, i)
    total_error = 0
    for j in range(train_extended.shape[0]):
        transformed = U.T @ train_extended[j] @ V
        reconstructed = U @ transformed @ V.T
        # Calculate the reconstruction error
        total_error += elementwise_difference(train_extended[j], reconstructed)
    print("Row eigenvectors:", i, "Column eigenvectors:", s, "Total error:", total_error)
    sse_s.append(total_error)

dsse = []
for idx, e in enumerate(sse_s):
    if idx < len(sse_s) - 1:
        dsse.append(float(abs(sse_s[idx+1] - sse_s[idx])/sse_s[idx]*100))
    else:
        dsse.append(0)

print(dsse)

# Plot the total error for each r to find the elbow point
plt.plot(range(1, 12), sse_s)
plt.xlabel("Number of row eigenvectors")
plt.ylabel("SSE")
plt.title("SSE vs number of row eigenvectors")
plt.savefig("figures/sse_vs_row_evs.pdf")
plt.show()


