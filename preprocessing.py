from read_data import read_data
import numpy as np

def set_seq_length(sequence, target):
    '''
    extend or truncate a sequence to match the target length
    '''
    if len(sequence) > target:
        return sequence[:target]
    if len(sequence) == target:
        return sequence
    if len(sequence) < target:
        return extend_sequence(sequence, target)

def extend_sequence(seq, target_length):
    '''
    extend a sequence by duplicationg some of the values
    '''
    current_length = len(seq)
    repeat_indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    extended_sequence = np.array(seq)[repeat_indices]
    if len(extended_sequence) != target_length:
        print(seq)
        print(extended_sequence)
        raise ValueError(f"Array does not match the target length")
    return extended_sequence

def extend_matrix(matrix, target):
    '''
    apply set_seq_length to each row of the matrix
    '''
    result = np.zeros((12, target))
    for idx, row in enumerate(matrix):
        extended_row = set_seq_length(row, target)
        result[idx] = extended_row
    return result

def unify_lengths(train, test):
    '''
    unify the lengths of the sequences in the train and test sets
    '''
    train_T = [i.T for i in train]
    test_T = [i.T for i in test]
    max_train = max([len(i[0]) for i in train_T])

    train_extended = [extend_matrix(m, max_train).T for m in train_T]
    test_extened = [extend_matrix(m, max_train).T for m in test_T]

    return np.array(train_extended), np.array(test_extened), max_train

def create_covariance_matrices(train_inputs, max_train_length):
    '''
    calculate the covariance matrices for the rows and columns of the input matrices
    '''
    mean_matrix = np.mean(train_inputs, axis=0)
    row_cov = np.zeros((max_train_length, max_train_length))
    col_cov = np.zeros((12, 12))
    
    for record in train_inputs:
        row_cov += (record - mean_matrix) @ (record - mean_matrix).T
        col_cov += (record - mean_matrix).T @ (record - mean_matrix)
        
    num_datapoints = len(train_inputs)
    row_cov /= num_datapoints
    col_cov /= num_datapoints

    return row_cov, col_cov


def get_eigenvectors(cov_matrix, n_evs):
    '''
    retunr the top n_evs eigenvectors of the covariance matrix
    '''
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    indices = np.argsort(eigenvalues)[::-1]  # [::-1] to reverse
    
    top_eigenvectors = eigenvectors[:, indices[:n_evs]] # : to select all rows indices[:n] to slice the first n elements from the indices array
    
    return top_eigenvectors
    
def preprocess_data(train_inputs, test_inputs, num_row_evs=5, num_col_evs=10):
    '''
    combine the preprocessing steps into a single function
    '''
    train_extended, test_extended, max_train_length = unify_lengths(train_inputs, test_inputs)
    row_cov, col_cov = create_covariance_matrices(train_extended, max_train_length)
    
    row_eigenvectors = get_eigenvectors(row_cov, num_row_evs)
    col_eigenvectors = get_eigenvectors(col_cov, num_col_evs)
   
    train_inputs_tranformed = np.array([row_eigenvectors.T @ original @ col_eigenvectors for original in train_extended])
    test_inputs_tranformed = np.array([row_eigenvectors.T @ original @ col_eigenvectors for original in test_extended])

    return train_inputs_tranformed, test_inputs_tranformed

def preprocess_part1(train_inputs, test_inputs):
'''
Split preprocess into two parts to compute number of eigenvectors in the middle
'''
    train_extended, test_extended, max_train_length = unify_lengths(train_inputs, test_inputs)

    row_cov, col_cov = create_covariance_matrices(train_extended, max_train_length)

    return train_extended, test_extended, row_cov, col_cov, max_train_length


def compute_eigenvectors(covariance_matrix, threshold=0.98):
'''
This returns the number of vectors that explains the threshold variance, it is 98% in the paper.
'''

    eigenvalues, _ = np.linalg.eigh(covariance_matrix)

    eigenvalues = sorted(eigenvalues, reverse=True)
    total_variance = sum(eigenvalues)
    explained_variance = 0
    count = 0
    for eigenvalue in eigenvalues:
        explained_variance += eigenvalue
        count += 1
        if explained_variance / total_variance >= threshold:
            break

    return count


def preprocess_part2(train_extended, test_extended, row_cov, col_cov, num_row_evs, num_col_evs):
    row_eigenvectors = get_eigenvectors(row_cov, num_row_evs)
    col_eigenvectors = get_eigenvectors(col_cov, num_col_evs)

    train_inputs_transformed = np.array([
        row_eigenvectors.T @ original @ col_eigenvectors for original in train_extended
    ])
    test_inputs_transformed = np.array([
        row_eigenvectors.T @ original @ col_eigenvectors for original in test_extended
    ])

    return train_inputs_transformed, test_inputs_transformed
