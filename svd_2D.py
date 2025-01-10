from preprocessing import preprocess_data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from functools import partial
from sklearn.model_selection import StratifiedKFold


def sum_squared_differences(X, Y, num_row_evs=3, num_col_evs=9):
    sum = 0
    for i in range(num_col_evs):
       x = np.array([X[i+num_col_evs*j] for j in range(num_row_evs)])
       y = np.array([Y[i+num_col_evs*j] for j in range(num_row_evs)])
       dist = np.linalg.norm(x-y)
       sum += dist
    return sum

def run_2dsvd():
    train_inputs, test_inputs, train_labels, test_labels = read_data()
    train_labels_int = np.argmax(train_labels, axis=1)
    test_labels_int  = np.argmax(test_labels, axis=1)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in [1, 3, 5, 7]:
        for row_evs in [3, 4, 5, 6]:
            for col_evs in [7, 8, 9, 10]:
                sum_squared_differences_partial = partial(sum_squared_differences, num_row_evs=row_evs, num_col_evs=col_evs)
                
                scores = []
                
                for train_index, val_index in kf.split(train_inputs, train_labels_int):

                    y_train, y_val = train_labels_int[train_index], train_labels_int[val_index]

                    X_train = [train_inputs[i] for i in train_index]
                    X_val = [train_inputs[i] for i in val_index]
                    
                    X_train_transformed, X_val_transformed = preprocess_data(X_train, X_val, row_evs, col_evs)
                    
                    X_train_flattened = np.array([np.array(matrix.flatten()) for matrix in X_train_transformed])
                    X_val_flattened = np.array([np.array(matrix.flatten()) for matrix in X_val_transformed])
                    
                    knn = KNeighborsClassifier(n_neighbors=k, metric=sum_squared_differences_partial)
                    knn.fit(X_train_flattened, y_train)
                
                    score = knn.score(X_val_flattened, y_val)
                    scores.append(score)
                print(f'Average val accuracy: {np.mean(scores)}, k={k}, row_evs={row_evs}, col_evs={col_evs}')
    
