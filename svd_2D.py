from preprocessing import preprocess_data, unify_lengths, preprocess_part1, preprocess_part2, compute_eigenvectors
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from read_data import read_data
import matplotlib.pyplot as plt
import matplotlib
import os

def sum_squared_differences(X, Y, num_row_evs=3, num_col_evs=9):
    sum = 0
    for i in range(num_col_evs):
        x = np.array([X[i+num_col_evs*j] for j in range(num_row_evs)])
        y = np.array([Y[i+num_col_evs*j] for j in range(num_row_evs)])
        dist = np.linalg.norm(x-y)
        sum += dist
    return sum

def find_best_params(cv_stats):
    max_acc = 0
    best_k = None
    best_row_ev = None
    best_col_ev = None

    for dict in cv_stats:
        if dict['acc'] > max_acc:
            max_acc = dict['acc']
            best_k = dict['k']
            best_row_ev = dict['row_evs']
            best_col_ev = dict['col_evs']

    return max_acc, best_k, best_row_ev, best_col_ev

'''
Tuning the number of row (r) and column (s) eigenvectors according to paper.
'''

def run_2dsvd(best_baseline_k = 3):
    train_inputs, test_inputs, train_labels, test_labels = read_data()
    train_labels_int = np.argmax(train_labels, axis=1)
    test_labels_int = np.argmax(test_labels, axis=1)
    max1_col_evs = train_inputs[0].shape[1]
    max2_col_evs = min(matrix.shape[1] for matrix in train_inputs)
    print(f'max1:{max1_col_evs}')
    print(f'max2:{max2_col_evs}')
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    k = best_baseline_k  # using the best k of the baseline knn

    row_evs_scores = []
    col_evs_scores = []
    best_r = None
    best_r_score = 0
    best_s = None
    best_s_score = 0
    train_extended, test_extended, row_cov, col_cov, max_train_length = preprocess_part1(train_inputs, test_inputs)
    cv_stats = []

    for row_evs in range(1, max_train_length):
        r_scores = []

        for train_index, val_index in kf.split(train_inputs, train_labels_int):
            y_train, y_val = train_labels_int[train_index], train_labels_int[val_index]

            X_train = [train_inputs[i] for i in train_index]
            X_val = [train_inputs[i] for i in val_index]

            train_extended, test_extended, row_cov, col_cov, max_train_length = preprocess_part1(X_train, X_val)
            s = compute_eigenvectors(row_cov)  # number of column eigenvectors
            X_train_transformed, X_val_transformed = preprocess_part2(train_extended, test_extended, row_cov, col_cov,
                                                                      row_evs, s)

            X_train_flattened = np.array([np.array(matrix.flatten()) for matrix in X_train_transformed])
            X_val_flattened = np.array([np.array(matrix.flatten()) for matrix in X_val_transformed])

            sum_squared_differences_partial = partial(sum_squared_differences, num_row_evs=row_evs, num_col_evs=s)
            knn = KNeighborsClassifier(n_neighbors=k, metric=sum_squared_differences_partial)
            knn.fit(X_train_flattened, y_train)

            score = knn.score(X_val_flattened, y_val)
            r_scores.append(score)
        print(f'Average val accuracy: {np.mean(r_scores)}, k={k}, row_evs={row_evs}, col_evs={s}')
        if r_scores:
            row_evs_scores.append((row_evs, np.mean(r_scores)))
            cv_stats.append({'acc': np.mean(r_scores), 'k': k, 'row_evs': row_evs, 'col_evs': s})
            if np.mean(r_scores) > best_r_score:
                best_r = row_evs
                best_r_score = np.mean(r_scores)

    for col_evs in range(1, max1_col_evs + 1):
        s_scores = []

        for train_index, val_index in kf.split(train_inputs, train_labels_int):
            y_train, y_val = train_labels_int[train_index], train_labels_int[val_index]

            X_train = [train_inputs[i] for i in train_index]
            X_val = [train_inputs[i] for i in val_index]

            train_extended, test_extended, row_cov, col_cov, max_train_length = preprocess_part1(X_train, X_val)
            r = compute_eigenvectors(col_cov)  # number of row eigenvectors
            X_train_transformed, X_val_transformed = preprocess_part2(train_extended, test_extended, row_cov, col_cov,
                                                                      r, col_evs)

            X_train_flattened = np.array([np.array(matrix.flatten()) for matrix in X_train_transformed])
            X_val_flattened = np.array([np.array(matrix.flatten()) for matrix in X_val_transformed])

            sum_squared_differences_partial = partial(sum_squared_differences, num_row_evs=r, num_col_evs=col_evs)
            knn = KNeighborsClassifier(n_neighbors=k, metric=sum_squared_differences_partial)
            knn.fit(X_train_flattened, y_train)

            score = knn.score(X_val_flattened, y_val)
            s_scores.append(score)
        print(f'Average val accuracy: {np.mean(s_scores)}, k={k}, row_evs={r}, col_evs={col_evs}')
        if s_scores:
            col_evs_scores.append((col_evs, np.mean(s_scores)))
            cv_stats.append({'acc': np.mean(s_scores), 'k': k, 'row_evs': r, 'col_evs': col_evs})
            if np.mean(s_scores) > best_s_score:
                best_s = col_evs
                best_s_score = np.mean(s_scores)

    print(f'Best r: {best_r} with accuracy: {best_r_score}')
    print(f'Best s: {best_s} with accuracy: {best_s_score}')

    return find_best_params(cv_stats)

'''For plotting
    matplotlib.use('TkAgg')

    if row_evs_scores:
        row_evs, row_scores = zip(*row_evs_scores)
        plt.plot(row_evs, row_scores, label="Row Evs vs Accuracy", marker="o")
        plt.xlabel("Row Eigenvectors")
        plt.ylabel("Validation Accuracy")
        plt.title("Row Eigenvectors vs Accuracy")
        plt.legend()
        plt.savefig("figures/row_eigenvectors_vs_accuracy.png")
        plt.show()

    if col_evs_scores:
        col_evs, col_scores = zip(*col_evs_scores)
        plt.plot(col_evs, col_scores, label="Col Evs vs Accuracy", marker="o")
        plt.xlabel("Column Eigenvectors")
        plt.ylabel("Validation Accuracy")
        plt.title("Column Eigenvectors vs Accuracy")
        plt.legend()
        plt.savefig("figures/col_eigenvectors_vs_accuracy.png")
        plt.show()
'''

def best_params_2dsvd(k, row_evs, col_evs):
    train_inputs, test_inputs, train_labels, test_labels = read_data()
    train_labels_int = np.argmax(train_labels, axis=1)
    test_labels_int  = np.argmax(test_labels, axis=1)

    sum_squared_differences_partial = partial(sum_squared_differences, num_row_evs=row_evs, num_col_evs=col_evs)

    # (270, 5, 10)       (370, 5, 10)
    X_train_transformed, X_test_transformed = preprocess_data(train_inputs, test_inputs, row_evs, col_evs)


    X_train_flattened = np.array([np.array(matrix.flatten()) for matrix in X_train_transformed])
    X_test_flattened = np.array([np.array(matrix.flatten()) for matrix in X_test_transformed])

    knn = KNeighborsClassifier(n_neighbors=k, metric=sum_squared_differences_partial)
    knn.fit(X_train_flattened, train_labels_int)

    predictions = knn.predict(X_test_flattened)
    
    acc = accuracy_score(test_labels_int, predictions)
    f1 = f1_score(test_labels_int, predictions, average="weighted")  # "weighted" to account for class imbalance
    
    # Print metrics
    print(f'Test accuracy: {acc*100}%, Test error rate {(1 - acc)*100:.2f}%, F1-score: {f1:2f}, k={k}, row_evs={row_evs}, col_evs={col_evs}')
    
    # Confusion matrix
    cm = confusion_matrix(test_labels_int, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(test_labels_int))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for a KNN model\nwith 2DSVD used for preprocessing')
    plt.savefig('figures/confusion_matrix_KNN_2dsvd_best_params.pdf')
    plt.show()



def knn_hyperparams():
    train_inputs, test_inputs, train_labels, test_labels = read_data()
    train_labels_int = np.argmax(train_labels, axis=1)
    test_labels_int  = np.argmax(test_labels, axis=1)
    cv_stats = []
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

                    predictions = knn.predict(X_val_flattened)
    
                    f1 = f1_score(y_val, predictions, average="weighted")  # "weighted" to account for class imbalance
                    
                    scores.append(f1)
                print(f'Average val f1: {np.mean(scores):.4f}, k={k}, row_evs={row_evs}, col_evs={col_evs}')
                cv_stats.append({'acc': np.mean(scores), 'k': k, 'row_evs': row_evs, 'col_evs': col_evs})
    print("Best params for KNN:")
    print(find_best_params(cv_stats))