from read_data import read_data
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

def extract_features(sequence):
    '''
    This function takes a sequence of shape (n, d) and returns a feature vector of shape (3*d,).
    The feature vector contains the mean, standard deviation, and range of each feature in the sequence.
    '''
    means = np.mean(sequence, axis=0)       
    stds = np.std(sequence, axis=0)         
    ranges = np.ptp(sequence, axis=0)       
    
    features = np.concatenate([means, stds, ranges], axis=0) 
    return features


def build_feature_matrix(X_inputs):
    '''
    This function takes a list of sequences and returns a feature matrix.
    '''
    feature_matrix = [extract_features(seq) for seq in X_inputs]
    return np.vstack(feature_matrix)


def best_performer(scores):
    '''
    This function takes a list of (k, score) tuples and returns the best k value and the best score.
    '''
    best_k = None
    best_score = 0.0

    for (k_i, score_i) in scores:
        if score_i > best_score:
            best_k = k_i
            best_score = score_i
    
    return best_k, best_score


def perform_baseline_model():
    '''
    This function performs the baseline model for the given dataset.
    It uses K-Nearest Neighbors (KNN) classifier with 5-fold cross-validation.
    The function returns the best k value, the average validation score for the best k value,
    and the test score for the best k value.
    '''
    train_inputs, test_inputs, train_labels, test_labels = read_data()

    train_labels_int = np.argmax(train_labels, axis=1)  # shape (270,)
    test_labels_int  = np.argmax(test_labels, axis=1)  # shape (270,)

    feature_matrix = build_feature_matrix(train_inputs)
    print(f'Feature matrix shape: {feature_matrix.shape}')
    scaler = StandardScaler()
    train_features = scaler.fit_transform(feature_matrix)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    final_scores = []
    for k in [1, 3, 5, 7]:
        scores = []
        print(f'k={k}')
        for train_index, val_index in kf.split(train_features, train_labels_int):
            X_train, X_val = train_features[train_index], train_features[val_index]
            y_train, y_val = train_labels_int[train_index], train_labels_int[val_index]

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            score = knn.score(X_val, y_val)
            scores.append(score)
            print(f'Validation score: {score}')
        print(f'Average score: {np.mean(scores)}')
        final_scores.append((k, np.mean(scores)))

    best_k, best_score = best_performer(final_scores)

    # Train the best model on the test set
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(train_features, train_labels_int)

    test_feature_matrix = build_feature_matrix(test_inputs)
    test_features = scaler.transform(test_feature_matrix)
    
    predictions = best_knn.predict(test_features)
    f1 = f1_score(test_labels_int, predictions, average='weighted') 
    print(f'Best model F1 score: {f1}')
    test_score = best_knn.score(test_features, test_labels_int)
    # print(f'Best k: {best_k}, Validation score: {best_score}, Test score: {test_score}')

    return best_k, best_score, test_score
