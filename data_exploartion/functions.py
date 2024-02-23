import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from scipy.stats import lognorm

# Assuming model, X_selected_standardized, y_train, and cv are defined

# Step 1: Define your custom scoring function for log likelihood
def get_score(estimator, x, y_true):
    eps = 1e-15  # To prevent log(0)

    # Fit the estimator
    estimator.fit(x, y_true)

    # Predict probabilities
    y_pred_proba = estimator.predict_proba(x)

    # Clip probabilities to avoid log(0) and ensure they sum to 1 across classes for each sample
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)

    ## Log Likelihood

    # Convert y_true class names to indices
    classes = estimator.classes_  # This assumes estimator is a classifier with the classes_ attribute
    y_true_indices = [np.where(classes == class_name)[0][0] for class_name in y_true]

    # One-hot encode y_true based on class indices
    y_true_one_hot = pd.get_dummies(y_true_indices).values

    # Calculate the log likelihood
    log_likelihood = np.sum(y_true_one_hot * np.log(y_pred_proba))


    ## Accuracy

    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracy = np.mean(y_true_indices == y_pred)

    return {'log_likelihood': log_likelihood, 'accuracy': accuracy}


def get_next_feature_with_BIC(model, X, y, features, cv):
    current_features = []
    # Initialize best BIC
    best_BIC = np.inf
    best_score = 0
    best_feature = None

    if len(features) == 1:
        return features[0]

    # Loop through the remaining features
    for feature in features:
        # Check if feature is already included
        if feature in current_features:
            continue

        # Add the feature to the current features
        current_features.append(feature)

        #get indicis of features
        feature_indices = [i for i, f in enumerate(features) if f in current_features]

        # Fit the model
        scores = cross_validate(model, X[feature_indices], y, cv=cv, scoring=make_scorer(get_score), return_train_score=False)

        # Calculate the BIC
        n = len(y)  # Number of samples
        p = len(current_features)  # Number of features
        log_likelihood = np.mean(scores['test_log_likelihood'])
        BIC = -2 * log_likelihood + p * np.log(n)

        # Check if BIC is the best so far
        if BIC < best_BIC:
            best_BIC = BIC
            best_feature = feature

        # Remove the feature from the current features
        current_features.remove(feature)

    return best_feature


