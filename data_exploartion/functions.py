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


    # BIC
    n = len(y_true)  # Number of samples
    k = x.shape[1]  # Number of features
    BIC = -2 * log_likelihood + k * np.log(n)

    return {'log_likelihood': round(log_likelihood,3), 'accuracy': round(accuracy,3), "BIC": round(BIC,3)}



