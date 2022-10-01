import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, max_iter=1000):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    lr = LogisticRegression(max_iter=max_iter)
    model = lr.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    accuracy = accuracy_score(y, preds)
    return precision, recall, fbeta, accuracy


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def slice_perf(data, model, encoder, lb, scaler, features: list, categorical: bool):
    """
    Outputs the performance of the model on slices of the data.
    Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
    
    args:
        data (dataframe): evaluation data raw, will be processed
        model (sklearn model): machine learning model to evaluate
        encoder (sklearn.preprocessing._encoders.OneHotEncoder): Trained sklearn OneHotEncoder, only used if training=False.          
        lb (sklearn.preprocessing._label.LabelBinarizer): Trained sklearn LabelBinarizer, only used if training=False.
        scaler (sklearn.preprocessing.StandardScaler): numerical value scaler
        features (list): list of features to evaluate on
        categorical (boolean): whether the features arg is categorical
    returns:
        None
    """
    
    X_eval, y_eval, _, _, _= process_data(
        data, categorical_features=features, label="salary", training=False, 
        encoder=encoder, lb=lb, scaler=scaler
    )
    y_pred = model.predict(X_eval)
    with open('../slice_output.txt', 'w') as f:
        for feat in features:
            f.write('-'*20 + '\n' + feat + '\n' + '-'*20 +  '\n')
            if categorical:
                for cat in data[feat].unique():
                    idx = data[feat] == cat
                    accuracy = np.mean(y_pred[idx] == y_eval[idx])
                    f.write(f"Performance on slice {feat}:{cat} is {round(accuracy, 3)}\n")
            f.write('\n')