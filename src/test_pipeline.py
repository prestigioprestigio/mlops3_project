import numpy as np
import pytest
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ml.data import process_data
import ml.model as m

@pytest.fixture
def mock_data():
    """ Simple function to generate some fake Pandas data."""
    X = pd.DataFrame({'gender': ['m', 'f', 'f', 'm', 'f'], 
                      'age': [10, 15, 25, 40, 60], 
                      'stage': ['youth', 'youth', 'adult', 'adult', 'adult']
                     })
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X['age'].to_frame())
    X_true = pd.DataFrame({'gender': [0, 1, 1, 0, 1], 
                      'age': scaled_data.ravel(),  
                     })
    y_true = [1, 1, 0, 0, 0]
    return X, X_true, y_true


@pytest.fixture
def model():
    return joblib.load('../models/lr.pkl')


@pytest.fixture
def eval_set():
    eval_set = pd.read_csv('../data/census_ref_eval.csv')
    encoder = joblib.load('../artifacts/encoder.pkl')
    lb = joblib.load('../artifacts/label_binarizer.pkl')
    scaler = joblib.load('../artifacts/scaler.pkl')

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    X_eval, y_eval, _, _, _ = process_data(
        eval_set, categorical_features=cat_features, label="salary", training=False, 
        encoder=encoder, lb=lb, scaler=scaler
        )
    return X_eval, y_eval


def test_process_data(mock_data):
    
    X, X_true, y_true = mock_data
    X_processed, y_processed, _, _, _= process_data(
        X, categorical_features=['gender'], label="stage", training=True
    )
    assert (y_true == y_processed).all()
    assert np.isclose(X_processed[:, 0], X_true['age'], rtol=0.1, atol=0.1).all()
    assert (X_processed[:, 1] == X_true['gender']).all()
    
    
def test_performance(eval_set, model):
    """
    Tests performance against a reference evaluation set
    
    Loads classifier, artifacts, and performs inference
    on an evaluation set and validating that the peforamance
    os adequate
    """
    clf = model
    X_eval, y_eval = eval_set
    y_pred = m.inference(clf, X_eval)
    precision, recall, fbeta, accuracy = m.compute_model_metrics(y_eval, y_pred)
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"Fbeta: {round(fbeta, 2)}")
    print(f"Accuracy: {round(accuracy, 2)}")
    assert accuracy > 0.8
    assert precision > 0.7
    
    
def test_fitted(eval_set, model):
    """
    Tests that the model saved during training is fitted
    """
    from sklearn.exceptions import NotFittedError
    
    clf = model
    X_eval, y_eval = eval_set
    
    try:
        clf.predict(X_eval)
    except NotFittedError as e:
        raise(e)

    

    
