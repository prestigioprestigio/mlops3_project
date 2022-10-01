# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import joblib
from ml.data import process_data
import ml.model as m

# Add code to load in the data.
data = pd.read_csv("../data/census_cleaned.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb, scaler= process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# joblib.dump(encoder, '../artifacts/encoder.pkl')
# joblib.dump(lb, '../artifacts/label_binarizer.pkl')
# joblib.dump(scaler, '../artifacts/scaler.pkl')

# Proces the test data with the process_data function.
X_test, y_test, _, _, _= process_data(
    test, categorical_features=cat_features, label="salary", training=False, 
    encoder=encoder, lb=lb, scaler=scaler
)


# Train and save a model.
clf = m.train_model(X_train, y_train)
y_pred = m.inference(clf, X_test)
precision, recall, fbeta, accuracy = m.compute_model_metrics(y_test, y_pred)
print(f"Precision: {round(precision, 2)}")
print(f"Recall: {round(recall, 2)}")
print(f"Fbeta: {round(fbeta, 2)}")
print(f"Accuracy: {round(accuracy, 2)}")

# Slice performance for categorical features
m.slice_perf(test, clf, features=cat_features, encoder=encoder, lb=lb, 
             scaler=scaler, categorical=True)

# joblib.dump(clf, '../models/lr.pkl') 