import pytest
# DONE: add necessary import
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

# DONE: implement the first test. Change the function name and input as needed
def test_train_model_returns_correct_type():
    """
    Test that the train_model function returns a RandomForestClassifier instance.
    """
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(2, size=100)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "train_model did not return a RandomForestClassifier."


# DONE: implement the second test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns the correct precision, recall, and F1 for known inputs.
    """
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    assert precision == 1.0, "Precision is not as expected."
    assert recall == 0.6666666666666666, "Recall is not as expected."
    assert f1 == 0.8, "F1 score is not as expected."



# DONE: implement the third test. Change the function name and input as needed
def test_process_data():
    """
    Test that process_data correctly processes data and returns expected shapes.
    """
    import pandas as pd

    data = pd.DataFrame({
        "feature1": ["A", "B", "A", "C"],
        "feature2": [1, 2, 3, 4],
        "salary": [">50K", "<=50K", ">50K", "<=50K"]
    })
    cat_features = ["feature1"]

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert X.shape == (4, len(encoder.get_feature_names_out(cat_features)) + 1), "Processed X has incorrect shape."
    assert y.shape == (4,), "Processed y has incorrect shape."


