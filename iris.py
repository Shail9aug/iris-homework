# iris.py
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data():
    iris = load_iris()
    return iris.data, iris.target

def validate_data(data):
    # Reject if not numpy ndarray
    if not isinstance(data, np.ndarray):
        return False
    # Check if 2D array and 4 features (columns)
    if data.ndim != 2 or data.shape[1] != 4:
        return False
    # Check that dtype is numeric
    if not np.issubdtype(data.dtype, np.number):
        return False
    return True

def evaluate_model(sample=None):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return {
        "accuracy": round(accuracy, 2),
        "model": model.__class__.__name__
    }


