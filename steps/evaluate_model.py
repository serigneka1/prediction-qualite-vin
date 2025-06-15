import pandas as pd

from sklearn.metrics import accuracy_score

from src.evaluate_model import ValidateModel
from steps.clean_data import clean_data
from steps.train_model import train_model



def evaluate_model():

    X_train_scaled, X_test_scaled, y_train, y_test = clean_data()
    
    model = train_model()
    y_true = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_true)

    return accuracy




