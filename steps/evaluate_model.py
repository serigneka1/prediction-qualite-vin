import pandas as pd

from sklearn.metrics import accuracy_score

from steps.clean_data import clean_data
from src.evaluate_model import Accuracy
from steps.train_model import train_model



def evaluate_model():
    X_train_scaled, X_test_scaled, y_train, y_test = clean_data()
    model = train_model()
    y_pred = model.predict(X_test_scaled)
    accuracy = Accuracy.calculate_metric(model, X_test_scaled, y_pred, y_test)
    
    return accuracy




    





