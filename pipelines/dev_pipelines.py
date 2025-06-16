import pandas as pd

from steps.get_data import get_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

def pipeline(data: pd.DataFrame):
     data = get_data()
     X_train_scaled, X_test_scaled, y_train, y_test = clean_data()
     model = train_model(X_train_scaled)
     evaluate_model(model, X_test_scaled, y_train)
     


