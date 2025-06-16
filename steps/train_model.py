import logging

from src.train_model import TrainLinearRegression
from steps.clean_data import clean_data



def train_model():
    try: 
        X_train_scaled, X_test_scaled, y_train, y_test = clean_data()
        linear_model = TrainLinearRegression()
        model = linear_model.train_model(X_train_scaled, y_train)
        logging.inf("Le model est bien entrainé...")
        return model
    except Exception as e:
        logging.error("L'entrainement du model a échoué")
        raise e