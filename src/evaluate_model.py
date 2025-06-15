import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score



class ValidateModel(ABC):
    @abstractmethod
    def calculate_metric(model, X_test, y_test, y_pred):
        pass

class Accuracy(ValidateModel):
    def calculate_metric(model, X, y_pred, y_true):
        y_pred = model.predict(X)

        accuracy = accuracy_score(y_true, y_pred)

        return accuracy
