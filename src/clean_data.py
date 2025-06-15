from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression


class CleaningStrategy(ABC):

    @abstractmethod
    def handle_data(self, data):
        pass

class PreprocessData(CleaningStrategy):

    def handle_data(self, data):


        return data
    
class DivideData(CleaningStrategy):
    def handle_data(self, data:pd.data):
        # Sépration des données 
        X = data.drop(columns = ["quality"], axis = 1)
        y = data["quality"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

        # Selection de features
        selector = SelectKBest(f_regression, k=5)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        # Normalisation des données
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)


        return X_train_scaled, X_test_scaled, y_train, y_test
    
class DataCleaning:
    def __init__(self, strategy:CleaningStrategy, data:pd.DataFrame):
        self.data = data
        self.strategy = strategy
    
    def handle_data(self):
        return self.strategy.handle_data()
    



