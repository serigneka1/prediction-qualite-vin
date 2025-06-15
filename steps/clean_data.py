from src.clean_data import PreprocessData, DivideData, DataCleaning
from get_data import get_data

import pandas as pd



def clean_data(data:pd.DataFrame):
    data = get_data()

    preprocess = PreprocessData()
    division = DivideData()
    preprocess_data  = DataCleaning(preprocess, data)
    X_train_scaled, X_test_scaled, y_train, y_test = DataCleaning(division, preprocess_data)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

