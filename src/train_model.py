from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin



class TrainModel(ABC):
    @abstractmethod
    def train_model(self, X, y) -> RegressorMixin:
        pass

class TrainLinearRegression(TrainModel):
    def train_model(self, X, y) -> RegressorMixin:
        model = LinearRegression()

        model.fit(X, y)

        return model