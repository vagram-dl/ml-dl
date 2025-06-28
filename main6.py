# Линейная регрессия

import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.linear_model import Ridge
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pytest

class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True, alpha=0.0):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.weights = None

    def _add_intercept(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.fit_intercept:
            X = self._add_intercept(X)

        I = np.eye(X.shape[1])
        if self.fit_intercept:
            I[0, 0] = 0
        self.weights = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y
        return self

    def predict(self, X):
        X = check_array(X)
        if self.fit_intercept:
            X = self._add_intercept(X)
        return X @ self.weights

    def score(self,X,y):
        y_pred=self.predict(X)
        ss_res=np.sum((y-y_pred) ** 2)
        ss_tot=np.sum((y-np.mean(y)) ** 2)
        return 1-(ss_res/ss_tot)

    def test_with_sklearn(self):
        X,y=fetch_california_housing(return_X_y=True)
        X=StandardScaler().fit_transform(X)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

        our_model=LinearRegression(alpha=0.1,fit_intercept=True)
        our_model.fit(X_train,y_train)
        our_r2=our_model.score(X_test,y_test)

        sklearn_model=Ridge(alpha=0.1,fit_intercept=True)
        sklearn_model.fit(X_train,y_train)
        sklearn_r2=sklearn_model.score(X_test,y_test)

        assert np.abs(our_r2-sklearn_r2) < 1e-6

    def example_visualization():
        np.random.seed(42)
        X = np.random.rand(100, 1) * 10
        y = 3 + 2 * X[:, 0] + np.random.randn(100) * 2

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression(fit_intercept=True, alpha=0.1)
        model.fit(X_train, y_train)

        print(f"Веса модели: w0 = {model.weights[0]:.2f}, w1 = {model.weights[1]:.2f}")
        print(f"R² на тесте: {model.score(X_test, y_test):.3f}")

        plt.scatter(X_test, y_test, label="Тестовые данные", alpha=0.7)
        plt.plot(X_test, model.predict(X_test), color='red', label="Предсказания")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Линейная регрессия")
        plt.legend()
        plt.grid()
        plt.show()

    if __name__ == "__main__":
        example_visualization()
        test_with_sklearn()
        print("Все тесты пройдены!")