import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1],[2],[3],[4]])
y = np.array([1,4,6,9])

def test_coefficients():
    model = LinearRegression()
    model.fit(X, y)
    assert model.coef_[0][0] == 2.2

def test_intercept():
    model = LinearRegression()
    model.fit(X, y)
    assert model.intercept_[0] == 0.0

def test_predictions():
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    assert round(pred[0],2) ==10.0