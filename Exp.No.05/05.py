import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
X = np.linspace(0, 10, 120).reshape(-1, 1)
y = 3 + 2*X.ravel() + 1.2*X.ravel()**2 + rng.normal(0, 8, 120)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

models = {
    "Linear": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Polynomial": make_pipeline(
        PolynomialFeatures(12), LinearRegression()
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_error = mean_squared_error(y_train, model.predict(X_train))
    test_error = mean_squared_error(y_test, model.predict(X_test))

    print(name, "Train MSE:", round(train_error, 2),
          "Test MSE:", round(test_error, 2))

plt.bar(
    list(models.keys()),
    [mean_squared_error(y_test, m.fit(X_train, y_train).predict(X_test))
     for m in models.values()]
)
plt.ylabel("Test Mean Squared Error")
plt.title("Model Comparison")
plt.show()