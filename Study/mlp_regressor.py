from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

housing_data = fetch_california_housing()

x_train_full, x_test, y_train_full, y_test = train_test_split(housing_data.data, housing_data.target, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, random_state=42)

mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)

pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(x_train, y_train)

print(f"score : {pipeline.score(x_train, y_train)}")

y_pred = pipeline.predict(x_valid)
print(f"y_pred : {y_pred}")

rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f"rmse : {rmse}")